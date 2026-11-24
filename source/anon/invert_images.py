"""Prépare des photos quelconques : détection, alignement FFHQ et inversion e4e.

Pour chaque visage détecté dans chaque photo de --input :
  1. détection + 68 landmarks (dlib),
  2. alignement/recadrage façon FFHQ, en 256x256,
  3. inversion dans W+ (14x512) avec l'encodeur e4e.

Sorties dans --out :
  images/0000.png, 0001.png, ...  visages alignés
  latents.pt                      tenseur (N, 14, 512), même ordre que les images
  index.txt                       correspondance index -> photo source / n° de visage

    python source/anon/invert_images.py --input <photos> --out <dir>
    python source/anon/anonymize.py encrypt --ckpt <ckpt.pt> --latents <dir>/latents.pt --seed 1234 --out <out>
"""
import argparse
import os

import dlib
import numpy as np
import scipy.ndimage
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps

import pretrained as pre

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def align_face(img, lm, output_size=256, transform_size=1024, enable_padding=True):
    """Alignement FFHQ (NVlabs/ffhq-dataset) à partir des 68 landmarks dlib."""
    lm_eye_left, lm_eye_right, lm_mouth_outer = lm[36:42], lm[42:48], lm[48:60]
    eye_left, eye_right = lm_eye_left.mean(0), lm_eye_right.mean(0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_avg = (lm_mouth_outer[0] + lm_mouth_outer[6]) * 0.5
    eye_to_mouth = mouth_avg - eye_avg
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(img.size[0] / shrink)), int(np.rint(img.size[1] / shrink)))
        img = img.resize(rsize, Image.Resampling.LANCZOS)
        quad /= shrink
        qsize /= shrink
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        yy, xx, _ = np.ogrid[:h, :w, :1]
        with np.errstate(divide='ignore', invalid='ignore'):
            mask = np.maximum(1.0 - np.minimum(np.float32(xx) / pad[0], np.float32(w - 1 - xx) / pad[2]), 1.0 - np.minimum(np.float32(yy) / pad[1], np.float32(h - 1 - yy) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]
    img = img.transform((transform_size, transform_size), Image.Transform.QUAD, (quad + 0.5).flatten(), Image.Resampling.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.Resampling.LANCZOS)
    return img


def detect_landmarks(img, detector, predictor, max_side=1600):
    """Renvoie une liste de tableaux (68, 2) en coordonnées de l'image d'origine."""
    scale = min(1.0, max_side / max(img.size))
    small = img.resize((int(img.size[0] * scale), int(img.size[1] * scale))) if scale < 1 else img
    arr = np.array(img)
    faces = detector(np.array(small), 1)
    out = []
    for r in faces:
        rect = dlib.rectangle(int(r.left() / scale), int(r.top() / scale), int(r.right() / scale), int(r.bottom() / scale))
        shape = predictor(arr, rect)
        out.append(np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float64))
    return out


def align_photos(input_dir, max_faces=0):
    """Photos d'un dossier -> liste de visages alignés (PIL 256x256) et leurs noms « photo<TAB>visage k »."""
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(pre.WEIGHTS / "shape_predictor_68_face_landmarks.dat"))
    aligned, index = [], []
    for name in sorted(os.listdir(input_dir)):
        if not name.lower().endswith(IMG_EXT):
            continue
        img = ImageOps.exif_transpose(Image.open(os.path.join(input_dir, name))).convert("RGB")
        lms = detect_landmarks(img, detector, predictor)
        if max_faces > 0:
            lms = lms[:max_faces]
        if not lms:
            print(f"[!] aucun visage détecté : {name}")
        for k, lm in enumerate(lms):
            aligned.append(align_face(img, lm))
            index.append(f"{name}\tvisage {k}")
        print(f"{name} : {len(lms)} visage(s)")
    return aligned, index


@torch.no_grad()
def invert(faces, device="cuda"):
    """Visages alignés (PIL) -> latents W+ (N, 14, 512)."""
    enc = pre.load_e4e(device)
    to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)])
    return torch.cat([enc(to_tensor(f).unsqueeze(0).to(device)) for f in faces])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="dossier de photos")
    ap.add_argument("--out", required=True, help="dossier de sortie")
    ap.add_argument("--max_faces", default=0, type=int, help="max visages par photo (0 = tous)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    aligned, index = align_photos(args.input, args.max_faces)
    if not aligned:
        raise SystemExit("Aucun visage trouvé, rien à inverser.")
    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    for i, face in enumerate(aligned):
        face.save(os.path.join(img_dir, f"{i:04d}.png"))
    latents = invert(aligned, args.device).cpu()
    torch.save(latents, os.path.join(args.out, "latents.pt"))
    with open(os.path.join(args.out, "index.txt"), "w") as f:
        f.writelines(f"{i:04d}\t{s}\n" for i, s in enumerate(index))
    print(f"{len(aligned)} visage(s) -> {img_dir}/ et {args.out}/latents.pt {tuple(latents.shape)}")


if __name__ == "__main__":
    main()
