import argparse
import pathlib

import torch

import pretrained as pre
from encryptor import LatentEncryptor


def load_key(G, args, n):
    if args.key:
        k = torch.load(args.key, map_location="cuda")
        k = k.view(1, 1, 512).repeat(1, pre.N_LAYERS, 1) if k.numel() == 512 else k.view(1, pre.N_LAYERS, 512)
        return k.expand(n, -1, -1)
    return pre.password_from_seed(G, args.seed, n)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["encrypt", "decrypt"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--latents", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--key", default=None)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    (out / args.mode).mkdir(parents=True, exist_ok=True)
    G = pre.load_generator()
    E = LatentEncryptor.from_checkpoint(args.ckpt)
    w = torch.load(args.latents, map_location="cuda")
    p = load_key(G, args, len(w))
    w_out = E(w, p)
    for i in range(len(w)):
        pre.save_grid(pre.render(G, w_out[i:i + 1]), out / args.mode / f"{i:04d}.png", nrow=1)
    if args.mode == "encrypt":
        torch.save(w_out.cpu(), out / "latents_enc.pt")
        torch.save(p[0, 0].cpu(), out / "key.pt")
        pre.save_grid(torch.cat([pre.render(G, w), pre.render(G, w_out)]), out / "grid_encrypt.png", nrow=len(w))
        print(f"{len(w)} visage(s) chiffré(s) -> {out}/encrypt/, {out}/latents_enc.pt, clé {out}/key.pt")
    else:
        torch.save(w_out.cpu(), out / "latents_dec.pt")
        pre.save_grid(torch.cat([pre.render(G, w), pre.render(G, w_out)]), out / "grid_decrypt.png", nrow=len(w))
        print(f"{len(w)} visage(s) déchiffré(s) -> {out}/decrypt/, {out}/latents_dec.pt")


if __name__ == "__main__":
    main()
