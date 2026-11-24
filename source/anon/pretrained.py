"""Réseaux pré-entraînés et figés : StyleGAN2 (générateur et discriminateur), ArcFace, FaceNet, LPIPS, e4e ; clés et rendu.

Rien ici n'est entraîné. Seuls les fichiers de poids sont lus, dans WEIGHTS ; le code des réseaux est dans
stylegan2.py, irse.py et les paquets `lpips` et `facenet-pytorch`.
ArcFace sert à l'évaluation, FaceNet aux pertes d'identité de l'entraînement.
"""
import pathlib

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from facenet_pytorch import InceptionResnetV1

from irse import ArcFace, E4EEncoder
from stylegan2 import Discriminator, Generator

ROOT = pathlib.Path(__file__).resolve().parents[2]          # racine du projet
DATA = ROOT / "weights" if (ROOT / "weights").exists() else ROOT / "RiDDLE"
WEIGHTS = DATA / "pretrained_models"
LATENTS = DATA / "embeddings" / "invert_w_256.pt"
N_LAYERS = 14            # StyleGAN2 256 px : nombre d'entrées de style


def _frozen(net):
    net.eval().requires_grad_(False)
    return net


def load_generator(device="cuda"):
    G = Generator(256)
    G.load_state_dict(torch.load(WEIGHTS / "stylegan2-ffhq-256.pt", map_location="cpu")["g_ema"])
    return _frozen(G.to(device))


def load_discriminator(device="cuda"):
    """Discriminateur StyleGAN2 entraîné avec le générateur : score de réalisme d'une image (haut = réaliste)."""
    D = Discriminator(256)
    D.load_state_dict(torch.load(WEIGHTS / "stylegan2-ffhq-256.pt", map_location="cpu")["d"])
    return _frozen(D.to(device))


def realism(D, x):
    """Score de réalisme de D sur l'image bornée à [-1, 1] (vrai bornage : gradient nul sur les pixels saturés).

    Sans bornage, D note bien des images hors domaine dont des pixels dépassent [-1, 1] (taches saturées) : il faut
    lui montrer l'image telle qu'elle sera enregistrée. Le bornage « straight-through » d'exp4a_bis laissait D pousser
    les pixels saturés toujours plus loin sans voir le résultat, jusqu'à la divergence (results.md §10). Ce qui
    ramène les pixels sortis est la perte --lambda_range (losses.range_excess).
    """
    return D(x.clamp(-1, 1))


def load_arcface(device="cuda"):
    net = ArcFace()
    net.load_state_dict(torch.load(WEIGHTS / "model_ir_se50.pth", map_location="cpu"))
    return _frozen(net.to(device))


class FaceNet(nn.Module):
    """InceptionResnetV1 entraîné sur VGGFace2 : visage 160×160 dans [-1, 1] -> embedding d'identité (B, 512) normalisé L2."""
    input_size = 160

    def __init__(self):
        super().__init__()
        self.net = InceptionResnetV1()

    def forward(self, x):
        return self.net(x * 127.5 / 128)


def load_facenet(device="cuda"):
    net = FaceNet()
    net.net.load_state_dict(torch.load(WEIGHTS / "facenet_vggface2.pt", map_location="cpu"))
    return _frozen(net.to(device))


class LPIPS(nn.Module):
    """LPIPS moyen sur le batch, images dans [-1, 1]."""
    def __init__(self, net_type="vgg"):
        super().__init__()
        self.net = lpips.LPIPS(net=net_type, verbose=False)

    def forward(self, x, y):
        return self.net(x, y).mean()


def load_lpips(net_type="vgg", device="cuda"):
    return _frozen(LPIPS(net_type).to(device))


def load_e4e(device="cuda"):
    ckpt = torch.load(WEIGHTS / "e4e_ffhq_encode_256.pt", map_location="cpu", weights_only=False)
    enc = E4EEncoder(N_LAYERS)
    enc.load_state_dict({k[len("encoder."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("encoder.")})
    if ckpt["opts"].get("start_from_latent_avg"):
        enc.latent_avg = ckpt["latent_avg"].to(device)
    return _frozen(enc.to(device))


@torch.no_grad()
def sample_password(G, batch, generator=None):
    """Clé = latent W aléatoire (mapping de StyleGAN2), répété sur les 14 couches : (B, 14, 512)."""
    device = next(G.parameters()).device
    z = torch.randn(batch, 512, device=device, generator=generator)
    return G.style(z).unsqueeze(1).repeat(1, N_LAYERS, 1)


def password_from_seed(G, seed, batch=1):
    """Clé reproductible à partir d'un entier."""
    device = next(G.parameters()).device
    gen = torch.Generator(device).manual_seed(seed)
    return sample_password(G, 1, generator=gen).expand(batch, -1, -1)


def render(G, w):
    """Latent W+ (B, 14, 512) -> image (B, 3, 256, 256) dans [-1, 1]."""
    return G(w)


def id_embed(net, x):
    """Embedding d'identité normalisé L2, (B, 512), par ArcFace ou FaceNet. x : visages alignés FFHQ 256 px."""
    x = x[:, :, 35:223, 32:220]
    x = F.adaptive_avg_pool2d(x, net.input_size)
    return net(x)


def save_grid(images, path, nrow):
    torchvision.utils.save_image(images, path, nrow=nrow, normalize=True, value_range=(-1, 1))
