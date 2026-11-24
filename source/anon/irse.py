"""Réseaux IR-SE50 : ArcFace (Deng et al., 2019) et encodeur e4e (Tov et al., 2021), compatibles avec leurs poids publiés."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from stylegan2 import EqualLinear


class SEModule(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.fc1 = nn.Conv2d(ch, ch // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(ch // reduction, ch, 1, bias=False)

    def forward(self, x):
        return x * torch.sigmoid(self.fc2(F.relu(self.fc1(F.adaptive_avg_pool2d(x, 1)))))


class BottleneckIRSE(nn.Module):
    def __init__(self, in_ch, depth, stride):
        super().__init__()
        self.shortcut_layer = nn.MaxPool2d(1, stride) if in_ch == depth else nn.Sequential(nn.Conv2d(in_ch, depth, 1, stride, bias=False), nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(nn.BatchNorm2d(in_ch), nn.Conv2d(in_ch, depth, 3, 1, 1, bias=False), nn.PReLU(depth), nn.Conv2d(depth, depth, 3, stride, 1, bias=False), nn.BatchNorm2d(depth), SEModule(depth))

    def forward(self, x):
        return self.shortcut_layer(x) + self.res_layer(x)


def input_layer():
    return nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU(64))


def ir_se50_body():
    """24 blocs résiduels en 4 étages (3, 4, 14, 3), chacun divisant la résolution par 2."""
    units = []
    for in_ch, depth, n in [(64, 64, 3), (64, 128, 4), (128, 256, 14), (256, 512, 3)]:
        units += [BottleneckIRSE(in_ch, depth, 2)] + [BottleneckIRSE(depth, depth, 1) for _ in range(n - 1)]
    return nn.Sequential(*units)


class ArcFace(nn.Module):
    """Visage 112×112 dans [-1, 1] -> embedding d'identité (B, 512) normalisé L2."""
    input_size = 112

    def __init__(self, dropout=0.6):
        super().__init__()
        self.input_layer = input_layer()
        self.body = ir_se50_body()
        self.output_layer = nn.Sequential(nn.BatchNorm2d(512), nn.Dropout(dropout), nn.Flatten(), nn.Linear(512 * 7 * 7, 512), nn.BatchNorm1d(512))

    def forward(self, x):
        return F.normalize(self.output_layer(self.body(self.input_layer(x))), dim=1)


class GradualStyleBlock(nn.Module):
    """Convolutions de pas 2 jusqu'à une carte 1×1, puis une couche linéaire : un vecteur de style."""
    def __init__(self, ch, spatial):
        super().__init__()
        layers = []
        for _ in range(int(math.log2(spatial))):
            layers += [nn.Conv2d(ch, ch, 3, 2, 1), nn.LeakyReLU()]
        self.convs = nn.Sequential(*layers)
        self.linear = EqualLinear(ch, ch)

    def forward(self, x):
        return self.linear(self.convs(x).view(x.shape[0], -1))


def upsample_add(x, y):
    return F.interpolate(x, size=y.shape[2:], mode="bilinear", align_corners=True) + y


class E4EEncoder(nn.Module):
    """Visage aligné 256×256 dans [-1, 1] -> latent W+ (B, n_styles, 512) : un code de base w0 plus un écart par couche.

    Les couches grossières (0-2) lisent la carte 16×16, les moyennes (3-6) la 32×32, les fines (7+) la 64×64 (pyramide de caractéristiques).
    """
    def __init__(self, n_styles=14):
        super().__init__()
        self.input_layer = input_layer()
        self.body = ir_se50_body()
        self.styles = nn.ModuleList(GradualStyleBlock(512, 16 if i < 3 else 32 if i < 7 else 64) for i in range(n_styles))
        self.latlayer1 = nn.Conv2d(256, 512, 1)
        self.latlayer2 = nn.Conv2d(128, 512, 1)
        self.latent_avg = None

    def forward(self, x):
        x = self.input_layer(x)
        for i, unit in enumerate(self.body):
            x = unit(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
        c3 = x
        p2 = upsample_add(c3, self.latlayer1(c2))
        p1 = upsample_add(p2, self.latlayer2(c1))
        w0 = self.styles[0](c3)
        deltas = [self.styles[i](c3 if i < 3 else p2 if i < 7 else p1) for i in range(1, len(self.styles))]
        w = w0.unsqueeze(1) + torch.cat([torch.zeros_like(w0).unsqueeze(1), torch.stack(deltas, 1)], 1)
        return w if self.latent_avg is None else w + self.latent_avg
