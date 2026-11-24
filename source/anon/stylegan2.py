"""Générateur et discriminateur StyleGAN2 (Karras et al., 2020), compatibles avec les poids `g_ema` et `d` au format rosinality.

PyTorch pur, sans opérations CUDA compilées. Le bruit de chaque couche est fixé par une graine.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NOISE_SEED = 0


def fused_leaky_relu(x, bias, slope=0.2, scale=2 ** 0.5):
    return F.leaky_relu(x + bias.view((1, -1) + (1,) * (x.dim() - 2)), slope) * scale


def upfirdn2d(x, kernel, up=1, pad=(0, 0)):
    """Suréchantillonnage par insertion de zéros, padding, puis filtre FIR appliqué canal par canal."""
    b, c, h, w = x.shape
    if up > 1:
        y = x.new_zeros(b, c, h * up, w * up)
        y[:, :, ::up, ::up] = x
        x = y
    x = F.pad(x, (pad[0], pad[1], pad[0], pad[1]))
    k = torch.flip(kernel, [0, 1])[None, None].expand(c, 1, *kernel.shape).to(x.dtype)
    return F.conv2d(x, k, groups=c)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    k = k[None, :] * k[:, None]
    return k / k.sum()


class Blur(nn.Module):
    def __init__(self, kernel, pad, factor=1):
        super().__init__()
        self.register_buffer("kernel", make_kernel(kernel) * factor ** 2)
        self.pad = pad

    def forward(self, x):
        return upfirdn2d(x, self.kernel, pad=self.pad)


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()
        self.factor = factor
        self.register_buffer("kernel", make_kernel(kernel) * factor ** 2)
        p = self.kernel.shape[0] - factor
        self.pad = ((p + 1) // 2 + factor - 1, p // 2)

    def forward(self, x):
        return upfirdn2d(x, self.kernel, up=self.factor, pad=self.pad)


class EqualLinear(nn.Module):
    """Couche linéaire à learning rate égalisé (poids mis à l'échelle à chaque passage)."""
    def __init__(self, in_dim, out_dim, bias_init=0.0, lr_mul=1.0, activation=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim) / lr_mul)
        self.bias = nn.Parameter(torch.full((out_dim,), float(bias_init)))
        self.scale, self.lr_mul, self.activation = lr_mul / math.sqrt(in_dim), lr_mul, activation

    def forward(self, x):
        out = F.linear(x, self.weight * self.scale)
        return fused_leaky_relu(out, self.bias * self.lr_mul) if self.activation else out + self.bias * self.lr_mul


class ModulatedConv2d(nn.Module):
    """Convolution dont les poids sont modulés par le style, puis démodulés."""
    def __init__(self, in_ch, out_ch, k, style_dim, demodulate=True, upsample=False, blur_kernel=(1, 3, 3, 1)):
        super().__init__()
        self.in_ch, self.out_ch, self.k, self.demodulate, self.upsample = in_ch, out_ch, k, demodulate, upsample
        if upsample:
            p = (len(blur_kernel) - 2) - (k - 1)
            self.blur = Blur(blur_kernel, pad=((p + 1) // 2 + 1, p // 2 + 1), factor=2)
        self.scale = 1 / math.sqrt(in_ch * k * k)
        self.padding = k // 2
        self.weight = nn.Parameter(torch.randn(1, out_ch, in_ch, k, k))
        self.modulation = EqualLinear(style_dim, in_ch, bias_init=1)

    def forward(self, x, style):
        b, c, h, w = x.shape
        weight = self.scale * self.weight * self.modulation(style).view(b, 1, c, 1, 1)
        if self.demodulate:
            weight = weight * torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8).view(b, self.out_ch, 1, 1, 1)
        x = x.reshape(1, b * c, h, w)
        if self.upsample:
            weight = weight.transpose(1, 2).reshape(b * c, self.out_ch, self.k, self.k)
            out = F.conv_transpose2d(x, weight, padding=0, stride=2, groups=b)
            return self.blur(out.view(b, self.out_ch, *out.shape[2:]))
        out = F.conv2d(x, weight.view(b * self.out_ch, c, self.k, self.k), padding=self.padding, groups=b)
        return out.view(b, self.out_ch, h, w)


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x, noise=None):
        if noise is None:
            noise = x.new_empty(x.shape[0], 1, *x.shape[2:]).normal_()
        return x + self.weight * noise


class FusedLeakyReLU(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(ch))

    def forward(self, x):
        return fused_leaky_relu(x, self.bias)


class StyledConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, style_dim, upsample=False, blur_kernel=(1, 3, 3, 1)):
        super().__init__()
        self.conv = ModulatedConv2d(in_ch, out_ch, k, style_dim, upsample=upsample, blur_kernel=blur_kernel)
        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_ch)

    def forward(self, x, style, noise=None):
        return self.activate(self.noise(self.conv(x, style), noise))


class ToRGB(nn.Module):
    def __init__(self, in_ch, style_dim, upsample=True, blur_kernel=(1, 3, 3, 1)):
        super().__init__()
        if upsample:
            self.upsample = Upsample(blur_kernel)
        self.conv = ModulatedConv2d(in_ch, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, x, style, skip=None):
        out = self.conv(x, style) + self.bias
        return out if skip is None else out + self.upsample(skip)


class ConstantInput(nn.Module):
    def __init__(self, ch, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, ch, size, size))

    def forward(self, b):
        return self.input.repeat(b, 1, 1, 1)


class Generator(nn.Module):
    """Réseau de mapping `style` (z -> w) et réseau de synthèse (W+ -> image dans [-1, 1])."""
    def __init__(self, size=256, style_dim=512, n_mlp=8, channel_multiplier=2, lr_mlp=0.01):
        super().__init__()
        self.style_dim = style_dim
        self.style = nn.Sequential(PixelNorm(), *(EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation=True) for _ in range(n_mlp)))
        ch = {4: 512, 8: 512, 16: 512, 32: 512, 64: 256 * channel_multiplier, 128: 128 * channel_multiplier, 256: 64 * channel_multiplier, 512: 32 * channel_multiplier, 1024: 16 * channel_multiplier}
        log_size = int(math.log2(size))
        self.n_latent = 2 * log_size - 2
        self.input = ConstantInput(ch[4])
        self.conv1 = StyledConv(ch[4], ch[4], 3, style_dim)
        self.to_rgb1 = ToRGB(ch[4], style_dim, upsample=False)
        self.convs, self.to_rgbs = nn.ModuleList(), nn.ModuleList()
        in_ch = ch[4]
        for i in range(3, log_size + 1):
            out_ch = ch[2 ** i]
            self.convs.append(StyledConv(in_ch, out_ch, 3, style_dim, upsample=True))
            self.convs.append(StyledConv(out_ch, out_ch, 3, style_dim))
            self.to_rgbs.append(ToRGB(out_ch, style_dim))
            in_ch = out_ch
        self.noises = nn.Module()
        gen = torch.Generator().manual_seed(NOISE_SEED)
        for i in range(2 * (log_size - 2) + 1):
            res = 2 ** ((i + 5) // 2)
            self.noises.register_buffer(f"noise_{i}", torch.randn(1, 1, res, res, generator=gen), persistent=False)

    @torch.no_grad()
    def mean_latent(self, n):
        return self.style(torch.randn(n, self.style_dim, device=self.input.input.device)).mean(0, keepdim=True)

    def forward(self, w, randomize_noise=False):
        """w : (B, n_latent, 512) -> image (B, 3, size, size)."""
        noise = [None] * (len(self.convs) + 1) if randomize_noise else [getattr(self.noises, f"noise_{i}") for i in range(len(self.convs) + 1)]
        out = self.conv1(self.input(w.shape[0]), w[:, 0], noise[0])
        skip = self.to_rgb1(out, w[:, 1])
        i = 1
        for conv1, conv2, n1, n2, to_rgb in zip(self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs):
            out = conv2(conv1(out, w[:, i], n1), w[:, i + 1], n2)
            skip = to_rgb(out, w[:, i + 2], skip)
            i += 2
        return skip


class PixelNorm(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(1, keepdim=True) + 1e-8)


class EqualConv2d(nn.Module):
    """Convolution à learning rate égalisé."""
    def __init__(self, in_ch, out_ch, k, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, k, k))
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
        self.scale, self.stride, self.padding = 1 / math.sqrt(in_ch * k * k), stride, padding

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias, self.stride, self.padding)


class ConvLayer(nn.Sequential):
    """[flou + convolution de pas 2 si downsample] puis activation (le biais est alors dans l'activation)."""
    def __init__(self, in_ch, out_ch, k, downsample=False, blur_kernel=(1, 3, 3, 1), bias=True, activate=True):
        layers = []
        if downsample:
            p = (len(blur_kernel) - 2) + (k - 1)
            layers.append(Blur(blur_kernel, pad=((p + 1) // 2, p // 2)))
            stride, padding = 2, 0
        else:
            stride, padding = 1, k // 2
        layers.append(EqualConv2d(in_ch, out_ch, k, stride, padding, bias=bias and not activate))
        if activate:
            layers.append(FusedLeakyReLU(out_ch))
        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvLayer(in_ch, in_ch, 3)
        self.conv2 = ConvLayer(in_ch, out_ch, 3, downsample=True)
        self.skip = ConvLayer(in_ch, out_ch, 1, downsample=True, activate=False, bias=False)

    def forward(self, x):
        return (self.conv2(self.conv1(x)) + self.skip(x)) / math.sqrt(2)


class Discriminator(nn.Module):
    """Discriminateur résiduel : image (B, 3, size, size) dans [-1, 1] -> score de réalisme (B,), haut = réaliste.

    L'écart-type de minibatch mélange les images par groupes de 4 : le batch doit être un multiple de min(B, 4).
    """
    def __init__(self, size=256, channel_multiplier=2):
        super().__init__()
        ch = {4: 512, 8: 512, 16: 512, 32: 512, 64: 256 * channel_multiplier, 128: 128 * channel_multiplier, 256: 64 * channel_multiplier, 512: 32 * channel_multiplier, 1024: 16 * channel_multiplier}
        log_size = int(math.log2(size))
        convs, in_ch = [ConvLayer(3, ch[size], 1)], ch[size]
        for i in range(log_size, 2, -1):
            convs.append(ResBlock(in_ch, ch[2 ** (i - 1)]))
            in_ch = ch[2 ** (i - 1)]
        self.convs = nn.Sequential(*convs)
        self.stddev_group = 4
        self.final_conv = ConvLayer(in_ch + 1, ch[4], 3)
        self.final_linear = nn.Sequential(EqualLinear(ch[4] * 4 * 4, ch[4], activation=True), EqualLinear(ch[4], 1))

    def forward(self, x):
        out = self.convs(x)
        b, c, h, w = out.shape
        g = min(b, self.stddev_group)
        sd = torch.sqrt(out.view(g, -1, 1, c, h, w).var(0, unbiased=False) + 1e-8).mean([2, 3, 4], keepdim=True).squeeze(2)
        out = torch.cat([out, sd.repeat(g, 1, h, w)], 1)
        return self.final_linear(self.final_conv(out).view(b, -1)).squeeze(1)
