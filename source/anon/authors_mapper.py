"""Chiffreur publié par les auteurs de RiDDLE (checkpoint iteration_90000.pt), reconstruit avec des couches PyTorch standard.

Sert uniquement de point de comparaison. Projection linéaire du latent et de la clé, embedding de position,
puis une couche nn.TransformerDecoderLayer par niveau de StyleGAN (4 / 4 / 6 couches).
"""
import torch
import torch.nn as nn


class AuthorsMapper(nn.Module):
    def __init__(self, split=(4, 4, 6), d=512, heads=4, ff=1024):
        super().__init__()
        self.split = list(split)
        self.linear_ori, self.linear_pwd = nn.Linear(d, d), nn.Linear(d, d)
        self.pos_embedding = nn.Parameter(torch.zeros(sum(self.split), 1, d))
        self.layers = nn.ModuleList(nn.TransformerDecoderLayer(d, heads, ff, dropout=0.1) for _ in self.split)

    def forward(self, w, p):                         # (B, 14, 512) x 2 -> (B, 14, 512)
        w, p = self.linear_ori(w.transpose(0, 1)), self.linear_pwd(p.transpose(0, 1))
        parts = zip(self.layers, torch.split(w, self.split), torch.split(p, self.split), torch.split(self.pos_embedding, self.split))
        return torch.cat([layer(a + e, b) for layer, a, b, e in parts]).transpose(0, 1)

    @classmethod
    def from_checkpoint(cls, path, device="cuda"):
        model = cls()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["mapper_state_dict"])
        return model.to(device).eval()
