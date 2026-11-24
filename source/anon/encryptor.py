import torch
import torch.nn as nn


class LevelBlock(nn.Module):
    """Add & Norm après self-attention, cross-attention et FFN."""
    def __init__(self, d=512, heads=4, ff=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, w, p):                         # w, p : (B, L, d)
        w = self.n1(w + self.drop(self.self_attn(w, w, w, need_weights=False)[0]))
        w = self.n2(w + self.drop(self.cross_attn(w, p, p, need_weights=False)[0]))
        return self.n3(w + self.drop(self.ff(w)))


class LatentEncryptor(nn.Module):
    def __init__(self, split=(4, 4, 6), d=512, heads=4, ff=1024, dropout=0.1, pos_embedding=True, final_fc=True):
        super().__init__()
        self.config = dict(split=list(split), d=d, heads=heads, ff=ff, dropout=dropout, pos_embedding=pos_embedding, final_fc=final_fc)
        self.split = list(split)
        self.blocks = nn.ModuleList(LevelBlock(d, heads, ff, dropout) for _ in self.split)
        self.pos = nn.Parameter(torch.zeros(1, sum(self.split), d)) if pos_embedding else None
        self.fc = nn.Linear(d, d) if final_fc else nn.Identity()

    def forward(self, w, p):                         # (B, 14, 512) x 2 -> (B, 14, 512)
        if self.pos is not None:
            w = w + self.pos
        ws = torch.split(w, self.split, dim=1)
        ps = torch.split(p, self.split, dim=1)
        out = torch.cat([blk(a, b) for blk, a, b in zip(self.blocks, ws, ps)], dim=1)
        return self.fc(out)

    def encrypt(self, w, p):
        return self(w, p)

    def decrypt(self, w_enc, p):
        return self(w_enc, p)

    @classmethod
    def from_checkpoint(cls, path, device="cuda"):
        ckpt = torch.load(path, map_location="cpu")
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["encryptor"])
        return model.to(device).eval()


if __name__ == "__main__":                           # test rapide des dimensions
    E = LatentEncryptor()
    print(f"{sum(p.numel() for p in E.parameters()) / 1e6:.2f} M paramètres")
    w, p = torch.randn(2, 14, 512), torch.randn(2, 14, 512)
    print(E.encrypt(w, p).shape, E.decrypt(E.encrypt(w, p), p).shape)
