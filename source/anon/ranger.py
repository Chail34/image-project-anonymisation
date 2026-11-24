"""Optimiseur Ranger : RAdam + Lookahead (tous les k pas, les poids rapides sont ramenés vers des poids lents)."""
import torch


class Ranger(torch.optim.RAdam):
    def __init__(self, params, lr=1e-3, alpha=0.5, k=6, betas=(0.95, 0.999), eps=1e-5, weight_decay=0):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.alpha, self.k = alpha, k

    @torch.no_grad()
    def step(self, closure=None):
        params = [p for g in self.param_groups for p in g["params"] if p.grad is not None]
        init = {p: p.detach().clone() for p in params if "slow" not in self.state[p]}
        loss = super().step(closure)
        for p in params:
            state = self.state[p]
            if p in init:
                state["slow"] = init[p]
            if int(state["step"]) % self.k == 0:
                state["slow"].add_(p - state["slow"], alpha=self.alpha)
                p.copy_(state["slow"])
        return loss
