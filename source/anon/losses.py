import torch
import torch.nn.functional as F

DEFAULT_LAMBDAS = dict(deid=1.0, wrong=2.0, div=1.0, rec=1.0, pix=0.05, lpips=1.0, latent=0.01, dom=0.0, real=0.0, range=0.0, qanc=0.0)


def cos(a, b):
    return (a * b).sum(-1)


@torch.no_grad()
def domain_stats(latents, q=0.5, chunk=4096):
    """Domaine des latents réels (inversions e4e) : centre (14, 512) et rayon par couche (14,), quantile q de la distance au centre."""
    mu = latents.mean(0)
    d = torch.cat([(latents[i:i + chunk] - mu).norm(dim=-1) for i in range(0, len(latents), chunk)])
    return mu, torch.quantile(d, q, dim=0)


def domain_excess(w, mu, r):
    """Dépassement moyen du rayon des latents réels, couche par couche (0 dans le domaine). w : (B, 14, 512)."""
    return ((w - mu).norm(dim=-1) - r).clamp(min=0).mean()


def range_excess(X):
    """Dépassement moyen de [-1, 1] par pixel (0 pour une image normale). Force de rappel des pixels saturés."""
    return (X.abs() - 1).clamp(min=0).mean()


@torch.no_grad()
def saturation(X):
    """Part des pixels hors de [-1, 1] (mesure ; exp1 ≈ 0,2 % sur les mauvaises clés)."""
    return (X.abs() > 1).float().mean().item()


def image_terms(X, x, lpips):
    """Pertes de qualité d'un groupe d'images générées X contre l'original x : (L1, LPIPS)."""
    return F.l1_loss(X, x), lpips(X, x)


def compute_losses(w, f, W, Fe, pix, lp, eps=0.0, lam=DEFAULT_LAMBDAS, wrong_margin=0.0, dom=None, dq=None, real_thr=0.0, X=None,
                   lq=None, qanc_latent=0.0):
    """Fe : embeddings des groupes e1, e2, d, q1, q2. pix, lp : (3,) L1 et LPIPS des groupes e1, e2, d (pas des mauvais déchiffrements).

    wrong_margin : la perte mauvaise clé s'arrête de pousser sous ce cosinus. dom : (centre, rayon) de domain_stats,
    pour la perte de domaine des mauvais déchiffrements q1, q2 (sans les rapprocher de l'original).
    dq : scores du discriminateur StyleGAN2 sur les images q1, q2 (2B,) ; la perte de réalisme pénalise ce qui
    manque pour atteindre real_thr (score typique d'une vraie image), sans pousser au-delà.
    X : images des 5 groupes (5B, 3, H, W), pour la perte de rappel des pixels hors de [-1, 1] (range).
    lq : (2,) LPIPS des mauvais déchiffrements q1, q2 contre le chiffré e1 (sans gradient vers e1), ou None.
    Ancrage sur le chiffré (qanc, exp5a) : lq.sum() + qanc_latent * distance moyenne par couche ||w_q - sg(w_e1)||.
    """
    f_e1, f_e2, f_d, f_q1, f_q2 = Fe.chunk(5)
    enc, wrong = torch.stack([f_e1, f_e2], 1), torch.stack([f_q1, f_q2], 1)
    anon = torch.cat([enc, wrong], 1)                              # (B, K, 512) : visages qui doivent être anonymes
    K = anon.shape[1]
    L_deid = cos(f.unsqueeze(1), enc).clamp(min=eps).mean()
    m = max(eps, wrong_margin)
    L_wrong = (cos(f.unsqueeze(1), wrong).clamp(min=m) + cos(f_d.unsqueeze(1), wrong).clamp(min=m)).mean() / 2
    L_rec = (1 - cos(f, f_d)).mean()
    M = (anon @ anon.transpose(1, 2)).clamp(min=eps)               # similarités entre visages anonymes
    off = 1 - torch.eye(K, device=M.device)
    L_div = (M * off).sum((1, 2)).mean() / (K * (K - 1))
    L_pix, L_lpips = pix.sum(), lp.sum()
    L_lat = 3 * (W[:3 * len(w)] - w.repeat(3, 1, 1)).norm(dim=-1).mean()   # e1, e2, d
    L_dom = domain_excess(W[3 * len(w):], *dom) if dom is not None else W.new_zeros(())   # q1, q2
    L_real = (real_thr - dq).clamp(min=0).mean() if dq is not None else W.new_zeros(())                  # q1, q2
    L_range = range_excess(X) if X is not None else W.new_zeros(())                                    # e1, e2, d, q1, q2
    B = len(w)
    L_qanc = qanc_latent * (W[3 * B:] - W[:B].detach().repeat(2, 1, 1)).norm(dim=-1).mean()            # q1, q2 vers sg(e1)
    if lq is not None:
        L_qanc = L_qanc + lq.sum()
    terms = dict(deid=L_deid, wrong=L_wrong, div=L_div, rec=L_rec, pix=L_pix, lpips=L_lpips, latent=L_lat, dom=L_dom, real=L_real, range=L_range, qanc=L_qanc)
    loss = sum(lam[k] * v for k, v in terms.items())
    return loss, {k: v.item() for k, v in terms.items()}


@torch.no_grad()
def id_metrics(f, Fe):
    """Similarités cosinus moyennes pour le suivi : de-id/wrong/div bas, rec haut ; qe : mauvaise clé / chiffré e1 (proche de 1 = copie)."""
    f_e1, f_e2, f_d, f_q1, f_q2 = Fe.chunk(5)
    return dict(
        cos_deid=((cos(f, f_e1) + cos(f, f_e2)) / 2).mean().item(),
        cos_rec=cos(f, f_d).mean().item(),
        cos_wrong=((cos(f, f_q1) + cos(f, f_q2)) / 2).mean().item(),
        cos_div=cos(f_e1, f_e2).mean().item(),
        cos_qe=((cos(f_e1, f_q1) + cos(f_e1, f_q2)) / 2).mean().item(),
    )
