"""Évaluation comparée de chiffreurs : ArcFace et FaceNet, LPIPS, distance latente.

Même protocole que runs/exp1/eval et runs/exp2/eval : N latents de validation (les 2000 réservés à
l'entraînement), --keys paires (bonne clé, mauvaise clé) fixes (5 par défaut). Réalisme des mauvais déchiffrements : score du
discriminateur StyleGAN2 (D q, comparé aux vraies images) et part sous le 5e centile des vraies images (D q bas) ;
pixels hors de [-1, 1] avant bornage (sat q). Env riddle. Un modèle dont le nom commence par
« auteurs » est chargé comme le checkpoint des auteurs (authors_mapper.py).

Avec --keys > 5, les 5 premières paires sont celles des anciennes évaluations (graine 1234), les suivantes sont tirées
ensuite : les mesures moyennes changent, mais les clés 0-4 restent comparables. La fuite dépend de la clé (results.md
§11) : regarder wrong_keys_leak (part des mauvaises clés dont le cosinus moyen dépasse le seuil) et wrong_per_key.

python source/anon/eval_compare.py --out source/anon/runs/exp3/eval/eval.json \
       exp1_20k=source/anon/runs/exp1/checkpoints/last.pt exp3a=source/anon/runs/exp3a/checkpoints/last.pt
"""
import argparse, json, pathlib, torch
import pretrained as ru
from encryptor import LatentEncryptor
from authors_mapper import AuthorsMapper
from losses import domain_stats

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("models", nargs="+", help="nom=chemin/du/checkpoint.pt")
ap.add_argument("--n", type=int, default=512, help="nombre de visages de validation")
ap.add_argument("--keys", type=int, default=5, help="nombre de paires (bonne clé, mauvaise clé) fixes ; les 5 premières sont toujours les mêmes")
ap.add_argument("--out", default=None, help="fichier JSON des résultats")
args = ap.parse_args()
models = dict(m.split("=", 1) for m in args.models)

N, P, B = args.n, args.keys, 8
dev = "cuda"
torch.manual_seed(0)
G, lp, D = ru.load_generator(), ru.load_lpips("vgg"), ru.load_discriminator()
nets = {"arcface": ru.load_arcface(), "facenet": ru.load_facenet()}
lat = torch.load(ru.LATENTS, map_location="cpu")
perm = torch.randperm(len(lat), generator=torch.Generator().manual_seed(0))
val = lat[perm[-2000:]][:N].clone()
MU, RAD = (t.to(dev) for t in domain_stats(lat[perm[:-2000]]))   # domaine des latents réels (médiane)
del lat
excess = lambda w: ((w - MU).norm(dim=-1) - RAD).clamp(min=0).mean(1)  # 0 dans le domaine
g = torch.Generator(dev).manual_seed(1234)
P0 = min(P, 5)                                                      # 5 premières paires : mêmes tirages qu'avant --keys
keys, wrong = ru.sample_password(G, 5, generator=g)[:P0], ru.sample_password(G, 5, generator=g)[:P0]
if P > 5:
    keys = torch.cat([keys, ru.sample_password(G, P - 5, generator=g)])
    wrong = torch.cat([wrong, ru.sample_password(G, P - 5, generator=g)])

@torch.no_grad()
def run(f):
    out = {k: {n: [] for n in nets} for k in ("o", "e", "d", "q")}
    LE, LD, LQ, LAT, DE, DQ, RE, RQ, SQ = [], [], [], [], [], [], [], [], []
    for i in range(0, N, B):
        w = val[i:i + B].to(dev); b = len(w)
        x = ru.render(G, w)
        for n, net in nets.items():
            out["o"][n].append(ru.id_embed(net, x))
        fe, fd, fq = ({n: [] for n in nets} for _ in range(3))
        le, ld, lq, la, de, dq, re, rq, sq = [], [], [], [], [], [], [], [], []
        for k in range(P):
            p, q = keys[k:k + 1].expand(b, -1, -1), wrong[k:k + 1].expand(b, -1, -1)
            we = f(w, p); xe = ru.render(G, we).clamp(-1, 1)
            wq = f(we, q)
            xd = ru.render(G, f(we, p)).clamp(-1, 1); xq = ru.render(G, wq)
            sq.append((xq.abs() > 1).float().mean((1, 2, 3))); xq = xq.clamp(-1, 1)
            for n, net in nets.items():
                fe[n].append(ru.id_embed(net, xe)); fd[n].append(ru.id_embed(net, xd)); fq[n].append(ru.id_embed(net, xq))
            le.append(lp.net(xe, x).reshape(-1)); ld.append(lp.net(xd, x).reshape(-1)); lq.append(lp.net(xq, x).reshape(-1))
            la.append((we - w).norm(dim=-1).mean(1)); de.append(excess(we)); dq.append(excess(wq))
            re.append(D(xe)); rq.append(D(xq))
        for n in nets:
            out["e"][n].append(torch.stack(fe[n], 1)); out["d"][n].append(torch.stack(fd[n], 1)); out["q"][n].append(torch.stack(fq[n], 1))
        LE.append(torch.stack(le, 1)); LD.append(torch.stack(ld, 1)); LQ.append(torch.stack(lq, 1)); LAT.append(torch.stack(la, 1))
        DE.append(torch.stack(de, 1)); DQ.append(torch.stack(dq, 1)); RE.append(torch.stack(re, 1)); RQ.append(torch.stack(rq, 1)); SQ.append(torch.stack(sq, 1))
    emb = {k: {n: torch.cat(v) for n, v in d.items()} for k, d in out.items()}
    return emb, torch.cat(LE), torch.cat(LD), torch.cat(LQ), torch.cat(LAT), torch.cat(DE), torch.cat(DQ), torch.cat(RE), torch.cat(RQ), torch.cat(SQ)

def offdiag(S):
    return S[~torch.eye(S.shape[-1], dtype=bool, device=S.device).expand_as(S)]

with torch.no_grad():                                           # réalisme de référence : vraies images (inversions rendues)
    D_REAL = torch.cat([D(ru.render(G, val[i:i + B].to(dev)).clamp(-1, 1)) for i in range(0, N, B)])
D_Q05 = torch.quantile(D_REAL, 0.05).item()
print(f"discriminateur, vraies images : moyenne {D_REAL.mean().item():.3f}, 5e centile {D_Q05:.3f}", flush=True)
res = {"_reel": {"d": D_REAL.mean().item(), "d_q05": D_Q05}}
for name, path in models.items():
    m = AuthorsMapper.from_checkpoint(path) if name.startswith("auteurs") else LatentEncryptor.from_checkpoint(path)
    emb, lpe, lpd, lpq, lat_d, dom_e, dom_q, d_e, d_q, sat_q = run(lambda w, p: m(w, p))
    r = dict(lpips_e=lpe.mean().item(), lpips_d=lpd.mean().item(), lpips_q=lpq.mean().item(), lat=lat_d.mean().item(),
             dom_e=dom_e.mean().item(), dom_q=dom_q.mean().item(), dom_q_out=(dom_q > 0.5).float().mean().item(),
             d_e=d_e.mean().item(), d_q=d_q.mean().item(), d_q_low=(d_q < D_Q05).float().mean().item(), sat_q=sat_q.mean().item())
    for n in nets:
        f0, fe, fd, fq = (emb[k][n] for k in "oedq")
        S0 = f0 @ f0.T
        thr = torch.quantile(offdiag(S0), 0.999).item()
        c_e, c_d, c_q = ((f0[:, None] * t).sum(-1) for t in (fe, fd, fq))
        div = offdiag(fe @ fe.transpose(1, 2))
        same_key = offdiag(fe.transpose(0, 1) @ fe.transpose(0, 1).transpose(1, 2))
        r[n] = dict(thr=thr, deid=c_e.mean().item(), rec=c_d.mean().item(), wrong=c_q.mean().item(),
                    qe=(fe * fq).sum(-1).mean().item(),                    # mauvaise clé / chiffré qu'elle déchiffre (1 = copie)
                    div=div.mean().item(), same_key=same_key.mean().item(),
                    deid_ok=(c_e < thr).float().mean().item(), rec_ok=(c_d > thr).float().mean().item(),
                    wrong_ok=(c_q < thr).float().mean().item(), div_ok=(div < thr).float().mean().item(),
                    rec_min=c_d.min().item(), wrong_max=c_q.max().item(), wrong_p95=torch.quantile(c_q.flatten(), 0.95).item(),
                    # par clé (moyenne sur les visages) : une mauvaise clé peut fuir pour presque tous les visages
                    wrong_keys_leak=(c_q.mean(0) > thr).float().mean().item(), wrong_key_max=c_q.mean(0).max().item(),
                    rec_key_min=c_d.mean(0).min().item(),
                    wrong_per_key=[round(v, 4) for v in c_q.mean(0).tolist()], rec_per_key=[round(v, 4) for v in c_d.mean(0).tolist()])
    res[name] = r
    print(name, json.dumps(r), flush=True)
    if args.out:                                                # sauvegarde après chaque modèle (évaluation longue)
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1)
    del m
if args.out:
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
print()
print(f"{'modèle':<14}{'réseau':<9}{'chiffré↓':>9}{'déchiffré↑':>11}{'mauv. clé↓':>11}{'mauv. non reconnues↑':>22}{'clés qui fuient↓':>17}{'2 clés↓':>9}{'LPIPS e↓':>10}{'LPIPS q':>9}{'dom q↓':>8}{'D q↑':>7}{'D q bas↓':>10}{'sat q↓':>9}")
for name, r in res.items():
    if name.startswith("_"):
        continue
    for n in nets:
        m = r[n]
        print(f"{name:<14}{n:<9}{m['deid']:>9.2f}{m['rec']:>11.2f}{m['wrong']:>11.2f}{100 * m['wrong_ok']:>21.1f}%{100 * m.get('wrong_keys_leak', float('nan')):>16.1f}%{m['div']:>9.2f}{r['lpips_e']:>10.3f}{r['lpips_q']:>9.3f}{r['dom_q']:>8.2f}{r['d_q']:>7.2f}{100 * r['d_q_low']:>9.1f}%{100 * r.get('sat_q', 0):>8.2f}%")
