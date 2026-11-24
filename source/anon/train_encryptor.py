import argparse
import csv
import pathlib
import time

import torch
from torch.utils.checkpoint import checkpoint

import pretrained as pre
from encryptor import LatentEncryptor
from losses import DEFAULT_LAMBDAS, compute_losses, domain_excess, domain_stats, id_metrics, image_terms, saturation
from ranger import Ranger


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp_dir", default=str(pre.ROOT / "source/anon/runs/exp"))
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=1, help="accumulation de gradient (batch effectif = batch * accum)")
    ap.add_argument("--optim", choices=["ranger", "adam"], default="ranger")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--eps", type=float, default=0.0, help="seuil de similarité (0 article, 0.1 script auteurs)")
    for k, v in DEFAULT_LAMBDAS.items():
        ap.add_argument(f"--lambda_{k}", type=float, default=v)
    ap.add_argument("--wrong_margin", type=float, default=0.0, help="la perte mauvaise clé ne pousse plus sous ce cosinus (0 : jusqu'à 0)")
    ap.add_argument("--dom_quantile", type=float, default=0.5, help="rayon du domaine par couche : quantile de la distance des latents réels à leur centre")
    ap.add_argument("--real_quantile", type=float, default=0.5, help="seuil de la perte de réalisme (--lambda_real) : quantile du score du discriminateur sur des vraies images")
    ap.add_argument("--wrong_warmup", type=int, default=0, help="montée linéaire du poids de la perte mauvaise clé de 0 à --lambda_wrong sur ce nombre d'itérations (0 : poids plein dès le début)")
    ap.add_argument("--stop_pix", type=float, default=2.0, help="arrêt d'urgence : perte pix (L1 de e1, e2, d ; normale 0,3 à 0,45) au-dessus de ce seuil, ou perte non finie... (0 : désactivé)")
    ap.add_argument("--stop_patience", type=int, default=20, help="... pendant ce nombre d'itérations de suite")
    ap.add_argument("--no_grad_ckpt", action="store_true", help="désactive le gradient checkpointing (plus rapide, beaucoup plus de mémoire)")
    ap.add_argument("--lpips_type", choices=["vgg", "alex"], default="vgg")
    ap.add_argument("--id_net", choices=["facenet", "arcface"], default="facenet", help="réseau d'identité des pertes ; la validation mesure avec ArcFace (cos_*) et FaceNet (fn_*)")
    ap.add_argument("--qanc_latent", type=float, default=0.0, help="ancrage des mauvaises clés sur le chiffré (--lambda_qanc) : poids alpha de la distance latente ||w_q - sg(w_e1)|| à côté de LPIPS(X_q, sg(X_e1))")
    ap.add_argument("--no_pos_embedding", action="store_true")
    ap.add_argument("--no_final_fc", action="store_true")
    ap.add_argument("--data_free", action="store_true", help="latents aléatoires au lieu des inversions e4e (§3.3)")
    ap.add_argument("--n_val", type=int, default=64, help="latents de validation évalués à chaque --val_interval")
    ap.add_argument("--log_interval", type=int, default=50)
    ap.add_argument("--image_interval", type=int, default=500)
    ap.add_argument("--val_interval", type=int, default=1000)
    ap.add_argument("--save_interval", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="checkpoint dont on reprend les poids du chiffreur pour un nouvel entraînement (ex. runs/exp1/checkpoints/last.pt)")
    return ap.parse_args()


def forward_step(E, G, id_net, lpips, w, p1, p2, q1, q2, grad_ckpt=True, D=None, qanc=False):
    """2 chiffrements, 1 bon déchiffrement, 2 mauvais.

    Chaque groupe de B images passe dans G et le réseau d'identité (et LPIPS contre l'original pour e1, e2, d ; le
    discriminateur D pour q1, q2 si D est donné ; LPIPS contre le chiffré e1, sans gradient vers e1, pour q1, q2 si
    qanc) sous gradient checkpointing.
    Retourne x, f (original), W, X, Fe (5 groupes concaténés), pix, lp (pertes de qualité de e1, e2, d, (3,)),
    dq (scores de D sur q1, q2, (2B,), ou None), lq (LPIPS q1, q2 contre e1, (2,), ou None).
    """
    with torch.no_grad():
        x = pre.render(G, w)
        f = pre.id_embed(id_net, x)
    w_e1, w_e2 = E(w, p1), E(w, p2)                  # chiffrement
    w_d = E(w_e1, p1)                                # bonne clé
    w_q1, w_q2 = E(w_e1, q1), E(w_e1, q2)            # mauvaises clés
    zero = x.new_zeros(())
    def group(w_c, quality, anchor):
        X_c = pre.render(G, w_c)
        f_c = pre.id_embed(id_net, X_c)
        if quality:
            return (X_c, f_c) + image_terms(X_c, x, lpips)
        return (X_c, f_c, pre.realism(D, X_c) if D is not None else zero, lpips(X_c, anchor) if anchor is not None else zero)
    def run(w_c, quality, anchor=None):
        if grad_ckpt and torch.is_grad_enabled():
            return checkpoint(group, w_c, quality, anchor, use_reentrant=False)
        return group(w_c, quality, anchor)
    outs = [run(w_c, True) for w_c in (w_e1, w_e2, w_d)]
    anchor = outs[0][0].detach() if qanc else None   # chiffré e1, cible de l'ancrage des mauvaises clés
    outs += [run(w_c, False, anchor) for w_c in (w_q1, w_q2)]
    X, Fe = torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])
    pix, lp = torch.stack([o[2] for o in outs[:3]]), torch.stack([o[3] for o in outs[:3]])
    dq = torch.cat([o[2] for o in outs[3:]]) if D is not None else None
    lq = torch.stack([o[3] for o in outs[3:]]) if qanc else None
    return x, f, torch.cat([w_e1, w_e2, w_d, w_q1, w_q2]), X, Fe, pix, lp, dq, lq


class LatentSource:
    """Batches de latents d'entraînement (inversions e4e de FFHQ) ou aléatoires (data-free)."""
    def __init__(self, G, data_free, n_val, seed, dom_quantile=0.5):
        self.G, self.data_free = G, data_free
        self.gen = torch.Generator().manual_seed(seed)
        if data_free:
            self.w_mean = G.mean_latent(100_000).detach()
            self.val = self.sample_random(n_val, torch.Generator("cuda").manual_seed(seed + 1))
            ref = torch.cat([self.sample_random(1000) for _ in range(20)]).cpu()
        else:
            latents = torch.load(pre.LATENTS, map_location="cpu")    # (70000, 14, 512)
            perm = torch.randperm(len(latents), generator=torch.Generator().manual_seed(seed))
            self.train = latents[perm[:-2000]]                       # 2000 derniers : validation
            self.val = latents[perm[-2000:]][:n_val].cuda()
            print(f"latents : {len(self.train)} entraînement, {len(self.val)} validation (sur 2000 réservés)")
            ref = self.train
        self.ref_real = ref[:512]                                    # vraies images de référence du discriminateur
        mu, r = domain_stats(ref, dom_quantile)
        self.dom = (mu.cuda(), r.cuda())                             # domaine des latents réels (perte et mesure dom)

    @torch.no_grad()
    def sample_random(self, n, generator=None):
        w = pre.sample_password(self.G, n, generator=generator)
        return self.w_mean + 0.7 * (w - self.w_mean)                 # troncature 0.7

    def batch(self, n):
        if self.data_free:
            return self.sample_random(n)
        idx = torch.randint(len(self.train), (n,), generator=self.gen)
        return self.train[idx].cuda(non_blocking=True)


@torch.no_grad()
def validate(E, G, arcface, facenet, lpips, D, val_w, val_keys, batch, dom):
    """Mesures ArcFace (cos_*) puis domaine, réalisme, LPIPS et saturation des mauvaises clés, puis FaceNet (fn_*)."""
    E.eval()
    sums, n = {}, 0
    for i in range(0, len(val_w), batch):
        sl = slice(i, i + batch)
        x, f, W, X, Fe, _, _, dq, _ = forward_step(E, G, arcface, lpips, val_w[sl], *(k[sl] for k in val_keys), D=D)
        m = id_metrics(f, Fe)
        m["dom_q"] = domain_excess(W[3 * len(f):], *dom).item()
        m["d_q"] = dq.mean().item()                                  # réalisme des mauvais déchiffrements (haut = réaliste)
        m["lpips_q"] = lpips(X.chunk(5)[3], x).item()                # mauvaise clé q1 contre l'original
        m["sat_q"] = saturation(X[3 * len(f):])                      # pixels hors de [-1, 1], mauvaises clés
        m_fn = id_metrics(pre.id_embed(facenet, x), pre.id_embed(facenet, X))   # second juge, indépendant si --id_net arcface
        m.update({k.replace("cos_", "fn_"): v for k, v in m_fn.items()})
        for k, v in m.items():
            sums[k] = sums.get(k, 0.0) + v * len(f)
        n += len(f)
        if i == 0:
            first = X
    E.train()
    return {k: v / n for k, v in sums.items()}, first


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    exp = pathlib.Path(args.exp_dir)
    (exp / "images").mkdir(parents=True, exist_ok=True)
    (exp / "checkpoints").mkdir(exist_ok=True)
    G, arcface, lpips = pre.load_generator(), pre.load_arcface(), pre.load_lpips(args.lpips_type)
    facenet = pre.load_facenet()
    id_net = facenet if args.id_net == "facenet" else arcface     # pertes d'identité ; la validation mesure avec les deux
    data = LatentSource(G, args.data_free, args.n_val, args.seed, args.dom_quantile)
    with torch.random.fork_rng(devices=[]):                          # n'entame pas le générateur aléatoire de la graine
        D = pre.load_discriminator()                                 # réalisme : perte (--lambda_real) et mesure d_q
    with torch.no_grad():
        ref = data.ref_real.cuda()
        ref_scores = torch.cat([pre.realism(D, pre.render(G, ref[i:i + 8])) for i in range(0, len(ref), 8)])
    real_thr = torch.quantile(ref_scores, args.real_quantile).item()
    print(f"discriminateur : score des vraies images médiane {ref_scores.median().item():.3f}, seuil de réalisme {real_thr:.3f} (quantile {args.real_quantile})")
    D_train = D if args.lambda_real > 0 else None
    gen_val = torch.Generator("cuda").manual_seed(1234)               # clés de validation fixes
    val_keys = [pre.sample_password(G, len(data.val), generator=gen_val) for _ in range(4)]
    E = LatentEncryptor(pos_embedding=not args.no_pos_embedding, final_fc=not args.no_final_fc).cuda().train()
    if args.optim == "ranger":
        opt = Ranger(E.parameters(), lr=args.lr)
    else:
        opt = torch.optim.Adam(E.parameters(), lr=args.lr)
    lam = {k: getattr(args, f"lambda_{k}") for k in DEFAULT_LAMBDAS}
    opt_impl = f"{type(opt).__module__}.{type(opt).__name__}"
    start, last = 0, exp / "checkpoints/last.pt"
    if last.exists():                                               # reprise
        ckpt = torch.load(last, map_location="cpu")
        E.load_state_dict(ckpt["encryptor"])
        if ckpt.get("opt_impl") == opt_impl:
            opt.load_state_dict(ckpt["opt"])
        else:
            print(f"optimiseur du checkpoint ({ckpt.get('opt_impl', 'Ranger de RiDDLE')}) différent de {opt_impl} : réinitialisé")
        start = ckpt["step"] + 1
        if "rng" in ckpt:
            torch.set_rng_state(ckpt["rng"]["cpu"])
            torch.cuda.set_rng_state(ckpt["rng"]["cuda"])
            data.gen.set_state(ckpt["rng"]["data"])
        else:
            torch.manual_seed(args.seed + start)
            data.gen.manual_seed(args.seed + start)
        log_path = exp / "log.csv"                                  # retire les lignes postérieures au checkpoint
        if log_path.exists():
            rows = list(csv.reader(open(log_path, newline="")))
            keep = rows[:1] + [r for r in rows[1:] if r and int(r[0]) < start]
            with open(log_path, "w", newline="") as fh:
                csv.writer(fh).writerows(keep)
        print(f"reprise à l'itération {start}")
    elif args.init:                                                 # nouvel entraînement à partir de poids existants
        E.load_state_dict(torch.load(args.init, map_location="cpu")["encryptor"])
        print(f"chiffreur initialisé depuis {args.init}")
    def save(step, path):
        rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(), "data": data.gen.get_state()}
        torch.save({"encryptor": E.state_dict(), "config": E.config, "opt": opt.state_dict(), "opt_impl": opt_impl, "step": step, "args": vars(args), "rng": rng}, path)
    ID = ["deid", "rec", "wrong", "div", "qe"]                       # ordre de losses.id_metrics
    header = ["step", "split", "loss", *DEFAULT_LAMBDAS, *(f"cos_{k}" for k in ID), "dom_q", "d_q", "lpips_q", "sat_q", *(f"fn_{k}" for k in ID), "it_s"]
    if start > 0 and (exp / "log.csv").exists():
        old = next(csv.reader(open(exp / "log.csv", newline="")), None)
        if old != header:
            print("attention : log.csv a été écrit par une autre version du code (colonnes différentes) ; utiliser un nouveau --exp_dir")
    log_f = open(exp / "log.csv", "a", newline="")
    log = csv.writer(log_f)
    if start == 0:
        log.writerow(header)
    print(f"chiffreur : {sum(p.numel() for p in E.parameters()) / 1e6:.2f} M paramètres")
    t0 = time.time()
    bad = 0                                                          # itérations divergentes de suite (arrêt d'urgence)
    for step in range(start, args.steps):
        lam_step = dict(lam)
        if args.wrong_warmup > 0:
            lam_step["wrong"] = lam["wrong"] * min(1.0, step / args.wrong_warmup)
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            w = data.batch(args.batch)
            keys = [pre.sample_password(G, args.batch) for _ in range(4)]   # p1, p2 bonnes ; q1, q2 mauvaises
            x, f, W, X, Fe, pix, lp, dq, lq = forward_step(E, G, id_net, lpips, w, *keys, grad_ckpt=not args.no_grad_ckpt, D=D_train, qanc=lam["qanc"] > 0)
            loss, terms = compute_losses(w, f, W, Fe, pix, lp, eps=args.eps, lam=lam_step, wrong_margin=args.wrong_margin, dom=data.dom, dq=dq, real_thr=real_thr, X=X,
                                         lq=lq, qanc_latent=args.qanc_latent)
            (loss / args.accum).backward()
        finite = bool(torch.isfinite(loss))
        if finite:
            opt.step()                                               # une perte non finie ne touche pas aux poids
        if args.stop_pix > 0 and (not finite or terms["pix"] > args.stop_pix):
            bad += 1
            print(f"[{step:6d}] ATTENTION : divergence ({bad}/{args.stop_patience}) : loss {loss.item():.3g}, pix {terms['pix']:.3g}", flush=True)
            if bad >= args.stop_patience:
                save(step, exp / "checkpoints/diverged.pt")         # last.pt garde le dernier état sain
                log_f.close()
                raise SystemExit(f"ARRÊT : l'entraînement diverge depuis {bad} itérations (pix {terms['pix']:.3g}). "
                                 f"last.pt et step_*.pt n'ont pas été écrasés ; état divergé dans checkpoints/diverged.pt.")
        else:
            bad = 0
        if step % args.log_interval == 0:
            m = id_metrics(f, Fe)
            m["dom_q"] = domain_excess(W[3 * args.batch:].detach(), *data.dom).item()
            m["d_q"] = dq.mean().item() if dq is not None else ""
            m["lpips_q"] = ""
            m["sat_q"] = saturation(X[3 * args.batch:].detach())
            it_s = (step - start + 1) / (time.time() - t0)
            print(f"[{step:6d}] loss {loss.item():.3f} | " + " ".join(f"{k} {v:.3f}" for k, v in terms.items()) + f" | cos rec {m['cos_rec']:.2f} deid {m['cos_deid']:.2f} wrong {m['cos_wrong']:.2f} qe {m['cos_qe']:.2f} | {it_s:.2f} it/s", flush=True)
            log.writerow([step, "train", loss.item(), *terms.values(), *m.values(), *[""] * len(ID), round(it_s, 3)])   # fn_* : validation seulement
            log_f.flush()
        if step % args.image_interval == 0:
            # lignes : original, chiffré (p1), chiffré (p2), déchiffré (p1), déchiffré mauvaise clé (q1)
            pre.save_grid(torch.cat([x, *X.detach().chunk(5)[:4]]), exp / f"images/train_{step:06d}.png", args.batch)
        if step % args.val_interval == 0 or step == args.steps - 1:
            vm, Xv = validate(E, G, arcface, facenet, lpips, D, data.val, val_keys, args.batch, data.dom)
            print(f"[{step:6d}] VALIDATION " + " ".join(f"{k} {v:.3f}" for k, v in vm.items()), flush=True)
            log.writerow([step, "val", *[""] * (1 + len(DEFAULT_LAMBDAS)), *vm.values(), ""])
            log_f.flush()
            pre.save_grid(torch.cat([pre.render(G, data.val[:args.batch]).detach(), *Xv.chunk(5)[:4]]), exp / f"images/val_{step:06d}.png", args.batch)
        if bad == 0 and (step % args.save_interval == 0 or step == args.steps - 1):   # pas de sauvegarde pendant une divergence
            save(step, last)
            if step % args.save_interval == 0 and step > 0:
                save(step, exp / f"checkpoints/step_{step:06d}.pt")
    log_f.close()


if __name__ == "__main__":
    main()
