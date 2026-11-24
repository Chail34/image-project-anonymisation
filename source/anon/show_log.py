"""Affiche un log.csv d'entraînement par nom de colonne (validation + quelques lignes d'entraînement).

python source/anon/show_log.py source/anon/runs/exp4a [--every 250]
"""
import argparse, csv, pathlib

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("exp_dir")
ap.add_argument("--every", type=int, default=250, help="une ligne d'entraînement toutes les N itérations")
args = ap.parse_args()
rows = list(csv.DictReader(open(pathlib.Path(args.exp_dir) / "log.csv", newline="")))
for split, cols, keep in (("train", ["loss", "wrong", "dom", "real", "range", "qanc", "pix", "latent", "cos_rec", "cos_wrong", "cos_qe", "dom_q", "d_q", "sat_q"], lambda s: s % args.every == 0),
                          ("val", ["cos_deid", "cos_rec", "cos_wrong", "cos_div", "cos_qe", "dom_q", "d_q", "lpips_q", "sat_q", "fn_rec", "fn_wrong", "fn_qe"], lambda s: True)):
    sel = [r for r in rows if r["split"] == split and keep(int(r["step"]))]
    cols = [c for c in cols if c in rows[0]]
    print(f"\n{split.upper():<8}" + "".join(f"{c:>11}" for c in cols))
    for r in sel:
        print(f"{r['step']:<8}" + "".join(f"{float(r[c]):>11.3f}" if r[c] else f"{'':>11}" for c in cols))
