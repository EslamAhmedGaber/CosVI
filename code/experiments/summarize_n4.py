"""Summarise the Nile runs (N4 v2: results/n4/n4v2_{rep}_seed{s}_s{s_obs}.json; N4c competitor: n4c_{mode}_seed{s}.json).

Primary yardstick (fixed 14 Nov after N4 v1): test posteriors refitted in two-scale coordinates under the exact law,
IWAE, per test window (3 test windows, years 1122-1284). The grid-coordinate refit is reported only as a count of
failures (it returned impossible positive values or diverged at H >= 0.6 in most representations).
A training group counts as diverged when its training bound per window is below -100 (sound groups: -61 to -78).

Usage: python summarize_n4.py [--s 0.1]      Output: markdown to stdout, results/n4/summary_n4.json
"""
import os, sys, json, glob, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results", "n4")
sys.path.insert(0, os.path.join(HERE, ".."))
from fnvi import labels as L          # display names (CosVI etc.), single source
HG = np.array([0.5, 0.6, 0.7, 0.8, 0.9])


def ms(v):
    v = np.asarray(v, float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else float("nan"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--s", default="0.1"); a = ap.parse_args()
    out = {}
    print(f"N4 v2, s_obs = {a.s}. Held-out = exact-law IWAE per test window, two-scale refit (mean ± sd over seeds).")
    print("| representation | seeds | profile Ĥ (training bound) | held-out at profile group | held-out per group H=0.5…0.9 (mean over seeds) |"
          " best group (held-out) | diverged groups | grid-refit failures |")
    print("|---|---|---|---|---|---|---|---|")
    reps = sorted({os.path.basename(f).split("_seed")[0][len("n4v2_"):] for f in glob.glob(os.path.join(RES, f"n4v2_*_seed*_s{a.s}.json"))})
    for rep in reps:
        files = sorted(glob.glob(os.path.join(RES, f"n4v2_{rep}_seed*_s{a.s}.json")))
        Hp, at_prof, per_group, div, gfail = [], [], [], 0, 0
        for f in files:
            d = json.load(open(f))
            ts = np.array(d["test_loglik_exact_law_twoscale_coords_mean"]); tr = np.array(d["elbo_train_per_window"])
            gr = np.array(d["test_loglik_exact_law_grid_coords_mean"])
            Hp.append(d["H_profile"]); ip = int(np.argmin(np.abs(HG - d["H_profile"])))
            at_prof.append(ts[ip]); per_group.append(ts)
            div += int(np.sum(tr < -100))
            # 18 Nov: a continuous log density can be positive, so a positive value alone is not a failure;
            # a grid refit counts as failed only when it lies more than 20 nats below the two-scale refit
            gfail += int(np.sum(gr < ts - 20))
        P = np.array(per_group); mg = P.mean(0); ib = int(np.argmax(mg))
        out[rep] = {"n": len(files), "H_profile": ms(Hp), "heldout_at_profile": ms(at_prof), "heldout_per_group_mean": mg.tolist(),
                    "heldout_per_group_sd": (P.std(0, ddof=1) if len(files) > 1 else np.full(5, np.nan)).tolist(),
                    "best_group_H": float(HG[ib]), "best_group_heldout": ms(P[:, ib]), "diverged_groups": div, "grid_refit_failures": gfail}
        f2 = lambda t: f"{t[0]:.3f} ± {t[1]:.3f}"
        out[rep]["name"] = L.rep_name(rep)
        print(f"| {L.rep_name(rep)} [{rep}] | {len(files)} | {f2(ms(Hp))} | {f2(ms(at_prof))} | {' / '.join(f'{v:.2f}' for v in mg)} | "
              f"H={HG[ib]:.1f}: {f2(ms(P[:, ib]))} | {div}/{5 * len(files)} | {gfail}/{5 * len(files)} |")
    # 18 Nov: the competitor's system (N4c) was run at s = 0.1 only; its rows belong to that summary alone
    comp = sorted(glob.glob(os.path.join(RES, "n4c_*_seed*.json"))) if float(a.s) == 0.1 else []
    if comp:
        print("\nCompetitor full system (N4c):")
        print("| mode | seed | Ĥ | held-out (two-scale refit) | held-out (grid refit) | σ (years, standard scale) |")
        print("|---|---|---|---|---|---|")
        for f in comp:
            d = json.load(open(f))
            print(f"| {L.competitor_name(d['mode'])} | {d['seed']} | {d['H_fit']:.3f} | {d['test_loglik_exact_law_twoscale_coords_mean']:.3f} | "
                  f"{d['test_loglik_exact_law_grid_coords_mean']:.3f} | {d['sigma_years_standard_scale']:.3f} |")
            out.setdefault("competitor", []).append({k: d[k] for k in ("mode", "seed", "H_fit", "test_loglik_exact_law_twoscale_coords_mean",
                                                                        "test_loglik_exact_law_grid_coords_mean", "sigma_years_standard_scale")})
    with open(os.path.join(RES, f"summary_n4_s{a.s}.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
