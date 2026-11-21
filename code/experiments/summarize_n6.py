r"""Summarise N6 (ablations) with the reading rule fixed on 15 Nov 2026 (about 21:38, before any run).

Per variant and H_true: paired Delta = ll(base) - ll(variant) over data seeds (grid yardstick, oracle H; base =
base_cos for 'classical' and 'raw', base otherwise), mean and sd (n = number of seeds).
  helps  : Delta >  sd(Delta)          hurts : Delta < -sd(Delta)          otherwise: no detectable effect
A component is claimed only where it helps at two or more of the three H_true values.  mf, low-rank and S rows are
cost-accuracy trade-offs: Delta is shown beside the wall-clock ratio of the jobs (variant / its base).
Also shown: the same Delta with the two-scale yardstick, and the drift-RMSE ratio (variant / base, mean over seeds).
Run from work\code\experiments:  python summarize_n6.py      Out: work/results/n6/n6_summary.json
"""
import os, glob, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results", "n6")
TRADE = {"mf", "lr2", "lr4", "lr8", "s4", "s8"}


def load():
    runs = {}
    for f in glob.glob(os.path.join(RES, "n6_*_seed*_H*.json")):
        with open(f) as fh:
            d = json.load(fh)
        runs[(d["variant"], round(d["H_true"], 2), d["seed"])] = d
    return runs


def main():
    runs = load()
    variants = sorted({v for v, _, _ in runs if not v.startswith("base")})
    Hs = sorted({h for _, h, _ in runs})
    out = {"rule": "helps if Delta > sd, hurts if Delta < -sd (paired over seeds); claim if helps at >= 2 of 3 H",
           "rows": {}}
    for v in variants:
        pair = next(d["pair"] for (vv, _, _), d in runs.items() if vv == v)
        row = {"pair": pair, "by_H": {}}
        n_help = 0
        for H in Hs:
            seeds = sorted(s for vv, hh, s in runs if vv == v and hh == H and (pair, H, s) in runs)
            if not seeds:
                continue
            g = lambda k: np.array([runs[(pair, H, s)][k] - runs[(v, H, s)][k] for s in seeds])
            dg = g("test_loglik_exact_grid_law_mean"); dt = g("test_loglik_exact_law_twoscale_coords_mean")
            sd = float(dg.std(ddof=1)) if len(dg) > 1 else float("nan")
            m = float(dg.mean())
            verdict = "helps" if m > sd else ("hurts" if m < -sd else "no detectable effect")
            n_help += verdict == "helps"
            rr = np.mean([runs[(v, H, s)]["drift_rmse_occupation"] / runs[(pair, H, s)]["drift_rmse_occupation"] for s in seeds])
            wc = np.mean([runs[(v, H, s)]["train_seconds_job"] / runs[(pair, H, s)]["train_seconds_job"] for s in seeds])
            row["by_H"][f"{H:.2f}"] = {"seeds": seeds, "delta_grid_mean": round(m, 3), "delta_grid_sd": round(sd, 3),
                                       "delta_grid_per_seed": [round(x, 3) for x in dg],
                                       "delta_twoscale_mean": round(float(dt.mean()), 3),
                                       "drift_rmse_ratio": round(float(rr), 3), "wallclock_ratio": round(float(wc), 3),
                                       "component_verdict": verdict}
        row["claimed"] = n_help >= 2
        row["kind"] = "trade-off" if v in TRADE else "component"
        out["rows"][v] = row
    with open(os.path.join(RES, "n6_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("Delta = ll(base) - ll(variant), nats per test series (grid yardstick); positive = the base component helps")
    for v, row in out["rows"].items():
        cells = "  ".join(f"H={H}: {c['delta_grid_mean']:+.2f} ± {c['delta_grid_sd']:.2f} ({c['component_verdict']}; "
                          f"drift x{c['drift_rmse_ratio']:.2f}, time x{c['wallclock_ratio']:.2f})" for H, c in row["by_H"].items())
        print(f"{v:9s} vs {row['pair']:8s} [{row['kind']}] claimed={row['claimed']}  {cells}")


if __name__ == "__main__":
    main()
