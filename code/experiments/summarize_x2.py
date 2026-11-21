r"""Summarise X2 (two-dimensional damped Duffing system; rule fixed 16 Nov 2026 07:03).

Rule: per H, paired Delta_r = ll(CosVI) - ll(r) over data seeds 0-2 (held-out log-likelihood per test series under
the exact grid law, grid-coordinate refit = primary; two-scale refit = secondary); CosVI "ahead" if Delta > sd,
"behind" if Delta < -sd, else "tied".  The one-dimensional conclusions CARRY OVER if no representation is ahead of
CosVI at any H AND CosVI's drift RMSE (occupation-weighted) is lower than both MA-fBm laws in at least 14 of the 18
paired comparisons (3 H x 3 seeds x 2 laws); otherwise "does not carry over", with the rows.
Run from work\code\experiments:  python -X utf8 summarize_x2.py      Out: work/results/n8/x2_summary.json
"""
import os, json, glob
import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "n8")
REPS = {"twoscale": "CosVI", "cosine": "CosVI-32", "grid": "exact grid", "markov": "MA-fBm law, published",
        "markov_best": "MA-fBm law, best"}
YARD = {"grid_law": "test_loglik_exact_grid_law_mean", "twoscale_coords": "test_loglik_exact_law_twoscale_coords_mean"}


def ms(v):
    v = np.asarray(v, float)
    return [round(float(v.mean()), 3), round(float(v.std(ddof=1)), 3) if len(v) > 1 else None]


def main():
    runs = {}
    for f in glob.glob(os.path.join(R, "n8_*_seed*_H*.json")):
        d = json.load(open(f))
        runs[(d["rep"], round(d["H_true"], 2), d["seed"])] = d
    res = {"rule": "X2", "n_records": len(runs), "by_H": {}}
    wins_total, ahead_any = 0, False
    for H in (0.6, 0.75, 0.9):
        row = {"reps": {}, "paired_vs_cosvi": {}}
        seeds = sorted(s for (r, h, s) in runs if r == "twoscale" and h == H)
        for rep, name in REPS.items():
            ds = [runs[(rep, H, s)] for s in seeds if (rep, H, s) in runs]
            row["reps"][rep] = {"name": name, "n": len(ds),
                                "ll_grid_law": ms([d[YARD["grid_law"]] for d in ds]),
                                "ll_twoscale_coords": ms([d[YARD["twoscale_coords"]] for d in ds]),
                                "drift_rmse": ms([d["drift_rmse_occupation"] for d in ds]),
                                "sigma": np.mean([d["sigma"] for d in ds], 0).round(3).tolist()}
            if rep == "twoscale":
                continue
            p = {}
            for yk, key in YARD.items():
                dl = np.array([runs[("twoscale", H, s)][key] - runs[(rep, H, s)][key] for s in seeds])
                sd = float(dl.std(ddof=1))
                p[yk] = {"mean": round(float(dl.mean()), 3), "sd": round(sd, 3), "per_seed": dl.round(3).tolist(),
                         "reading": "ahead" if dl.mean() > sd else ("behind" if dl.mean() < -sd else "tied")}
            w = sum(int(runs[("twoscale", H, s)]["drift_rmse_occupation"] < runs[(rep, H, s)]["drift_rmse_occupation"])
                    for s in seeds)
            p["drift_wins"] = f"{w}/{len(seeds)}"
            row["paired_vs_cosvi"][rep] = p
            if p["grid_law"]["reading"] == "behind":
                ahead_any = True
            if rep in ("markov", "markov_best"):
                wins_total += w
        res["by_H"][str(H)] = row
    res["gate"] = {"no_representation_ahead_of_cosvi": not ahead_any, "drift_wins_vs_lifts": wins_total,
                   "drift_wins_required": 14, "of": 18,
                   "verdict": "carries over" if (not ahead_any and wins_total >= 14) else "does not carry over"}
    with open(os.path.join(R, "x2_summary.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    for H, row in res["by_H"].items():
        print(f"H = {H}")
        for rep, v in row["reps"].items():
            print(f"   {v['name']:24s} n={v['n']} ll {v['ll_grid_law'][0]:7.2f} +- {v['ll_grid_law'][1]}  "
                  f"(two-scale refit {v['ll_twoscale_coords'][0]:7.2f})  drift {v['drift_rmse'][0]:.3f} +- {v['drift_rmse'][1]}")
        for rep, p in row["paired_vs_cosvi"].items():
            print(f"   CosVI - {rep:12s} {p['grid_law']['mean']:+.2f} +- {p['grid_law']['sd']:.2f} [{p['grid_law']['reading']}]"
                  f"  two-scale refit {p['twoscale_coords']['mean']:+.2f} [{p['twoscale_coords']['reading']}]  drift wins {p['drift_wins']}")
    print("gate:", res["gate"])


if __name__ == "__main__":
    main()
