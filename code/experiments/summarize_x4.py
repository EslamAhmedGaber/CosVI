r"""Summarise X4 (calibration of the latent-path uncertainty; 05_EXPERIMENTS.md section 3b, rule fixed 16 Nov 07:03,
seeds amended 07:12 before any run).

Per representation and H_true: mean +- sd over the available seeds of coverage of the central 50 % and 90 % intervals,
calibration error e = mean(|cov50 - 0.5|, |cov90 - 0.9|), CRPS and the 90 % interval score.  Paired over seeds against
CosVI: CosVI "better calibrated" than r if mean(e_r - e_CosVI) > sd, "worse" if < -sd, else "tied"; the same rule for
CRPS (lower is better).  Seeds 1-2 for all six representations; seeds 3-4 (tag _hg95) added for CosVI, the exact grid
and the published and best MA-fBm laws when present.  With two seeds the sd is crude: the reading is final only with
the seeds 3-4 runs (n = 4).
Run from work\code\experiments:  python -X utf8 summarize_x4.py      Out: work/results/calib/x4_summary.json
"""
import os, json, glob
import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "calib")
REPS = ["twoscale", "cosine", "grid", "markov", "markov_tuned", "markov_best"]
NAMES = {"twoscale": "CosVI", "cosine": "CosVI-32", "grid": "exact grid", "markov": "MA-fBm law, published",
         "markov_tuned": "MA-fBm law, tuned", "markov_best": "MA-fBm law, best"}


def runs(rep, H):
    out = {}
    for f in glob.glob(os.path.join(R, f"calib_{rep}_seed*_H{H:.2f}*.json")):
        if "smoke" in f:
            continue
        d = json.load(open(f))
        out[d["seed"]] = d
    return out


def verdict(diffs):
    d = np.asarray(diffs, float)
    if len(d) < 2:
        return "n < 2"
    m, sd = d.mean(), d.std(ddof=1)
    return "CosVI better" if m > sd else ("CosVI worse" if m < -sd else "tied")


def main():
    res = {"rule": "05 3b X4", "by": {}}
    for H in (0.6, 0.75, 0.9):
        base = runs("twoscale", H)
        for rep in REPS:
            rr = runs(rep, H)
            if not rr:
                continue
            seeds = sorted(rr)
            e = np.array([rr[s]["calibration_error"] for s in seeds])
            c50 = np.array([rr[s]["coverage50"] for s in seeds]); c90 = np.array([rr[s]["coverage90"] for s in seeds])
            crps = np.array([rr[s]["crps_mean"] for s in seeds]); isc = np.array([rr[s]["interval_score90_mean"] for s in seeds])
            common = [s for s in seeds if s in base]
            entry = {"name": NAMES[rep], "seeds": seeds,
                     "cov50": [round(c50.mean(), 3), round(c50.std(ddof=1), 3) if len(seeds) > 1 else None],
                     "cov90": [round(c90.mean(), 3), round(c90.std(ddof=1), 3) if len(seeds) > 1 else None],
                     "cal_error": [round(e.mean(), 4), round(e.std(ddof=1), 4) if len(seeds) > 1 else None],
                     "crps": [round(crps.mean(), 5), round(crps.std(ddof=1), 5) if len(seeds) > 1 else None],
                     "is90": [round(isc.mean(), 4), round(isc.std(ddof=1), 4) if len(seeds) > 1 else None],
                     "pit_hist20": np.mean([rr[s]["pit_hist20"] for s in seeds], 0).round(4).tolist(),
                     "coverage_curve": {lev: round(float(np.mean([rr[s]["coverage_central"][lev] for s in seeds])), 4)
                                        for lev in rr[seeds[0]]["coverage_central"]}}
            if rep != "twoscale" and common:
                entry["vs_cosvi_cal_error"] = verdict([rr[s]["calibration_error"] - base[s]["calibration_error"] for s in common])
                entry["vs_cosvi_crps"] = verdict([rr[s]["crps_mean"] - base[s]["crps_mean"] for s in common])
            res["by"][f"{rep}|{H}"] = entry
    with open(os.path.join(R, "x4_summary.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("coverage 50/90 % (target 0.50/0.90), calibration error, CRPS; paired verdict against CosVI (n = seeds)")
    for k, v in res["by"].items():
        print(f"  {v['name']:24s} H={k.split('|')[1]:5s} n={len(v['seeds'])} cov {v['cov50'][0]:.3f}/{v['cov90'][0]:.3f}"
              f"  err {v['cal_error'][0]:.3f}  CRPS {v['crps'][0]:.4f}  "
              f"{v.get('vs_cosvi_cal_error', '')}  {('CRPS: ' + v['vs_cosvi_crps']) if 'vs_cosvi_crps' in v else ''}")


if __name__ == "__main__":
    main()
