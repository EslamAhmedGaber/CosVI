r"""Summarise X9 (evaluator check on the real series; 05_EXPERIMENTS.md section 3b, protocol fixed 18 Nov 2026 15:20).

Per series and comparator r: paired Delta_r = ll(CosVI) - ll(r) over training seeds 0-2 under each evaluator
(smooth = CosVI's solver, pl = the grid's solver); ahead / tied / behind at +-1 sd.  "Robust to the evaluation map"
if every classification and the leading representation agree under both evaluators.  Also: per evaluator the mean
held-out score of each representation, the smallest effective sample size and the largest normalised weight.
Run from work\code\experiments:  python -X utf8 summarize_x9.py      Out: work/results/x9/x9_summary.json
"""
import os, json, glob
import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "x9")
NAMES = {"twoscale": "CosVI", "cosine64": "CosVI-64", "grid": "exact grid", "markov": "MA-fBm law, published",
         "markov_tuned": "MA-fBm law, tuned"}


def main():
    res = {"rule": "05 3b X9", "by_series": {}}
    robust_all = True
    for series in ("nile", "temp"):
        runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, f"x9_{series}_seed[0-9].json")))]
        if not runs:
            continue
        reps = [m["rep"] for m in runs[0]["models"]]
        row = {"seeds": [d["seed"] for d in runs], "evaluators": {}, "paired_vs_cosvi": {}}
        for e in ("smooth", "pl"):
            m = np.array([d["evaluators"][e]["mean"] for d in runs])                       # (seeds, reps)
            row["evaluators"][e] = {"mean": dict(zip(reps, m.mean(0).round(3).tolist())),
                                    "sd": dict(zip(reps, m.std(0, ddof=1).round(3).tolist())) if len(runs) > 1 else None,
                                    "leading": NAMES[reps[int(np.argmax(m.mean(0)))]],
                                    "min_ess": float(min(np.min(d["evaluators"][e]["ess"]) for d in runs)),
                                    "max_weight": float(max(np.max(d["evaluators"][e]["max_weight"]) for d in runs))}
        i0 = reps.index("twoscale")
        same = row["evaluators"]["smooth"]["leading"] == row["evaluators"]["pl"]["leading"]
        for j, r in enumerate(reps):
            if r == "twoscale":
                continue
            p = {}
            for e in ("smooth", "pl"):
                dl = np.array([d["evaluators"][e]["mean"][i0] - d["evaluators"][e]["mean"][j] for d in runs])
                sd = float(dl.std(ddof=1)) if len(dl) > 1 else float("nan")
                p[e] = {"mean": round(float(dl.mean()), 3), "sd": round(sd, 3), "per_seed": dl.round(3).tolist(),
                        "reading": "ahead" if dl.mean() > sd else ("behind" if dl.mean() < -sd else "tied")}
            p["same_reading"] = p["smooth"]["reading"] == p["pl"]["reading"]
            same = same and p["same_reading"]
            row["paired_vs_cosvi"][r] = p
        row["robust_to_evaluation_map"] = bool(same)
        robust_all = robust_all and same
        res["by_series"][series] = row
    res["robust_all"] = bool(robust_all) if res["by_series"] else None
    with open(os.path.join(R, "x9_summary.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    for s, row in res["by_series"].items():
        print(s, "seeds", row["seeds"], "robust", row["robust_to_evaluation_map"])
        for e, v in row["evaluators"].items():
            print(f"   {e:6s} leading {v['leading']:12s} min ESS {v['min_ess']:.1f} max weight {v['max_weight']:.3f}  "
                  + "  ".join(f"{NAMES[k]} {m:.2f}" for k, m in v["mean"].items()))
        for r, p in row["paired_vs_cosvi"].items():
            print(f"   CosVI - {r:12s} smooth {p['smooth']['mean']:+.2f}+-{p['smooth']['sd']:.2f} [{p['smooth']['reading']}]"
                  f"  pl {p['pl']['mean']:+.2f}+-{p['pl']['sd']:.2f} [{p['pl']['reading']}]")


if __name__ == "__main__":
    main()
