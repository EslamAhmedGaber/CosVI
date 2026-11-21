r"""Summarise X8 (coordinates versus solver; protocol fixed 18 Nov 2026 14:35).

(a) exact linear-Gaussian check: pass flags and differences (x8a_linear_check.json).
(b) saved N1b-protocol models under two evaluators (smooth = CosVI's solver; pl = the grid's solver in CosVI-PL
    coordinates): paired Delta_r = ll(CosVI) - ll(r) over the seeds both have; ahead / tied / behind at +-1 sd;
    "robust to the evaluation map" if every classification agrees under both evaluators.
(c) training control: per H, contrasts paired over seeds (positive = the first arm is better):
      coordinate effect = CosVI-PL - grid      solver effect = CosVI - CosVI-PL      total = CosVI - grid
    for the held-out log-likelihood under the pl evaluator (primary) and the smooth evaluator, and for the drift RMSE
    (as rmse(second) - rmse(first)); an effect is "present" if |mean| > 2 SE (SE = sd / sqrt(n)), else
    "not distinguishable".  Decision on the primary metric, per H and overall (an effect counts overall if present
    with the same sign at two of the three H).
Run from work\code\experiments:  python -X utf8 summarize_x8.py      Out: work/results/x8/x8_summary.json
"""
import os, json, glob
import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "x8")
HS = (0.6, 0.75, 0.9)
ARM = {"twoscale": 0, "twoscale_pl": 1, "grid": 2}


def stat(d):
    d = np.asarray(d, float); n = len(d)
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean": round(float(d.mean()), 3), "sd": round(sd, 3), "se": round(se, 3), "per_seed": d.round(3).tolist(),
            "verdict": ("present, +" if d.mean() > 2 * se else "present, -" if d.mean() < -2 * se else "not distinguishable")
            if n > 1 else "n = 1"}


def part_b():
    out = {}
    for H in HS:
        recs = {}
        for f in glob.glob(os.path.join(R, f"x8b_seed*_H{H:.2f}.json")):
            d = json.load(open(f))
            for i, m in enumerate(d["models"]):
                recs[(m["rep"], d["seed"])] = {e: d["test_loglik"][e]["mean"][i] for e in ("smooth", "pl")} | \
                    {"drift": d["drift_rmse_occupation"][i], "source": m["source"]}
        row = {}
        for r in ("cosine", "grid", "markov", "markov_tuned", "markov_best"):
            seeds = sorted(s for (rr, s) in recs if rr == r and ("twoscale", s) in recs)
            if not seeds:
                continue
            row[r] = {"seeds": seeds}
            for e in ("smooth", "pl"):
                dl = np.array([recs[("twoscale", s)][e] - recs[(r, s)][e] for s in seeds])
                sd = float(dl.std(ddof=1)) if len(dl) > 1 else float("nan")
                row[r][e] = {"mean": round(float(dl.mean()), 3), "sd": round(sd, 3), "per_seed": dl.round(3).tolist(),
                             "reading": "ahead" if dl.mean() > sd else ("behind" if dl.mean() < -sd else "tied")}
            row[r]["same_reading"] = row[r]["smooth"]["reading"] == row[r]["pl"]["reading"]
        out[str(H)] = row
    allrows = [v for row in out.values() for v in row.values()]
    return {"by_H": out, "robust_to_evaluation_map": bool(allrows) and all(v["same_reading"] for v in allrows),
            "changed_rows": [f"H={H} {r}" for H, row in out.items() for r, v in row.items() if not v["same_reading"]]}


def part_c():
    out = {}
    for H in HS:
        runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, f"x8c_seed*_H{H:.2f}.json")))]
        if not runs:
            continue
        g = lambda d, key, arm: d["test_loglik"][key]["mean"][ARM[arm]] if key in ("pl", "smooth") else d[key][ARM[arm]]
        row = {"seeds": [d["seed"] for d in runs], "arms": {}}
        for arm in ARM:
            row["arms"][arm] = {k: [round(float(np.mean([g(d, k, arm) for d in runs])), 3),
                                    round(float(np.std([g(d, k, arm) for d in runs], ddof=1)), 3) if len(runs) > 1 else None]
                                for k in ("pl", "smooth", "drift_rmse_occupation", "elbo_train_own_model", "sigma")}
        row["train_seconds"] = round(float(np.mean([d["train_seconds"] for d in runs])), 1)
        row["inert_block_max_change"] = max(max(d["inert_block_check"].values()) for d in runs)
        con = {}
        for name, a, b in (("coordinate", "twoscale_pl", "grid"), ("solver", "twoscale", "twoscale_pl"), ("total", "twoscale", "grid")):
            con[name] = {"pl": stat([g(d, "pl", a) - g(d, "pl", b) for d in runs]),
                         "smooth": stat([g(d, "smooth", a) - g(d, "smooth", b) for d in runs]),
                         "drift": stat([g(d, "drift_rmse_occupation", b) - g(d, "drift_rmse_occupation", a) for d in runs])}
        row["contrasts"] = con
        c, s = con["coordinate"]["pl"]["verdict"], con["solver"]["pl"]["verdict"]
        row["decision"] = ("coordinates" if c.startswith("present") and not s.startswith("present") else
                           "solver" if s.startswith("present") and not c.startswith("present") else
                           "both" if c.startswith("present") and s.startswith("present") else "neither")
        out[str(H)] = row
    overall = {}
    for name in ("coordinate", "solver"):
        v = [out[H]["contrasts"][name]["pl"]["verdict"] for H in out]
        overall[name] = ("present, +" if sum(x == "present, +" for x in v) >= 2 else
                         "present, -" if sum(x == "present, -" for x in v) >= 2 else "not established")
    c, s = overall["coordinate"].startswith("present"), overall["solver"].startswith("present")
    overall["decision"] = "coordinates" if c and not s else "solver" if s and not c else "both" if c and s else "neither"
    return {"by_H": out, "overall": overall}


def main():
    res = {"rule": "X8"}
    fa = os.path.join(R, "x8a_linear_check.json")
    if os.path.exists(fa):
        a = json.load(open(fa))
        res["a"] = {"pass_all": a["pass_all"],
                    "max_abs_diff_pl_vs_grid": max(max(v["max_abs_diff_vs_grid"]["twoscale_pl"].values()) for v in a["by_H"].values()),
                    "max_abs_diff_smooth_vs_grid": {H: v["max_abs_diff_vs_grid"]["twoscale"] for H, v in a["by_H"].items()},
                    "cov_relerr_grid_values": {H: v["cov_relerr_grid_values"] for H, v in a["by_H"].items()},
                    "product_family_gap_mean": {H: round(float(np.mean(v["gap"]["grid"])), 3) for H, v in a["by_H"].items()}}
    res["b"] = part_b()
    res["c"] = part_c()
    with open(os.path.join(R, "x8_summary.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    if "a" in res:
        print("(a)", res["a"]["pass_all"], "max |PL - grid|", res["a"]["max_abs_diff_pl_vs_grid"], "gap", res["a"]["product_family_gap_mean"])
    for H, row in res["b"]["by_H"].items():
        for r, v in row.items():
            print(f"(b) H={H} CosVI-{r:12s} seeds {v['seeds']}  smooth {v['smooth']['mean']:+.2f}+-{v['smooth']['sd']:.2f} [{v['smooth']['reading']}]"
                  f"  pl {v['pl']['mean']:+.2f}+-{v['pl']['sd']:.2f} [{v['pl']['reading']}]")
    print("(b) robust:", res["b"]["robust_to_evaluation_map"], res["b"]["changed_rows"])
    for H, row in res["c"]["by_H"].items():
        print(f"(c) H={H} seeds {row['seeds']} decision {row['decision']}")
        for name, v in row["contrasts"].items():
            print(f"     {name:10s} pl {v['pl']['mean']:+.2f} (se {v['pl']['se']}) {v['pl']['verdict']:22s} smooth {v['smooth']['mean']:+.2f} {v['smooth']['verdict']:22s}"
                  f" drift {v['drift']['mean']:+.3f} {v['drift']['verdict']}")
    print("(c) overall:", res["c"]["overall"])


if __name__ == "__main__":
    main()
