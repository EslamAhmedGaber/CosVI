r"""Real series: how stable are the paired held-out scores? (descriptive, post hoc, 19 Nov 2026; reads stored records
only, no new fits).

For every paired comparison CosVI minus r on the two real series (r = CosVI-64, exact grid, MA-fBm law with published
and tuned nodes; this includes the leader and runner-up of each series), each training seed and each test window, five
readouts of the same frozen trained models are compared:
  A        smooth-stage evaluator, original run (refit 4000 steps, K = 1024)     results/n4 (Nile), results/n4t (temp.)
  B        smooth-stage evaluator, repeated run (fresh refit, fresh draws)        results/x9
  C        PL evaluator (the grid's solver)                                       results/x9
  D1, D2   coupled Gauss-Newton proposal, two independent sets of 512 draws       results/x10 (x10c_*)
Reported per comparison: the gap (mean over test windows, nats per window) per seed and readout, and its mean over
seeds; the replicate differences A - B (fresh refit and draws) and D1 - D2 (fresh draws) of the gap; the number of
window readouts with effective sample size below 10 and the largest normalised weight; the gap of readouts A and B
without the windows whose ESS in B is below 10 for either model; and the smallest and largest seed-mean gap over all
leave-one-window-out deletions, with the window whose deletion moves the gap of readout B the most (test windows are
numbered in time order). A ranking is called stable if the seed-mean gap keeps its sign in every readout, after the ESS
filter and under every single-window deletion.
Run from work\code\experiments:  python -X utf8 x9_window_audit.py      Out: work/results/x9/x9_window_audit.json
"""
import os, json
import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
NAMES = {"twoscale": "CosVI", "cosine64": "CosVI-64", "grid": "exact grid", "markov": "MA-fBm law, published",
         "markov_tuned": "MA-fBm law, tuned"}
SEEDS = (0, 1, 2)


def stored_smooth(series, rep, seed, group_H):
    f = (os.path.join(RES, "n4", f"n4v2_{rep}_seed{seed}_s0.1.json") if series == "nile"
         else os.path.join(RES, "n4t", f"n4t_{rep}_seed{seed}_s0.1.json"))
    r = json.load(open(f))
    gi = r["groups"].index(f"H={group_H:.2f}")
    return np.array(r["test_loglik_exact_law_twoscale_coords_per_window"][gi], dtype=float)


def readouts(series, seed):
    x9 = json.load(open(os.path.join(RES, "x9", f"x9_{series}_seed{seed}.json")))
    out = {}
    for g, m in enumerate(x9["models"]):
        rep = m["rep"]
        ev = x9["evaluators"]
        x10 = json.load(open(os.path.join(RES, "x10", f"x10c_{series}_{rep}_seed{seed}.json")))
        d = np.array([[w["sets"][k]["log_score"] for w in x10["windows"]] for k in (0, 1)])
        d_ess = np.array([[w["sets"][k]["ess"] for w in x10["windows"]] for k in (0, 1)])
        d_wmax = np.array([[w["sets"][k]["max_weight"] for w in x10["windows"]] for k in (0, 1)])
        out[rep] = {"A": stored_smooth(series, rep, seed, m["group_H"]),
                    "B": np.array(ev["smooth"]["per_window"][g]), "B_ess": np.array(ev["smooth"]["ess"][g]),
                    "B_wmax": np.array(ev["smooth"]["max_weight"][g]),
                    "C": np.array(ev["pl"]["per_window"][g]), "C_ess": np.array(ev["pl"]["ess"][g]),
                    "C_wmax": np.array(ev["pl"]["max_weight"][g]),
                    "D1": d[0], "D2": d[1], "D_ess": d_ess, "D_wmax": d_wmax}
    return out


def check_series(series):
    data = {s: readouts(series, s) for s in SEEDS}
    res = {}
    for r in ("cosine64", "grid", "markov", "markov_tuned"):
        gaps = {k: [] for k in ("A", "B", "C", "D1", "D2")}
        filt, loo, low, wmax, n_win = [], [], 0, 0.0, None
        for s in SEEDS:
            a, b = data[s]["twoscale"], data[s][r]
            n_win = len(a["B"])
            for k in gaps:
                gaps[k].append(float(np.mean(a[k] - b[k])))
            keep = (a["B_ess"] >= 10) & (b["B_ess"] >= 10)
            filt.append({"windows_kept": int(keep.sum()),
                         "A": float(np.mean((a["A"] - b["A"])[keep])) if keep.any() else None,
                         "B": float(np.mean((a["B"] - b["B"])[keep])) if keep.any() else None})
            for x in (a, b):
                low += int((x["B_ess"] < 10).sum() + (x["C_ess"] < 10).sum() + (x["D_ess"] < 10).sum())
                wmax = max(wmax, float(x["B_wmax"].max()), float(x["C_wmax"].max()), float(x["D_wmax"].max()))
        win, without, on = {}, {}, {}
        for k in ("A", "B", "C", "D1", "D2"):          # leave one window out, seed-mean gap
            per_w = np.array([[data[s]["twoscale"][k][w] - data[s][r][k][w] for w in range(n_win)] for s in SEEDS])
            loo += [float(np.mean(np.delete(per_w, w, axis=1))) for w in range(n_win)]
            win[k] = [float(x) for x in per_w.mean(axis=0)]
        # the window whose removal moves the seed-mean gap of readout B the most, and the gap with and without it
        wb = np.array(win["B"]); full_b = float(wb.mean())
        w_star = int(np.argmax([abs(float(np.delete(wb, w).mean()) - full_b) for w in range(n_win)]))
        for k in win:
            on[k] = win[k][w_star]
            without[k] = float(np.mean(np.delete(np.array(win[k]), w_star)))
        mean = {k: float(np.mean(v)) for k, v in gaps.items()}
        signs = [np.sign(v) for v in mean.values()] + [np.sign(f[k]) for f in filt for k in ("A", "B") if f[k] is not None]
        signs += [np.sign(np.mean([f[k] for f in filt if f[k] is not None])) for k in ("A", "B")]
        signs += [np.sign(min(loo)), np.sign(max(loo))]
        res[r] = {"comparison": f"CosVI minus {NAMES[r]}", "gap_per_seed": gaps, "gap_seed_mean": mean,
                  "gap_seed_sd_B": float(np.std(gaps["B"], ddof=1)),
                  "replicate_difference_A_minus_B": [x - y for x, y in zip(gaps["A"], gaps["B"])],
                  "replicate_difference_D1_minus_D2": [x - y for x, y in zip(gaps["D1"], gaps["D2"])],
                  "readouts_with_ess_below_10": low, "readouts_total": 2 * len(SEEDS) * n_win * 4,
                  "largest_normalised_weight": wmax, "ess_filtered_B_lt_10_removed": filt,
                  "leave_one_window_out_seed_mean": {"min": min(loo), "max": max(loo)},
                  "window_gap_seed_mean": win, "most_influential_window": w_star,
                  "gap_on_most_influential_window": on, "gap_without_most_influential_window": without,
                  "stable_sign": bool(len(set(signs)) == 1)}
    return {"windows": n_win, "by_comparison": res}


def main():
    out = {"rule": "descriptive, post hoc (19 Nov 2026): stable if the seed-mean gap keeps its sign in every readout, "
                   "after the ESS filter and under every single-window deletion",
           "series": {s: check_series(s) for s in ("nile", "temp")}}
    path = os.path.join(RES, "x9", "x9_window_audit.json")
    json.dump(out, open(path, "w"), indent=1)
    for s, v in out["series"].items():
        print(f"== {s} ({v['windows']} test windows)")
        for r, c in v["by_comparison"].items():
            m = c["gap_seed_mean"]; lo = c["leave_one_window_out_seed_mean"]
            print(f"  {c['comparison']:<36} A {m['A']:+.2f}  B {m['B']:+.2f}  C {m['C']:+.2f}  D {m['D1']:+.2f}/{m['D2']:+.2f}"
                  f" | sd(B) {c['gap_seed_sd_B']:.2f} | A-B {np.round(c['replicate_difference_A_minus_B'], 2)}"
                  f" D1-D2 {np.round(c['replicate_difference_D1_minus_D2'], 2)} | ESS<10 {c['readouts_with_ess_below_10']}/"
                  f"{c['readouts_total']} wmax {c['largest_normalised_weight']:.3f} | LOO [{lo['min']:+.2f}, {lo['max']:+.2f}]"
                  f" | ESS-filtered B {[None if f['B'] is None else round(f['B'], 2) for f in c['ess_filtered_B_lt_10_removed']]}"
                  f" kept {[f['windows_kept'] for f in c['ess_filtered_B_lt_10_removed']]} | stable {c['stable_sign']}")


if __name__ == "__main__":
    main()
