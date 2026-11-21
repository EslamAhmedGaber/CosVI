r"""Summarise X1 (rules fixed 16 Nov 2026 07:03, amended 07:41 before any X1 result was read).

Part (i): runs n1b_{rep}_seed{s}_H{H}_hg95.json with the profile grid {0.5, 0.6, 0.7, 0.8, 0.9, 0.95}.
Two estimators of H per run:
  H_profile : vertex of the TRAINING bound over the grid groups (as N1b; unequal spacing allowed)
  H_val     : vertex of the HELD-OUT exact-law likelihood over the grid groups (grid yardstick; the test series act
              as the validation set, so the likelihood at H_val is not reported as a test score)
Rule: "recovered" if |H_hat - H_true| <= 0.05; an estimate on the top grid value 0.95 is reported as "at the edge".
Part (ii) (seeds 3-4) is summarised when present: paired Delta = ll(CosVI) - ll(r) over seeds 0-4 at the oracle group,
"gain" if mean > 2 SE, "loss" if mean < -2 SE, else "not distinguishable" (seeds 0-2 from N1b, 3-4 from X1).
Run from work\code\experiments:  python -X utf8 summarize_x1.py      Out: work/results/n1b/x1_summary.json
"""
import os, sys, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from n1b_representations import vertex

R = os.path.join(HERE, "..", "..", "results", "n1b")
NAMES = {"twoscale": "CosVI", "cosine": "CosVI-32", "grid": "exact grid", "markov": "MA-fBm law, published",
         "markov_tuned": "MA-fBm law, tuned", "markov_best": "MA-fBm law, best"}


def est(hg, vals):
    v = np.asarray(vals, float); i = int(np.nanargmax(v))
    edge = i in (0, len(hg) - 1)
    return (float(hg[i]) if edge else vertex(hg, v, i)), edge


def load(rep, H, s, tag):
    f = os.path.join(R, f"n1b_{rep}_seed{s}_H{H:.2f}{tag}.json")
    return json.load(open(f)) if os.path.exists(f) else None


def part_i():
    out = {}
    for rep in NAMES:
        for H in (0.6, 0.9):
            rows = []
            for s in range(5):
                d = load(rep, H, s, "_hg95")
                if d is None or s > 2:
                    continue
                hg = d["hgrid"]; G = len(hg)
                hp, ep = est(hg, d["elbo_train_own_model"][:G])
                hv, ev = est(hg, d["test_loglik_exact_grid_law_mean"][:G])
                old = load(rep, H, s, "")
                rows.append({"seed": s, "H_profile": round(hp, 4), "profile_at_edge": ep, "H_val": round(hv, 4),
                             "val_at_edge": ev, "N1b_H_profile_old_grid": round(old["H_profile_own_model"], 4) if old else None,
                             "heldout_oracle": round(d["test_loglik_exact_grid_law_mean"][-1], 3),
                             "heldout_per_group": [round(x, 3) for x in d["test_loglik_exact_grid_law_mean"]]})
            if not rows:
                continue
            hp = np.array([r["H_profile"] for r in rows]); hv = np.array([r["H_val"] for r in rows])
            out[f"{rep}|{H}"] = {"name": NAMES[rep], "H_true": H, "n": len(rows), "runs": rows,
                                 "H_profile_mean": round(hp.mean(), 3), "H_profile_sd": round(hp.std(ddof=1), 3) if len(hp) > 1 else None,
                                 "H_val_mean": round(hv.mean(), 3), "H_val_sd": round(hv.std(ddof=1), 3) if len(hv) > 1 else None,
                                 "profile_recovered_runs": int(np.sum(np.abs(hp - H) <= 0.05)),
                                 "val_recovered_runs": int(np.sum(np.abs(hv - H) <= 0.05)),
                                 "profile_at_edge_runs": int(sum(r["profile_at_edge"] for r in rows)),
                                 "val_at_edge_runs": int(sum(r["val_at_edge"] for r in rows))}
    return out


def part_ii():
    out = {}
    for H in (0.6, 0.75, 0.9):
        for r in ("grid", "markov", "markov_best"):
            dd = []
            for s in range(5):
                tag = "" if s <= 2 else "_hg95"
                a, b = load("twoscale", H, s, tag), load(r, H, s, tag)
                if a and b:
                    dd.append(a["test_loglik_exact_grid_law_mean"][-1] - b["test_loglik_exact_grid_law_mean"][-1])
            if len(dd) >= 2:
                m, sd = float(np.mean(dd)), float(np.std(dd, ddof=1)); se = sd / np.sqrt(len(dd))
                verdict = "gain" if m > 2 * se else ("loss" if m < -2 * se else "not distinguishable")
                out[f"{r}|{H}"] = {"n": len(dd), "delta_mean": round(m, 3), "delta_sd": round(sd, 3), "se": round(se, 3),
                                   "per_seed": [round(x, 3) for x in dd], "verdict_n5_rule": verdict if len(dd) == 5 else "pending (n < 5)"}
    return out


if __name__ == "__main__":
    res = {"rule": "X1 (rule fixed 16 Nov 2026 07:03, amended 07:41, before any result was read)", "part_i": part_i(), "part_ii": part_ii()}
    with open(os.path.join(R, "x1_summary.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("Part (i): estimates of H (mean +- sd over seeds 0-2; recovered = |H - H_true| <= 0.05; edge = top value 0.95)")
    for k, v in res["part_i"].items():
        print(f"  {v['name']:24s} H_true={v['H_true']}:  profile {v['H_profile_mean']:.3f} +- {v['H_profile_sd']}"
              f" (recovered {v['profile_recovered_runs']}/{v['n']}, edge {v['profile_at_edge_runs']})  |  held-out"
              f" {v['H_val_mean']:.3f} +- {v['H_val_sd']} (recovered {v['val_recovered_runs']}/{v['n']}, edge {v['val_at_edge_runs']})"
              f"  old-grid profile {[r['N1b_H_profile_old_grid'] for r in v['runs']]}")
    print("Part (ii): paired Delta over seeds (n < 5 until seeds 3-4 run)")
    for k, v in res["part_ii"].items():
        print(f"  {k:18s} n={v['n']} {v['delta_mean']:+.3f} +- {v['delta_sd']:.3f}  {v['verdict_n5_rule']}")
