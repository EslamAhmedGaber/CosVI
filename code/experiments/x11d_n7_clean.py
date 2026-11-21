"""X11d: Clean N7 summary separating standard-scale qualifying lifts from the un-rescaled
Mandelbrot-Van Ness cancellation artifact at H=0.90 (CPU, reads results/n7/n7_exact.json and n7_cost.json).

Out: results/x11/n7_clean_summary.json
"""
import os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results")
OUT = os.path.join(RES, "x11")
os.makedirs(OUT, exist_ok=True)

ex = json.load(open(os.path.join(RES, "n7", "n7_exact.json")))
co = json.load(open(os.path.join(RES, "n7", "n7_cost.json")))["rows"]

def step_cost_2000(method, rep, T, J=None, K=None):
    for r in co:
        if r["method"] == method and abs(r["T"] - T) < 1e-9:
            if method == "ours" and r.get("rep") == rep:
                if rep != "cosine" or r.get("J") == J:
                    return 2000.0 * r["step_s"]
            elif method == "port" and r.get("K") == K:
                return 2000.0 * r["step_s"]
    raise KeyError((method, rep, T, J, K))

by_H = {}
for H in (0.6, 0.75, 0.9):
    rows = []
    for r in ex["rows"]:
        if abs(r["H"] - H) > 1e-6:
            continue
        T = r["T"]
        Jstar = r["J_star_2pct"]
        # Cheapest qualifying CosVI (standard fBm scale)
        cands_c = [("two-scale", step_cost_2000("ours", "twoscale", T))]
        if Jstar is not None:
            cands_c.append((f"cosine J={Jstar}", step_cost_2000("ours", "cosine", T, J=Jstar)))
        best_c = min(cands_c, key=lambda x: x[1])

        # Cheapest qualifying STANDARD-SCALE lift (excluding un-rescaled published_coded)
        cands_l_std = []
        if r["err_lift_published_rescaled"] <= 0.02:
            cands_l_std.append(("published (rescaled, K=5)", step_cost_2000("port", None, T, K=5)))
        if r["err_lift_best"] <= 0.02:
            cands_l_std.append(("best (rescaled, K=12)", step_cost_2000("port", None, T, K=12)))
        best_l_std = min(cands_l_std, key=lambda x: x[1])

        rows.append({
            "T": T, "m": r["m"], "J_star": Jstar,
            "err_lift_published_coded": r["err_lift_published_coded"],
            "err_lift_published_rescaled": r["err_lift_published_rescaled"],
            "err_lift_best": r["err_lift_best"],
            "cosvi_cheapest": list(best_c),
            "lift_std_cheapest": list(best_l_std),
            "ratio_std_lift_over_cosvi": best_l_std[1] / best_c[1],
        })
    by_H[f"{H:.2f}"] = rows

out = {"note": "Standard-scale qualifying lifts only (published_coded at H=0.90 is 1.28% only via cancellation of un-rescaled c_H with kernel error; rescaled published is 9.35%).",
       "by_H": by_H}
json.dump(out, open(os.path.join(OUT, "n7_clean_summary.json"), "w"), indent=1)
for Hk, rows in by_H.items():
    print(f"H={Hk}: " + " | ".join(f"T={int(r['T'])}: {r['cosvi_cheapest'][0]}({r['cosvi_cheapest'][1]:.0f}s) vs {r['lift_std_cheapest'][0]}({r['lift_std_cheapest'][1]:.0f}s) [{r['ratio_std_lift_over_cosvi']:.2f}x]" for r in rows))
