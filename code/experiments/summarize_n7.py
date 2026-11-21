r"""Summarise N7 with the crossover rule fixed on 15 Nov 2026 (21:59, before any run).

Inputs: results/n7/n7_exact.json (accuracy at convergence, CPU) and results/n7/n7_cost.json (time per step, GPU).
Cost at matched accuracy = time per step x 2000 for every configuration whose error at convergence is <= 2 %:
  CosVI side : cosine at J* (if J* exists) and two-scale (exact on the grid; error from the time step only)
  lift side  : published as coded, published rescaled (both K = 5 cost), best (K = 12 cost), each only if <= 2 %
Crossover: the smallest T at which the cheapest qualifying CosVI configuration costs more than the cheapest
qualifying lift; "no crossover in range" otherwise; a T where no lift qualifies has no lift cost at matched accuracy.
Run from work\code\experiments:  python summarize_n7.py      Out: work/results/n7/n7_summary.json
"""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results", "n7")
STEPS, TOL = 2000, 0.02


def main():
    ex = json.load(open(os.path.join(RES, "n7_exact.json")))["rows"]
    cost = [r for r in json.load(open(os.path.join(RES, "n7_cost.json")))["rows"] if "step_s" in r]
    step = lambda **k: next((r["step_s"] for r in cost if all(r.get(a) == b for a, b in k.items())), None)
    out = {"rule": "N7, rule fixed 15 Nov 2026 21:59, before the runs", "by_H": {}}
    for H in sorted({r["H"] for r in ex}):
        rows, cross = [], None
        for r in sorted((r for r in ex if r["H"] == H), key=lambda r: r["T"]):
            T = r["T"]
            ours = {}
            if r["J_star_2pct"] is not None:
                s = step(method="ours", rep="cosine", J=r["J_star_2pct"], T=T)
                if s is not None:
                    ours[f"cosine J={r['J_star_2pct']}"] = s * STEPS
            s = step(method="ours", rep="twoscale", T=T)
            if s is not None:
                ours["two-scale"] = s * STEPS
            lift = {}
            for lab, e, K in (("published (coded)", r["err_lift_published_coded"], 5),
                              ("published (rescaled)", r["err_lift_published_rescaled"], 5),
                              ("best K=12", r["err_lift_best"], 12)):
                s = step(method="port", K=K, T=T)
                if e <= TOL and s is not None:
                    lift[lab] = s * STEPS
            o = min(ours.items(), key=lambda kv: kv[1]) if ours else None
            l = min(lift.items(), key=lambda kv: kv[1]) if lift else None
            ours_more = bool(o and l and o[1] > l[1])
            if ours_more and cross is None:
                cross = T
            rows.append({"T": T, "J_star": r["J_star_2pct"], "err_lift": {"published_coded": r["err_lift_published_coded"],
                         "published_rescaled": r["err_lift_published_rescaled"], "best": r["err_lift_best"]},
                         "cosvi_cheapest": o, "lift_cheapest_qualifying": l, "cosvi_costs_more": ours_more})
        out["by_H"][f"{H:.2f}"] = {"rows": rows, "crossover_T": cross if cross is not None else "no crossover in range"}
    with open(os.path.join(RES, "n7_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    for H, d in out["by_H"].items():
        print(f"H = {H}: crossover {d['crossover_T']}")
        for r in d["rows"]:
            o = f"{r['cosvi_cheapest'][0]} {r['cosvi_cheapest'][1]:.0f} s" if r["cosvi_cheapest"] else "-"
            l = f"{r['lift_cheapest_qualifying'][0]} {r['lift_cheapest_qualifying'][1]:.0f} s" if r["lift_cheapest_qualifying"] else "none <= 2 %"
            e = r["err_lift"]
            print(f"  T={r['T']:>4}: J*={r['J_star']}; CosVI {o}; lift {l}; lift errors coded {e['published_coded']:.3f}"
                  f" rescaled {e['published_rescaled']:.3f} best {e['best']:.4f}")


if __name__ == "__main__":
    main()
