r"""Summarise N2 (ELBO tightness against the same exact evidence) over data seeds and H_true.

Reads work/results/n2/n2_seed*_H*.json (written by n2_elbo_tightness.py) and prints, per representation:
  gap      = reference - bound (nats per series; reference = two-scale IWAE at K max, the protocol's reference)
  own-ref  = own-law IWAE (K = 512) - reference (how far the representation's own model evidence is from the exact one)
for the oracle group (H = H_true, the last group) and averaged over all six groups; mean +- sd over data seeds.
Reference diagnostics: |ref(K max) - ref(K previous)| (convergence check) and grid IWAE - two-scale IWAE at K max
(two lower bounds of the same exact evidence).  Sensitivity (added after seeing the results, labelled as such):
the gap against the larger of the two K-max references.
Run from work\code\experiments:  python summarize_n2.py
Out: work/results/n2/n2_summary.json
"""
import os, glob, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results", "n2")
ORDER = ["twoscale", "cosine64", "cosine32", "cosine16", "grid", "markov", "markov_tuned", "markov_best"]


def ms(x):
    x = np.asarray(x, float)
    return [round(float(x.mean()), 3), round(float(x.std(ddof=1)) if x.size > 1 else 0.0, 3), int(x.size)]


def main():
    allruns = []
    for f in sorted(glob.glob(os.path.join(RES, "n2_seed*_H*.json"))):
        with open(f) as fh:
            allruns.append(json.load(fh))
    out = {"files": len(allruns), "units": "nats per series (mean over the 16 test series)", "by_model": {}}
    # frozen model: twoscale (CosVI) = N2 protocol; grid = N2-x cross-check (files *_model-grid.json)
    for model in sorted({r.get("model_rep", "twoscale") for r in allruns}, key=lambda m: m != "twoscale"):
        runs = [r for r in allruns if r.get("model_rep", "twoscale") == model]
        out["by_model"][model] = summarise(runs)
    with open(os.path.join(RES, "n2_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    for model, s in out["by_model"].items():
        print(f"\n===== frozen model trained with: {model}  ({'N2 protocol' if model == 'twoscale' else 'N2-x cross-check'})")
        show(s)


def summarise(runs):
    Hs = sorted({r["H_true"] for r in runs})
    out = {"runs": len(runs), "by_H": {}}
    for H in Hs:
        rr = [r for r in runs if r["H_true"] == H]
        Ks = sorted(int(k) for k in rr[0]["reference_twoscale_iwae"])
        conv, gvt = [], []
        for r in rr:
            a = np.array(r["reference_twoscale_iwae"][str(Ks[-1])])
            b = np.array(r["reference_twoscale_iwae"][str(Ks[-2])])
            conv.append(np.abs(a - b))
            gvt.append(np.array(r["reference_grid_iwae_Kmax"]) - a)
        conv, gvt = np.concatenate(conv), np.concatenate(gvt)
        d = {"seeds": [r["seed"] for r in rr],
             "reference_change_Kmax_vs_Kprev": {"mean": round(float(conv.mean()), 3), "max": round(float(conv.max()), 3)},
             "grid_minus_twoscale_reference_Kmax": {"mean": round(float(gvt.mean()), 3), "min": round(float(gvt.min()), 3),
                                                    "max": round(float(gvt.max()), 3)},
             "reps": {}}
        for rep in ORDER:
            g_or, g_all, o_or, o_all, gb_or = [], [], [], [], []
            for r in rr:
                p = r["representations"][rep]
                ref = np.array(r["reference_twoscale_iwae"][str(Ks[-1])])
                best = np.maximum(ref, np.array(r["reference_grid_iwae_Kmax"]))
                gap = np.array(p["gap_to_reference"]); own = np.array(p["own_evidence_minus_reference"])
                g_or.append(gap[-1]); g_all.append(gap.mean()); o_or.append(own[-1]); o_all.append(own.mean())
                gb_or.append(best[-1] - p["elbo_own"][-1])
            d["reps"][rep] = {"D": rr[0]["representations"][rep]["D"],
                              "gap_oracle": ms(g_or), "gap_all_groups": ms(g_all),
                              "own_minus_ref_oracle": ms(o_or), "own_minus_ref_all_groups": ms(o_all),
                              "gap_oracle_vs_best_reference": ms(gb_or)}
        out["by_H"][f"{H:.2f}"] = d
    return out


def show(out):
    print(f"N2 summary from {out['runs']} runs; nats per series; mean (sd over data seeds)")
    for H, d in out["by_H"].items():
        print(f"\nH_true = {H}  seeds {d['seeds']}  reference |K max - K prev| mean {d['reference_change_Kmax_vs_Kprev']['mean']}"
              f" max {d['reference_change_Kmax_vs_Kprev']['max']};  grid - two-scale ref at K max mean"
              f" {d['grid_minus_twoscale_reference_Kmax']['mean']} [{d['grid_minus_twoscale_reference_Kmax']['min']},"
              f" {d['grid_minus_twoscale_reference_Kmax']['max']}]")
        print(f"  {'rep':13s} {'D':>3s}  {'gap oracle':>14s}  {'gap all':>14s}  {'own-ref oracle':>15s}  {'own-ref all':>14s}"
              f"  {'gap vs best ref':>15s}")
        for rep, v in d["reps"].items():
            f = lambda x: f"{x[0]:7.2f} ({x[1]:.2f})"
            print(f"  {rep:13s} {v['D']:3d}  {f(v['gap_oracle']):>14s}  {f(v['gap_all_groups']):>14s}"
                  f"  {f(v['own_minus_ref_oracle']):>15s}  {f(v['own_minus_ref_all_groups']):>14s}"
                  f"  {f(v['gap_oracle_vs_best_reference']):>15s}")


if __name__ == "__main__":
    main()
