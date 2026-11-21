r"""Summarise X3 (Northern-Hemisphere temperature; rule fixed 16 Nov 2026 07:03, amended 07:12).

Per representation: profile H (vertex of the training bound over {0.5,...,0.9,0.95}) and the held-out log-likelihood per
test window under the exact law (two-scale refit) at the grid group nearest the profile H; mean +- sd over seeds 0-2.
The competitor's system (port, N4c harness): held-out log-likelihood with H learned (joint) and fixed (oracle 0.84).
Rule: CosVI is "best or tied" among the exact representations (tied = within the seed sd of the paired difference) and
"ahead" of each Markovian law and of the competitor's system by more than the seed sd; otherwise the leading
representation is reported.  The profile H is compared with the classical estimates, not judged against them.
Run from work\code\experiments:  python -X utf8 summarize_x3.py      Out: work/results/n4t/x3_summary.json
"""
import os, json, glob
import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "n4t")
PROV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "n4", "data", "provenance.json")
NAMES = {"twoscale": "CosVI", "cosine64": "CosVI-64", "grid": "exact grid", "markov": "MA-fBm law, published",
         "markov_tuned": "MA-fBm law, tuned"}


def ours(rep):
    out = {}
    for f in glob.glob(os.path.join(R, f"n4t_{rep}_seed*_s0.1.json")):
        d = json.load(open(f))
        hg = np.array(d["hgrid"]); ll = np.array(d["test_loglik_exact_law_twoscale_coords_mean"][:len(hg)])
        i = int(np.argmin(np.abs(hg - d["H_profile_general"])))
        out[d["seed"]] = {"H": d["H_profile_general"], "ll": float(ll[i]), "ll_groups": ll.round(2).tolist()}
    return out


def theirs(mode):
    out = {}
    for f in glob.glob(os.path.join(R, f"n4t_competitor_{mode}_seed*.json")):
        d = json.load(open(f))
        out[d["seed"]] = {"H": d["H_fit"], "ll": d["test_loglik_exact_law_twoscale_coords_mean"]}
    return out


def ms(v):
    v = np.asarray(v, float)
    return [round(float(v.mean()), 2), round(float(v.std(ddof=1)), 2) if len(v) > 1 else None]


def paired(a, b):
    s = sorted(set(a) & set(b))
    d = np.array([a[k]["ll"] - b[k]["ll"] for k in s])
    return {"n": len(s), "mean": round(float(d.mean()), 2), "sd": round(float(d.std(ddof=1)), 2) if len(s) > 1 else None,
            "per_seed": d.round(2).tolist()}


def main():
    res = {"rule": "X3", "by": {}, "paired_vs_cosvi": {}}
    base = ours("twoscale")
    for rep in NAMES:
        r = ours(rep)
        res["by"][rep] = {"name": NAMES[rep], "seeds": sorted(r), "H": ms([v["H"] for v in r.values()]),
                          "ll": ms([v["ll"] for v in r.values()])}
        if rep != "twoscale":
            res["paired_vs_cosvi"][rep] = paired(base, r)
    for mode in ("joint", "oracle"):
        t = theirs(mode)
        if t:
            res["by"][f"competitor_{mode}"] = {"name": f"MA-fBm system ({mode})", "seeds": sorted(t),
                                               "H": ms([v["H"] for v in t.values()]), "ll": ms([v["ll"] for v in t.values()])}
            res["paired_vs_cosvi"][f"competitor_{mode}"] = paired(base, t)
    ok_exact = all(p["mean"] >= -(p["sd"] or 0) for k, p in res["paired_vs_cosvi"].items() if k == "grid")
    ahead = {k: p["mean"] > (p["sd"] or 0) for k, p in res["paired_vs_cosvi"].items() if k in ("markov", "markov_tuned") or k.startswith("competitor")}
    lead = max(res["by"], key=lambda k: res["by"][k]["ll"][0])
    res["reading"] = {"cosvi_best_or_tied_among_exact": ok_exact, "cosvi_ahead_of": ahead, "leading_representation": res["by"][lead]["name"]}
    prov = json.load(open(PROV))["series"]["nhemi_temp"]
    res["classical_H"] = {k: round(prov[k], 3) for k in ("H_whittle_fgn", "H_aggregated_variance", "H_RS")}
    with open(os.path.join(R, "x3_summary.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    for k, v in res["by"].items():
        print(f"  {v['name']:26s} n={len(v['seeds'])}  H {v['H'][0]:.3f} +- {v['H'][1]}  held-out {v['ll'][0]:.2f} +- {v['ll'][1]}")
    for k, p in res["paired_vs_cosvi"].items():
        print(f"  CosVI - {k:20s} {p['mean']:+.2f} +- {p['sd']}  {p['per_seed']}")
    print("  reading:", res["reading"], "| classical H:", res["classical_H"])


if __name__ == "__main__":
    main()
