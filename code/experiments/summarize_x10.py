r"""Summarise X10 (proposal sensitivity; protocol fixed 18 Nov 2026, 19:28; this
summariser written before any X10 result was read).

(a) N2 cases (CosVI-trained oracle-H model, seeds 1-2, H in {0.6, 0.75, 0.9}); joint-proposal evidences ev_J of the
    smooth-stage (CosVI), grid and published-lift (markov K = 5) models.  md_J(r) = ev_J(r) - ev_J(CosVI) beside the
    archived md(r) = iwae_own_K512(r) - reference_K32768; g_J(r) = ev_J(r) - ELBO_own(r) beside the archived
    iwae_own_K512(r) - ELBO_own(r).  N2's cross-model statement holds in a case if ev_J(markov) < ev_J(grid) - u,
    u = |set difference| (grid) + |set difference| (markov); it holds overall if it holds in all six cases.
(b) Saved X8(b) models (CosVI, grid, markov, markov_best) under the grid evaluator with the joint proposal; paired
    Delta_r = ll(CosVI) - ll(r) per seed; ahead / tied / behind at +-1 sd; compared with the archived primary scores
    (product refit proposal, K = 512) of the same models and seeds; robust if all nine classifications agree.
Run from work\code\experiments:  python -X utf8 summarize_x10.py      Out: work/results/x10/x10_summary.json
"""
import os, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results")
X10 = os.path.join(RES, "x10")


def rec(part, rep, emap, H, s):
    f = os.path.join(X10, f"x10{part}_{rep}_on-{emap}_seed{s}_H{H:.2f}.json")
    return json.load(open(f)) if os.path.exists(f) else None


def cls(v):
    m, sd = float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")
    return m, sd, ("ahead" if m > sd else "behind" if m < -sd else "tied")


def part_a():
    out, holds = {}, []
    cand = os.path.join(X10, "n2_joint_candidates.json")   # joint-minus-product candidate contrasts, six N2 cases
    rv = {(c["H"], c["seed"]): c for c in json.load(open(cand))} if os.path.exists(cand) else {}
    for s in (1, 2):
        for H in (0.6, 0.75, 0.9):
            r = {m: rec("a", "twoscale", m, H, s) for m in ("twoscale", "grid", "markov")}
            if any(v is None or v["windows_ok"] < v["windows_total"] for v in r.values()):
                out[f"{H}|{s}"] = {"incomplete": {m: (None if v is None else f"{v['windows_ok']}/{v['windows_total']}")
                                                 for m, v in r.items()}}
                continue
            n2 = json.load(open(os.path.join(RES, "n2", f"n2_seed{s}_H{H:.2f}.json")))
            ref = n2["reference_twoscale_iwae"]["32768"][-1]
            reps = n2["representations"]
            ev = {m: r[m]["score"] for m in r}
            u = r["grid"]["set_difference"] + r["markov"]["set_difference"]
            ok = ev["markov"] < ev["grid"] - u
            holds.append(ok)
            row = {"ev_joint": ev, "reference_K32768": ref, "shift_of_reference": ev["twoscale"] - ref,
                   "u": u, "holds": ok,
                   "md_joint": {m: ev[m] - ev["twoscale"] for m in ("grid", "markov")},
                   "md_archived": {m: reps[m]["iwae_own_K512"][-1] - ref for m in ("twoscale", "grid", "markov")},
                   "gap_joint": {m: ev[m] - reps[m]["elbo_own"][-1] for m in ev},
                   "gap_archived": {m: reps[m]["iwae_own_K512"][-1] - reps[m]["elbo_own"][-1] for m in ev},
                   "min_ess": {m: r[m]["min_ess"] for m in r}, "max_weight": {m: r[m]["max_weight"] for m in r}}
            if (H, s) in rv and "independent_scores" in rv[(H, s)]:
                row["candidate_twoscale_score"] = float(np.mean(rv[(H, s)]["independent_scores"]))
            out[f"{H}|{s}"] = row
    verdict = ("holds" if holds and len(holds) == 6 and all(holds) else
               "incomplete" if len(holds) < 6 else "qualified")
    return {"by_case": out, "cases_complete": len(holds), "cases_holding": int(sum(holds)), "verdict": verdict}


def archived_primary(model_file):
    f = os.path.join(RES, "n1b", model_file.replace(".pt", ".json"))
    return json.load(open(f))["test_loglik_exact_grid_law_mean"][-1]


def part_b():
    by, changed, shifts = {}, [], {}
    for H in (0.6, 0.75, 0.9):
        seeds = (0, 1, 2, 3, 4) if H == 0.9 else (1, 2, 3, 4)
        sc, ar, ess, mw, flags = {}, {}, [], [], 0
        for rep in ("twoscale", "grid", "markov", "markov_best"):
            for s in seeds:
                d = rec("b", rep, "grid", H, s)
                if d is None or d["windows_ok"] < d["windows_total"]:
                    continue
                sc[(rep, s)] = d["score"]; ar[(rep, s)] = archived_primary(d["model_file"])
                ess.append(d["min_ess"]); mw.append(d["max_weight"]); flags += len(d["flagged_windows_ess_below_5pct"])
        rows = {}
        for r in ("grid", "markov", "markov_best"):
            ss = [s for s in seeds if ("twoscale", s) in sc and (r, s) in sc]
            if len(ss) < 2:
                rows[r] = {"seeds": ss, "incomplete": True}
                continue
            dj = [sc[("twoscale", s)] - sc[(r, s)] for s in ss]
            da = [ar[("twoscale", s)] - ar[(r, s)] for s in ss]
            mj, sdj, cj = cls(dj); ma, sda, ca = cls(da)
            rows[r] = {"seeds": ss, "joint": {"mean": mj, "sd": sdj, "reading": cj, "per_seed": dj},
                       "archived": {"mean": ma, "sd": sda, "reading": ca, "per_seed": da}, "same": cj == ca}
            if cj != ca:
                changed.append(f"H={H} {r}")
        for rep in ("twoscale", "grid", "markov", "markov_best"):
            v = [sc[(rep, s)] - ar[(rep, s)] for s in seeds if (rep, s) in sc]
            if v:
                shifts[f"{rep}|{H}"] = {"mean_shift": float(np.mean(v)), "min": float(np.min(v)),
                                        "max": float(np.max(v)), "n": len(v)}
        by[str(H)] = {"rows": rows, "min_ess": min(ess) if ess else None, "max_weight": max(mw) if mw else None,
                      "flagged_windows": flags}
    complete = all(not v.get("incomplete") for h in by.values() for v in h["rows"].values())
    return {"by_H": by, "score_shift_joint_minus_archived": shifts, "changed": changed,
            "robust": (complete and not changed) if complete else None}


def part_c():
    """X10(c), protocol 19:46 (summariser part written before any X10(c) result was read): real series, smooth-stage
    evaluator; paired Delta_r = ll(CosVI) - ll(r) over seeds 0-2 at +-1 sd, joint vs archived; robust if all eight
    classifications and the leader of each series agree."""
    reps = ["twoscale", "cosine64", "grid", "markov", "markov_tuned"]
    by, changed = {}, []
    for series in ("nile", "temp"):
        sc, ar, ess, mw, flags, inc = {}, {}, [], [], 0, []
        for rep in reps:
            for s in (0, 1, 2):
                f = os.path.join(X10, f"x10c_{series}_{rep}_seed{s}.json")
                if not os.path.exists(f):
                    inc.append(f"{rep}|{s} missing"); continue
                d = json.load(open(f))
                if d["windows_ok"] < d["windows_total"]:
                    inc.append(f"{rep}|{s} {d['windows_ok']}/{d['windows_total']}"); continue
                sc[(rep, s)] = d["score"]; ar[(rep, s)] = d["archived_smooth_ll"]
                ess.append(d["min_ess"]); mw.append(d["max_weight"]); flags += len(d["flagged_windows_ess_below_5pct"])
        rows = {}
        for r in reps[1:]:
            ss = [s for s in (0, 1, 2) if ("twoscale", s) in sc and (r, s) in sc]
            if len(ss) < 2:
                rows[r] = {"seeds": ss, "incomplete": True}; continue
            dj = [sc[("twoscale", s)] - sc[(r, s)] for s in ss]
            da = [ar[("twoscale", s)] - ar[(r, s)] for s in ss]
            mj, sdj, cj = cls(dj); ma, sda, ca = cls(da)
            rows[r] = {"seeds": ss, "joint": {"mean": mj, "sd": sdj, "reading": cj, "per_seed": dj},
                       "archived": {"mean": ma, "sd": sda, "reading": ca, "per_seed": da}, "same": cj == ca}
            if cj != ca:
                changed.append(f"{series} {r}")
        mean_j = {r: float(np.mean([sc[(r, s)] for s in (0, 1, 2) if (r, s) in sc])) for r in reps
                  if any((r, s) in sc for s in (0, 1, 2))}
        mean_a = {r: float(np.mean([ar[(r, s)] for s in (0, 1, 2) if (r, s) in ar])) for r in reps
                  if any((r, s) in ar for s in (0, 1, 2))}
        lead_j = max(mean_j, key=mean_j.get) if mean_j else None
        lead_a = max(mean_a, key=mean_a.get) if mean_a else None
        if lead_j != lead_a:
            changed.append(f"{series} leader {lead_a} -> {lead_j}")
        shifts = {r: float(np.mean([sc[(r, s)] - ar[(r, s)] for s in (0, 1, 2) if (r, s) in sc])) for r in mean_j}
        by[series] = {"rows": rows, "mean_joint": mean_j, "mean_archived": mean_a, "leader_joint": lead_j,
                      "leader_archived": lead_a, "shift_joint_minus_archived": shifts, "incomplete": inc,
                      "min_ess": min(ess) if ess else None, "max_weight": max(mw) if mw else None, "flagged_windows": flags}
    complete = all(not v["incomplete"] for v in by.values())
    return {"by_series": by, "changed": changed, "robust": (not changed) if complete else None}


def main():
    a, b = part_a(), part_b()
    c = part_c() if glob.glob(os.path.join(X10, "x10c_*.json")) else None
    json.dump({"a": a, "b": b, "c": c}, open(os.path.join(X10, "x10_summary.json"), "w"), indent=1)
    if c:
        print(f"(c) robust: {c['robust']}  changed: {c['changed']}")
        for sr, v in c["by_series"].items():
            print(f"    {sr}: leader joint {v['leader_joint']} archived {v['leader_archived']}; min ESS {v['min_ess']}, "
                  f"max weight {v['max_weight']}, flagged {v['flagged_windows']}, incomplete {v['incomplete']}")
            for r, row in v["rows"].items():
                if row.get("incomplete"):
                    print(f"      CosVI-{r}: incomplete"); continue
                j, a_ = row["joint"], row["archived"]
                print(f"      CosVI-{r:12s} joint {j['mean']:+.2f}+-{j['sd']:.2f} [{j['reading']}]  archived "
                      f"{a_['mean']:+.2f}+-{a_['sd']:.2f} [{a_['reading']}]")
            print("      means joint", {k: round(x, 2) for k, x in v["mean_joint"].items()})
            print("      shift      ", {k: round(x, 2) for k, x in v["shift_joint_minus_archived"].items()})
    print(f"(a) verdict {a['verdict']} ({a['cases_holding']}/{a['cases_complete']} cases)")
    for k, v in a["by_case"].items():
        if "incomplete" in v:
            print("   ", k, "incomplete", v["incomplete"]); continue
        print(f"    H|seed {k}: ev_J CosVI {v['ev_joint']['twoscale']:+.3f} grid {v['ev_joint']['grid']:+.3f} "
              f"pub {v['ev_joint']['markov']:+.3f} | ref K32768 {v['reference_K32768']:+.3f} (shift "
              f"{v['shift_of_reference']:+.3f}) | md_J grid {v['md_joint']['grid']:+.3f} pub {v['md_joint']['markov']:+.3f}"
              f" vs archived grid {v['md_archived']['grid']:+.3f} pub {v['md_archived']['markov']:+.3f} | u {v['u']:.3f}"
              f" holds {v['holds']} | gap_J {', '.join(f'{m} {g:.2f}' for m, g in v['gap_joint'].items())}"
              f" | min ESS {', '.join(f'{m} {e:.0f}' for m, e in v['min_ess'].items())}")
    print(f"(b) robust: {b['robust']}  changed: {b['changed']}")
    for H, v in b["by_H"].items():
        for r, row in v["rows"].items():
            if row.get("incomplete"):
                print(f"    H={H} {r}: incomplete {row['seeds']}"); continue
            j, a_ = row["joint"], row["archived"]
            print(f"    H={H} CosVI-{r:12s} n={len(row['seeds'])} joint {j['mean']:+.2f}+-{j['sd']:.2f} [{j['reading']}]"
                  f"  archived {a_['mean']:+.2f}+-{a_['sd']:.2f} [{a_['reading']}]")
        print(f"    H={H}: min ESS {v['min_ess']}, max weight {v['max_weight']}, flagged windows {v['flagged_windows']}")
    for k, v in b["score_shift_joint_minus_archived"].items():
        print(f"    shift {k:20s} mean {v['mean_shift']:+.3f} [{v['min']:+.3f}, {v['max']:+.3f}] n={v['n']}")


if __name__ == "__main__":
    main()
