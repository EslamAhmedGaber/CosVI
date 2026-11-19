"""Summarise N1 v2, N1b and N1c result files into one table (markdown to stdout and JSON to results/summary_trained.json).

For each (representation or system, H_true): mean ± sd over seeds of
  * held-out log-likelihood per test series under the exact grid law (oracle-H group, and profile-selected group),
  * drift RMSE (occupation weighted) at the oracle group,
  * profile Ĥ (own bound) and, where available, joint Ĥ,
  * sigma at the oracle group (true 0.5).
"""
import os, sys, json, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results")
sys.path.insert(0, os.path.join(HERE, ".."))
from fnvi import labels as L          # display names (CosVI etc.), single source


def ms(v):
    v = np.asarray(v, float)
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan"), int(len(v)))


def file_tag(f):
    """Suffix after the H value in the file name (e.g. '_steps10000'), so ablation runs never pool with main runs."""
    base = os.path.basename(f)[:-5]; after = base[base.rindex("_H") + 2:]
    return after[after.index("_"):] if "_" in after else ""


def new_row(D=None, name=None):
    return {"ll_oracle": [], "ll_profile_sel": [], "ll2_oracle": [], "ll2_profile_sel": [], "rmse_oracle": [], "H_profile": [],
            "sigma_oracle": [], "D": D, "secs": [], "name": name}


def main():
    # 14 Nov: rows are keyed by representation/system AND run variant (file tag, competitor config, lr_h);
    # the first version pooled the 10000-step ablation and the best-configuration competitor runs with the main runs.
    # ll = exact grid-law yardstick (test posteriors refitted in grid coordinates);
    # ll2 = exact-law yardstick with test posteriors refitted in two-scale coordinates (present for runs after 14 Nov).
    rows = {}
    for f in glob.glob(os.path.join(RES, "n1b", "n1b_*_seed*_H*.json")):
        d = json.load(open(f)); key = (d["rep"] + file_tag(f), d["H_true"])
        ll = d["test_loglik_exact_grid_law_mean"]; G = len(d["groups"])
        ll2 = d.get("test_loglik_exact_law_twoscale_coords_mean")
        hg = [0.5, 0.6, 0.7, 0.8, 0.9]
        ip = int(np.argmin(np.abs(np.array(hg) - d["H_profile_own_model"])))
        r = rows.setdefault(key, new_row(d["D"], L.rep_name(key[0])))
        r["ll_oracle"].append(ll[G - 1]); r["ll_profile_sel"].append(ll[ip]); r["rmse_oracle"].append(d["drift_rmse_occupation"][G - 1])
        if ll2 is not None:
            r["ll2_oracle"].append(ll2[G - 1]); r["ll2_profile_sel"].append(ll2[ip])
        r["H_profile"].append(d["H_profile_own_model"]); r["sigma_oracle"].append(d["sigma"][G - 1]); r["secs"].append(d["train_seconds"])
    for f in glob.glob(os.path.join(RES, "n1c", "n1c_*_seed*_H*.json")):
        d = json.load(open(f))
        name = f"competitor_{d['mode']}_{d.get('config', 'published')}" + ("" if d.get("lr_h") is None else f"_lrH{d['lr_h']:g}")
        key = (name, d["H_true"])
        r = rows.setdefault(key, new_row(None, L.competitor_name(d["mode"], d.get("config", "published"), d.get("lr_h"))))
        r["ll_oracle"].append(d["test_loglik_exact_grid_law_mean"]); r["rmse_oracle"].append(d["drift_rmse_occupation"])
        if "test_loglik_exact_law_twoscale_coords_mean" in d:
            r["ll2_oracle"].append(d["test_loglik_exact_law_twoscale_coords_mean"])
        r["H_profile"].append(d["H_fit"]); r["sigma_oracle"].append(d["sigma_standard_fbm_scale"]); r["secs"].append(d["train_seconds"])
    out = {}
    print("| method | H_true | n | held-out loglik, grid yardstick (oracle H / own fit) | (profile-selected H) | two-scale yardstick (oracle / own fit) | (profile-selected) | drift RMSE | H estimate | sigma | D | train s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(rows, key=lambda k: (k[1], k[0])):
        r = rows[key]; s = {k: ms(v) for k, v in r.items() if isinstance(v, list) and v}
        out[f"{key[0]}|{key[1]}"] = dict(s, name=r["name"])
        f = lambda k: (f"{s[k][0]:.3f} ± {s[k][1]:.3f}" + ("" if s[k][2] == s['ll_oracle'][2] else f" (n={s[k][2]})")) if k in s else "—"
        print(f"| {r['name']} [{key[0]}] | {key[1]} | {s['ll_oracle'][2]} | {f('ll_oracle')} | {f('ll_profile_sel')} | {f('ll2_oracle')} | {f('ll2_profile_sel')} | {f('rmse_oracle')} | {f('H_profile')} | {f('sigma_oracle')} | {r['D']} | {np.mean(r['secs']):.0f} |")
    with open(os.path.join(RES, "summary_trained.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
