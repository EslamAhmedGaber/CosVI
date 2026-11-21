r"""X3 (16 Nov 2026): a second real series — Northern-Hemisphere monthly temperature anomalies, 1854-1989.

Protocol fixed before any run.  Data: `results/n4/data/nhemi_temp.csv`, parsed
as text by n4_prepare_data.py from CRAN longmemo 1.1-4 (the archive downloaded on 13 Nov for the Nile; nothing from it
is executed).  The 1632 monthly values form 32 consecutive windows of 51 months (as the Nile's 51-year windows: the
first value of a window is its initial value, the next 50 are observations).  Test windows: every fourth window
(k = 3, 7, ..., 31; 8 windows), training: the other 24 (interleaved because the series warms; a chronological split
would test on the warmest decades only).  Standardised with the training windows' mean and sd.
Everything else is the N4 v2 pipeline unchanged (n4v2_nile.run: T = 50, N = 200 RK4 steps, s = 0.1, drift MLP
1-32-32-1, sigma learned, full q, S = 16, 5000 steps, yardstick = exact law with the two-scale refit, 4000 steps,
K = 1024; grid refit secondary) except the H grid {0.5, ..., 0.9, 0.95}; the profile vertex is recomputed with the
actual (unequal) spacing and stored as H_profile_general.  The competitor's full system runs with the N4c harness
(n4c_competitor_nile.run, port; modes joint and oracle) on the same windows; its time unit becomes 20 months.
Run from work\code\experiments:
    python n4t_temperature.py --reps twoscale --seeds 0          (ours, one representation and seed per job)
    python n4t_temperature.py --competitor joint --seeds 0       (their system)
Out: work/results/n4t/n4t_{rep}_seed{seed}_s{s}.json, n4t_competitor_{mode}_seed{seed}.json, models/
"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import n4_nile as v1
import n4v2_nile as v2
import n4c_competitor_nile as c4
from n1b_representations import vertex

WORK = os.path.abspath(os.path.join(HERE, "..", ".."))
OUTT = os.path.join(WORK, "results", "n4t"); os.makedirs(os.path.join(OUTT, "models"), exist_ok=True)
WIN, NWIN = 51, 32
TEST = list(range(3, NWIN, 4))
HGRID_T = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def load_windows_temp():
    y = np.loadtxt(os.path.join(v1.OUT, "data", "nhemi_temp.csv"), delimiter=",", skiprows=1)
    assert y.shape == (WIN * NWIN,), y.shape
    W = y.reshape(NWIN, WIN)
    tr = W[[k for k in range(NWIN) if k not in TEST]]
    te = W[TEST]
    mu, sd = tr.mean(), tr.std()
    tr, te = (tr - mu) / sd, (te - mu) / sd
    return tr[:, 1:], te[:, 1:], tr[:, 0], te[:, 0], (float(mu), float(sd))


def patch():
    v1.load_windows = load_windows_temp          # used by n4v2_nile.run and n4c_competitor_nile.run
    v2.OUT = OUTT
    v2.HGRID = list(HGRID_T)
    c4.OUT = OUTT


def ours(rep, seed, s_obs):
    v2.run(rep, seed, s_obs)
    src = os.path.join(OUTT, f"n4v2_{rep}_seed{seed}_s{s_obs}.json")
    with open(src) as fh:
        res = json.load(fh)
    eg = np.array(res["elbo_train_per_window"], float)
    finite = np.isfinite(eg); i = int(np.nanargmax(np.where(finite, eg, -np.inf)))
    ok = 0 < i < len(HGRID_T) - 1 and finite[i - 1] and finite[i + 1]
    res.update({"version": "n4t (X3)", "data": "NhemiTemp monthly anomalies 1854-1989, 32 windows of 51 months",
                "test_windows": TEST, "hgrid": HGRID_T,
                "H_profile_general": vertex(HGRID_T, eg, i) if ok else float(HGRID_T[i]),
                "H_profile_at_grid_edge": bool(i in (0, len(HGRID_T) - 1)), "time_unit": "month"})
    dst = os.path.join(OUTT, f"n4t_{rep}_seed{seed}_s{s_obs}.json")
    with open(dst, "w") as fh:
        json.dump(res, fh, indent=1)
    os.remove(src)                               # the file written by n4v2_nile.run in this run, now renamed
    print(json.dumps({"rep": rep, "seed": seed, "H_profile_general": round(res["H_profile_general"], 3),
                      "test_ts_mean_best_group": round(max(res["test_loglik_exact_law_twoscale_coords_mean"]), 2)}), flush=True)


def competitor(mode, seed, steps):
    c4.run(mode, seed, steps)
    src = os.path.join(OUTT, f"n4c_{mode}_seed{seed}.json")
    with open(src) as fh:
        res = json.load(fh)
    res.update({"version": "n4t competitor (X3)", "test_windows": TEST, "time_unit_note": "20 months per unit"})
    with open(os.path.join(OUTT, f"n4t_competitor_{mode}_seed{seed}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    os.remove(src)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="*", default=[])
    ap.add_argument("--competitor", nargs="*", default=[], help="modes of their system: joint, oracle")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--s", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=5000)
    a = ap.parse_args()
    patch()
    for sd_ in a.seeds:
        for rep in a.reps:
            ours(rep, sd_, a.s)
        for mode in a.competitor:
            competitor(mode, sd_, a.steps)
