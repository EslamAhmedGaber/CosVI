r"""N7(b), post hoc (19 Nov 2026; suggested by an independent review of 18 Nov and recomputed here): does CosVI meet the
N7 accuracy criterion at every horizon, instead of being assumed to?

The N7 protocol (05_EXPERIMENTS.md, fixed 15 Nov 2026 21:59; n7_horizon_exact.py) scores a representation by the mean
over the observation times of |s - s_exact| / s_exact, where s is the posterior standard deviation of X at the
observation times under the representation's exact (linear-Gaussian) inference and s_exact that of the true model
dX = -X dt + dB_H, X(0) = 0, observations every 0.2 on (0, T], s_obs = 0.1; a representation qualifies if the error is
at most 2 %.  N7 costed CosVI at N = 50 T Runge-Kutta steps (observations every 10 steps) with J = 16 (n7_cost.json)
and assumed that it qualifies.  Here the discrete CosVI model with exactly that configuration (the implemented
smooth-stage map, RK4, theta = 1, sigma = 1) is scored by the same error at T in {2, 5, 10, 20, 50} and H in
{0.6, 0.75, 0.9} (15 cases), CPU, float64.
Run from work\code\experiments:  python -X utf8 n7b_cosvi_accuracy.py      Out: work/results/n7/n7_cosvi_accuracy.json
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
import sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, HERE)
from fnvi.noise_maps import build
from n7_horizon_exact import exact_cov, post_std_cov, err, S_OBS, THETA, DT_OBS

OUT = os.path.join(HERE, "..", "..", "results", "n7", "n7_cosvi_accuracy.json")
J, STEPS_PER_UNIT, OBS_EVERY = 16, 50, 10


def obs_rows(maps, N, T, every):
    """Coefficients of X(t_obs) on z under RK4 with drift -theta x, X(0) = 0 and the map's stage rates (sigma = 1)."""
    left, mid, right = maps
    h = T / N
    x = np.zeros(left.shape[1]); rows = []
    for n in range(N):
        k1 = -THETA * x + left[n]
        k2 = -THETA * (x + 0.5 * h * k1) + mid[n]
        k3 = -THETA * (x + 0.5 * h * k2) + mid[n]
        k4 = -THETA * (x + h * k3) + right[n]
        x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if (n + 1) % every == 0:
            rows.append(x.copy())
    return np.array(rows)


def post_std_rows(F):
    """Posterior std of F z given y = F z + eps, z ~ N(0, I), eps ~ N(0, s^2 I): sqrt diag(F (I + F'F/s^2)^-1 F')."""
    D = F.shape[1]
    A = np.eye(D) + F.T @ F / S_OBS ** 2
    L = np.linalg.cholesky(A)
    W = np.linalg.solve(L, F.T)                          # D x m
    return np.sqrt(np.einsum("ij,ij->j", W, W))


def main():
    rows = []
    for T in (2.0, 5.0, 10.0, 20.0, 50.0):
        for H in (0.6, 0.75, 0.9):
            t0 = time.time()
            N = int(round(STEPS_PER_UNIT * T)); m = int(round(T / DT_OBS))
            times = DT_OBS * np.arange(1, m + 1)
            s_ex = post_std_cov(exact_cov(times, H, THETA, 1.0))
            maps = [np.asarray(a) for a in build("twoscale", H, T, N, J=J)]
            F = obs_rows(maps, N, T, OBS_EVERY)
            assert F.shape[0] == m, (F.shape, m)
            e = err(post_std_rows(F), s_ex)
            rows.append({"T": T, "H": H, "N": N, "J": J, "D": int(maps[0].shape[1]), "m": m, "err": e,
                         "qualifies_2pct": e <= 0.02, "seconds": time.time() - t0})
            print(f"T={T:5.1f} H={H}: N={N}, D={maps[0].shape[1]}, error {100 * e:.5f} %  ({time.time() - t0:.0f}s)",
                  flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"protocol": "N7 criterion (05_EXPERIMENTS.md, 15 Nov 2026 21:59), CosVI configuration of n7_cost.json",
               "rows": rows, "all_qualify": all(r["qualifies_2pct"] for r in rows),
               "max_err": max(r["err"] for r in rows)}, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
