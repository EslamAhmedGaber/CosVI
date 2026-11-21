r"""X8(a): exact linear-Gaussian check of the common-interpolation control (protocol
fixed 18 Nov 2026 14:35).  CPU, float64, no training.

Model: dX = -X dt + sigma dB_H, X(0) ~ N(0, 1), sigma = 0.5, T = 5, RK4 with N = 80 steps whose stages are forced by
the map's stage rates (left, mid, mid, right), observations of X at the N1 times (every 4th step) with s = 0.05.
Every state is linear in w = (z, x0) ~ N(0, I), so y = Jw + e and everything is closed form:
  log evidence           log N(y; 0, J J^T + s^2 I)
  posterior precision    Lam = I + J^T J / s^2
  optimal ELBO of the implemented family q(z) q(x0) (full Gaussian over z, separate Gaussian over x0):
      ELBO* = log p(y) - 0.5 [logdet Lam_zz + log Lam_00 - logdet Lam]     (structured mean field, exact means)
Maps: grid (Cholesky of fGn, D = 80), twoscale_pl (CosVI coordinates with the grid's piecewise-constant forcing,
D = 95), twoscale (CosVI as implemented: smooth cosine stage rates + piecewise-constant residual, D = 95).
Pass (fixed before the run): twoscale_pl and grid agree within 1e-6 nats in log evidence and in ELBO* on every data set.
Run from work\code\experiments:  python -X utf8 x8_linear_check.py      Out: work/results/x8/x8a_linear_check.json
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from fnvi.noise_maps import grid_map, twoscale_map, twoscale_pl_map, path_cov_on_grid

OUT = os.path.join(HERE, "..", "..", "results", "x8"); os.makedirs(OUT, exist_ok=True)
T, N, J, SIG, S_OBS, N_OBS, NDATA = 5.0, 80, 16, 0.5, 0.05, 20, 20
OBS = np.arange(1, N_OBS + 1) * (N // N_OBS)


def obs_matrix(maps):
    """Rows: coefficients of X(t_obs) on w = (z, x0) under RK4 with drift -x and the map's stage rates."""
    left, mid, right = maps
    D = left.shape[1]; h = T / N
    x = np.zeros(D + 1); x[D] = 1.0                             # X(0) = x0
    pad = lambda r: np.r_[SIG * r, 0.0]
    rows = []
    for n in range(N):
        rl, rm, rr = pad(left[n]), pad(mid[n]), pad(right[n])
        k1 = -x + rl
        k2 = -(x + 0.5 * h * k1) + rm
        k3 = -(x + 0.5 * h * k2) + rm
        k4 = -(x + h * k3) + rr
        x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if n + 1 in set(OBS.tolist()):
            rows.append(x.copy())
    return np.array(rows)


def scores(Jm, y):
    m = Jm.shape[0]; d = Jm.shape[1]
    C = Jm @ Jm.T + S_OBS ** 2 * np.eye(m)
    Lc = np.linalg.cholesky(C)
    a = np.linalg.solve(Lc, y)
    logev = -0.5 * a @ a - np.log(np.diag(Lc)).sum() - 0.5 * m * np.log(2 * np.pi)
    Lam = np.eye(d) + Jm.T @ Jm / S_OBS ** 2
    ld = lambda A: 2 * np.log(np.diag(np.linalg.cholesky(A))).sum()
    gap = 0.5 * (ld(Lam[:-1, :-1]) + np.log(Lam[-1, -1]) - ld(Lam))
    return logev, logev - gap, gap


def main():
    res = {"rule": "X8(a)", "settings": {"T": T, "N": N, "J": J, "sigma": SIG, "s_obs": S_OBS, "n_data": NDATA},
           "by_H": {}}
    for H in (0.6, 0.75, 0.9):
        h = T / N; t = h * np.arange(1, N + 1)
        K = 0.5 * (t[:, None] ** (2 * H) + t[None, :] ** (2 * H) - np.abs(t[:, None] - t[None, :]) ** (2 * H))
        maps = {"grid": grid_map(H, T, N), "twoscale_pl": twoscale_pl_map(H, T, N, J), "twoscale": twoscale_map(H, T, N, J)}
        Js = {k: obs_matrix(v) for k, v in maps.items()}
        rng = np.random.default_rng(20261118 + int(100 * H))
        row = {"cov_relerr_grid_values": {}, "obs_cov_relerr_vs_grid": {}, "logev": {}, "elbo_opt": {}, "gap": {}}
        for k, v in maps.items():
            P = path_cov_on_grid(*v, T)
            row["cov_relerr_grid_values"][k] = float(np.linalg.norm(P - K) / np.linalg.norm(K))
            Cg = Js["grid"] @ Js["grid"].T; Ck = Js[k] @ Js[k].T
            row["obs_cov_relerr_vs_grid"][k] = float(np.linalg.norm(Ck - Cg) / np.linalg.norm(Cg))
        Jg = Js["grid"]
        sc = {k: [] for k in maps}
        for _ in range(NDATA):
            w = rng.standard_normal(Jg.shape[1])
            y = Jg @ w + S_OBS * rng.standard_normal(Jg.shape[0])
            for k in maps:
                sc[k].append(scores(Js[k], y))
        for k in maps:
            arr = np.array(sc[k])
            row["logev"][k] = arr[:, 0].round(6).tolist(); row["elbo_opt"][k] = arr[:, 1].round(6).tolist()
            row["gap"][k] = arr[:, 2].round(6).tolist()
        diff = lambda k, f: np.abs(np.array(sc[k])[:, f] - np.array(sc["grid"])[:, f])
        row["max_abs_diff_vs_grid"] = {k: {"logev": float(diff(k, 0).max()), "elbo_opt": float(diff(k, 1).max())}
                                       for k in ("twoscale_pl", "twoscale")}
        row["mean_signed_diff_vs_grid"] = {k: {"logev": float((np.array(sc[k])[:, 0] - np.array(sc["grid"])[:, 0]).mean()),
                                               "elbo_opt": float((np.array(sc[k])[:, 1] - np.array(sc["grid"])[:, 1]).mean())}
                                           for k in ("twoscale_pl", "twoscale")}
        row["pass"] = bool(row["max_abs_diff_vs_grid"]["twoscale_pl"]["logev"] <= 1e-6 and
                           row["max_abs_diff_vs_grid"]["twoscale_pl"]["elbo_opt"] <= 1e-6)
        res["by_H"][f"{H}"] = row
        print(f"H={H}: grid-value cov error " + ", ".join(f"{k} {v:.3g}" for k, v in row["cov_relerr_grid_values"].items()))
        print(f"   observed-state cov vs grid: " + ", ".join(f"{k} {v:.3g}" for k, v in row["obs_cov_relerr_vs_grid"].items()))
        print(f"   max |diff| vs grid: {row['max_abs_diff_vs_grid']}   mean signed: {row['mean_signed_diff_vs_grid']}")
        print(f"   mean log evidence {np.mean(row['logev']['grid']):.4f}, mean product-family gap {np.mean(row['gap']['grid']):.4f}"
              f"  PASS={row['pass']}")
    res["pass_all"] = all(r["pass"] for r in res["by_H"].values())
    with open(os.path.join(OUT, "x8a_linear_check.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("X8(a) pass at every H:", res["pass_all"])


if __name__ == "__main__":
    main()
