r"""N7, accuracy at convergence (exact, no training; protocol fixed in 05_EXPERIMENTS.md on 15 Nov 2026, 21:59,
before any run).

Model: dX = -X dt + dB_H (standard fBm), X(0) = 0, observations every 0.2 on (0, T] (m = 5T), s_obs = 0.1.
Posterior std at the m observation times (it does not depend on y) for
  exact  : the true model (linear_exact.exact_cov)
  cosine : J projection coordinates, J in {16, 32, 64, 128, 256} (features nested in J: computed once at J = 256)
  lift_published_coded    : K = 5, their gamma rule (gamma_max = 20), omega on horizon 6, their scaling (as coded)
  lift_published_rescaled : the same rescaled to standard fBm
  lift_best               : K = 12 geometric on [1e-6, 40], omega fitted on [0, 3T], standard scale
Error = mean over the m points of |s - s_exact| / s_exact.  J* = smallest J with error <= 2 %.
Run from work\code\experiments:  python n7_horizon_exact.py      (--T 2 --H 0.75 --no-save for a smoke test)
Out: work/results/n7/n7_exact.json
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
import sys, json, time, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from fnvi.linear_exact import exact_cov, ma_cov, gp_posterior
from fnvi.coords import CosineCoords
from fnvi.quad import gl, graded
from fnvi.mafbm_torch import gamma_by_gamma_max, omega_type1
from fnvi.ma_fbm import MAFit

OUT = os.path.join(HERE, "..", "..", "results", "n7")
S_OBS, THETA, DT_OBS = 0.1, 1.0, 0.2
JS = [16, 32, 64, 128, 256]


def cosine_bank(times, H, T, Jmax, n=12, levels=30):
    """Phi_J(t_i) = int_0^{t_i} e^{-theta (t_i - s)} a(s) ds for every J <= Jmax from ONE quadrature (projection
    coordinates are nested in J), as g2_linear_exact.cosine_feature_bank: Gauss-Legendre panels between consecutive
    observation times, split so that a panel spans at most about 2/3 of a period of the highest cosine (the rule of
    linear_exact.cosine_features), graded toward t = 0 and t = T; accumulated recursively in time (no large exponents)."""
    cc = CosineCoords(H, T, Jmax)
    edges = np.concatenate([[0.0], times]); m = len(times)
    s_list, w_list, own = [], [], []
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        k = max(1, int(np.ceil(1.5 * Jmax * (b - a) / T)))
        sub = np.linspace(a, b, k + 1)
        for j, (c, d) in enumerate(zip(sub[:-1], sub[1:])):
            if i == 0 and j == 0:
                s, w = graded(c, d, levels, n, 'a')
            elif i == m - 1 and j == k - 1:
                s, w = graded(c, d, levels, n, 'b')
            else:
                s, w = gl(c, d, n)
            s_list.append(s); w_list.append(w); own.append(np.full(len(s), i))
    s = np.concatenate(s_list); w = np.concatenate(w_list); own = np.concatenate(own)
    A = cc.rate_funcs(s, 'projection')                                    # (ns, Jmax)
    P = np.zeros((m, Jmax))
    np.add.at(P, own, (w * np.exp(-THETA * (times[own] - s)))[:, None] * A)
    Phi = np.zeros((m, Jmax)); prev = np.zeros(Jmax); tprev = 0.0
    for i in range(m):
        prev = np.exp(-THETA * (times[i] - tprev)) * prev + P[i]; tprev = times[i]; Phi[i] = prev
    return Phi, len(s)


def post_std_cov(C):
    _, v = gp_posterior(C, C, np.diag(C), np.zeros(len(C)), S_OBS ** 2)
    return np.sqrt(np.clip(v, 1e-300, None))


def post_std_features(Phi):
    J = Phi.shape[1]
    Sig = np.linalg.inv(np.eye(J) + Phi.T @ Phi / S_OBS ** 2)
    return np.sqrt(np.clip(np.einsum("ij,jk,ik->i", Phi, Sig, Phi), 1e-300, None))


def err(s, s0):
    return float(np.mean(np.abs(s - s0) / s0))


def one(T, H):
    t0 = time.time()
    m = int(round(T / DT_OBS)); times = DT_OBS * np.arange(1, m + 1)
    s_ex = post_std_cov(exact_cov(times, H, THETA, 1.0)); t_ex = time.time() - t0
    Phi, n_nodes = cosine_bank(times, H, T, max(JS)); t_cos = time.time() - t0 - t_ex
    cos = {str(J): err(post_std_features(Phi[:, :J]), s_ex) for J in JS}
    Jstar = next((J for J in JS if cos[str(J)] <= 0.02), None)
    g5 = gamma_by_gamma_max(5, 20.0); w5 = omega_type1(g5, H, 6.0)
    pub_coded = err(post_std_cov(ma_cov(times, H, THETA, 1.0, g5, w5, rescale=False)), s_ex)
    pub_resc = err(post_std_cov(ma_cov(times, H, THETA, 1.0, g5, w5, rescale=True)), s_ex)
    g12 = np.geomspace(1e-6, 40.0, 12); w12 = MAFit(H, 3 * T, g12, "I").w
    best = err(post_std_cov(ma_cov(times, H, THETA, 1.0, g12, w12, rescale=True)), s_ex)
    r = {"T": T, "H": H, "m": m, "posterior_std_exact_mean": float(s_ex.mean()),
         "err_cosine": cos, "J_star_2pct": Jstar,
         "err_lift_published_coded": pub_coded, "err_lift_published_rescaled": pub_resc, "err_lift_best": best,
         "cosine_quadrature_nodes": int(n_nodes),
         "seconds": {"exact": t_ex, "cosine_features": t_cos, "total": time.time() - t0}}
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "seconds"}
                     | {"s": round(r["seconds"]["total"])}), flush=True)
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, nargs="+", default=[2.0, 5.0, 10.0, 20.0, 50.0])
    ap.add_argument("--H", type=float, nargs="+", default=[0.6, 0.75, 0.9])
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    rows = []
    for T in a.T:
        for H in a.H:
            rows.append(one(T, H))
            if not a.no_save:                                   # write after every case (long run)
                os.makedirs(OUT, exist_ok=True)
                with open(os.path.join(OUT, "n7_exact.json"), "w") as fh:
                    json.dump({"protocol": "05_EXPERIMENTS.md, fixed 15 Nov 2026 21:59", "rows": rows}, fh, indent=1)
