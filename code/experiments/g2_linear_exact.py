"""Gate G2: linear model with a known posterior (fractional Ornstein-Uhlenbeck), everything exact (no Monte Carlo).

Model  dX = -theta X dt + sigma dB_H, X(0) = 0, standard fBm, T = 2, theta = sigma = 1.
Case A 'bridge'  : one observation y = X(T) + eps, eps ~ N(0, 0.1^2), y = 0 (the setting of Daems et al. 2024, Sec. 5).
Case B 'sparse'  : y_i = X(0.2 i) + eps_i, i = 1..10, 20 data sets drawn from the exact model (seed 2026).
Compared with the exact evidence and posterior:
  cosine model (projection and classical coordinates, J = 4..64), with its exact evidence, the Titsias-type
  exact-model bound (projection only), and the variational gaps of mean-field / low-rank+diagonal Gaussians;
  MA-fBm model (type I, Daems et al.), K = 2..16, default rule (gamma in [1/20, 20], weights fitted on [0, 3T])
  and tuned rule (gamma_max chosen per K on the path criterion over [0, T]).
Also: the stationary fOU conditioned on X(0) = 0 (the analytic reference read in Daems et al., eqs. 23-24; check).
Out: work/results/g2/g2_linear_exact.json
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "2")
import sys, json, time
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from fnvi.vendor import hfc
from fnvi.coords import CosineCoords
from fnvi.quad import gl, graded
from fnvi.ma_fbm import MAFit, nodes
from fnvi.linear_exact import exact_cov, ma_cov, log_normal, gp_posterior

T, THETA, SIGMA, S_OBS = 2.0, 1.0, 1.0, 0.1
HS = [0.6, 0.75, 0.9]
JS = [4, 6, 8, 12, 16, 24, 32, 48, 64]
KS = [2, 3, 4, 5, 6, 8, 12, 16]
GMAX_GRID = [2, 5, 10, 20, 50, 100, 300, 1e3, 3e3, 1e4, 3e4, 1e5, 1e6, 1e7, 1e8]
NGRID = 40
TIMES = np.arange(1, NGRID + 1) * T / NGRID                  # 0.05, ..., 2.0
OBS_B = np.arange(4, NGRID + 1, 4) - 1                       # indices of 0.2, 0.4, ..., 2.0
OBS_A = np.array([NGRID - 1])                                # index of T
NDATA = 20
OUT = os.path.join(HERE, "..", "..", "results", "g2"); os.makedirs(OUT, exist_ok=True)


def cosine_feature_bank(H, Jmax=64, n=16, levels=30):
    """Phi_J(t_m) for every J <= Jmax and both modes from one quadrature (projection coordinates are nested)."""
    cc = CosineCoords(H, T, Jmax)
    edges = np.concatenate([[0.0], TIMES])
    s_list, w_list, own = [], [], []
    for m, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        if m == 0:
            s, w = graded(a, b, levels, n, 'a')
        elif m == NGRID - 1:
            s, w = graded(a, b, levels, n, 'b')
        else:
            s, w = gl(a, b, n)
        s_list.append(s); w_list.append(w); own.append(np.full(len(s), m))
    s = np.concatenate(s_list); w = np.concatenate(w_list); own = np.concatenate(own)
    a_proj = cc.rate_funcs(s, 'projection')                  # (ns, Jmax), nested in J
    hh = hfc.h('cos', Jmax, T, s)                            # (ns, Jmax)
    ew = w * np.exp(THETA * s)
    def cum(A):                                              # Phi(t_m) = e^{-theta t_m} sum_{panels <= m} int e^{theta s} a(s)
        P = np.zeros((NGRID, A.shape[1]))
        np.add.at(P, own, ew[:, None] * A)
        return np.exp(-THETA * TIMES)[:, None] * np.cumsum(P, axis=0)
    bank = {"projection": cum(a_proj)}
    for J in JS:
        bank[("classical", J)] = cum(hh[:, :J] @ cc.L[:J, :J])
    return bank


def lowrank_gap(Lam, r):
    """min over S = D + V V^T of KL(N(m,S) || N(m, Lam^{-1})) (exact posterior precision Lam), in nats."""
    J = Lam.shape[0]
    ld_Lam = np.linalg.slogdet(Lam)[1]
    rng = np.random.default_rng(0)
    x0 = np.concatenate([-np.log(np.diag(Lam)), 0.01 * rng.standard_normal(J * r)])
    def f(x):
        d = np.exp(x[:J]); V = x[J:].reshape(J, r)
        S = np.diag(d) + V @ V.T
        Ls = np.linalg.cholesky(S); Si = np.linalg.inv(S)
        val = 0.5 * (np.sum(Lam * S) - 2 * np.log(np.diag(Ls)).sum() - J - ld_Lam)
        Gs = 0.5 * (Lam - Si)
        return val, np.concatenate([np.diag(Gs) * d, (2 * Gs @ V).ravel()])
    res = minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 5000, "gtol": 1e-10})
    return float(res.fun)


INTERIOR = TIMES >= T / 4


def var_metrics(v_a, v_e):
    r = np.abs(v_a / v_e - 1.0)
    return {"max": float(r.max()), "max_interior": float(r[INTERIOR].max()),
            "integrated": float(np.sum(np.abs(v_a - v_e)) / np.sum(v_e))}


def posterior_errors(mu_a, v_a, mu_e, v_e):
    sd = np.sqrt(np.maximum(v_e, 1e-300))
    return float(np.max(np.abs(mu_a - mu_e) / sd)), var_metrics(v_a, v_e)


def main():
    t0 = time.time()
    rng = np.random.default_rng(2026)
    out = {"setting": {"T": T, "theta": THETA, "sigma": SIGMA, "s_obs": S_OBS, "times": TIMES.tolist(),
                       "obs_A": TIMES[OBS_A].tolist(), "obs_B": TIMES[OBS_B].tolist(), "n_datasets_B": NDATA},
           "results": {}}
    s2 = S_OBS ** 2
    for H in HS:
        key = f"{H:.2f}"; R = {}
        C = exact_cov(TIMES, H, THETA, SIGMA)                 # exact prior covariance on the grid
        # stationary fOU conditioned on X(0)=0 (reference read in Daems et al.; computed as a long-run limit)
        Lb = 40.0
        Cs_full = exact_cov(np.concatenate([[Lb], Lb + TIMES]), H, THETA, SIGMA)
        k0 = Cs_full[0, 1:]; Cst = Cs_full[1:, 1:] - np.outer(k0, k0) / Cs_full[0, 0]
        datasets = {"A": [np.zeros(1)],
                    "B": [np.linalg.cholesky(C[np.ix_(OBS_B, OBS_B)] + s2 * np.eye(len(OBS_B))) @ rng.standard_normal(len(OBS_B)) for _ in range(NDATA)]}
        exact = {}
        for case, obs in (("A", OBS_A), ("B", OBS_B)):
            Koo = C[np.ix_(obs, obs)]; Kgo = C[:, obs]
            exact[case] = []
            for y in datasets[case]:
                lp = log_normal(y, Koo + s2 * np.eye(len(obs)))
                mu, var = gp_posterior(Kgo, Koo, np.diag(C), y, s2)
                exact[case].append((lp, mu, var))
        # reference mismatch: stationary-conditioned process vs started-at-zero process (bridge case)
        muS, varS = gp_posterior(Cst[:, OBS_A], Cst[np.ix_(OBS_A, OBS_A)], np.diag(Cst), np.zeros(1), s2)
        R["stationary_conditioned_vs_started_bridge_var_maxrel"] = float(np.max(np.abs(varS / exact["A"][0][2] - 1)))
        R["stationary_conditioned_vs_started_prior_var_at_T"] = [float(Cst[-1, -1]), float(C[-1, -1])]
        curves = {"times": TIMES.tolist(), "exact": exact["A"][0][2].tolist(), "stationary_conditioned_reference": varS.tolist()}
        # ---------------- cosine
        bank = cosine_feature_bank(H)
        cos = {}
        for J in JS:
            for mode in ("projection", "classical"):
                Phi = SIGMA * (bank["projection"][:, :J] if mode == "projection" else bank[("classical", J)])
                rec = {}
                for case, obs in (("A", OBS_A), ("B", OBS_B)):
                    Po = Phi[obs]; Qoo = Po @ Po.T
                    dl, me, ve, titsias, mf, lr1, lr2 = [], [], [], [], [], [], []
                    for y, (lp, mu_e, v_e) in zip(datasets[case], exact[case]):
                        lpJ = log_normal(y, Qoo + s2 * np.eye(len(obs)))
                        dl.append(lpJ - lp)
                        Lam = np.eye(J) + Po.T @ Po / s2
                        Sig = np.linalg.inv(Lam); m = Sig @ Po.T @ y / s2
                        mu_a = Phi @ m; v_a = np.einsum("ij,jk,ik->i", Phi, Sig, Phi)
                        e1, e2 = posterior_errors(mu_a, v_a, mu_e, v_e); me.append(e1); ve.append(e2)
                        if mode == "projection":
                            resid = np.diag(C)[obs] - np.sum(Po ** 2, axis=1)
                            titsias.append(lpJ - resid.sum() / (2 * s2) - lp)
                        mf.append(0.5 * (np.sum(np.log(np.diag(Lam))) - np.linalg.slogdet(Lam)[1]))
                    rec[case] = {"dlogp_mean": float(np.mean(dl)), "dlogp_sd": float(np.std(dl)),
                                 "post_mean_err_sd_units_max": float(np.max(me)),
                                 "post_var_relerr": {k: float(max(v[k] for v in ve)) for k in ve[0]},
                                 "mf_gap_mean": float(np.mean(mf))}
                    if mode == "projection":
                        rec[case]["titsias_minus_exact_mean"] = float(np.mean(titsias))
                    if case == "B":
                        y0 = datasets["B"][0]
                        Lam = np.eye(J) + Po.T @ Po / s2
                        rec[case]["lowrank_gap_r1_data0"] = lowrank_gap(Lam, 1)
                        rec[case]["lowrank_gap_r2_data0"] = lowrank_gap(Lam, 2)
                cos[f"{mode}_J{J}"] = rec
                if mode == "projection" and J in (8, 16, 32, 64):
                    Po = Phi[OBS_A]; Sg = np.linalg.inv(np.eye(J) + Po.T @ Po / s2)
                    curves[f"cosine_projection_J{J}"] = np.einsum("ij,jk,ik->i", Phi, Sg, Phi).tolist()
            print(f"H={H} J={J} done ({time.time() - t0:.0f}s)", flush=True)
        R["cosine"] = cos
        # ---------------- MA-fBm type I
        ma = {}
        for K in KS:
            best = None
            for gm in GMAX_GRID:
                fit = MAFit(H, T, nodes(K, gm), "I")
                if best is None or fit.rel_int_err < best[1]:
                    best = (gm, fit.rel_int_err, fit.w)
            rules = {"default": (20.0, MAFit(H, 3 * T, nodes(K, 20.0), "I").w), "tuned": (best[0], best[2])}
            for lab, (gm, w) in rules.items():
                CM = ma_cov(TIMES, H, THETA, SIGMA, nodes(K, gm), w)
                rec = {"gamma_max": gm, "prior_var_relerr": var_metrics(np.diag(CM), np.diag(C))}
                for case, obs in (("A", OBS_A), ("B", OBS_B)):
                    dl, me, ve = [], [], []
                    for y, (lp, mu_e, v_e) in zip(datasets[case], exact[case]):
                        Moo = CM[np.ix_(obs, obs)]
                        dl.append(log_normal(y, Moo + s2 * np.eye(len(obs))) - lp)
                        mu_a, v_a = gp_posterior(CM[:, obs], Moo, np.diag(CM), y, s2)
                        e1, e2 = posterior_errors(mu_a, v_a, mu_e, v_e); me.append(e1); ve.append(e2)
                    rec[case] = {"dlogp_mean": float(np.mean(dl)), "dlogp_sd": float(np.std(dl)),
                                 "post_mean_err_sd_units_max": float(np.max(me)),
                                 "post_var_relerr": {k: float(max(v[k] for v in ve)) for k in ve[0]}}
                    if case == "A":
                        curves[f"ma_{lab}_K{K}"] = gp_posterior(CM[:, obs], CM[np.ix_(obs, obs)], np.diag(CM), datasets["A"][0], s2)[1].tolist()
                ma[f"{lab}_K{K}"] = rec
            if K == 5:   # the authors' own model exactly as coded (no rescaling to standard fBm), default rule
                CMa = ma_cov(TIMES, H, THETA, SIGMA, nodes(5, 20.0), MAFit(H, 3 * T, nodes(5, 20.0), "I").w, rescale=False)
                curves["ma_authors_as_coded_K5"] = gp_posterior(CMa[:, OBS_A], CMa[np.ix_(OBS_A, OBS_A)], np.diag(CMa), datasets["A"][0], s2)[1].tolist()
                R["ma_authors_as_coded_K5_prior_var_relerr"] = var_metrics(np.diag(CMa), np.diag(C))
            print(f"H={H} MA K={K} done ({time.time() - t0:.0f}s)", flush=True)
        R["ma_typeI"] = ma
        # cosine prior variance error on the grid (law-level, same metric as MA)
        R["cosine_prior_var_relerr"] = {f"{mode}_J{J}": var_metrics(
            np.sum((SIGMA * (bank['projection'][:, :J] if mode == 'projection' else bank[('classical', J)])) ** 2, 1), np.diag(C))
            for J in JS for mode in ("projection", "classical")}
        R["bridge_posterior_var_curves"] = curves
        out["results"][key] = R
    out["runtime_s"] = time.time() - t0
    with open(os.path.join(OUT, "g2_linear_exact.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", os.path.join(OUT, "g2_linear_exact.json"))


if __name__ == "__main__":
    main()
