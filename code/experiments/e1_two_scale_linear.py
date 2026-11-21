"""E1-a: two-scale whitened coordinates (TSW) in the linear model, exact computation.

Representation (Section 3.1 of the paper): on a fine grid t_n = n h (h = T/N),
    B(t_n) = A(t_n) xi + R(t_n),  xi ~ N(0, I_J)  (projection cosine coordinates, CosChaos),
    R_grid = L_R eta,  eta ~ N(0, I_{N-1}),  L_R = chol(Cov R on the interior grid),  R(0) = R(T) = 0,
R independent of xi, R piecewise linear between grid points.  On the grid the prior of B is EXACT (no truncation);
the only model error is the linear interpolation of the small residual between grid points.

Question: with this exact prior, how much is lost by a cheap STRUCTURED posterior?
    full      : q full Gaussian over (xi, eta)                     (exact posterior, gap 0)
    arrow-b   : q with precision = arrowhead (xi dense) + band b in eta (no fill-in ordering: eta first, xi last)
    blockdiag : q(xi) full x q(eta) diagonal (ignores xi-eta coupling)
compared with cosine-only coordinates with J_big in {32, 64, 128, 256} and a full q (G2b).
Model/data: G2 case B (fOU, theta = sigma = 1, T = 2, 10 observations every 0.2, s = 0.1), H in {0.6, 0.75, 0.9},
J in {8, 16, 32}, N = 160 (h = 0.0125); 5 data sets.  Out: work/results/e1/e1a_two_scale_linear.json
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "2")
import sys, json, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import g2_linear_exact as g2
from fnvi.coords import CosineCoords
from fnvi.linear_exact import exact_cov, log_normal, gp_posterior

OUT = os.path.join(HERE, "..", "..", "results", "e1"); os.makedirs(OUT, exist_ok=True)
T, TH, SIG, S2 = g2.T, g2.THETA, g2.SIGMA, g2.S_OBS ** 2
N = 160; h = T / N
TF = np.arange(N + 1) * h                       # fine grid, includes 0 and T
STRIDE = N // g2.NGRID                          # 4 fine steps per coarse grid step (0.05)
torch.set_default_dtype(torch.float64)


def residual_operator(theta):
    """Psi (40, N+1): X_R(t_m) = sigma * sum_n (R_{n+1} - R_n) e^{-theta(t_m - t_{n+1})} (1 - e^{-theta h})/(theta h)."""
    g = (1 - np.exp(-theta * h)) / (theta * h)
    Psi = np.zeros((g2.NGRID, N + 1))
    for m in range(g2.NGRID):
        tm = g2.TIMES[m]; nmax = (m + 1) * STRIDE                    # fine steps up to t_m
        for n in range(nmax):
            c = SIG * np.exp(-theta * (tm - TF[n + 1])) * g
            Psi[m, n + 1] += c; Psi[m, n] -= c
    return Psi


def arrow_band_gap(Lam, J, b, iters=400):
    """min KL(q || N(mu, Lam^{-1})) over q with precision M M^T, M lower-triangular with pattern:
    order (eta_1..eta_n, xi_1..xi_J); eta rows: band b; xi rows: dense.  Returns gap in nats."""
    n = Lam.shape[0] - J
    perm = np.concatenate([np.arange(J, J + n), np.arange(J)])      # eta first, xi last
    Lp = torch.as_tensor(Lam[np.ix_(perm, perm)])
    D = n + J
    mask = torch.zeros(D, D)
    for i in range(D):
        if i < n:
            mask[i, max(0, i - b):i + 1] = 1.0
        else:
            mask[i, :i + 1] = 1.0
    # initialise with the exact Cholesky of Lam masked
    M0 = torch.linalg.cholesky(Lp) * mask
    d0 = torch.log(torch.diagonal(M0))
    off = (M0 - torch.diag(torch.diagonal(M0))).clone().requires_grad_(True)
    dlog = d0.clone().requires_grad_(True)
    ld_Lam = torch.logdet(Lp)
    opt = torch.optim.LBFGS([off, dlog], max_iter=iters, tolerance_grad=1e-10, tolerance_change=1e-14, line_search_fn="strong_wolfe")
    tri = torch.tril(torch.ones(D, D), -1) * mask
    def kl():
        M = off * tri + torch.diag(torch.exp(dlog))
        # Sigma = M^{-T} M^{-1};  tr(Lam Sigma) = ||M^{-1} C||_F^2 with Lam = C C^T
        C = torch.linalg.cholesky(Lp)
        Y = torch.linalg.solve_triangular(M, C, upper=False)
        return 0.5 * ((Y ** 2).sum() + 2 * dlog.sum() - D - ld_Lam)  # log det Sigma = -2 sum log diag M
    def closure():
        opt.zero_grad(); v = kl(); v.backward(); return v
    opt.step(closure)
    return float(kl().detach())


def main():
    t0 = time.time(); rng = np.random.default_rng(2027)
    Psi = residual_operator(TH)
    res = {"setting": {"T": T, "theta": TH, "sigma": SIG, "s_obs": g2.S_OBS, "N_fine": N, "obs": g2.TIMES[g2.OBS_B].tolist()},
           "results": {}}
    for H in g2.HS:
        C = exact_cov(g2.TIMES, H, TH, SIG)
        obs = g2.OBS_B
        ys = [np.linalg.cholesky(C[np.ix_(obs, obs)] + S2 * np.eye(len(obs))) @ rng.standard_normal(len(obs)) for _ in range(5)]
        bank = g2.cosine_feature_bank(H, Jmax=32)
        out = {}
        for J in (8, 16, 32):
            cc = CosineCoords(H, T, J)
            tin = TF[1:-1]
            A = cc.path_funcs(tin, "projection")
            CR = cc.fbm_cov(tin, tin) - A @ A.T
            CR = 0.5 * (CR + CR.T)
            ev = np.linalg.eigvalsh(CR)
            LR = np.linalg.cholesky(CR + 1e-12 * np.trace(CR) / len(tin) * np.eye(len(tin)))
            PsiI = Psi[:, 1:-1]                                     # R(0) = R(T) = 0
            Phi = SIG * bank["projection"][:, :J]                    # (40, J)
            Fr = PsiI @ LR                                           # (40, N-1)
            F_all = np.hstack([Phi, Fr])                             # X = F_all @ (xi, eta)
            prior_C = F_all @ F_all.T
            prior_err = float(np.sum(np.abs(np.diag(prior_C) - np.diag(C))) / np.sum(np.diag(C)))
            rec = {"prior_var_integrated_err_TSW": prior_err, "resid_cov_min_eig": float(ev.min()), "resid_var_share": float(np.trace(CR) / np.trace(cc.fbm_cov(tin, tin))),
                   "datasets": []}
            for y in ys:
                lp_exact = log_normal(y, C[np.ix_(obs, obs)] + S2 * np.eye(len(obs)))
                _, v_e = gp_posterior(C[:, obs], C[np.ix_(obs, obs)], np.diag(C), y, S2)
                Fo = F_all[obs]
                lp_tsw = log_normal(y, Fo @ Fo.T + S2 * np.eye(len(obs)))
                Lam = np.eye(F_all.shape[1]) + Fo.T @ Fo / S2
                Sig = np.linalg.inv(Lam)
                v_full = np.einsum("ij,jk,ik->i", F_all, Sig, F_all)
                # block-diagonal: xi full x eta diagonal
                Lxx = Lam[:J, :J]; d_ee = np.diag(Lam)[J:]
                gap_bd = 0.5 * (np.linalg.slogdet(Lxx)[1] + np.sum(np.log(d_ee)) - np.linalg.slogdet(Lam)[1])
                S_bd = np.zeros_like(Lam); S_bd[:J, :J] = np.linalg.inv(Lxx); S_bd[J:, J:] = np.diag(1 / d_ee)
                v_bd = np.einsum("ij,jk,ik->i", F_all, S_bd, F_all)
                # cosine-only with the same J (truncated model) for reference
                Po = Phi[obs]; Lc = np.eye(J) + Po.T @ Po / S2
                v_cos = np.einsum("ij,jk,ik->i", Phi, np.linalg.inv(Lc), Phi)
                d = {"dlogp_TSW_minus_exact": lp_tsw - lp_exact,
                     "post_var_err_full": float(np.sum(np.abs(v_full - v_e)) / np.sum(v_e)),
                     "post_var_err_blockdiag": float(np.sum(np.abs(v_bd - v_e)) / np.sum(v_e)),
                     "post_var_err_cosine_only_sameJ": float(np.sum(np.abs(v_cos - v_e)) / np.sum(v_e)),
                     "gap_blockdiag": float(gap_bd)}
                for b in (1, 2, 4):
                    d[f"gap_arrow_b{b}"] = arrow_band_gap(Lam, J, b)
                rec["datasets"].append(d)
            agg = {k: float(np.mean([x[k] for x in rec["datasets"]])) for k in rec["datasets"][0]}
            rec["mean"] = agg
            out[f"J{J}"] = rec
            print(f"H={H} J={J} prior_err={prior_err:.2e} " + " ".join(f"{k}={v:.4f}" for k, v in agg.items()) + f"  ({time.time() - t0:.0f}s)", flush=True)
        res["results"][f"{H:.2f}"] = out
    with open(os.path.join(OUT, "e1a_two_scale_linear.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote e1a_two_scale_linear.json")


if __name__ == "__main__":
    main()
