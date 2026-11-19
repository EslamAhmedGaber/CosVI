"""V1: the trained CVI pipeline (rate table + RK4 random-ODE solve + Gaussian q + Monte Carlo ELBO + Adam) on the
linear fOU model, where the answer is known exactly.  Checks that training reaches the exact evidence of the
truncated model log p_J(y) and its exact posterior, and reports the remaining gap to the true model.

Adam with cosine learning-rate decay (0.05 -> 0.001), 2000 steps, S = 64.
Model and data as G2 case B (T = 2, theta = sigma = 1, 10 observations, s = 0.1), one data set per H (seed 11).
Out: work/results/v1/v1_trained_vi_linear.json
"""
import os, sys, json, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import g2_linear_exact as g2
from fnvi.cvi import rate_table, GaussianFamily, solve_rk4, noise_rates
from fnvi.linear_exact import exact_cov, log_normal, gp_posterior

OUT = os.path.join(HERE, "..", "..", "results", "v1"); os.makedirs(OUT, exist_ok=True)
torch.set_default_dtype(torch.float64)
def n_steps_for(J):                            # resolve the highest mode: about 5 steps per unit of J, multiple of 40
    return int(40 * max(4, int(np.ceil(5 * J / 40))))


def run(H, J, kind="full", steps=1500, S=64, lr=0.05, seed=0):
    torch.manual_seed(seed)
    s2 = g2.S_OBS ** 2
    C = exact_cov(g2.TIMES, H, g2.THETA, g2.SIGMA)
    obs = g2.OBS_B
    y = np.linalg.cholesky(C[np.ix_(obs, obs)] + s2 * np.eye(len(obs))) @ np.random.default_rng(11).standard_normal(len(obs))
    lp_exact = log_normal(y, C[np.ix_(obs, obs)] + s2 * np.eye(len(obs)))
    N_STEPS = n_steps_for(J); STEP_OF_GRID = np.arange(1, g2.NGRID + 1) * (N_STEPS // g2.NGRID)
    table = rate_table(H, g2.T, J, N_STEPS)
    f = lambda x, t: -g2.THETA * x
    g = lambda x, t: g2.SIGMA
    q = GaussianFamily(1, J, kind=kind, rank=4)
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.02)
    yt = torch.as_tensor(y)
    idx = torch.as_tensor(STEP_OF_GRID[obs])
    t0 = time.time()
    for it in range(steps):
        xi = q.rsample(S)                                                   # (S, 1, J)
        R = noise_rates(table, xi)                                          # (2N+1, S, 1)
        X = solve_rk4(f, g, torch.zeros(S, 1), R, g2.T, N_STEPS)            # (N+1, S, 1)
        Xo = X[idx, :, 0].T                                                 # (S, m)
        ll = (-0.5 * ((yt - Xo) ** 2).sum(1) / s2 - 0.5 * len(obs) * np.log(2 * np.pi * s2)).mean()
        loss = -(ll - q.kl())
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    train_s = time.time() - t0
    # exact ELBO of the trained q: the solve is linear in xi, so Phi is obtained by solving with xi = e_j
    with torch.no_grad():
        E = torch.eye(J)[:, None, :]                                        # (J, 1, J)
        Xb = solve_rk4(f, g, torch.zeros(J, 1), noise_rates(table, E), g2.T, N_STEPS)[:, :, 0]   # (N+1, J)
        Phi_all = Xb[torch.as_tensor(STEP_OF_GRID)].numpy()                 # (40, J) on the grid
        mu = q.mu[0].numpy(); L = q.scale_tril()[0].numpy(); Sq = L @ L.T
    Po = Phi_all[obs]
    r = y - Po @ mu
    ell = -0.5 * (r @ r + np.trace(Po @ Sq @ Po.T)) / s2 - 0.5 * len(obs) * np.log(2 * np.pi * s2)
    kl = 0.5 * (np.trace(Sq) + mu @ mu - J - np.linalg.slogdet(Sq)[1])
    elbo = ell - kl
    lp_J = log_normal(y, Po @ Po.T + s2 * np.eye(len(obs)))
    Phi_q = g2.SIGMA * g2.cosine_feature_bank(H, Jmax=J)["projection"][:, :J]   # quadrature route (independent)
    v_trained = np.einsum("ij,jk,ik->i", Phi_all, Sq, Phi_all)
    Lam = np.eye(J) + Po.T @ Po / s2; v_trunc = np.einsum("ij,jk,ik->i", Phi_all, np.linalg.inv(Lam), Phi_all)
    _, v_exact = gp_posterior(C[:, obs], C[np.ix_(obs, obs)], np.diag(C), y, s2)
    return {"H": H, "J": J, "kind": kind, "n_steps": N_STEPS, "train_seconds": train_s, "elbo_trained_exact_eval": float(elbo),
            "logp_J": float(lp_J), "logp_exact": float(lp_exact), "gap_elbo_to_logpJ": float(lp_J - elbo),
            "gap_logpJ_to_exact": float(lp_J - lp_exact),
            "rk4_vs_quadrature_Phi_maxabs": float(np.abs(Phi_all - Phi_q).max()),
            "post_var_trained_vs_truncated_integrated": float(np.sum(np.abs(v_trained - v_trunc)) / np.sum(v_trunc)),
            "post_var_trained_vs_exact_integrated": float(np.sum(np.abs(v_trained - v_exact)) / np.sum(v_exact))}


def main():
    res = []
    for H in g2.HS:
        for J in (32,):
            for kind in ("full", "meanfield"):
                r = run(H, J, kind, steps=2000)
                res.append(r)
                print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    with open(os.path.join(OUT, "v1_trained_vi_linear.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
