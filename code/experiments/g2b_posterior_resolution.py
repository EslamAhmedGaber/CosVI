"""G2b (diagnosis after the G2 posterior-variance failure): how many cosine coordinates does the posterior need?

Same model and data as g2_linear_exact.py, case B (10 observations, s = 0.1), J up to 256, plus a sweep over the
number of observations m in {2, 5, 10, 20, 40} at fixed J, to estimate the rule J ~ c * m (resolution between
observations).  Exact computations only.  Out: work/results/g2/g2b_posterior_resolution.json
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "2")
import sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import g2_linear_exact as g2
from fnvi.linear_exact import exact_cov, log_normal, gp_posterior

JS = [16, 32, 64, 96, 128, 192, 256]
MS = [2, 5, 10, 20, 40]


def main():
    t0 = time.time(); s2 = g2.S_OBS ** 2; out = {"J": JS, "m": MS, "results": {}}
    rng = np.random.default_rng(7)
    for H in g2.HS:
        C = exact_cov(g2.TIMES, H, g2.THETA, g2.SIGMA)
        bank = g2.cosine_feature_bank(H, Jmax=max(JS))
        res = {}
        for m in MS:
            obs = np.round(np.linspace(0, g2.NGRID - 1, m + 1)[1:]).astype(int) if m < g2.NGRID else np.arange(g2.NGRID)
            Koo = C[np.ix_(obs, obs)]
            ys = [np.linalg.cholesky(Koo + s2 * np.eye(m)) @ rng.standard_normal(m) for _ in range(10)]
            ex = [(log_normal(y, Koo + s2 * np.eye(m)), *gp_posterior(C[:, obs], Koo, np.diag(C), y, s2)) for y in ys]
            for J in JS:
                Phi = g2.SIGMA * bank["projection"][:, :J]; Po = Phi[obs]
                dl, me, vi = [], [], []
                for y, (lp, mu_e, v_e) in zip(ys, ex):
                    dl.append(log_normal(y, Po @ Po.T + s2 * np.eye(m)) - lp)
                    Sig = np.linalg.inv(np.eye(J) + Po.T @ Po / s2); mu = Phi @ (Sig @ Po.T @ y / s2)
                    v = np.einsum("ij,jk,ik->i", Phi, Sig, Phi)
                    me.append(np.max(np.abs(mu - mu_e) / np.sqrt(v_e)))
                    vi.append(np.sum(np.abs(v - v_e)) / np.sum(v_e))
                res[f"m{m}_J{J}"] = {"dlogp_mean": float(np.mean(dl)), "post_mean_err_sd_units_max": float(np.max(me)),
                                     "post_var_relerr_integrated_max": float(np.max(vi))}
            print(f"H={H} m={m} done {time.time() - t0:.0f}s", flush=True)
        out["results"][f"{H:.2f}"] = res
    with open(os.path.join(g2.OUT, "g2b_posterior_resolution.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote g2b_posterior_resolution.json")


if __name__ == "__main__":
    main()
