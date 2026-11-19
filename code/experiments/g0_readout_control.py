r"""G0 read-out control (18 Nov 2026): an independent exact computation of the own-lift posterior and a sampling
control of the 256-path read-out, stored as a result for the manuscript.  CPU, float64, no training.

For the bridge runs of the authors' code that exist for two seeds (theta = 1, H = 0.7 and 0.9, seeds 0 and 1):
  1. integrated relative error of the stored fitted posterior SD against three references stored in the run file:
     the authors' analytic benchmark curve (stationary fOU conditioned at zero), the standard fOU started at zero
     (the simulated model; exact), and the exact posterior of the lift as coded ("own lift");
  2. the deterministic discrepancy between the own-lift and standard references;
  3. the exact own-lift posterior recomputed independently (matrix exponentials of the linear state (X, Y_1..Y_K),
     stationary OU initial states, exact Gaussian conditioning on the noisy end point), checked against the stored curve;
  4. the sampling distribution of the integrated error of a 256-path read-out under the EXACT own-lift posterior
     (512 independent batches of whole correlated paths; also 4096 paths), i.e. the error a perfect fit would show.
Error metric everywhere: 100 * sum_t |s(t) - s_ref(t)| / sum_t s_ref(t) over the stored positive times.
Run from work\code\experiments:  python -X utf8 g0_readout_control.py      Out: work/results/g0/readout_control.json
"""
import os, json
import numpy as np
from scipy.linalg import expm

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "g0")


def ierr(s, ref):
    return float(100 * np.abs(np.asarray(s) - np.asarray(ref)).sum() / np.asarray(ref).sum())


def own_lift_posterior_cov(rec, s_obs=0.1):
    """Exact posterior covariance of X on the stored times for the lift as coded (unit diffusion, X(0) = 0,
    stationary Y(0)), conditioned on Y_obs = X(T) + N(0, s_obs^2)."""
    g = np.array(rec["gamma"]); w = np.array(rec["omega"]); m = len(g) + 1
    A = np.zeros((m, m)); A[0, 0] = -rec["theta"]; A[0, 1:] = -g * w; A[1:, 1:] = -np.diag(g)
    b = np.r_[w.sum(), np.ones(m - 1)]
    n = len(rec["times"]); dt = 2.0 / n
    V = expm(dt * np.block([[A, np.outer(b, b)], [np.zeros_like(A), -A.T]]))
    F = V[:m, :m]; Q = V[:m, m:] @ F.T
    P = np.zeros_like(A); P[1:, 1:] = 1.0 / (g[:, None] + g[None, :])
    lag = np.empty((n, m)); lag[0] = np.eye(m)[0]
    for k in range(1, n):
        lag[k] = lag[k - 1] @ F
    C = np.zeros((n, n))
    for j in range(n):
        P = F @ P @ F.T + Q
        C[j:, j] = lag[:n - j] @ P[:, 0]
        C[j, j:] = C[j:, j]
    return C - np.outer(C[:, -1], C[:, -1]) / (C[-1, -1] + s_obs ** 2)


def main():
    out = {"rule": "audit 18 Nov section 2, adopted", "runs": {}, "by_H": {}}
    for H in (0.7, 0.9):
        for seed in (0, 1):
            rec = json.load(open(os.path.join(RES, f"bridge_H{H:.2f}_theta1.0_seed{seed}_steps2000.json")))
            s = rec["std_empirical"]
            out["runs"][f"{H}|{seed}"] = {"authors_curve": ierr(s, rec["std_ref_authors"]),
                                          "standard_fou": ierr(s, rec["std_ref_true"]),
                                          "own_lift": ierr(s, rec["std_ref_ownmodel"]),
                                          "train_hours": rec["train_seconds"] / 3600, "n_paths": 256}
        own, std = np.array(rec["std_ref_ownmodel"]), np.array(rec["std_ref_true"])
        C = own_lift_posterior_cov(rec)
        sd = np.sqrt(np.diag(C))
        rng = np.random.default_rng(20261118 + int(100 * H))
        Lc = np.linalg.cholesky(C + 1e-14 * np.eye(len(sd)))
        ctrl = {}
        for npath in (256, 4096):
            errs = []
            for _ in range(512):
                paths = rng.standard_normal((npath, len(sd))) @ Lc.T
                errs.append(ierr(paths.std(axis=0), sd))
            ctrl[str(npath)] = np.percentile(errs, [2.5, 50, 97.5]).round(3).tolist()
        out["by_H"][str(H)] = {"own_vs_standard_reference_discrepancy": ierr(own, std),
                               "recomputed_own_lift_max_abs_sd_diff": float(np.abs(sd - own).max()),
                               "readout_error_percentiles_2.5_50_97.5": ctrl}
    with open(os.path.join(RES, "readout_control.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    for k, v in out["runs"].items():
        print(k, {a: round(b, 2) for a, b in v.items()})
    for k, v in out["by_H"].items():
        print(k, v)


if __name__ == "__main__":
    main()
