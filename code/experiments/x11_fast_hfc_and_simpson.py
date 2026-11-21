"""X11 (Tier 3): Verify fast_hfc.py against 30-digit mpmath, run fair 1-thread speedup benchmarks
against vendor/hfc.py, and verify the Simpson-exact smooth-stage map (twoscale_exact_map) against
twoscale_map, twoscale_pl_map, and exact grid.

Run from code/experiments:  python -X utf8 x11_fast_hfc_and_simpson.py
Out: results/x11/fast_hfc_benchmark.json, results/x11/simpson_exact_check.json
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import sys, json, time
import numpy as np
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from fnvi import fast_hfc
from fnvi.vendor import hfc as slow_hfc
from fnvi.coords import CosineCoords
from fnvi.noise_maps import twoscale_map, twoscale_exact_map, twoscale_pl_map, grid_map, path_cov_on_grid
from x8_linear_check import T as T_LIN, N as N_LIN, J as J_LIN, S_OBS, NDATA, obs_matrix, scores

OUT = os.path.join(HERE, "..", "..", "results", "x11")
os.makedirs(OUT, exist_ok=True)


def verify_mpmath():
    mp.mp.dps = 30
    Hs = [0.5, 0.505, 0.55, 0.6, 0.75, 0.9, 0.95, 0.99]
    xs = np.array([0.0, 1e-4, 0.5, 3.141592653589793, 25.0, 55.9, 56.1, 100.0, 400.0, 2000.0])
    acc = {}
    worst_M = 0.0
    worst_N = 0.0
    worst_G = 0.0
    worst_Q = 0.0
    worst_dQ = 0.0
    for H in Hs:
        m_fast = fast_hfc.M_H(H, xs)
        n_fast = fast_hfc.N_H(H, xs)
        m_ref = np.array([1.0 + 0j if abs(H - 0.5) < 1e-15 else complex(mp.hyp1f1(2 * H - 1, 2 * H, 1j * x)) for x in xs])
        n_ref = np.array([complex(mp.hyp1f1(2 * H, 2 * H + 1, 1j * x)) for x in xs])
        err_M = float(np.max(np.abs(m_fast - m_ref)))
        err_N = float(np.max(np.abs(n_fast - n_ref)))
        # Compare Gram, Q, dQ against slow_hfc (Gauss-Jacobi) at T=5, J=32
        T, J = 5.0, 32
        ts = np.linspace(0.0, T, 81)
        eG = float(np.max(np.abs(fast_hfc.gram(H, T, "cos", J) - slow_hfc.gram(H, T, "cos", J))))
        eQ = float(np.max(np.abs(fast_hfc.Q(H, T, "cos", J, ts) - slow_hfc.Q(H, T, "cos", J, ts))))
        edQ = float(np.max(np.abs(fast_hfc.dQ(H, T, "cos", J, ts) - slow_hfc.dQ(H, T, "cos", J, ts))))
        worst_M = max(worst_M, err_M)
        worst_N = max(worst_N, err_N)
        worst_G = max(worst_G, eG)
        worst_Q = max(worst_Q, eQ)
        worst_dQ = max(worst_dQ, edQ)
        acc[f"{H:.3f}"] = {"max_abs_M_vs_mpmath": err_M, "max_abs_N_vs_mpmath": err_N,
                           "max_abs_G_vs_ref": eG, "max_abs_Q_vs_ref": eQ, "max_abs_dQ_vs_ref": edQ}
    return {"by_H": acc, "worst_M": worst_M, "worst_N": worst_N,
            "worst_G": worst_G, "worst_Q": worst_Q, "worst_dQ": worst_dQ}


def bench_one(fn, reps=5):
    ts = []
    for _ in range(reps):
        fast_hfc.clear_caches()
        slow_hfc._rules.clear()
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def benchmark_speed():
    cases = [
        ("twoscale", 0.75, 5.0, 80, 16, 5),
        ("twoscale", 0.75, 50.0, 200, 16, 5),
        ("cosine", 0.75, 50.0, 200, 64, 3),
        ("cosine", 0.75, 50.0, 200, 256, 3),
        ("twoscale", 0.75, 50.0, 2500, 16, 3),
        ("cosine", 0.75, 50.0, 2500, 64, 3),
        ("cosine", 0.90, 50.0, 200, 256, 3),
        ("gram", 0.75, 1.0, 0, 512, 5),
    ]
    rows = []
    for kind, H, T, N, J, reps in cases:
        tk = np.linspace(0.0, T, 2 * N + 1) if N > 0 else None
        if kind == "gram":
            t_slow = bench_one(lambda: slow_hfc.gram(H, T, "cos", J), reps=reps)
            t_fast = bench_one(lambda: fast_hfc.gram(H, T, "cos", J), reps=reps)
            max_diff = float(np.max(np.abs(fast_hfc.gram(H, T, "cos", J) - slow_hfc.gram(H, T, "cos", J))))
        elif kind == "cosine":
            t_slow = bench_one(lambda: CosineCoords(H, T, J, fast=False).rate_funcs(tk, "projection"), reps=min(reps, 2 if (N >= 2500 or J >= 256) else reps))
            t_fast = bench_one(lambda: CosineCoords(H, T, J, fast=True).rate_funcs(tk, "projection"), reps=reps)
            max_diff = float(np.max(np.abs(
                CosineCoords(H, T, J, fast=True).rate_funcs(tk, "projection")
                - CosineCoords(H, T, J, fast=False).rate_funcs(tk, "projection")
            )))
        else:
            def run_ts(fast_flag):
                cc = CosineCoords(H, T, J, fast=fast_flag)
                _ = cc.rate_funcs(tk, "projection")
                tg = np.arange(1, N) * (T / N)
                A = cc.path_funcs(tg, "projection")
                CR = cc.fbm_cov(tg, tg) - A @ A.T
                CR = 0.5 * (CR + CR.T)
                return np.linalg.cholesky(CR)
            t_slow = bench_one(lambda: run_ts(False), reps=min(reps, 2 if N >= 2500 else reps))
            t_fast = bench_one(lambda: run_ts(True), reps=reps)
            max_diff = float(np.max(np.abs(run_ts(True) - run_ts(False))))
        sp = t_slow / t_fast
        row = {"kind": kind, "H": H, "T": T, "N": N, "J": J,
               "slow_ms": 1e3 * t_slow, "fast_ms": 1e3 * t_fast, "speedup": sp, "max_abs_diff": max_diff}
        rows.append(row)
        print(f"  {kind:8s} H={H:.2f} T={T:4.0f} N={N:4d} J={J:3d}: slow={row['slow_ms']:8.1f} ms | fast={row['fast_ms']:6.2f} ms | {sp:6.1f}x | diff={max_diff:.2e}", flush=True)
    # Also time H-bank (99 values of H in [0.5, 0.99], N=200, J=64)
    Hs_bank = np.round(np.arange(0.5, 0.99 + 1e-9, 0.005), 6)
    tk200 = np.linspace(0.0, 50.0, 401)
    fast_hfc.clear_caches()
    t0 = time.perf_counter()
    for h in Hs_bank:
        CosineCoords(float(h), 50.0, 64, fast=True).rate_funcs(tk200, "projection")
    bank_fast_s = time.perf_counter() - t0
    print(f"  h_bank (99 H, T=50, N=200, J=64): fast={bank_fast_s:.2f} s (est slow ~95 s)", flush=True)
    return {"threads": 1, "rows": rows, "h_bank_99H_N200_J64_fast_s": bank_fast_s}


def check_simpson_exact():
    out = {"T": T_LIN, "N": N_LIN, "J": J_LIN, "by_H": {}}
    tg = np.arange(1, N_LIN + 1, dtype=float) * (T_LIN / N_LIN)
    for H in [0.5, 0.55, 0.6, 0.75, 0.9, 0.95]:
        cc = CosineCoords(H, T_LIN, J_LIN, fast=True)
        K_ex = cc.fbm_cov(tg, tg)
        # Strict PD check of C_R (zero jitter!)
        t_int = tg[:-1]
        A_int = cc.path_funcs(t_int, "projection")
        CR = 0.5 * ((cc.fbm_cov(t_int, t_int) - A_int @ A_int.T) + (cc.fbm_cov(t_int, t_int) - A_int @ A_int.T).T)
        ev_CR = np.linalg.eigvalsh(CR)
        ev_G = np.linalg.eigvalsh(cc.G)

        M_raw = twoscale_map(H, T_LIN, N_LIN, J_LIN)
        M_ex = twoscale_exact_map(H, T_LIN, N_LIN, J_LIN, jitter=0.0)
        M_pl = twoscale_pl_map(H, T_LIN, N_LIN, J_LIN)
        M_gr = grid_map(H, T_LIN, N_LIN)

        cov_err = {
            "raw_twoscale": float(np.linalg.norm(path_cov_on_grid(*M_raw, T_LIN) - K_ex) / np.linalg.norm(K_ex)),
            "simpson_exact": float(np.linalg.norm(path_cov_on_grid(*M_ex, T_LIN) - K_ex) / np.linalg.norm(K_ex)),
            "twoscale_pl": float(np.linalg.norm(path_cov_on_grid(*M_pl, T_LIN) - K_ex) / np.linalg.norm(K_ex)),
            "grid": float(np.linalg.norm(path_cov_on_grid(*M_gr, T_LIN) - K_ex) / np.linalg.norm(K_ex)),
        }
        lin = {}
        if H in (0.6, 0.75, 0.9):
            F_gr = obs_matrix(M_gr); F_pl = obs_matrix(M_pl); F_raw = obs_matrix(M_raw); F_ex = obs_matrix(M_ex)
            rng = np.random.default_rng(20261118 + int(100 * H))
            Ys = [F_gr @ rng.standard_normal(F_gr.shape[1]) + S_OBS * rng.standard_normal(F_gr.shape[0]) for _ in range(NDATA)]
            e_gr = np.array([scores(F_gr, y)[0] for y in Ys])
            e_pl = np.array([scores(F_pl, y)[0] for y in Ys])
            e_raw = np.array([scores(F_raw, y)[0] for y in Ys])
            e_ex = np.array([scores(F_ex, y)[0] for y in Ys])
            # Also compute exact linear-Gaussian KL gap for Product Full, Joint Full (0), Mean-Field, and BlockJoint!
            d_pl = F_pl.shape[1]  # J + (N-1) + 1 = 96, last col is x0
            Lam = np.eye(d_pl) + F_pl.T @ F_pl / (S_OBS ** 2)
            ld = lambda A: float(2.0 * np.log(np.diag(np.linalg.cholesky(A))).sum())
            ld_full = ld(Lam)
            # Reorder indices so [cosine (0..J-1), x0 (d_pl-1)] is the joint block, rest is eta (J..d_pl-2)
            idx_blk = list(range(J_LIN)) + [d_pl - 1]
            idx_eta = list(range(J_LIN, d_pl - 1))
            gap_prod_full = 0.5 * (ld(Lam[:-1, :-1]) + np.log(Lam[-1, -1]) - ld_full)
            gap_block_joint = 0.5 * (ld(Lam[np.ix_(idx_blk, idx_blk)]) + float(np.log(np.diag(Lam)[idx_eta]).sum()) - ld_full)
            gap_mf_joint = 0.5 * (float(np.log(np.diag(Lam)).sum()) - ld_full)
            # Mean-field on grid coordinates for comparison
            d_gr = F_gr.shape[1]
            Lam_gr = np.eye(d_gr) + F_gr.T @ F_gr / (S_OBS ** 2)
            gap_mf_grid = 0.5 * (float(np.log(np.diag(Lam_gr)).sum()) - ld(Lam_gr))
            lin = {
                "max_logev_diff_pl_vs_grid": float(np.max(np.abs(e_pl - e_gr))),
                "max_logev_diff_raw_vs_grid": float(np.max(np.abs(e_raw - e_gr))),
                "max_logev_diff_simpson_exact_vs_grid": float(np.max(np.abs(e_ex - e_gr))),
                "exact_gap_joint_full": 0.0,
                "exact_gap_product_full": float(gap_prod_full),
                "exact_gap_block_joint_J16": float(gap_block_joint),
                "exact_gap_mf_cosvi": float(gap_mf_joint),
                "exact_gap_mf_grid": float(gap_mf_grid),
            }
        out["by_H"][f"{H:.2f}"] = {
            "cond_G": float(ev_G[-1] / ev_G[0]),
            "min_eig_CR": float(ev_CR[0]),
            "cond_CR": float(ev_CR[-1] / ev_CR[0]),
            "cov_relerr": cov_err,
            "linear_check": lin,
        }
        print(f"  H={H:.2f}: min(CR)={ev_CR[0]:.2e} | cov_err raw={cov_err['raw_twoscale']:.2e} -> simp_exact={cov_err['simpson_exact']:.2e} | lin_diff raw={lin.get('max_logev_diff_raw_vs_grid', float('nan')):.2e} -> simp_exact={lin.get('max_logev_diff_simpson_exact_vs_grid', float('nan')):.2e}", flush=True)
    return out


if __name__ == "__main__":
    print("=== 1. Verifying fast_hfc against 30-digit mpmath ===", flush=True)
    acc = verify_mpmath()
    print(f"  WORST ERRORS: M={acc['worst_M']:.2e}, N={acc['worst_N']:.2e}, G={acc['worst_G']:.2e}, Q={acc['worst_Q']:.2e}, dQ={acc['worst_dQ']:.2e}", flush=True)
    print("\n=== 2. Fair 1-thread benchmark (fast_hfc vs vendor/hfc) ===", flush=True)
    spd = benchmark_speed()
    json.dump({"accuracy": acc, "benchmark": spd}, open(os.path.join(OUT, "fast_hfc_benchmark.json"), "w"), indent=1)
    print("\n=== 3. Checking Simpson-exact map & strict PD of C_R ===", flush=True)
    simp = check_simpson_exact()
    json.dump(simp, open(os.path.join(OUT, "simpson_exact_check.json"), "w"), indent=1)
    print("DONE X11 CPU checks.")
