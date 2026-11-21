"""Every Gaussian noise representation as a linear map from whitened coordinates z ~ N(0, I_D) to the noise rate
dB/dt at the three RK4 stage positions of each of N steps on [0, T] (left end, midpoint, right end).

    rate_stage[n] = M_stage[n, :] @ z,   M_stage in R^{N x D},  stage in {left, mid, right}

With the same variational family over z, representations differ only in (law accuracy, D, smoothness, cost),
so trained comparisons isolate the representation.  Representations (all target STANDARD fBm, Var t^{2H}):
  cosine     : J projection cosine coordinates (CosChaos); smooth rates; D = J; truncated law
  twoscale   : cosine (J) + whitened residual on the interior grid, piecewise linear; D = J + N - 1;
               EXACT law on the grid (construction of Section 3.1 of the paper)
  grid       : exact fGn increments on the grid, Cholesky whitening (law of Davies-Harte, as used by
               Beskos, Dureau, Kalogeropoulos 2015); piecewise-linear path; D = N; EXACT on the grid
  markov     : Markovian lift of Daems et al. 2024 (type I, stationary OU start, weights of their Prop. 5,
               rescaled to standard fBm), exponential-Euler OU steps driven by the grid increments of W;
               D = K + N; inexact law (history floor)
At H = 1/2 'grid' and 'cosine' are Brownian motion (the Brownian baseline).
"""
import os
import numpy as np
from .coords import CosineCoords
from .cvi import rate_table_np
from .ma_fbm import MAFit, nodes, c_mvn

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "cache")


def _stages_from_table(tab):
    return tab[0:-1:2], tab[1::2], tab[2::2]                # left, mid, right: (N, D) each


def cosine_map(H, T, N, J, mode="projection"):
    return _stages_from_table(rate_table_np(H, T, J, N, mode))


def fgn_cov(H, N, h):
    k = np.arange(N, dtype=float)
    r = 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * k ** (2 * H) + np.abs(k - 1) ** (2 * H)) * h ** (2 * H)
    return r[np.abs(np.subtract.outer(np.arange(N), np.arange(N)))]


def grid_map(H, T, N):
    h = T / N
    L = np.linalg.cholesky(fgn_cov(H, N, h))
    M = L / h
    return M, M, M


def twoscale_map(H, T, N, J):
    h = T / N
    left, mid, right = cosine_map(H, T, N, J)
    cc = CosineCoords(H, T, J)
    t = np.arange(1, N) * h
    A = cc.path_funcs(t, "projection")
    CR = cc.fbm_cov(t, t) - A @ A.T
    CR = 0.5 * (CR + CR.T)
    LR = np.linalg.cholesky(CR + 1e-12 * np.trace(CR) / len(t) * np.eye(len(t)))
    Dm = np.zeros((N, N - 1))                                  # Delta R_n = R_{n+1} - R_n, R_0 = R_N = 0
    for n in range(N):
        if n <= N - 2:
            Dm[n, n] += 1.0
        if n >= 1:
            Dm[n, n - 1] -= 1.0
    Rr = Dm @ LR / h                                            # (N, N-1), piecewise-constant residual rate
    return tuple(np.hstack([s, Rr]) for s in (left, mid, right))


def twoscale_exact_map(H, T, N, J, jitter=0.0):
    """Exact-increment Simpson-corrected smooth-stage CosVI map.

    Retains the smooth cosine stage rates at left and right endpoints of each step and adjusts the
    midpoint stage rate by (3/2) * (exact_inc - simpson_inc) so that Simpson's rule reproduces the
    exact grid increments B^J(t_{n+1}) - B^J(t_n) to machine precision (~1e-15), eliminating the
    O(h^{2H}) endpoint quadrature defect of raw smooth stages. Because C_R is strictly positive
    definite on {t_1, ..., t_{N-1}} for every J >= 1 and N >= 2, Cholesky succeeds without jitter
    (falling back to tiny relative jitter only if needed for extreme grids).
    """
    h = T / N
    left, mid, right = cosine_map(H, T, N, J)
    cc = CosineCoords(H, T, J)
    tg = np.arange(N + 1, dtype=float) * h
    B_cos = np.zeros((N + 1, J))
    B_cos[1:] = cc.path_funcs(tg[1:], "projection")
    exact_inc = (B_cos[1:] - B_cos[:-1]) / h
    simp_inc = (left + 4.0 * mid + right) / 6.0
    mid_corr = mid + 1.5 * (exact_inc - simp_inc)

    A = B_cos[1:N]
    t = tg[1:N]
    CR = cc.fbm_cov(t, t) - A @ A.T
    CR = 0.5 * (CR + CR.T)
    if jitter > 0.0:
        CR = CR + jitter * np.trace(CR) / len(t) * np.eye(len(t))
    try:
        LR = np.linalg.cholesky(CR)
    except np.linalg.LinAlgError:
        LR = np.linalg.cholesky(CR + 1e-12 * np.trace(CR) / len(t) * np.eye(len(t)))
    Dm = np.zeros((N, N - 1))
    Dm[:-1, :] += np.eye(N - 1)
    Dm[1:, :] -= np.eye(N - 1)
    Rr = (Dm @ LR) / h
    return tuple(np.hstack([s, Rr]) for s in (left, mid_corr, right))


def twoscale_pl_map(H, T, N, J):
    """X8 control (18 Nov 2026): CosVI's coordinates (xi, eta), but the grid's solver.  Grid values
    B(t_n) = A(t_n) xi + R(t_n) from the analytic projection loading and the same residual factor as twoscale_map
    (R(0) = R(T) = 0; B(T) is a function of the constant cosine coordinate), and every RK4 stage forced by the
    piecewise-constant rate (B(t_{n+1}) - B(t_n)) / h, exactly as grid_map.  So twoscale_pl and grid share the law on
    the grid and the likelihood map; they differ only in the coordinates (D = J + N - 1 versus N)."""
    h = T / N
    cc = CosineCoords(H, T, J)
    t = np.arange(1, N) * h
    A = cc.path_funcs(t, "projection")
    CR = cc.fbm_cov(t, t) - A @ A.T
    CR = 0.5 * (CR + CR.T)
    LR = np.linalg.cholesky(CR + 1e-12 * np.trace(CR) / len(t) * np.eye(len(t)))
    V = np.zeros((N + 1, J + N - 1))                            # B at t_0 .. t_N as rows of coefficients
    V[1:N, :J] = A
    V[1:N, J:] = LR
    V[N, :J] = cc.path_funcs(np.array([T]), "projection")[0]
    M = (V[1:] - V[:-1]) / h
    return M, M, M


def markov_map(H, T, N, K=5, gamma_max=20.0, fit_horizon=None, gamma_min=None):
    """Exponential-Euler discretisation of the type-I lift; coordinates (zeta in R^K for Y(0), eta in R^N for W).
    gamma_min=None: their symmetric geometric rule on [1/gamma_max, gamma_max]; otherwise geometric on
    [gamma_min, gamma_max] (the 'best' configuration found on 14 Nov: long history, stability-respecting gamma_max)."""
    h = T / N
    g = nodes(K, gamma_max) if gamma_min is None else np.geomspace(gamma_min, gamma_max, K)
    w = MAFit(H, fit_horizon or 3 * T, g, "I").w / np.sqrt(c_mvn(H))
    C0 = 1.0 / (g[:, None] + g[None, :])
    ev, V = np.linalg.eigh(C0)
    L0 = V * np.sqrt(np.clip(ev, 0, None))
    D = K + N
    Y = np.zeros((K, D)); Y[:, :K] = L0                        # Y_0 = L0 zeta
    dec = np.exp(-g * h); phi = (1 - dec) / (g * h)
    B_prev = np.zeros(D); M = np.zeros((N, D))
    Y0w = w @ Y
    for n in range(N):
        dW = np.zeros(D); dW[K + n] = np.sqrt(h)
        Y = dec[:, None] * Y + phi[:, None] * dW[None, :]
        B = w @ Y - Y0w
        M[n] = (B - B_prev) / h
        B_prev = B
    return M, M, M


def build(rep, H, T, N, **kw):
    os.makedirs(CACHE, exist_ok=True)
    key = f"map_{rep}_H{H:.4f}_T{T}_N{N}_" + "_".join(f"{k}{v}" for k, v in sorted(kw.items())) + ".npz"
    fn = os.path.join(CACHE, key)
    if os.path.exists(fn):
        z = np.load(fn); return z["left"], z["mid"], z["right"]
    if rep == "cosine":
        out = cosine_map(H, T, N, kw.get("J", 32))
    elif rep == "twoscale":
        out = twoscale_map(H, T, N, kw.get("J", 16))
    elif rep == "twoscale_exact":
        out = twoscale_exact_map(H, T, N, kw.get("J", 16))
    elif rep == "twoscale_pl":
        out = twoscale_pl_map(H, T, N, kw.get("J", 16))
    elif rep == "grid":
        out = grid_map(H, T, N)
    elif rep == "markov":
        out = markov_map(H, T, N, kw.get("K", 5), kw.get("gamma_max", 20.0), gamma_min=kw.get("gamma_min"))
    else:
        raise ValueError(rep)
    np.savez(fn, left=out[0], mid=out[1], right=out[2])
    return out


def path_cov_on_grid(M_left, M_mid, M_right, T):
    """Covariance of B at the grid points implied by a map, integrating the stage rates with Simpson's rule
    (exact for piecewise-constant rates); returns (N, N)."""
    N = M_left.shape[0]; h = T / N
    inc = h / 6.0 * (M_left + 4 * M_mid + M_right)             # (N, D) increments
    P = np.cumsum(inc, axis=0)
    return P @ P.T
