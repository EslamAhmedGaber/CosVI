"""Exact second-order quantities of the fractional Ornstein-Uhlenbeck model and of its two finite approximations.

Model  dX = -theta X dt + sigma dB_H,  X(0) = 0,  standard fBm (Var B_H(t) = t^{2H}), 1/2 < H < 1.

* exact covariance: Cov(X(t),X(u)) = sigma^2 H(2H-1) int_0^t int_0^u e^{-theta(t-a)} e^{-theta(u-b)} |a-b|^{2H-2}
  evaluated with incomplete-gamma inner integrals and graded Gauss rules (no Monte Carlo);
* cosine model: X_J(t) = sigma * Phi(t) @ xi, Phi_j(t) = int_0^t e^{-theta(t-s)} a_j(s) ds, a = rate functions;
* MA-fBm model (type I, Daems et al. 2024): X_hat(t) = (sigma/sqrt(c_I)) int_0^t e^{-theta(t-s)} dB_hat(s), as a Wiener
  integral over (-inf, t] with closed-form kernels; covariance by graded quadrature on a common grid.
"""
import numpy as np
from scipy.special import gammainc, gamma as Gamma
from .quad import gl, gj_left, graded, graded_both, composite
from .coords import CosineCoords
from .ma_fbm import c_mvn


def _Gm(x, H, th):          # int_0^x e^{-th r} r^{2H-2} dr
    x = np.asarray(x, float)
    return th ** (1 - 2 * H) * Gamma(2 * H - 1) * gammainc(2 * H - 1, th * x)


def _Gp(x, H, th, n=40):    # int_0^x e^{+th r} r^{2H-2} dr = x^{2H-1} int_0^1 e^{th x rho} rho^{2H-2} drho
    x = np.atleast_1d(np.asarray(x, float))
    rho, w = gj_left(0.0, 1.0, 2 * H - 2, n)
    return x ** (2 * H - 1) * (np.exp(th * np.outer(x, rho)) @ w)


def exact_cov(times, H, theta, sigma, levels=30, n=12):
    times = np.asarray(times, float); m = len(times); C = np.zeros((m, m))
    for i in range(m):
        t = times[i]
        for j in range(i, m):
            u = times[j]
            # outer integral over a in [0, t], kinks at a = 0 and a = u
            brk = sorted(set([0.0, t] + ([u] if 0 < u < t else [])))
            val = 0.0
            for a0, a1 in zip(brk[:-1], brk[1:]):
                a, w = graded_both(a0, a1, levels, n)
                I = np.where(a <= u,
                             np.exp(-theta * (u - a)) * (_Gm(a, H, theta) + _Gp(np.clip(u - a, 0, None), H, theta)),
                             np.exp(-theta * (u - a)) * (_Gm(a, H, theta) - _Gm(np.clip(a - u, 0, None), H, theta)))
                val += w @ (np.exp(-theta * (t - a)) * I)
            C[i, j] = C[j, i] = sigma ** 2 * H * (2 * H - 1) * val
    return C


def cosine_features(times, H, T, J, theta, mode='projection', levels=24, n=12):
    """Phi (len(times), J) with X_J(t) = sigma * Phi(t) @ xi."""
    cc = CosineCoords(H, T, J)
    out = np.zeros((len(times), J))
    for i, t in enumerate(times):
        if t <= 0:
            continue
        s, w = composite(0.0, t, max(2, int(np.ceil(1.5 * J * t / T)) + 1), n, levels)
        out[i] = (w * np.exp(-theta * (t - s))) @ cc.rate_funcs(s, mode)
    return out


def _E(p, q, t):            # int_0^t e^{-p(t-s)} e^{-q s} ds, broadcasting over q and t
    q = np.asarray(q, float); t = np.asarray(t, float); d = p - q
    small = np.abs(d) * np.maximum(t, 1e-300) < 1e-8
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        v = (np.exp(-q * t) - np.exp(-p * t)) / d
    return np.where(small, t * np.exp(-p * t), v)


def ma_cov(times, H, theta, sigma, gammas, w, levels=30, n=8, rescale=True):
    """Covariance of the MA-fBm (type I) driven OU model at `times` (sorted ascending, > 0).
    rescale=True divides the noise by sqrt(c_I) so that the target is standard fBm (Var t^{2H});
    rescale=False reproduces the authors' code, whose target is the Mandelbrot-Van Ness scaling c_I t^{2H}."""
    times = np.asarray(times, float); g = np.asarray(gammas, float)
    s_eff = sigma / np.sqrt(c_mvn(H)) if rescale else sigma
    # past part: kernel  -g_k e^{g_k u} E(theta, g_k, t)  for u < 0
    Psi = np.stack([-(g * _E(theta, g, t)) for t in times])           # (m, K)
    past = Psi @ (np.outer(w, w) / (g[:, None] + g[None, :])) @ Psi.T
    # present part on a common grid graded toward every time point
    brk = np.concatenate([[0.0], times])
    us, ws = [], []
    for a0, a1 in zip(brk[:-1], brk[1:]):
        if a1 > a0:
            u, wu = graded_both(a0, a1, levels, n); us.append(u); ws.append(wu)
    u = np.concatenate(us); wu = np.concatenate(ws)
    KM = np.zeros((len(times), len(u)))
    for i, t in enumerate(times):
        msk = u <= t; d = t - u[msk]
        # kernel of int_0^t e^{-theta(t-s)} dY_k(s) at u in [0, t]: e^{-theta d} - g_k E(theta, g_k, d)
        kk = np.exp(-theta * d)[:, None] - g[None, :] * _E(theta, g[None, :], d[:, None])
        KM[i, msk] = kk @ w
    pres = (KM * wu) @ KM.T
    return s_eff ** 2 * (past + pres)


def log_normal(y, C):
    L = np.linalg.cholesky(C); a = np.linalg.solve(L, y)
    return float(-0.5 * a @ a - np.log(np.diag(L)).sum() - 0.5 * len(y) * np.log(2 * np.pi))


def gp_posterior(Kgo, Koo, Kgg_diag, y, s2):
    A = Koo + s2 * np.eye(len(y)); L = np.linalg.cholesky(A)
    V = np.linalg.solve(L, Kgo.T)
    mean = V.T @ np.linalg.solve(L, y)
    var = Kgg_diag - np.sum(V ** 2, axis=0)
    return mean, var
