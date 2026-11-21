"""Markov approximation of fBm by K Ornstein-Uhlenbeck processes driven by one Brownian motion (MA-fBm).

Re-implementation, for exact error computations only, of the construction of Daems et al. (ICLR 2024, Def. 3 and
Prop. 5), read in arXiv:2310.12975v1 on 13 Nov 2026:
    B_hat(t) = sum_k w_k (Y_k(t) - Y_k(0)),  dY_k = -g_k Y_k dt + dW,
    Type I  (standard fBm, Mandelbrot-Van Ness, prefactor 1/Gamma(H+1/2)): Y_k(0) stationary, Cov = 1/(g_i+g_j)
    Type II (Riemann-Liouville fBm):                                      Y_k(0) = 0
nodes g_k = r^(k-n), n=(K+1)/2 (geometric, symmetric about 1); weights w minimise the coupled L2 error
int_0^T E|B_hat(t) - B(t)|^2 dt (their Prop. 5).  Here w is obtained by a weighted least-squares fit of the
Wiener-integral kernels (numerically stable equivalent of their normal equations A w = b).

All errors are exact up to quadrature (no Monte Carlo).  The trained competitor itself is NOT this file: it is
the authors' public code, which is not redistributed here (see the G0 runs).
"""
import numpy as np
from scipy.special import gamma as Gamma
from .quad import graded, half_line, gl


def nodes(K, gamma_max):
    if K == 1:
        return np.array([1.0])
    k = np.arange(1, K + 1)
    return gamma_max ** ((2 * k - K - 1) / (K - 1))


def c_mvn(H):
    """Var B(1) for the Mandelbrot-Van Ness integral with prefactor 1/Gamma(H+1/2)."""
    return 1.0 / (Gamma(2 * H + 1) * np.sin(np.pi * H))


class MAFit:
    """Least-squares MA-fBm fit on [0, T] for type 'I' or 'II'; stores w and the exact errors."""

    def __init__(self, H, T, gammas, typ='I', nt=96, levels=40, n=16):
        self.H, self.T, self.g, self.typ = H, T, np.asarray(gammas, float), typ
        self.a = H - 0.5
        self.c = 1.0 / Gamma(H + 0.5)
        tt, vt = graded(0.0, T, 14, nt // 14 + 1, 'a')              # t-nodes for the integrated criterion
        self.t, self.vt = tt, vt
        rows, rhs, owner, wts = [], [], [], []
        tails = np.zeros(len(tt))
        if typ == 'I':
            xs, wx, X = half_line(levels, 44, n)
        for i, t in enumerate(tt):
            u, wu = graded(0.0, t, levels, n, 'a')                    # present part, u = t - s
            rows.append(np.exp(-np.outer(u, self.g))); rhs.append(self.c * u ** self.a)
            wts.append(wu); owner.append(np.full(len(u), i))
            if typ == 'I':                                          # past part, s = -x < 0
                rows.append(np.exp(-np.outer(xs, self.g)) * (np.exp(-self.g * t) - 1.0)[None, :])
                rhs.append(self.c * ((t + xs) ** self.a - xs ** self.a))
                wts.append(wx); owner.append(np.full(len(xs), i))
                tails[i] = (self.c * self.a * t) ** 2 * X ** (2 * self.a - 1) / (1 - 2 * self.a)
        Phi = np.vstack(rows); y = np.concatenate(rhs); w = np.concatenate(wts); own = np.concatenate(owner)
        sw = np.sqrt(w * vt[own])
        self.w, *_ = np.linalg.lstsq(Phi * sw[:, None], y * sw, rcond=None)
        r2 = (y - Phi @ self.w) ** 2 * w
        self.e_t = np.bincount(own, r2, minlength=len(tt)) + tails   # E|B(t)-B_hat(t)|^2 at the t-nodes
        self.var_t = self._var_target(tt)
        self.int_err = float(vt @ self.e_t)
        self.int_var = float(vt @ self.var_t)

    def _var_target(self, t):
        if self.typ == 'I':
            return c_mvn(self.H) * t ** (2 * self.H)
        return self.c ** 2 * t ** (2 * self.H) / (2 * self.H)

    @property
    def rel_int_err(self):
        return self.int_err / self.int_var

    @property
    def sup_rel_err(self):
        return float(np.max(self.e_t) / self._var_target(np.array([self.T]))[0])

    # ---- law (covariance) of the type I approximation, closed form ----
    def _EYY(self, a, b):
        gi = self.g[:, None]; gj = self.g[None, :]
        m = np.minimum(a, b)
        return np.exp(-gi * (a - m) - gj * (b - m)) / (gi + gj)

    def cov(self, s, t):
        """Cov(B_hat(s), B_hat(t)) for type I (stationary start)."""
        assert self.typ == 'I'
        out = np.empty((len(s), len(t)))
        for i, a in enumerate(s):
            for j, b in enumerate(t):
                C = self._EYY(a, b) - self._EYY(a, 0.0) - self._EYY(0.0, b) + self._EYY(0.0, 0.0)
                out[i, j] = self.w @ C @ self.w
        return out

    # ---- linear functionals int_0^T f dB  (type I), strong error and variance ----
    def functional(self, f, fp, nq=48, levels=40, n=16):
        """f, fp: callables (f and f').  Returns (Var true [MVN scale], Var approx, coupled error)."""
        H, T, a, c, g = self.H, self.T, self.a, self.c, self.g
        tq, wq = gl(0.0, T, 400)
        F = (wq * f(tq)) @ np.exp(-np.outer(tq, g))                  # F_k = int_0^T f e^{-g t}
        # past u = -x
        xs, wx, X = half_line(levels, 44, n)
        # kappa(-x) = c * a * int_0^T f(t)(t+x)^(a-1) dt = c*[f(T)(T+x)^a - f(0)x^a - int f'(t)(t+x)^a dt]
        kap_p = c * (f(T) * (T + xs) ** a - f(0.0) * xs ** a - ((wq * fp(tq))[None, :] * (tq[None, :] + xs[:, None]) ** a).sum(1))
        psi_p = -(np.exp(-np.outer(xs, g)) * (g * F)[None, :]) @ self.w
        # present u in [0, T]
        uq, wu = graded(0.0, T, levels, n, 'b')
        kap_n = np.empty(len(uq)); psi_n = np.empty(len(uq))
        from .quad import gj_left
        for m, u in enumerate(uq):
            if T - u > 0:
                ts, ws = gj_left(u, T, a - 1.0, nq)
                kap_n[m] = c * a * (ws @ f(ts))
                t2, w2 = gl(u, T, 64)
                inner = (w2 * f(t2)) @ np.exp(-np.outer(t2 - u, g))
            else:
                kap_n[m] = 0.0; inner = np.zeros(len(g))
            psi_n[m] = self.w @ (f(u) - g * inner)
        intf = float(wq @ f(tq))
        tail = (c * a * intf) ** 2 * X ** (2 * a - 1) / (1 - 2 * a)      # x > X, where psi is negligible
        var_true = wx @ kap_p ** 2 + wu @ kap_n ** 2 + tail
        var_hat = wx @ psi_p ** 2 + wu @ psi_n ** 2
        err = wx @ (kap_p - psi_p) ** 2 + wu @ (kap_n - psi_n) ** 2 + tail
        return float(var_true), float(var_hat), float(err)
