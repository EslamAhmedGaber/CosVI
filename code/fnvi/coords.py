"""Whitened cosine coordinates of standard fractional Brownian motion on [0, T], 1/2 <= H < 1.

CosChaos objects (Gram matrix, Cholesky factor, nested coordinates) computed with the vendored solver file
`vendor/hfc.py` (SHA-256 75dd3c90...d39274, identical to the CosChaos v1.1.3 snapshot).

    Z_k = int_0^T h_k dB_H,  G = E[Z Z^T] = L L^T,  xi = L^{-1} Z ~ N(0, I_J)   (nested in J)

Two truncations of the noise, both functions of xi only:
    'classical'  : B^J(t) = sum_{k<J} m_k(t) Z_k = (L^T m(t))^T xi          (CosChaos, noise series)
    'projection' : B^J(t) = E[B_H(t) | Z_0..Z_{J-1}] = (L^{-1} Q(t))^T xi     (best approximation in V_J)
The projection residual B_H - B^J is independent of xi (jointly Gaussian and uncorrelated); the classical
residual is not.  Rates (time derivatives) are  L^T h(t)  and  L^{-1} Q'(t).
"""
import numpy as np
from scipy.linalg import solve_triangular
from .vendor import hfc as _vendor_hfc
from . import fast_hfc as _fast_hfc


class CosineCoords:
    def __init__(self, H, T, J, fast=True):
        self.H, self.T, self.J = float(H), float(T), int(J)
        self._hfc = _fast_hfc if fast else _vendor_hfc
        self.G = self._hfc.gram(self.H, self.T, 'cos', self.J)
        self.L = np.linalg.cholesky(self.G)

    # B^J(t) = A(t) @ xi ; returns (len(t), J)
    def path_funcs(self, t, mode='projection'):
        t = np.atleast_1d(np.asarray(t, float))
        if mode == 'projection':
            Qt = self._hfc.Q(self.H, self.T, 'cos', self.J, t)
            return solve_triangular(self.L, Qt.T, lower=True).T
        if mode == 'classical':
            return self._hfc.Mcoef('cos', self.J, self.T, t) @ self.L
        raise ValueError(mode)

    # dB^J/dt = a(t) @ xi ; returns (len(t), J)
    def rate_funcs(self, t, mode='projection'):
        t = np.atleast_1d(np.asarray(t, float))
        if mode == 'projection':
            dQ = self._hfc.dQ(self.H, self.T, 'cos', self.J, t)
            return solve_triangular(self.L, dQ.T, lower=True).T
        if mode == 'classical':
            return self._hfc.h('cos', self.J, self.T, t) @ self.L
        raise ValueError(mode)

    def fbm_cov(self, s, t):
        s = np.asarray(s, float)[:, None]; t = np.asarray(t, float)[None, :]
        H2 = 2 * self.H
        return 0.5 * (s ** H2 + t ** H2 - np.abs(s - t) ** H2)

    # E|B_H(t) - B^J(t)|^2 (exact, closed form)
    def resid_var(self, t, mode='projection'):
        t = np.atleast_1d(np.asarray(t, float))
        A = self.path_funcs(t, mode)
        if mode == 'projection':
            return t ** (2 * self.H) - np.sum(A ** 2, axis=1)
        Qt = self._hfc.Q(self.H, self.T, 'cos', self.J, t)
        Mt = self._hfc.Mcoef('cos', self.J, self.T, t)
        return t ** (2 * self.H) - 2 * np.sum(Mt * Qt, axis=1) + np.sum(A ** 2, axis=1)

    # Cov(B^J(s), B^J(t)) - Cov(B_H(s), B_H(t))  (law error of the truncated noise)
    def cov_error(self, s, t, mode='projection'):
        As = self.path_funcs(s, mode); At = self.path_funcs(t, mode)
        return As @ At.T - self.fbm_cov(s, t)

    # coordinates of a linear functional int f dB_H:  c_k = <f, h_k>_H = int f Q'_k  (needs quadrature nodes)
    def functional_coords(self, f_vals, nodes, weights):
        dQ = self._hfc.dQ(self.H, self.T, 'cos', self.J, nodes)          # (n, J)
        a = self._hfc.h('cos', self.J, self.T, nodes)                     # (n, J)
        c = (weights * f_vals) @ dQ                                 # <f, h_k>_H
        aL2 = (weights * f_vals) @ a                                # <f, h_k>_{L2}
        return c, aL2
