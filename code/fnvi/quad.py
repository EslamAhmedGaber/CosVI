"""Quadrature helpers used by every exact (deterministic) computation in Paper 4.

All rules are Gauss-Legendre or Gauss-Jacobi; graded rules cluster nodes geometrically toward an endpoint where
the integrand has a power-type singularity (t^{2H}, (t-s)^{H-1/2}, ...).
"""
import numpy as np
from scipy.special import roots_legendre, roots_jacobi

_cache = {}


def gl(a, b, n):
    """n-point Gauss-Legendre nodes and weights on [a, b]."""
    key = ('gl', n)
    if key not in _cache:
        _cache[key] = roots_legendre(n)
    x, w = _cache[key]
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def gj_left(a, b, beta, n):
    """Nodes/weights for int_a^b (x-a)^beta g(x) dx  (weight included in the weights)."""
    key = ('gj', n, round(beta, 14))
    if key not in _cache:
        _cache[key] = roots_jacobi(n, 0.0, beta)
    x, w = _cache[key]
    h = 0.5 * (b - a)
    return a + h * (x + 1.0), w * h ** (beta + 1.0)


def graded(a, b, levels=40, n=16, toward='a'):
    """Composite Gauss-Legendre on geometrically graded panels clustering toward endpoint a (or b)."""
    L = b - a
    if L <= 0:
        return np.zeros(0), np.zeros(0)
    fr = [2.0 ** (-m) for m in range(levels, 0, -1)] + [1.0]
    edges = [0.0] + fr
    xs, ws = [], []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        x, w = gl(e0, e1, n)
        xs.append(x); ws.append(w)
    x = np.concatenate(xs); w = np.concatenate(ws)
    if toward == 'a':
        return a + L * x, L * w
    return b - L * x, L * w


def graded_both(a, b, levels=40, n=16):
    """Graded toward both endpoints (split at the midpoint)."""
    m = 0.5 * (a + b)
    x1, w1 = graded(a, m, levels, n, 'a')
    x2, w2 = graded(m, b, levels, n, 'b')
    return np.concatenate([x1, x2]), np.concatenate([w1, w2])


def composite(a, b, npan, n=12, levels=24):
    """npan uniform Gauss-Legendre panels on [a, b]; the first and last panels are graded toward a and b.
    Use npan >= (highest angular frequency) * (b - a) / 2 for oscillatory integrands."""
    npan = max(1, int(npan))
    if npan == 1:
        return graded_both(a, b, levels, n)
    e = np.linspace(a, b, npan + 1)
    xs, ws = [], []
    x, w = graded(e[0], e[1], levels, n, 'a'); xs.append(x); ws.append(w)
    for e0, e1 in zip(e[1:-2], e[2:-1]):
        x, w = gl(e0, e1, n); xs.append(x); ws.append(w)
    x, w = graded(e[-2], e[-1], levels, n, 'b'); xs.append(x); ws.append(w)
    return np.concatenate(xs), np.concatenate(ws)


def half_line(levels_in=40, levels_out=44, n=16):
    """Nodes/weights on (0, 2**levels_out): graded toward 0 on (0,1], doubling panels on [1, 2**levels_out]."""
    x0, w0 = graded(0.0, 1.0, levels_in, n, 'a')
    xs, ws = [x0], [w0]
    for m in range(levels_out):
        x, w = gl(2.0 ** m, 2.0 ** (m + 1), n)
        xs.append(x); ws.append(w)
    return np.concatenate(xs), np.concatenate(ws), 2.0 ** levels_out
