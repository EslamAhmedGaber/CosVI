"""Reference data generation that uses no cosine coordinates: exact fBm on a fine grid (Davies-Harte circulant
embedding) and a Heun scheme for SDEs with additive or multiplicative noise (pathwise / Young sense, H > 1/2)."""
import numpy as np


def fgn(N, H, M, rng):
    """(M, N) fractional Gaussian noise with unit spacing (Var = 1), exact (Davies-Harte)."""
    k = np.arange(N + 1)
    r = 0.5 * (np.abs(k + 1) ** (2 * H) - 2 * k.astype(float) ** (2 * H) + np.abs(k - 1) ** (2 * H))
    c = np.concatenate([r, r[-2:0:-1]])
    lam = np.fft.fft(c).real
    if lam.min() < -1e-10:
        raise RuntimeError("circulant embedding not nonnegative")
    lam = np.clip(lam, 0, None)
    Z = rng.standard_normal((M, 2 * N)) + 1j * rng.standard_normal((M, 2 * N))
    return np.fft.fft(np.sqrt(lam / (2 * N)) * Z, axis=1).real[:, :N]


def fbm_increments(T, N, H, M, rng):
    return fgn(N, H, M, rng) * (T / N) ** H


def simulate(f, g, x0, T, N, H, rng, dB=None):
    """Heun scheme on N steps for dX = f(X) dt + g(X) dB_H; x0 (M,) or (M, d) with independent noises per dim.
    Returns X at all grid points, shape (N+1, *x0.shape)."""
    x = np.array(x0, float); M = x.shape[0]; h = T / N
    if dB is None:
        if x.ndim == 1:
            dB = fbm_increments(T, N, H, M, rng).T                       # (N, M)
        else:
            dB = np.stack([fbm_increments(T, N, H, M, rng).T for _ in range(x.shape[1])], -1)   # (N, M, d)
    X = np.empty((N + 1,) + x.shape); X[0] = x
    for n in range(N):
        k1 = f(x); g1 = g(x)
        xp = x + h * k1 + g1 * dB[n]
        x = x + 0.5 * h * (k1 + f(xp)) + 0.5 * (g1 + g(xp)) * dB[n]
        X[n + 1] = x
    return X
