"""Cosine-coordinate variational inference (CVI) for neural SDEs driven by fractional Brownian motion, H >= 1/2.

    dX = f_theta(X, t) dt + g_theta(X, t) dB_H(t)   on [0, T]
    B_H  ~  B^J(t) = A(t) xi,  xi ~ N(0, I_J)  (whitened cosine coordinates, Paper 1)

so that X is the solution of a random ODE  dX/dt = f + g * (a(t) @ xi)  with a = dA/dt.  For H > 1/2 the smooth
approximation converges to the pathwise (Young) solution; no correction term (see 02_THEORY.md, Sec. T4).

Variational posterior: q(xi) Gaussian (mean-field, full, or low-rank + diagonal), closed-form KL to N(0, I).
ELBO_J = E_q[log p(y | X(xi))] - KL(q || N(0, I))  <= log p_J(y)  (evidence of the truncated model).
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .coords import CosineCoords


# ------------------------------------------------------------------ noise tables (numpy -> torch), H-differentiable
def rate_table_np(H, T, J, n_steps, mode="projection"):
    """a(t) at the RK4 half-step times t_k = k*h/2, k = 0..2N; shape (2N+1, J)."""
    tk = np.linspace(0.0, T, 2 * n_steps + 1)
    if abs(H - 0.5) < 1e-12:                  # Brownian motion: cosine Karhunen-type basis, G = I
        from .vendor import hfc
        return hfc.h("cos", J, T, tk)
    return CosineCoords(H, T, J).rate_funcs(tk, mode)


class _RateTable(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, T, J, n_steps, mode, dH, device, dtype):
        h = float(H)
        a = rate_table_np(h, T, J, n_steps, mode)
        lo, hi = max(0.5 + 1e-6, h - dH), min(0.999, h + dH)
        da = (rate_table_np(hi, T, J, n_steps, mode) - rate_table_np(lo, T, J, n_steps, mode)) / (hi - lo)
        ctx.save_for_backward(torch.as_tensor(da, device=device, dtype=dtype))
        return torch.as_tensor(a, device=device, dtype=dtype)

    @staticmethod
    def backward(ctx, g):
        (da,) = ctx.saved_tensors
        return (g * da).sum().reshape(1), None, None, None, None, None, None, None


def rate_table(H, T, J, n_steps, mode="projection", dH=1e-4, device="cpu", dtype=None):
    """Differentiable in H (tensor of shape (1,)) through a central finite difference of the closed-form table."""
    dtype = dtype or torch.get_default_dtype()
    if not torch.is_tensor(H):
        return torch.as_tensor(rate_table_np(H, T, J, n_steps, mode), device=device, dtype=dtype)
    return _RateTable.apply(H, T, J, n_steps, mode, dH, device, dtype)


# ------------------------------------------------------------------ Gaussian variational families over xi
class GaussianFamily(nn.Module):
    """Batched Gaussian q(xi) for `nb` independent blocks (e.g. trajectories x state dims), each of dimension J."""

    def __init__(self, nb, J, kind="full", rank=4, init_scale=1.0):
        super().__init__()
        self.nb, self.J, self.kind, self.rank = nb, J, kind, rank
        self.mu = nn.Parameter(torch.zeros(nb, J))
        s0 = math.log(math.expm1(init_scale))
        if kind == "meanfield":
            self.rho = nn.Parameter(torch.full((nb, J), s0))
        elif kind == "full":
            self.rho = nn.Parameter(torch.full((nb, J), s0))
            self.off = nn.Parameter(torch.zeros(nb, J, J))
        elif kind == "lowrank":
            self.rho = nn.Parameter(torch.full((nb, J), s0))
            self.V = nn.Parameter(1e-3 * torch.randn(nb, J, rank))
        else:
            raise ValueError(kind)

    def scale_tril(self):
        d = F.softplus(self.rho)
        if self.kind == "meanfield":
            return torch.diag_embed(d)
        if self.kind == "full":
            return torch.tril(self.off, -1) + torch.diag_embed(d)
        S = torch.diag_embed(d ** 2) + self.V @ self.V.transpose(-1, -2)
        return torch.linalg.cholesky(S)

    def rsample(self, S):
        """S samples per block -> (S, nb, J)."""
        eps = torch.randn(S, self.nb, self.J, device=self.mu.device, dtype=self.mu.dtype)
        d = F.softplus(self.rho)
        if self.kind == "meanfield":
            return self.mu + d * eps
        if self.kind == "full":
            L = self.scale_tril()
            return self.mu + torch.einsum("bij,sbj->sbi", L, eps)
        eps2 = torch.randn(S, self.nb, self.rank, device=self.mu.device, dtype=self.mu.dtype)
        return self.mu + d * eps + torch.einsum("bjr,sbr->sbj", self.V, eps2)

    def kl(self):
        """KL(q || N(0, I)) summed over blocks (closed form)."""
        d = F.softplus(self.rho)
        quad = (self.mu ** 2).sum()
        if self.kind == "meanfield":
            return 0.5 * ((d ** 2).sum() + quad - self.nb * self.J - 2 * torch.log(d).sum())
        if self.kind == "full":
            L = self.scale_tril()
            return 0.5 * ((L ** 2).sum() + quad - self.nb * self.J - 2 * torch.log(d).sum())
        # low rank + diagonal: log det via the determinant lemma
        D2 = d ** 2
        tr = D2.sum() + (self.V ** 2).sum()
        M = torch.eye(self.rank, device=d.device, dtype=d.dtype) + self.V.transpose(-1, -2) @ (self.V / D2[..., None])
        logdet = torch.log(D2).sum() + torch.logdet(M).sum()
        return 0.5 * (tr + quad - self.nb * self.J - logdet)


# ------------------------------------------------------------------ fixed-step solvers for the random ODE
def solve_rk4(f, g, x0, R, T, n_steps, save_idx=None):
    """dx/dt = f(x, t) + g(x, t) * r(t);  R: noise rates at half-step times, shape (2N+1, *x0.shape).
    Returns states at the step times (N+1, *x0.shape) or at save_idx."""
    h = T / n_steps
    x = x0
    xs = [x0]
    for n in range(n_steps):
        t = n * h
        r0, r1, r2 = R[2 * n], R[2 * n + 1], R[2 * n + 2]
        k1 = f(x, t) + g(x, t) * r0
        k2 = f(x + 0.5 * h * k1, t + 0.5 * h) + g(x + 0.5 * h * k1, t + 0.5 * h) * r1
        k3 = f(x + 0.5 * h * k2, t + 0.5 * h) + g(x + 0.5 * h * k2, t + 0.5 * h) * r1
        k4 = f(x + h * k3, t + h) + g(x + h * k3, t + h) * r2
        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        xs.append(x)
    X = torch.stack(xs, 0)
    return X if save_idx is None else X[save_idx]


def solve_heun(f, g, x0, R, T, n_steps, save_idx=None):
    h = T / n_steps
    x = x0; xs = [x0]
    for n in range(n_steps):
        t = n * h
        k1 = f(x, t) + g(x, t) * R[2 * n]
        xp = x + h * k1
        k2 = f(xp, t + h) + g(xp, t + h) * R[2 * n + 2]
        x = x + 0.5 * h * (k1 + k2)
        xs.append(x)
    X = torch.stack(xs, 0)
    return X if save_idx is None else X[save_idx]


def noise_rates(table, xi):
    """table (2N+1, J), xi (S, nb, J) -> (2N+1, S, nb)."""
    return torch.einsum("kj,sbj->ksb", table, xi)


# ------------------------------------------------------------------ small networks
class MLP(nn.Module):
    def __init__(self, din, dout, width=64, depth=2, act=nn.Tanh):
        super().__init__()
        layers, d = [], din
        for _ in range(depth):
            layers += [nn.Linear(d, width), act()]
            d = width
        layers.append(nn.Linear(d, dout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
