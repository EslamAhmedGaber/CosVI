"""Structured and joint Gaussian variational families for CosVI.

Extends ``LocalsD`` (product full Gaussian ``q(z) q(x0)``) with two key families:
  1. ``LocalsJointFull``: joint full-covariance Gaussian ``q(z, x0)`` of dimension ``D + 1``,
     eliminating the factorisation gap of Corollary A.1 (1.2-1.8 nats in the linear analogue)
     during full training at negligible extra cost.
  2. ``LocalsBlockJoint``: two-scale structured joint Gaussian
         q(xi, eta, x0) = N((xi, x0); mu_c, U_c U_c^T) * prod_{r=1}^{D-J} N(eta_r; mu_r, d_r^2),
     placing full covariance on the J cosine coordinates jointly with x0 (which carry
     96.6-99.8% of the observation sensitivity) and an independent diagonal Gaussian on the
     N-1 high-frequency residual coordinates. Parameter and sampling complexity drops from
     O((J + N)^2) to O(J^2 + N) — linear in the grid length N.
  3. ``Locals2BlockJoint``: multi-channel (d = 2) extension placing a joint full Gaussian over
     (xi_1, ..., xi_d, x0) of dimension d*J + d and diagonal Gaussian over the d*(N-1) residuals.

All families share the exact prior z ~ N(0, I_D), x0 ~ N(0, I_d) with closed-form KL,
exact reparameterised sampling, and exact log_q for importance-weighted evaluation.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG2PI = float(math.log(2.0 * math.pi))


class LocalsJointFull(nn.Module):
    """Joint full-covariance Gaussian q(z, x0) of dimension D + 1 per group and series."""

    def __init__(self, G, M, D, init_x0):
        super().__init__()
        self.D = int(D)
        Dp = self.D + 1
        self.mu = nn.Parameter(torch.zeros(G, M, Dp))
        self.rho = nn.Parameter(torch.full((G, M, Dp), 0.5413))
        self.off = nn.Parameter(torch.zeros(G, M, Dp, Dp))
        with torch.no_grad():
            self.mu[:, :, self.D].copy_(init_x0.clone().expand(G, M))
            self.rho[:, :, self.D].fill_(-2.0)

    def num_local_params_per_series(self):
        Dp = self.D + 1
        return Dp + Dp * (Dp + 1) // 2

    def scale(self):
        d = F.softplus(self.rho)
        return torch.tril(self.off, -1) + torch.diag_embed(d), d

    def sample(self, S, common=True):
        L, d = self.scale()
        G, M, Dp = self.mu.shape
        eps = torch.randn((S, 1 if common else G, M, Dp), device=self.mu.device, dtype=self.mu.dtype).expand(S, G, M, Dp)
        w = self.mu + torch.einsum("gmij,sgmj->sgmi", L, eps)
        z = w[..., :self.D]
        x0 = w[..., self.D]
        kl = 0.5 * ((L ** 2).sum((-1, -2)) + (self.mu ** 2).sum(-1) - Dp - 2.0 * torch.log(d).sum(-1))
        return z, x0, kl, eps[..., :self.D], eps[..., self.D]

    def log_q(self, z, x0):
        L, d = self.scale()
        w = torch.cat([z, x0.unsqueeze(-1)], dim=-1)
        r = (w - self.mu).unsqueeze(-1)
        sol = torch.linalg.solve_triangular(L.unsqueeze(0).expand(w.shape[0], *L.shape), r, upper=False)[..., 0]
        return -0.5 * (sol ** 2).sum(-1) - torch.log(d).sum(-1) - 0.5 * (self.D + 1) * LOG2PI


class LocalsBlockJoint(nn.Module):
    """Two-scale O(J^2 + N) joint Gaussian: full on (xi, x0) in R^{J+1}, diagonal on residual eta in R^{D-J}."""

    def __init__(self, G, M, J, D, init_x0):
        super().__init__()
        self.J = int(J)
        self.D = int(D)
        self.R = self.D - self.J
        assert self.R >= 0
        Jc = self.J + 1
        self.mu_c = nn.Parameter(torch.zeros(G, M, Jc))
        self.rho_c = nn.Parameter(torch.full((G, M, Jc), 0.5413))
        self.off_c = nn.Parameter(torch.zeros(G, M, Jc, Jc))
        with torch.no_grad():
            self.mu_c[:, :, self.J].copy_(init_x0.clone().expand(G, M))
            self.rho_c[:, :, self.J].fill_(-2.0)
        self.mu_r = nn.Parameter(torch.zeros(G, M, self.R))
        self.rho_r = nn.Parameter(torch.full((G, M, self.R), 0.5413))

    def num_local_params_per_series(self):
        Jc = self.J + 1
        return (Jc + Jc * (Jc + 1) // 2) + 2 * self.R

    def scale_c(self):
        dc = F.softplus(self.rho_c)
        return torch.tril(self.off_c, -1) + torch.diag_embed(dc), dc

    def sample(self, S, common=True):
        Lc, dc = self.scale_c()
        dr = F.softplus(self.rho_r)
        G, M, Jc = self.mu_c.shape
        dev, dt = self.mu_c.device, self.mu_c.dtype
        ec = torch.randn((S, 1 if common else G, M, Jc), device=dev, dtype=dt).expand(S, G, M, Jc)
        wc = self.mu_c + torch.einsum("gmij,sgmj->sgmi", Lc, ec)
        xi = wc[..., :self.J]
        x0 = wc[..., self.J]
        kl_c = 0.5 * ((Lc ** 2).sum((-1, -2)) + (self.mu_c ** 2).sum(-1) - Jc - 2.0 * torch.log(dc).sum(-1))
        if self.R > 0:
            er = torch.randn((S, 1 if common else G, M, self.R), device=dev, dtype=dt).expand(S, G, M, self.R)
            eta = self.mu_r + dr * er
            z = torch.cat([xi, eta], dim=-1)
            kl_r = 0.5 * ((dr ** 2).sum(-1) + (self.mu_r ** 2).sum(-1) - self.R - 2.0 * torch.log(dr).sum(-1))
            eps_z = torch.cat([ec[..., :self.J], er], dim=-1)
        else:
            z = xi
            kl_r = 0.0
            eps_z = ec[..., :self.J]
        return z, x0, kl_c + kl_r, eps_z, ec[..., self.J]

    def log_q(self, z, x0):
        Lc, dc = self.scale_c()
        wc = torch.cat([z[..., :self.J], x0.unsqueeze(-1)], dim=-1)
        rc = (wc - self.mu_c).unsqueeze(-1)
        sol = torch.linalg.solve_triangular(Lc.unsqueeze(0).expand(wc.shape[0], *Lc.shape), rc, upper=False)[..., 0]
        lq_c = -0.5 * (sol ** 2).sum(-1) - torch.log(dc).sum(-1) - 0.5 * (self.J + 1) * LOG2PI
        if self.R > 0:
            dr = F.softplus(self.rho_r)
            eta = z[..., self.J:]
            lq_r = -0.5 * (((eta - self.mu_r) / dr) ** 2).sum(-1) - torch.log(dr).sum(-1) - 0.5 * self.R * LOG2PI
            return lq_c + lq_r
        return lq_c


class Locals2BlockJoint(nn.Module):
    """2D two-scale O(J^2 + N) joint Gaussian: full on (xi_1, xi_2, x0_1, x0_2) in R^{2J+2},
    diagonal on (eta_1, eta_2) in R^{2(D-J)}. Reduces 2D local params from 18,339 to 915 per series."""

    def __init__(self, G, M, J, D, init_x0):
        super().__init__()
        self.J = int(J)
        self.D = int(D)
        self.R = self.D - self.J
        Jc = 2 * self.J + 2
        self.mu_c = nn.Parameter(torch.zeros(G, M, Jc))
        self.rho_c = nn.Parameter(torch.full((G, M, Jc), 0.5413))
        self.off_c = nn.Parameter(torch.zeros(G, M, Jc, Jc))
        with torch.no_grad():
            self.mu_c[:, :, 2 * self.J:].copy_(init_x0.clone())
            self.rho_c[:, :, 2 * self.J:].fill_(-2.0)
        self.mu_r = nn.Parameter(torch.zeros(G, M, 2 * self.R))
        self.rho_r = nn.Parameter(torch.full((G, M, 2 * self.R), 0.5413))

    def num_local_params_per_series(self):
        Jc = 2 * self.J + 2
        return (Jc + Jc * (Jc + 1) // 2) + 4 * self.R

    def scale_c(self):
        dc = F.softplus(self.rho_c)
        return torch.tril(self.off_c, -1) + torch.diag_embed(dc), dc

    def sample(self, S):
        Lc, dc = self.scale_c()
        dr = F.softplus(self.rho_r)
        G, M, Jc = self.mu_c.shape
        dev, dt = self.mu_c.device, self.mu_c.dtype
        ec = torch.randn((S, G, M, Jc), device=dev, dtype=dt)
        wc = self.mu_c + torch.einsum("gmij,sgmj->sgmi", Lc, ec)
        xi1 = wc[..., :self.J]
        xi2 = wc[..., self.J:2 * self.J]
        x0 = wc[..., 2 * self.J:]
        kl_c = 0.5 * ((Lc ** 2).sum((-1, -2)) + (self.mu_c ** 2).sum(-1) - Jc - 2.0 * torch.log(dc).sum(-1))
        er = torch.randn((S, G, M, 2 * self.R), device=dev, dtype=dt)
        eta = self.mu_r + dr * er
        eta1 = eta[..., :self.R]
        eta2 = eta[..., self.R:]
        z = torch.cat([xi1, eta1, xi2, eta2], dim=-1)
        kl_r = 0.5 * ((dr ** 2).sum(-1) + (self.mu_r ** 2).sum(-1) - 2 * self.R - 2.0 * torch.log(dr).sum(-1))
        return z, x0, kl_c + kl_r

    def log_q(self, z, x0):
        Lc, dc = self.scale_c()
        xi1 = z[..., :self.J]
        eta1 = z[..., self.J:self.D]
        xi2 = z[..., self.D:self.D + self.J]
        eta2 = z[..., self.D + self.J:]
        wc = torch.cat([xi1, xi2, x0], dim=-1)
        rc = (wc - self.mu_c).unsqueeze(-1)
        sol = torch.linalg.solve_triangular(Lc.unsqueeze(0).expand(wc.shape[0], *Lc.shape), rc, upper=False)[..., 0]
        lq_c = -0.5 * (sol ** 2).sum(-1) - torch.log(dc).sum(-1) - 0.5 * (2 * self.J + 2) * LOG2PI
        dr = F.softplus(self.rho_r)
        eta = torch.cat([eta1, eta2], dim=-1)
        lq_r = -0.5 * (((eta - self.mu_r) / dr) ** 2).sum(-1) - torch.log(dr).sum(-1) - self.R * LOG2PI
        return lq_c + lq_r


class LocalsPrecLR(nn.Module):
    """Precision-low-rank (Woodbury) joint Gaussian q(z, x0) with precision
        Lambda = diag(d^{-2}) + V V^T,   V in R^{(D+1) x r}.

    Exact for any linear-Gaussian model with m observations whenever r >= m (since the exact
    posterior precision in whitened coordinates is I + F^T R^{-1} F of rank <= m above I),
    while requiring only O(r D) = O(r N) local parameters and O(r^2 D + r^3) per step.

    CUDA-graph-capturable exact sampling via Matheron's rule (no eigendecomposition):
      Let W = diag(d) V,  M = I_r + W^T W = L_M L_M^T  (r x r Cholesky via cholesky_ex).
      Draw eps1 ~ N(0, I_{D+1}) and eps2 ~ N(0, I_r) independently; then
          u = diag(d) * (eps1 - W M^{-1} (W^T eps1 + eps2))
      has exact covariance diag(d) (I - W M^{-1} W^T) diag(d) = (diag(d^{-2}) + V V^T)^{-1}.
      Closed-form KL:
          log det(Sigma) = 2 sum(log d) - 2 sum(log diag(L_M)),
          tr(Sigma)      = sum(d^2) - || L_M^{-1} (W^T diag(d)) ||_F^2,
      and log q(w) uses Lambda = diag(d^{-2}) + V V^T directly with zero matrix inversions.
    """

    def __init__(self, G, M, D, init_x0, rank=16):
        super().__init__()
        self.D = int(D)
        self.rank = int(rank)
        Dp = self.D + 1
        self.mu = nn.Parameter(torch.zeros(G, M, Dp))
        self.rho = nn.Parameter(torch.full((G, M, Dp), 0.5413))
        self.V = nn.Parameter(0.01 * torch.randn(G, M, Dp, self.rank))
        with torch.no_grad():
            self.mu[:, :, self.D].copy_(init_x0.clone().expand(G, M))
            self.rho[:, :, self.D].fill_(-2.0)

    def num_local_params_per_series(self):
        Dp = self.D + 1
        return 2 * Dp + Dp * self.rank

    def _cholesky_M(self):
        d = F.softplus(self.rho)                                # (G, M, Dp)
        W = d.unsqueeze(-1) * self.V                            # (G, M, Dp, r)
        eye_r = torch.eye(self.rank, device=d.device, dtype=d.dtype)
        M_mat = eye_r + torch.matmul(W.transpose(-1, -2), W)    # (G, M, r, r)
        L_M, _ = torch.linalg.cholesky_ex(M_mat)                # 100% CUDA-graph capturable
        return d, W, L_M

    def sample(self, S, common=True):
        d, W, L_M = self._cholesky_M()
        G, M, Dp = self.mu.shape
        dev, dt = self.mu.device, self.mu.dtype
        eps1 = torch.randn((S, 1 if common else G, M, Dp), device=dev, dtype=dt).expand(S, G, M, Dp)
        eps2 = torch.randn((S, 1 if common else G, M, self.rank), device=dev, dtype=dt).expand(S, G, M, self.rank)
        # rhs = W^T eps1 + eps2: (S, G, M, r) -> permute to (G, M, r, S) for batched triangular solve
        rhs = (torch.einsum("gmdr,sgmd->sgmr", W, eps1) + eps2).permute(1, 2, 3, 0)
        y = torch.linalg.solve_triangular(L_M, rhs, upper=False)
        sol = torch.linalg.solve_triangular(L_M.transpose(-1, -2), y, upper=True).permute(3, 0, 1, 2)  # (S, G, M, r)
        corr = torch.einsum("gmdr,sgmr->sgmd", W, sol)
        w = self.mu + d.unsqueeze(0) * (eps1 - corr)
        z = w[..., :self.D]
        x0 = w[..., self.D]
        # Exact KL(q || N(0, I_{D+1}))
        dW_t = (d.unsqueeze(-1) * W).transpose(-1, -2)          # (G, M, r, Dp)
        Z = torch.linalg.solve_triangular(L_M, dW_t, upper=False)
        tr = (d ** 2).sum(-1) - (Z ** 2).sum((-1, -2))
        logdet = 2.0 * torch.log(d).sum(-1) - 2.0 * torch.log(torch.diagonal(L_M, dim1=-2, dim2=-1)).sum(-1)
        kl = 0.5 * (tr + (self.mu ** 2).sum(-1) - Dp - logdet)
        return z, x0, kl, eps1[..., :self.D], eps1[..., self.D]

    def log_q(self, z, x0):
        d, _, L_M = self._cholesky_M()
        w = torch.cat([z, x0.unsqueeze(-1)], dim=-1)
        diff = w - self.mu                                      # (S, G, M, Dp)
        quad_d = ((diff / d) ** 2).sum(-1)
        Vt_diff = torch.einsum("gmdr,sgmd->sgmr", self.V, diff)
        quad_v = (Vt_diff ** 2).sum(-1)
        logdet = 2.0 * torch.log(d).sum(-1) - 2.0 * torch.log(torch.diagonal(L_M, dim1=-2, dim2=-1)).sum(-1)
        return -0.5 * (quad_d + quad_v + logdet + (self.D + 1) * LOG2PI)

