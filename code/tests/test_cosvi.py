"""Comprehensive unit test suite for CosVI (mathematics, fast_hfc, exact maps, and variational families)."""
import math
import os
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fnvi import fast_hfc
from fnvi.vendor import hfc as slow_hfc
from fnvi.coords import CosineCoords
from fnvi.noise_maps import twoscale_map, twoscale_exact_map, twoscale_pl_map, grid_map, path_cov_on_grid
from fnvi.families import LocalsJointFull, LocalsBlockJoint, Locals2BlockJoint, LocalsPrecLR


@pytest.mark.parametrize("H", [0.5, 0.55, 0.6, 0.75, 0.9, 0.95])
def test_fast_hfc_matches_mpmath_and_reference(H):
    import mpmath as mp
    mp.mp.dps = 30
    xs = np.array([0.0, 1e-3, 1.5, 10.0, 55.5, 56.5, 120.0, 600.0])
    m_ref = np.array([1.0 + 0j if abs(H - 0.5) < 1e-15 else complex(mp.hyp1f1(2 * H - 1, 2 * H, 1j * x)) for x in xs])
    n_ref = np.array([complex(mp.hyp1f1(2 * H, 2 * H + 1, 1j * x)) for x in xs])
    assert np.max(np.abs(fast_hfc.N_H(H, xs) - n_ref)) < 1e-13
    assert np.max(np.abs(fast_hfc.M_H(H, xs) - m_ref)) < 1e-12
    T, J = 5.0, 24
    ts = np.linspace(0.0, T, 41)
    assert np.max(np.abs(fast_hfc.gram(H, T, "cos", J) - slow_hfc.gram(H, T, "cos", J))) < 1e-11
    assert np.max(np.abs(fast_hfc.Q(H, T, "cos", J, ts) - slow_hfc.Q(H, T, "cos", J, ts))) < 1e-12
    assert np.max(np.abs(fast_hfc.dQ(H, T, "cos", J, ts) - slow_hfc.dQ(H, T, "cos", J, ts))) < 1e-11


@pytest.mark.parametrize("H", [0.5, 0.6, 0.75, 0.9])
@pytest.mark.parametrize("N,J", [(10, 4), (10, 24), (80, 16)])
def test_residual_covariance_strictly_pd_even_when_J_exceeds_N(H, N, J):
    """Lemma 3.1 / Proposition 3.2: C_R is strictly positive definite on interior grid {t_1..t_{N-1}}
    for EVERY J >= 1 and N >= 2, including J > N."""
    T = 5.0
    cc = CosineCoords(H, T, J, fast=True)
    t_int = np.arange(1, N, dtype=float) * (T / N)
    A = cc.path_funcs(t_int, "projection")
    CR = cc.fbm_cov(t_int, t_int) - A @ A.T
    CR = 0.5 * (CR + CR.T)
    eigs = np.linalg.eigvalsh(CR)
    assert eigs[0] > 0.0, f"C_R not strictly PD at H={H}, N={N}, J={J}: min eig={eigs[0]}"
    # Cholesky succeeds with zero jitter
    LR = np.linalg.cholesky(CR)
    assert np.max(np.abs(LR @ LR.T - CR)) < 1e-13


@pytest.mark.parametrize("H", [0.5, 0.6, 0.75, 0.9, 0.95])
def test_simpson_exact_and_pl_grid_covariance_exact(H):
    T, N, J = 5.0, 80, 16
    cc = CosineCoords(H, T, J, fast=True)
    tg = np.arange(1, N + 1, dtype=float) * (T / N)
    K_ex = cc.fbm_cov(tg, tg)
    for fn in (twoscale_exact_map, twoscale_pl_map):
        M = fn(H, T, N, J)
        K_hat = path_cov_on_grid(*M, T)
        rel = np.linalg.norm(K_hat - K_ex) / np.linalg.norm(K_ex)
        assert rel < 1e-12, f"{fn.__name__} failed at H={H}: rel={rel}"


def test_corollary_a1_woodbury_identity():
    """Verify Corollary A.1 exact product-family gap matches both precision-block and Woodbury forms."""
    rng = np.random.default_rng(42)
    m, D = 12, 30
    F = rng.normal(size=(m, D)) * 0.3
    b = rng.normal(size=(m,))
    R = np.diag(0.05 ** 2 * np.ones(m))
    Rinv = np.linalg.inv(R)
    P = np.eye(D) + F.T @ Rinv @ F
    a = F.T @ Rinv @ b
    c = 1.0 + b @ Rinv @ b
    S = c - a @ np.linalg.solve(P, a)
    delta_prec = 0.5 * math.log(c / S)
    delta_obs = 0.5 * math.log((1.0 + b @ Rinv @ b) / (1.0 + b @ np.linalg.solve(R + F @ F.T, b)))
    assert abs(delta_prec - delta_obs) < 1e-13


def _exact_kl_and_logq_check(loc, cov_fn, mean_fn, D, is_2d=False):
    torch.manual_seed(0)
    if is_2d:
        z, x0, kl = loc.sample(4)
        w = torch.cat([z[..., :loc.J], z[..., loc.D:loc.D + loc.J], x0,
                       z[..., loc.J:loc.D], z[..., loc.D + loc.J:]], dim=-1)
    else:
        z, x0, kl, _, _ = loc.sample(4, common=False)
        w = torch.cat([z, x0.unsqueeze(-1)], dim=-1)
    mu = mean_fn(loc)
    Sigma = cov_fn(loc)
    Dp = mu.shape[-1]
    prior = torch.distributions.MultivariateNormal(torch.zeros(Dp, dtype=mu.dtype), torch.eye(Dp, dtype=mu.dtype))
    q_dist = torch.distributions.MultivariateNormal(mu[0, 0], Sigma[0, 0])
    kl_ref = torch.distributions.kl_divergence(q_dist, prior)
    assert torch.allclose(kl[0, 0], kl_ref, atol=1e-5, rtol=1e-4)
    lq = loc.log_q(z, x0)
    lq_ref = q_dist.log_prob(w[:, 0, 0, :])
    assert torch.allclose(lq[:, 0, 0], lq_ref, atol=1e-5, rtol=1e-4)
    # Check empirical covariance from large sample
    torch.manual_seed(123)
    if is_2d:
        zs, x0s, _ = loc.sample(20000)
        ws = torch.cat([zs[..., :loc.J], zs[..., loc.D:loc.D + loc.J], x0s,
                        zs[..., loc.J:loc.D], zs[..., loc.D + loc.J:]], dim=-1)[:, 0, 0, :]
    else:
        zs, x0s, _, _, _ = loc.sample(20000, common=False)
        ws = torch.cat([zs, x0s.unsqueeze(-1)], dim=-1)[:, 0, 0, :]
    wc = ws - ws.mean(0, keepdim=True)
    emp_cov = (wc.T @ wc) / (ws.shape[0] - 1)
    assert torch.max(torch.abs(emp_cov - Sigma[0, 0])).item() < 0.04


def test_locals_joint_full_exact():
    torch.manual_seed(1)
    loc = LocalsJointFull(1, 2, 10, torch.zeros(1, 2)).double()
    with torch.no_grad():
        loc.mu.normal_(0, 0.3); loc.off.normal_(0, 0.15); loc.rho.normal_(0.1, 0.2)
    def cov_fn(m):
        L, _ = m.scale()
        return L @ L.transpose(-1, -2)
    _exact_kl_and_logq_check(loc, cov_fn, lambda m: m.mu, 10)


def test_locals_block_joint_exact():
    torch.manual_seed(2)
    J, D = 4, 10
    loc = LocalsBlockJoint(1, 2, J, D, torch.zeros(1, 2)).double()
    with torch.no_grad():
        loc.mu_c.normal_(0, 0.3); loc.off_c.normal_(0, 0.15); loc.rho_c.normal_(0.1, 0.2)
        loc.mu_r.normal_(0, 0.3); loc.rho_r.normal_(0.1, 0.2)
    def mean_fn(m):
        return torch.cat([m.mu_c[..., :J], m.mu_r, m.mu_c[..., J:]], dim=-1)
    def cov_fn(m):
        Lc, _ = m.scale_c()
        Sc = Lc @ Lc.transpose(-1, -2)
        dr = torch.nn.functional.softplus(m.rho_r)
        Dp = D + 1
        S = torch.zeros(1, 2, Dp, Dp, dtype=Sc.dtype)
        idx_c = list(range(J)) + [D]
        for a, ia in enumerate(idx_c):
            for b, ib in enumerate(idx_c):
                S[..., ia, ib] = Sc[..., a, b]
        for r in range(D - J):
            S[..., J + r, J + r] = dr[..., r] ** 2
        return S
    _exact_kl_and_logq_check(loc, cov_fn, mean_fn, D)


def test_locals_preclr_exact_and_autograd():
    torch.manual_seed(3)
    D, r = 12, 5
    loc = LocalsPrecLR(1, 2, D, torch.zeros(1, 2), rank=r).double()
    with torch.no_grad():
        loc.mu.normal_(0, 0.3); loc.rho.normal_(0.2, 0.2); loc.V.normal_(0, 0.5)
    def cov_fn(m):
        d = torch.nn.functional.softplus(m.rho)
        Lam = torch.diag_embed(1.0 / (d ** 2)) + m.V @ m.V.transpose(-1, -2)
        return torch.linalg.inv(Lam)
    _exact_kl_and_logq_check(loc, cov_fn, lambda m: m.mu, D)
    # Autograd backward check
    z, x0, kl, _, _ = loc.sample(4, common=False)
    loss = (z ** 2).sum() + (x0 ** 2).sum() + kl.sum()
    loss.backward()
    assert torch.isfinite(loc.V.grad).all() and torch.isfinite(loc.rho.grad).all()


def test_locals_2d_block_joint_exact():
    torch.manual_seed(4)
    J, D = 3, 7
    loc = Locals2BlockJoint(1, 2, J, D, torch.zeros(1, 2, 2)).double()
    with torch.no_grad():
        loc.mu_c.normal_(0, 0.2); loc.off_c.normal_(0, 0.12); loc.rho_c.normal_(0.1, 0.2)
        loc.mu_r.normal_(0, 0.2); loc.rho_r.normal_(0.1, 0.2)
    def mean_fn(m):
        return torch.cat([m.mu_c, m.mu_r], dim=-1)
    def cov_fn(m):
        Lc, _ = m.scale_c()
        Sc = Lc @ Lc.transpose(-1, -2)
        dr = torch.nn.functional.softplus(m.rho_r)
        Jc = 2 * J + 2; R2 = 2 * (D - J)
        S = torch.zeros(1, 2, Jc + R2, Jc + R2, dtype=Sc.dtype)
        S[..., :Jc, :Jc] = Sc
        for k in range(R2):
            S[..., Jc + k, Jc + k] = dr[..., k] ** 2
        return S
    _exact_kl_and_logq_check(loc, cov_fn, mean_fn, D, is_2d=True)
