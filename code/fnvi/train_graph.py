"""Batched, CUDA-graph-captured training of cosine-coordinate latent SDE models (scalar state, additive noise).

Groups g = 0..G-1 are independent models trained side by side (different fixed Hurst indices, a learned-H group).
One training step = sample (xi, x0) from q, build noise rates from the (fixed or H-interpolated) rate tables,
RK4-solve the random ODE, Gaussian log-likelihood at the observation steps, closed-form KL, Adam update.  The
whole step is captured once as a CUDA graph and replayed (about 8x faster than eager mode on this machine).
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .cvi import rate_table_np

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "cache")


def h_bank(T, J, N, dH=0.005, Hmax=0.99, mode="projection"):
    """Rate tables at H = 0.5, 0.5+dH, ..., Hmax: array (nH, 2N+1, J), cached on disk."""
    os.makedirs(CACHE, exist_ok=True)
    fn = os.path.join(CACHE, f"bank_T{T}_J{J}_N{N}_dH{dH}_{mode}.npz")
    Hs = np.round(np.arange(0.5, Hmax + 1e-9, dH), 6)
    if os.path.exists(fn):
        z = np.load(fn)
        if np.allclose(z["Hs"], Hs):
            return z["Hs"], z["bank"]
    bank = np.stack([rate_table_np(h, T, J, N, mode) for h in Hs], 0)
    np.savez(fn, Hs=Hs, bank=bank)
    return Hs, bank


class GroupMLP(nn.Module):
    """G independent MLPs evaluated in one batched call: x (G, B, din) -> (G, B, dout)."""
    def __init__(self, G, din=1, width=32, depth=2, dout=1):
        super().__init__()
        dims = [din] + [width] * depth + [dout]
        self.W = nn.ParameterList(); self.b = nn.ParameterList()
        for a, c in zip(dims[:-1], dims[1:]):
            w0 = torch.empty(a, c)
            nn.init.xavier_uniform_(w0)                      # common initial weights for every group
            w = w0[None].repeat(G, 1, 1)
            self.W.append(nn.Parameter(w)); self.b.append(nn.Parameter(torch.zeros(G, 1, c)))

    def forward(self, x):
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            x = torch.baddbmm(b, x, w)
            if i < len(self.W) - 1:
                x = torch.tanh(x)
        return x


class Locals(nn.Module):
    """Per-group, per-series q(xi) (full Gaussian, dimension J) and q(x0) (Gaussian)."""
    def __init__(self, G, M, J, init_x0):
        super().__init__()
        self.J = J
        self.mu = nn.Parameter(torch.zeros(G, M, J))
        self.rho = nn.Parameter(torch.full((G, M, J), 0.5413))
        self.off = nn.Parameter(torch.zeros(G, M, J, J))
        self.m0 = nn.Parameter(init_x0.clone().expand(G, M).clone())
        self.r0 = nn.Parameter(torch.full((G, M), -2.0))

    def sample(self, S):
        """Common random numbers: the same standard normals are used by every group."""
        d = F.softplus(self.rho)
        L = torch.tril(self.off, -1) + torch.diag_embed(d)
        G, M, J = self.mu.shape
        eps = torch.randn((S, 1, M, J), device=self.mu.device).expand(S, G, M, J)
        xi = self.mu + torch.einsum("gmij,sgmj->sgmi", L, eps)
        x0 = self.m0 + F.softplus(self.r0) * torch.randn((S, 1, M), device=self.mu.device).expand(S, G, M)
        kl_xi = 0.5 * ((L ** 2).sum((-1, -2)) + (self.mu ** 2).sum(-1) - self.J - 2 * torch.log(d).sum(-1)).sum(-1)
        v0 = F.softplus(self.r0) ** 2
        kl_x0 = 0.5 * (v0 + self.m0 ** 2 - 1 - torch.log(v0)).sum(-1)
        return xi, x0, kl_xi + kl_x0


class LatentSDE(nn.Module):
    """Globals: drift networks, noise scales, a learnable Hurst index (used by the last group if learn_H)."""
    def __init__(self, G, width=32, depth=2, H_init=0.7, sigma_init=0.3):
        super().__init__()
        self.net = GroupMLP(G, 1, width, depth, 1)
        self.s_raw = nn.Parameter(torch.full((G,), float(np.log(np.expm1(sigma_init)))))
        self.h_raw = nn.Parameter(torch.tensor([float(np.log((H_init - 0.5) / (0.99 - H_init)))]))

    def H(self):
        return 0.5 + 0.49 * torch.sigmoid(self.h_raw)

    def sigma(self):
        return F.softplus(self.s_raw)


def solve(model, tabs, xi, x0, T, N, obs_steps):
    """tabs (G, 2N+1, J), xi (S, G, M, J), x0 (S, G, M) -> X at obs steps (m, G, S, M)."""
    S, G, M, J = xi.shape
    R = torch.einsum("gkj,sgmj->kgsm", tabs, xi).reshape(2 * N + 1, G, S * M, 1)
    sig = model.sigma()[:, None, None]
    x = x0.permute(1, 0, 2).reshape(G, S * M, 1)
    h = T / N
    out = []
    obs_set = set(int(o) for o in obs_steps)
    for n in range(N):
        r0, r1, r2 = R[2 * n], R[2 * n + 1], R[2 * n + 2]
        k1 = model.net(x) + sig * r0
        k2 = model.net(x + 0.5 * h * k1) + sig * r1
        k3 = model.net(x + 0.5 * h * k2) + sig * r1
        k4 = model.net(x + h * k3) + sig * r2
        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if n + 1 in obs_set:
            out.append(x)
    return torch.stack(out, 0)[..., 0].reshape(len(out), G, S, M)


def make_tabs(model, tabs_fixed, bank, Hs, learn_H):
    """Fixed tables for the first groups; the last group uses the bank interpolated at the learned H."""
    Hj = model.H() if learn_H else model.H().detach()
    dH = float(Hs[1] - Hs[0])
    u = (Hj - float(Hs[0])) / dH
    k = torch.clamp(torch.floor(u), 0, len(Hs) - 2)
    w = (u - k)
    ki = k.long()
    tab = (1 - w)[:, None, None] * bank.index_select(0, ki) + w[:, None, None] * bank.index_select(0, ki + 1)
    return torch.cat([tabs_fixed, tab], 0)


def elbo_terms(model, loc, Y, tabs, T, N, obs_steps, s_obs, S):
    xi, x0, kl = loc.sample(S)
    X = solve(model, tabs, xi, x0, T, N, obs_steps)                         # (m, G, S, M)
    m, G, S_, M = X.shape
    ll = (-0.5 * (Y.T[:, None, None, :] - X) ** 2 / s_obs ** 2).sum(0).mean(1).sum(-1) \
        - 0.5 * m * M * float(np.log(2 * np.pi * s_obs ** 2))
    return ll - kl                                                            # (G,)


def graph_trainer(model, loc, Y, tabs_fixed, bank, Hs, T, N, obs_steps, s_obs, S, train_globals, lrs, learn_H=True):
    """Returns (replay(), out) where replay() performs one captured training step and out['elbo'] holds (G,)."""
    dev = Y.device
    T_ = lambda v: torch.tensor(float(v), device=dev)
    groups = [{"params": list(loc.parameters()), "lr": T_(lrs["local"]), "base": lrs["local"]}]
    if train_globals:
        groups += [{"params": list(model.net.parameters()), "lr": T_(lrs["net"]), "base": lrs["net"]},
                   {"params": [model.s_raw], "lr": T_(lrs["sigma"]), "base": lrs["sigma"]}]
        if learn_H:
            groups += [{"params": [model.h_raw], "lr": T_(lrs["H"]), "base": lrs["H"]}]
    opt = torch.optim.Adam(groups, capturable=True)
    out = {"elbo": torch.zeros(tabs_fixed.shape[0] + 1, device=Y.device)}
    M = Y.shape[0]

    def step():
        tabs = make_tabs(model, tabs_fixed, bank, Hs, learn_H and train_globals)
        elbo = elbo_terms(model, loc, Y, tabs, T, N, obs_steps, s_obs, S)
        loss = -elbo.sum() / M
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=False)
        out["elbo"].copy_(elbo.detach())

    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    def set_lr_factor(fac):
        for grp in opt.param_groups:
            grp["lr"].fill_(grp["base"] * fac)
    out["set_lr_factor"] = set_lr_factor
    return g.replay, out


def cosine_factor(it, n, floor=0.05):
    return floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * min(it, n) / n))


@torch.no_grad()
def elbo_eval(model, loc, Y, tabs_fixed, bank, Hs, T, N, obs_steps, s_obs, S=64, reps=8):
    tabs = make_tabs(model, tabs_fixed, bank, Hs, False)
    acc = 0.0
    for _ in range(reps):
        acc = acc + elbo_terms(model, loc, Y, tabs, T, N, obs_steps, s_obs, S // reps)
    return (acc / reps / Y.shape[0]).cpu().numpy()
