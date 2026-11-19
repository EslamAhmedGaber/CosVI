"""Batched CUDA-graph training of latent SDE models whose noise is a linear map of whitened coordinates
(`noise_maps.py`).  Groups share the coordinate dimension D (one representation per run) and differ in H.
Also: exact-model held-out log-likelihood by importance sampling (IWAE) under the exact grid law, used as the
common yardstick for every representation.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .train_graph import GroupMLP, cosine_factor


class LocalsD(nn.Module):
    """q(z) full Gaussian (dimension D) and q(x0) Gaussian, per group and series."""
    def __init__(self, G, M, D, init_x0):
        super().__init__()
        self.D = D
        self.mu = nn.Parameter(torch.zeros(G, M, D))
        self.rho = nn.Parameter(torch.full((G, M, D), 0.5413))
        self.off = nn.Parameter(torch.zeros(G, M, D, D))
        self.m0 = nn.Parameter(init_x0.clone().expand(G, M).clone())
        self.r0 = nn.Parameter(torch.full((G, M), -2.0))

    def scale(self):
        d = F.softplus(self.rho)
        return torch.tril(self.off, -1) + torch.diag_embed(d), d

    def sample(self, S, common=True):
        L, d = self.scale()
        G, M, D = self.mu.shape
        eps = torch.randn((S, 1 if common else G, M, D), device=self.mu.device).expand(S, G, M, D)
        z = self.mu + torch.einsum("gmij,sgmj->sgmi", L, eps)
        e0 = torch.randn((S, 1 if common else G, M), device=self.mu.device).expand(S, G, M)
        x0 = self.m0 + F.softplus(self.r0) * e0
        kl_z = 0.5 * ((L ** 2).sum((-1, -2)) + (self.mu ** 2).sum(-1) - D - 2 * torch.log(d).sum(-1))   # (G, M)
        v0 = F.softplus(self.r0) ** 2
        kl_x0 = 0.5 * (v0 + self.m0 ** 2 - 1 - torch.log(v0))
        return z, x0, kl_z + kl_x0, eps, e0

    def log_q(self, z, x0):
        """log q(z, x0) for samples z (S, G, M, D), x0 (S, G, M) -> (S, G, M)."""
        L, d = self.scale()
        r = (z - self.mu).unsqueeze(-1)
        sol = torch.linalg.solve_triangular(L.unsqueeze(0).expand(z.shape[0], *L.shape), r, upper=False)[..., 0]
        lq = -0.5 * (sol ** 2).sum(-1) - torch.log(d).sum(-1) - 0.5 * self.D * np.log(2 * np.pi)
        s0 = F.softplus(self.r0)
        lq0 = -0.5 * ((x0 - self.m0) / s0) ** 2 - torch.log(s0) - 0.5 * np.log(2 * np.pi)
        return lq + lq0


class Globals(nn.Module):
    def __init__(self, G, width=32, depth=2, sigma_init=0.3):
        super().__init__()
        self.net = GroupMLP(G, 1, width, depth, 1)
        self.s_raw = nn.Parameter(torch.full((G,), float(np.log(np.expm1(sigma_init)))))

    def sigma(self):
        return F.softplus(self.s_raw)


def solve(glob, Ms, z, x0, T, obs_steps):
    """Ms (G, 3, N, D); z (S, G, M, D); x0 (S, G, M) -> X at obs steps (m, G, S, M)."""
    S, G, M, D = z.shape
    N = Ms.shape[2]
    R = torch.einsum("gcnd,sgmd->cngsm", Ms, z).reshape(3, N, G, S * M, 1)
    sig = glob.sigma()[:, None, None]
    x = x0.permute(1, 0, 2).reshape(G, S * M, 1)
    h = T / N
    out = []; obs_set = set(int(o) for o in obs_steps)
    for n in range(N):
        rl, rm, rr = R[0, n], R[1, n], R[2, n]
        k1 = glob.net(x) + sig * rl
        k2 = glob.net(x + 0.5 * h * k1) + sig * rm
        k3 = glob.net(x + 0.5 * h * k2) + sig * rm
        k4 = glob.net(x + h * k3) + sig * rr
        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if n + 1 in obs_set:
            out.append(x)
    return torch.stack(out, 0)[..., 0].reshape(len(out), G, S, M)


def loglik(Y, X, s_obs):
    """Y (M, m), X (m, G, S, M) -> per-sample log-likelihood (S, G, M)."""
    return (-0.5 * (Y.T[:, None, None, :] - X) ** 2 / s_obs ** 2 - 0.5 * np.log(2 * np.pi * s_obs ** 2)).sum(0).permute(1, 0, 2)


def graph_trainer(glob, loc, Y, Ms, T, obs_steps, s_obs, S, train_globals, lrs):
    dev = Y.device
    T_ = lambda v: torch.tensor(float(v), device=dev)
    groups = [{"params": list(loc.parameters()), "lr": T_(lrs["local"]), "base": lrs["local"]}]
    if train_globals:
        groups += [{"params": list(glob.net.parameters()), "lr": T_(lrs["net"]), "base": lrs["net"]},
                   {"params": [glob.s_raw], "lr": T_(lrs["sigma"]), "base": lrs["sigma"]}]
    opt = torch.optim.Adam(groups, capturable=True)
    out = {"elbo": torch.zeros(Ms.shape[0], device=dev)}

    def step():
        z, x0, kl, _, _ = loc.sample(S)
        X = solve(glob, Ms, z, x0, T, obs_steps)
        ll = loglik(Y, X, s_obs).mean(0)                                 # (G, M)
        elbo = (ll - kl).sum(-1)                                          # (G,)
        loss = -elbo.sum() / Y.shape[0]
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=False)
        out["elbo"].copy_(elbo.detach())

    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()

    def set_lr(fac):
        for grp in opt.param_groups:
            grp["lr"].fill_(grp["base"] * fac)
    return g.replay, out, set_lr


def train(glob, loc, Y, Ms, T, obs_steps, s_obs, S, steps, train_globals, lrs, log_every=200):
    replay, out, set_lr = graph_trainer(glob, loc, Y, Ms, T, obs_steps, s_obs, S, train_globals, lrs)
    hist = []
    for it in range(steps - 3):
        if it % 50 == 0:
            set_lr(cosine_factor(it, steps))
        replay()
        if it % log_every == 0:
            hist.append([it] + (out["elbo"] / Y.shape[0]).cpu().tolist())
    return hist


@torch.no_grad()
def elbo_eval(glob, loc, Y, Ms, T, obs_steps, s_obs, S=256, reps=16):
    acc = 0.0
    for _ in range(reps):
        z, x0, kl, _, _ = loc.sample(S // reps)
        X = solve(glob, Ms, z, x0, T, obs_steps)
        acc = acc + (loglik(Y, X, s_obs).mean(0) - kl).sum(-1)
    return (acc / reps / Y.shape[0]).cpu().numpy()


@torch.no_grad()
def iwae(glob, loc, Y, Ms, T, obs_steps, s_obs, K=512, chunk=64):
    """Importance-weighted estimate of log p(y) per group and series under the model defined by Ms (exact grid law
    when Ms is the 'grid' map): log mean_k exp(log p(y|z_k) + log N(z_k) + log N(x0_k) - log q(z_k, x0_k))."""
    logw = []
    D = Ms.shape[-1]
    for _ in range(K // chunk):
        z, x0, _, _, _ = loc.sample(chunk, common=False)
        X = solve(glob, Ms, z, x0, T, obs_steps)
        lp = loglik(Y, X, s_obs)                                                          # (S, G, M)
        lprior = -0.5 * (z ** 2).sum(-1) - 0.5 * D * np.log(2 * np.pi) - 0.5 * x0 ** 2 - 0.5 * np.log(2 * np.pi)
        logw.append(lp + lprior - loc.log_q(z, x0))
    lw = torch.cat(logw, 0)                                                               # (K, G, M)
    return (torch.logsumexp(lw, 0) - np.log(lw.shape[0])).cpu().numpy()                  # (G, M)
