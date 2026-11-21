r"""N6: ablations of CosVI's components on the N1 double-well task (protocol fixed on
15 Nov 2026, about 21:38, before any run; batching amended about 21:56, before any run).

Settings of N1b (data, drift MLP, sigma, N = 80, 5000 Adam steps, N1b learning rates and cosine decay), oracle H only
(H fixed at H_true), one component changed at a time:
  base        CosVI: two-scale (J = 16 + residual), whitened coordinates, q full, RK4, S = 16
  base_cos    CosVI-J: cosine J = 32, projection coordinates, otherwise base
  classical   cosine J = 32 in CosChaos's classical coordinates                                 (pair: base_cos)
  raw         cosine J = 32, q over Z = L xi with prior N(0, G), initialised at the prior         (pair: base_cos)
  mf          q mean-field (diagonal)                                                             (pair: base)
  lr2/4/8     q low-rank plus diagonal, rank r; U initialised at 0.01 N(0, 1)                     (pair: base)
  heun        Heun steps (left and right stage rates) instead of RK4                              (pair: base)
  s4, s8      S = 4 or 8 samples per step                                                         (pair: base)
Batching: one job per variant trains all (H_true, data seed) combinations side by side as independent groups (own data,
drift network, sigma and posteriors per group; Adam is per parameter, so each group's updates depend only on its own
ELBO); independent Monte Carlo noise per group.  The GPU here is step-latency bound, so G = 9 costs about as much as
G = 1.  Results are equal in law to separate runs.
Yardstick (fixed for every variant): held-out log-likelihood of the 16 test series under the exact grid law, test
posteriors refitted in grid coordinates (2000 steps, RK4, q full) and IWAE K = 512; second yardstick in two-scale
coordinates.  Also drift RMSE (occupation weighted), sigma, training wall-clock (of the whole batched job).
Run from work\code\experiments:  python n6_ablations.py --variants base heun
Out: work/results/n6/n6_{variant}_seed{seed}_H{H}.json  (one file per group)
"""
import os, sys, json, time, argparse, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import n1_drift_recovery as n1
import n1b_representations as nb
from fnvi.noise_maps import build, cosine_map
from fnvi.coords import CosineCoords
from fnvi.train_maps import Globals, LocalsD, solve as solve_rk4
from fnvi.train_graph import cosine_factor

OUT = os.path.join(HERE, "..", "..", "results", "n6"); os.makedirs(OUT, exist_ok=True)
T, N, OBS, S_OBS, LRS = nb.T, nb.N, nb.OBS, nb.S_OBS, nb.LRS
TS = dict(rep="twoscale", J=16, mode="projection", q="full", solver="rk4", S=16)
CO = dict(TS, rep="cosine", J=32)
VARIANTS = {"base": TS, "base_cos": CO, "classical": dict(CO, mode="classical"), "raw": dict(CO, q="raw"),
            "mf": dict(TS, q="lr0"), "lr2": dict(TS, q="lr2"), "lr4": dict(TS, q="lr4"), "lr8": dict(TS, q="lr8"),
            "heun": dict(TS, solver="heun"), "s4": dict(TS, S=4), "s8": dict(TS, S=8)}
PAIR = {v: ("base_cos" if v in ("classical", "raw") else "base") for v in VARIANTS if not v.startswith("base")}


# ---------------------------------------------------------------- variational families (interface of LocalsD)
def _x0_part(m0, r0, S, G, M, common, dev):
    e0 = torch.randn((S, 1 if common else G, M), device=dev).expand(S, G, M)
    x0 = m0 + F.softplus(r0) * e0
    v0 = F.softplus(r0) ** 2
    return x0, 0.5 * (v0 + m0 ** 2 - 1 - torch.log(v0)), e0


class LocalsRaw(LocalsD):
    """q full Gaussian over the raw coefficients Z = L xi, prior N(0, G = L L^T) per group; initialised at the prior."""
    def __init__(self, G, M, D, init_x0, Lps):
        super().__init__(G, M, D, init_x0)
        self.register_buffer("Linv", torch.as_tensor(np.stack([np.linalg.inv(L) for L in Lps]), dtype=torch.float32))
        self.register_buffer("logdetG", torch.tensor([float(2 * np.log(np.diag(L)).sum()) for L in Lps]))
        Lt = torch.as_tensor(np.stack(Lps), dtype=torch.float32)                         # (G, D, D)
        with torch.no_grad():
            self.rho.copy_(torch.log(torch.expm1(torch.diagonal(Lt, dim1=-2, dim2=-1)))[:, None, :].expand_as(self.rho))
            self.off.copy_(torch.tril(Lt, -1)[:, None].expand_as(self.off))

    def sample(self, S, common=True):
        L, d = self.scale()
        G, M, D = self.mu.shape
        eps = torch.randn((S, 1 if common else G, M, D), device=self.mu.device).expand(S, G, M, D)
        z = self.mu + torch.einsum("gmij,sgmj->sgmi", L, eps)
        A = torch.einsum("gij,gmjk->gmik", self.Linv, L)                     # L_G^{-1} C
        m = torch.einsum("gij,gmj->gmi", self.Linv, self.mu)                 # L_G^{-1} mu
        kl_z = 0.5 * ((A ** 2).sum((-1, -2)) + (m ** 2).sum(-1) - D + self.logdetG[:, None] - 2 * torch.log(d).sum(-1))
        x0, kl_x0, e0 = _x0_part(self.m0, self.r0, S, G, M, common, self.mu.device)
        return z, x0, kl_z + kl_x0, eps, e0


class LocalsLR(nn.Module):
    """q Gaussian with covariance diag(d^2) + U U^T, U in R^{D x r}; r = 0 is mean-field."""
    def __init__(self, G, M, D, init_x0, r):
        super().__init__()
        self.D, self.r = D, r
        self.mu = nn.Parameter(torch.zeros(G, M, D))
        self.rho = nn.Parameter(torch.full((G, M, D), 0.5413))
        self.U = nn.Parameter(0.01 * torch.randn(G, M, D, r)) if r > 0 else None
        self.m0 = nn.Parameter(init_x0.clone().expand(G, M).clone())
        self.r0 = nn.Parameter(torch.full((G, M), -2.0))

    def sample(self, S, common=True):
        d = F.softplus(self.rho)
        G, M, D = self.mu.shape
        dev = self.mu.device
        eps = torch.randn((S, 1 if common else G, M, D), device=dev).expand(S, G, M, D)
        z = self.mu + d * eps
        tr = (d ** 2).sum(-1); logdet = 2 * torch.log(d).sum(-1)
        if self.r > 0:
            e2 = torch.randn((S, 1 if common else G, M, self.r), device=dev).expand(S, G, M, self.r)
            z = z + torch.einsum("gmdr,sgmr->sgmd", self.U, e2)
            tr = tr + (self.U ** 2).sum((-1, -2))
            W = self.U / d[..., None]
            Kc, _ = torch.linalg.cholesky_ex(torch.eye(self.r, device=dev) + W.transpose(-1, -2) @ W)
            logdet = logdet + 2 * torch.log(torch.diagonal(Kc, dim1=-2, dim2=-1)).sum(-1)
        kl_z = 0.5 * (tr + (self.mu ** 2).sum(-1) - D - logdet)
        x0, kl_x0, e0 = _x0_part(self.m0, self.r0, S, G, M, common, dev)
        return z, x0, kl_z + kl_x0, eps, e0


# ---------------------------------------------------------------- per-group data, solvers, trainer, IWAE
def loglik_g(Y, X):
    """Y (G, M, m) per-group data, X (m, G, S, M) -> (S, G, M)."""
    Yb = Y.permute(2, 0, 1)[:, :, None, :]
    return (-0.5 * (Yb - X) ** 2 / S_OBS ** 2 - 0.5 * np.log(2 * np.pi * S_OBS ** 2)).sum(0).permute(1, 0, 2)


def solve_heun(glob, Ms, z, x0, T, obs_steps):
    """Heun steps with the left and right stage rates; Ms (G, 3, N, D) (the mid row is not used)."""
    S, G, M, D = z.shape
    Nn = Ms.shape[2]
    R = torch.einsum("gcnd,sgmd->cngsm", Ms[:, 0::2], z).reshape(2, Nn, G, S * M, 1)
    sig = glob.sigma()[:, None, None]
    x = x0.permute(1, 0, 2).reshape(G, S * M, 1)
    h = T / Nn
    out = []; obs_set = set(int(o) for o in obs_steps)
    for n in range(Nn):
        k1 = glob.net(x) + sig * R[0, n]
        k2 = glob.net(x + h * k1) + sig * R[1, n]
        x = x + 0.5 * h * (k1 + k2)
        if n + 1 in obs_set:
            out.append(x)
    return torch.stack(out, 0)[..., 0].reshape(len(out), G, S, M)


def train_g(glob, loc, Y, Ms, S, steps, solve_fn, train_globals=True, lrs=LRS, log_every=200):
    """train_maps.train with per-group data, a choice of solver and independent noise per group (same optimiser,
    learning rates, cosine schedule and CUDA-graph capture)."""
    dev = Y.device
    T_ = lambda v: torch.tensor(float(v), device=dev)
    groups = [{"params": list(loc.parameters()), "lr": T_(lrs["local"]), "base": lrs["local"]}]
    if train_globals:
        groups += [{"params": list(glob.net.parameters()), "lr": T_(lrs["net"]), "base": lrs["net"]},
                   {"params": [glob.s_raw], "lr": T_(lrs["sigma"]), "base": lrs["sigma"]}]
    opt = torch.optim.Adam(groups, capturable=True)
    out = {"elbo": torch.zeros(Ms.shape[0], device=dev)}

    def step():
        z, x0, kl, _, _ = loc.sample(S, common=False)
        X = solve_fn(glob, Ms, z, x0, T, OBS)
        elbo = (loglik_g(Y, X).mean(0) - kl).sum(-1)                               # (G,)
        (-elbo.sum() / Y.shape[1]).backward(); opt.step(); opt.zero_grad(set_to_none=False)
        out["elbo"].copy_(elbo.detach())

    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    hist = []
    for it in range(steps - 3):
        if it % 50 == 0:
            for grp in opt.param_groups:
                grp["lr"].fill_(grp["base"] * cosine_factor(it, steps))
        g.replay()
        if it % log_every == 0:
            hist.append([it] + (out["elbo"] / Y.shape[1]).cpu().tolist())
    return hist


@torch.no_grad()
def iwae_g(glob, loc, Y, Ms, K=512, chunk=64):
    """train_maps.iwae with per-group data: log p(y) per group and series (G, M)."""
    logw = []
    D = Ms.shape[-1]
    for _ in range(K // chunk):
        z, x0, _, _, _ = loc.sample(chunk, common=False)
        X = solve_rk4(glob, Ms, z, x0, T, OBS)
        lprior = -0.5 * (z ** 2).sum(-1) - 0.5 * D * np.log(2 * np.pi) - 0.5 * x0 ** 2 - 0.5 * np.log(2 * np.pi)
        logw.append(loglik_g(Y, X) + lprior - loc.log_q(z, x0))
    lw = torch.cat(logw, 0)
    return (torch.logsumexp(lw, 0) - np.log(lw.shape[0])).cpu().numpy()


# ---------------------------------------------------------------- maps
def variant_maps(cfg, Hs, dev):
    Ms, Lps = [], []
    for H in Hs:
        if cfg["rep"] == "twoscale":
            M = np.stack(build("twoscale", H, T, N, J=cfg["J"]), 0)
        elif cfg["mode"] == "classical":
            M = np.stack(cosine_map(H, T, N, cfg["J"], "classical"), 0)
        else:
            M = np.stack(build("cosine", H, T, N, J=cfg["J"]), 0)
            if cfg["q"] == "raw":                               # rates = M xi = (M L^{-1}) Z
                L = CosineCoords(H, T, cfg["J"]).L
                M = M @ np.linalg.inv(L); Lps.append(L)
        Ms.append(M)
    return torch.as_tensor(np.stack(Ms, 0), dtype=torch.float32, device=dev), Lps


def run(variant, Hs_true, seeds, steps=5000, refit=2000, save=True):
    cfg = VARIANTS[variant]
    dev = torch.device("cuda"); torch.manual_seed(12345)
    combos = list(itertools.product(Hs_true, seeds)); G = len(combos)
    data = [n1.make_data(H, s) for H, s in combos]
    Ytr = torch.as_tensor(np.stack([d[0] for d in data]), dtype=torch.float32, device=dev)      # (G, 48, 20)
    Yte = torch.as_tensor(np.stack([d[1] for d in data]), dtype=torch.float32, device=dev)      # (G, 16, 20)
    Hs = [H for H, _ in combos]
    Ms, Lps = variant_maps(cfg, Hs, dev); D = Ms.shape[-1]
    glob = Globals(G).to(dev)                                   # created first: same initial weights in every variant
    q = cfg["q"]
    if q == "full":
        loc = LocalsD(G, Ytr.shape[1], D, Ytr[:, :, 0]).to(dev)
    elif q == "raw":
        loc = LocalsRaw(G, Ytr.shape[1], D, Ytr[:, :, 0], Lps).to(dev)
    else:
        loc = LocalsLR(G, Ytr.shape[1], D, Ytr[:, :, 0], int(q[2:])).to(dev)
    solve_fn = solve_rk4 if cfg["solver"] == "rk4" else solve_heun
    t0 = time.time()
    hist = train_g(glob, loc, Ytr, Ms, cfg["S"], steps, solve_fn)
    torch.cuda.synchronize(); train_s = time.time() - t0
    for p in glob.parameters():
        p.requires_grad_(False)
    t1 = time.time()
    Mg = nb.maps_tensor("grid", Hs, dev)
    loc_te = LocalsD(G, Yte.shape[1], N, Yte[:, :, 0]).to(dev)
    train_g(glob, loc_te, Yte, Mg, 16, refit, solve_rk4, train_globals=False)
    ll_grid = iwae_g(glob, loc_te, Yte, Mg)                                                     # (G, 16)
    Mt = nb.maps_tensor("twoscale", Hs, dev)
    loc_te2 = LocalsD(G, Yte.shape[1], Mt.shape[-1], Yte[:, :, 0]).to(dev)
    train_g(glob, loc_te2, Yte, Mt, 16, refit, solve_rk4, train_globals=False)
    ll_ts = iwae_g(glob, loc_te2, Yte, Mt)
    eval_s = time.time() - t1
    xg = np.linspace(-1.5, 1.5, 301)
    with torch.no_grad():
        fg = glob.net(torch.as_tensor(xg, dtype=torch.float32, device=dev)[None, :, None].expand(G, -1, 1))[..., 0].cpu().numpy()
    ft = n1.f_true(xg)
    sig = glob.sigma().cpu().numpy()
    for gi, (H, s) in enumerate(combos):
        w_occ, edges = np.histogram(np.clip(data[gi][3], -1.5, 1.5), bins=300, range=(-1.5, 1.5), density=True)
        w = np.interp(xg, 0.5 * (edges[1:] + edges[:-1]), w_occ); w = w / w.sum()
        res = {"variant": variant, "config": cfg, "pair": PAIR.get(variant), "D": int(D), "H_true": H, "seed": s,
               "steps": steps, "refit": refit, "groups_in_job": [f"H={h:.2f}/seed{t}" for h, t in combos],
               "train_seconds_job": train_s, "eval_seconds_job": eval_s,
               "sigma": float(sig[gi]), "drift_rmse_occupation": float(np.sqrt((((fg[gi] - ft) ** 2) * w).sum())),
               "test_loglik_exact_grid_law_mean": float(ll_grid[gi].mean()),
               "test_loglik_exact_grid_law_per_series": ll_grid[gi].tolist(),
               "test_loglik_exact_law_twoscale_coords_mean": float(ll_ts[gi].mean()),
               "test_loglik_exact_law_twoscale_coords_per_series": ll_ts[gi].tolist(),
               "drift_curve": {"x": xg[::10].tolist(), "true": ft[::10].tolist(), "fit": fg[gi, ::10].tolist()},
               "history_elbo_per_series": [[h[0], h[1 + gi]] for h in hist]}
        if save:
            with open(os.path.join(OUT, f"n6_{variant}_seed{s}_H{H:.2f}.json"), "w") as fh:
                json.dump(res, fh, indent=1)
        print(json.dumps({"variant": variant, "H_true": H, "seed": s, "D": int(D), "sigma": round(res["sigma"], 3),
                          "drift_rmse": round(res["drift_rmse_occupation"], 3),
                          "ll_grid": round(res["test_loglik_exact_grid_law_mean"], 3),
                          "ll_twoscale": round(res["test_loglik_exact_law_twoscale_coords_mean"], 3)}), flush=True)
    print(json.dumps({"variant": variant, "groups": G, "train_s": round(train_s), "eval_s": round(eval_s)}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--H", type=float, nargs="+", default=[0.6, 0.75, 0.9])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--refit", type=int, default=2000)
    ap.add_argument("--no-save", action="store_true", help="smoke test: do not write the result files")
    a = ap.parse_args()
    for v in a.variants:
        run(v, a.H, a.seeds, a.steps, a.refit, not a.no_save)
