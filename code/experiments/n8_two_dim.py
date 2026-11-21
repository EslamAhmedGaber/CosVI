r"""X2 (16 Nov 2026): a two-dimensional neural SDE — the damped Duffing system, the 2-D extension of the N1 double well.

Protocol fixed before any run.
Truth : dX1 = X2 dt + sigma dB1,  dX2 = (X1 - X1^3 - 0.5 X2) dt + sigma dB2,  independent standard fBm channels with
        the same H, sigma = 0.5 (as N1), X(0) ~ U([-1.5, 1.5]^2), T = 5.
Data  : exact fBm (Davies-Harte, 8192 steps) + Heun; 20 observations of both coordinates (every 0.25) with s = 0.05;
        48 training and 16 test series; data seeds 0-2 (random stream 3000 + seed).
Fit   : drift MLP 2-32-32-2 (tanh), sigma learned per channel, X(0) ~ N(0, I); q full Gaussian JOINTLY over the noise
        coordinates of both channels (2D) and Gaussian over x0; RK4 with N = 80 (the map's three stage rates per channel);
        S = 16; 5000 Adam steps with the N1b learning rates and cosine decay; H fixed at H_true.
Groups: one job per representation trains the nine (H_true, seed) combinations side by side as independent groups
        (as N6): own data, network, sigma and posteriors; common initial weights across representations.
Yardstick: held-out log-likelihood of the 16 test series under the exact grid law (test posteriors refitted in grid
        coordinates, 2000 steps, primary; two-scale coordinates, secondary; IWAE K = 512); drift RMSE weighted by the
        occupation measure on [-2, 2]^2 (41 x 41 grid); sigma; wall-clock of the job.
Run from work\code\experiments:  python n8_two_dim.py --reps twoscale        (--steps 30 --refit 20 --K 32 --no-save: smoke)
Out: work/results/n8/n8_{rep}_seed{seed}_H{H}.json (one file per group)
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
from fnvi.simulate import simulate
from fnvi.train_graph import GroupMLP, cosine_factor

OUT = os.path.join(HERE, "..", "..", "results", "n8"); os.makedirs(OUT, exist_ok=True)
T, N, OBS, S_OBS, LRS = nb.T, nb.N, nb.OBS, nb.S_OBS, nb.LRS
SIG, DAMP = n1.SIG_TRUE, 0.5
M_TR, M_TE, N_OBS = n1.M_TR, n1.M_TE, n1.N_OBS
REPS = ["twoscale", "cosine", "grid", "markov", "markov_best"]
LOG2PI = float(np.log(2 * np.pi))


def f2(x):
    return np.stack([x[..., 1], x[..., 0] - x[..., 0] ** 3 - DAMP * x[..., 1]], -1)


def make_data2(H, seed):
    rng = np.random.default_rng(3000 + seed)
    M = M_TR + M_TE; Nf = 8192
    x0 = rng.uniform(-1.5, 1.5, (M, 2))
    X = simulate(f2, lambda x: SIG * np.ones_like(x), x0, T, Nf, H, rng)                 # (Nf+1, M, 2)
    idx = np.arange(1, N_OBS + 1) * (Nf // N_OBS)
    Y = X[idx].transpose(1, 0, 2) + S_OBS * rng.standard_normal((M, N_OBS, 2))           # (M, m, 2)
    return Y[:M_TR], Y[M_TR:], X[::8].reshape(-1, 2)


class Locals2(nn.Module):
    """q full Gaussian over the 2D noise coordinates (both channels jointly) and diagonal Gaussian q(x0) in R^2."""
    def __init__(self, G, M, D, init_x0):
        super().__init__()
        self.D2 = 2 * D
        self.mu = nn.Parameter(torch.zeros(G, M, 2 * D))
        self.rho = nn.Parameter(torch.full((G, M, 2 * D), 0.5413))
        self.off = nn.Parameter(torch.zeros(G, M, 2 * D, 2 * D))
        self.m0 = nn.Parameter(init_x0.clone())
        self.r0 = nn.Parameter(torch.full(tuple(init_x0.shape), -2.0))

    def scale(self):
        d = F.softplus(self.rho)
        return torch.tril(self.off, -1) + torch.diag_embed(d), d

    def sample(self, S):
        L, d = self.scale()
        G, M, D2 = self.mu.shape
        dev = self.mu.device
        eps = torch.randn((S, G, M, D2), device=dev)
        z = self.mu + torch.einsum("gmij,sgmj->sgmi", L, eps)
        x0 = self.m0 + F.softplus(self.r0) * torch.randn((S, G, M, 2), device=dev)
        kl_z = 0.5 * ((L ** 2).sum((-1, -2)) + (self.mu ** 2).sum(-1) - D2 - 2 * torch.log(d).sum(-1))
        v0 = F.softplus(self.r0) ** 2
        kl_x0 = 0.5 * (v0 + self.m0 ** 2 - 1 - torch.log(v0)).sum(-1)
        return z, x0, kl_z + kl_x0

    def log_q(self, z, x0):
        L, d = self.scale()
        r = (z - self.mu).unsqueeze(-1)
        sol = torch.linalg.solve_triangular(L.unsqueeze(0).expand(z.shape[0], *L.shape), r, upper=False)[..., 0]
        lq = -0.5 * (sol ** 2).sum(-1) - torch.log(d).sum(-1) - 0.5 * self.D2 * LOG2PI
        s0 = F.softplus(self.r0)
        lq0 = (-0.5 * ((x0 - self.m0) / s0) ** 2 - torch.log(s0) - 0.5 * LOG2PI).sum(-1)
        return lq + lq0


class Globals2(nn.Module):
    def __init__(self, G, width=32, depth=2, sigma_init=0.3):
        super().__init__()
        self.net = GroupMLP(G, 2, width, depth, 2)
        self.s_raw = nn.Parameter(torch.full((G, 2), float(np.log(np.expm1(sigma_init)))))

    def sigma(self):
        return F.softplus(self.s_raw)


def solve2(glob, Ms, z, x0, obs_steps):
    """Ms (G, 3, N, D) per-channel map; z (S, G, M, 2D); x0 (S, G, M, 2) -> X at obs steps (m, G, S, M, 2)."""
    S, G, M, D2 = z.shape
    D = D2 // 2; Nn = Ms.shape[2]
    R = torch.stack([torch.einsum("gcnd,sgmd->cngsm", Ms, z[..., c * D:(c + 1) * D]) for c in range(2)], -1)
    R = R.reshape(3, Nn, G, S * M, 2)
    sig = glob.sigma()[:, None, :]
    x = x0.permute(1, 0, 2, 3).reshape(G, S * M, 2)
    h = T / Nn
    out = []; obs_set = set(int(o) for o in obs_steps)
    for n in range(Nn):
        rl, rm, rr = R[0, n], R[1, n], R[2, n]
        k1 = glob.net(x) + sig * rl
        k2 = glob.net(x + 0.5 * h * k1) + sig * rm
        k3 = glob.net(x + 0.5 * h * k2) + sig * rm
        k4 = glob.net(x + h * k3) + sig * rr
        x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if n + 1 in obs_set:
            out.append(x)
    return torch.stack(out, 0).reshape(len(out), G, S, M, 2)


def loglik2(Y, X):
    """Y (G, M, m, 2), X (m, G, S, M, 2) -> (S, G, M)."""
    Yb = Y.permute(2, 0, 1, 3)[:, :, None]
    return (-0.5 * (Yb - X) ** 2 / S_OBS ** 2 - 0.5 * np.log(2 * np.pi * S_OBS ** 2)).sum((0, -1)).permute(1, 0, 2)


def train2(glob, loc, Y, Ms, S, steps, train_globals=True, lrs=LRS, log_every=200):
    dev = Y.device
    T_ = lambda v: torch.tensor(float(v), device=dev)
    groups = [{"params": list(loc.parameters()), "lr": T_(lrs["local"]), "base": lrs["local"]}]
    if train_globals:
        groups += [{"params": list(glob.net.parameters()), "lr": T_(lrs["net"]), "base": lrs["net"]},
                   {"params": [glob.s_raw], "lr": T_(lrs["sigma"]), "base": lrs["sigma"]}]
    opt = torch.optim.Adam(groups, capturable=True)
    out = {"elbo": torch.zeros(Ms.shape[0], device=dev)}

    def step():
        z, x0, kl = loc.sample(S)
        X = solve2(glob, Ms, z, x0, OBS)
        elbo = (loglik2(Y, X).mean(0) - kl).sum(-1)
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
def iwae2(glob, loc, Y, Ms, K=512, chunk=16):
    logw = []
    for _ in range(K // chunk):
        z, x0, _ = loc.sample(chunk)
        X = solve2(glob, Ms, z, x0, OBS)
        lprior = -0.5 * (z ** 2).sum(-1) - 0.5 * z.shape[-1] * LOG2PI - 0.5 * (x0 ** 2).sum(-1) - LOG2PI
        logw.append(loglik2(Y, X) + lprior - loc.log_q(z, x0))
    lw = torch.cat(logw, 0)
    return (torch.logsumexp(lw, 0) - np.log(lw.shape[0])).cpu().numpy()                 # (G, M)


def run(rep, Hs_true, seeds, steps=5000, refit=2000, K=512, save=True):
    dev = torch.device("cuda"); torch.manual_seed(12345)
    combos = list(itertools.product(Hs_true, seeds)); G = len(combos)
    data = [make_data2(H, s) for H, s in combos]
    Ytr = torch.as_tensor(np.stack([d[0] for d in data]), dtype=torch.float32, device=dev)      # (G, 48, 20, 2)
    Yte = torch.as_tensor(np.stack([d[1] for d in data]), dtype=torch.float32, device=dev)      # (G, 16, 20, 2)
    Hs = [H for H, _ in combos]
    Ms = nb.maps_tensor(rep, Hs, dev); D = Ms.shape[-1]
    glob = Globals2(G).to(dev)                                  # created first: same initial weights for every rep
    loc = Locals2(G, Ytr.shape[1], D, Ytr[:, :, 0, :]).to(dev)
    t0 = time.time()
    hist = train2(glob, loc, Ytr, Ms, 16, steps)
    torch.cuda.synchronize(); train_s = time.time() - t0
    for p in glob.parameters():
        p.requires_grad_(False)
    t1 = time.time()
    Mg = nb.maps_tensor("grid", Hs, dev)
    loc_te = Locals2(G, Yte.shape[1], Mg.shape[-1], Yte[:, :, 0, :]).to(dev)
    train2(glob, loc_te, Yte, Mg, 16, refit, train_globals=False)
    ll_grid = iwae2(glob, loc_te, Yte, Mg, K)
    Mt = nb.maps_tensor("twoscale", Hs, dev)
    loc_te2 = Locals2(G, Yte.shape[1], Mt.shape[-1], Yte[:, :, 0, :]).to(dev)
    train2(glob, loc_te2, Yte, Mt, 16, refit, train_globals=False)
    ll_ts = iwae2(glob, loc_te2, Yte, Mt, K)
    eval_s = time.time() - t1
    xs = np.linspace(-2, 2, 41)
    XX, YY = np.meshgrid(xs, xs)
    Pt = np.stack([XX.ravel(), YY.ravel()], -1)                                                # (1681, 2): x fastest
    with torch.no_grad():
        Ff = glob.net(torch.as_tensor(Pt, dtype=torch.float32, device=dev)[None].expand(G, -1, -1).contiguous()).cpu().numpy()
    Ft = f2(Pt)
    sig = glob.sigma().cpu().numpy()
    coarse = np.ix_(np.arange(0, 41, 2), np.arange(0, 41, 2))
    for gi, (H, s) in enumerate(combos):
        occ = data[gi][2]
        Hh, _, _ = np.histogram2d(occ[:, 0], occ[:, 1], bins=41, range=[[-2.05, 2.05], [-2.05, 2.05]])
        w = Hh.T.ravel(); w = w / w.sum()                                                      # w[k] for Pt[k] = (x_j, y_i)
        err2 = ((Ff[gi] - Ft) ** 2).sum(-1)
        res = {"rep": rep, "rep_args": nb.REPS[rep], "D_per_channel": int(D), "H_true": H, "seed": s, "steps": steps,
               "refit": refit, "K": K, "groups_in_job": [f"H={h:.2f}/seed{t}" for h, t in combos],
               "train_seconds_job": train_s, "eval_seconds_job": eval_s, "sigma": sig[gi].tolist(),
               "drift_rmse_occupation": float(np.sqrt((err2 * w).sum())),
               "drift_rmse_box": float(np.sqrt(err2.mean())),
               "test_loglik_exact_grid_law_mean": float(ll_grid[gi].mean()),
               "test_loglik_exact_grid_law_per_series": ll_grid[gi].tolist(),
               "test_loglik_exact_law_twoscale_coords_mean": float(ll_ts[gi].mean()),
               "test_loglik_exact_law_twoscale_coords_per_series": ll_ts[gi].tolist(),
               "field": {"x": xs[::2].tolist(), "true": Ft.reshape(41, 41, 2)[coarse].tolist(),
                         "fit": Ff[gi].reshape(41, 41, 2)[coarse].tolist(),
                         "occupation": (Hh.T / Hh.sum())[coarse].tolist()},
               "history_elbo_per_series": [[h[0], h[1 + gi]] for h in hist]}
        if save:
            with open(os.path.join(OUT, f"n8_{rep}_seed{s}_H{H:.2f}.json"), "w") as fh:
                json.dump(res, fh, indent=1)
        print(json.dumps({"rep": rep, "H_true": H, "seed": s, "D": int(D), "sigma": [round(v, 3) for v in res["sigma"]],
                          "drift_rmse": round(res["drift_rmse_occupation"], 3),
                          "ll_grid": round(res["test_loglik_exact_grid_law_mean"], 3),
                          "ll_twoscale": round(res["test_loglik_exact_law_twoscale_coords_mean"], 3)}), flush=True)
    print(json.dumps({"rep": rep, "groups": G, "train_s": round(train_s), "eval_s": round(eval_s)}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="+", default=REPS, choices=REPS)
    ap.add_argument("--H", type=float, nargs="+", default=[0.6, 0.75, 0.9])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--refit", type=int, default=2000)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    for r in a.reps:
        run(r, a.H, a.seeds, a.steps, a.refit, a.K, not a.no_save)
