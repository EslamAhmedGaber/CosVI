"""N4c: the competitor's full system (validated port: type-I Markovian lift K = 5, gamma in [1/20, 20], weights on
[0, 3T]; Girsanov control network 2x200 with the window's 50 observations as context; joint H learning through
omega(H)) on the Nile windows of N4 v2, evaluated with the same exact-law yardstick (test posteriors refitted in
two-scale coordinates, 4000 steps, K = 1024; plus the grid-coordinate refit as secondary).

Time unit: a user of their code normalises time to O(1) (their video model: T = 2.4). We map 50 years to T_c = 2.5
(c = 20 years per unit), dt = 0.01 (250 steps; gamma_max * dt = 0.2 < 0.5, their stability check). fBm is
self-similar, so this is a change of units: a model fitted in tau = t / c with drift f_c and noise sigma_c corresponds in
years to f = f_c / c and sigma = sigma_c * c^(-H) (times sqrt(c_H) for their Mandelbrot-Van Ness scale), which is how
the fitted model is passed to the yardstick.
Modes: joint (their published procedure) and oracle (H fixed at the classical Whittle estimate 0.84, since the true
H is unknown). Seeds 0-2.  Out: results/n4/n4c_{mode}_seed{seed}.json
"""
import os, sys, json, time, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import n4_nile as v1
import n4v2_nile as v2
from fnvi.mafbm_torch import gamma_by_gamma_max, omega_type1
from fnvi.ma_fbm import c_mvn
from fnvi.train_graph import GroupMLP, cosine_factor
from fnvi.train_maps import Globals

OUT = v1.OUT
TC, NS, CU = 2.5, 250, 20.0                       # normalised horizon, steps, years per unit
OBS = np.arange(1, 51) * (NS // 50)
K, GMAX, S_OBS = 5, 20.0, 0.1


class CompetitorNile(nn.Module):
    def __init__(self, H_init, learn_H, M, width=200, ctx_dim=50):
        super().__init__()
        g = gamma_by_gamma_max(K, GMAX)
        self.register_buffer("gamma", torch.tensor(g, dtype=torch.float32))
        self.register_buffer("L0", torch.tensor(np.linalg.cholesky(1.0 / (g[:, None] + g[None, :])), dtype=torch.float32))
        self.h_raw = nn.Parameter(torch.tensor([math.log((H_init - 0.5) / (0.995 - H_init))]), requires_grad=learn_H)
        self.drift = GroupMLP(1, 1, 32, 2, 1)
        self.s_raw = nn.Parameter(torch.tensor(float(np.log(np.expm1(0.3)))))
        self.u1 = nn.Linear(2 + 1 + K + ctx_dim, width); self.u2 = nn.Linear(width, width); self.u3 = nn.Linear(width, 1)
        nn.init.zeros_(self.u3.weight); nn.init.zeros_(self.u3.bias)
        self.m0 = nn.Parameter(torch.zeros(M)); self.r0 = nn.Parameter(torch.full((M,), -2.0))
        Hb = np.round(np.arange(0.5, 0.995 + 1e-9, 0.0025), 6)
        self.register_buffer("omega_bank", torch.tensor(np.stack([omega_type1(g, h, 3 * TC) for h in Hb]), dtype=torch.float32))

    def H(self):
        return 0.5 + 0.495 * torch.sigmoid(self.h_raw)

    def omega(self):
        u = (self.H() - 0.5) / 0.0025
        k = torch.clamp(torch.floor(u), 0, self.omega_bank.shape[0] - 2); w = u - k; ki = k.long()
        return ((1 - w) * self.omega_bank.index_select(0, ki) + w * self.omega_bank.index_select(0, ki + 1))[0]

    def forward(self, Y, S):
        M = Y.shape[0]; dev = Y.device; h = TC / NS
        w = self.omega(); sw = w.sum(); sig = F.softplus(self.s_raw)
        ctx = Y.repeat(S, 1)
        x = (self.m0 + F.softplus(self.r0) * torch.randn(S, M, device=dev)).reshape(S * M, 1)
        v0 = F.softplus(self.r0) ** 2
        kl0 = 0.5 * (v0 + self.m0 ** 2 - 1 - torch.log(v0))
        y = torch.randn(S * M, K, device=dev) @ self.L0.T
        kl = torch.zeros(S * M, device=dev); out = []; obs_set = set(int(o) for o in OBS)
        for n in range(NS):
            tt = torch.full((S * M, 1), n * h, device=dev)
            u = self.u3(torch.tanh(self.u2(torch.tanh(self.u1(torch.cat([torch.sin(tt), torch.cos(tt), x, y, ctx], -1))))))
            dW = math.sqrt(h) * torch.randn(S * M, 1, device=dev)
            dyd = -self.gamma * y + u
            x = x + (self.drift(x[None])[0] + sig * (dyd * w).sum(-1, keepdim=True)) * h + sig * sw * dW
            y = y + dyd * h + dW
            kl = kl + 0.5 * u[:, 0] ** 2 * h
            if n + 1 in obs_set:
                out.append(x[:, 0])
        X = torch.stack(out, 0).reshape(len(out), S, M)
        ll = (-0.5 * (Y.T[:, None, :] - X) ** 2 / S_OBS ** 2 - 0.5 * math.log(2 * math.pi * S_OBS ** 2)).sum(0)
        return ll, kl.reshape(S, M) + kl0[None]


def run(mode, seed, steps=5000, S=4, ys=(4000, 2000), ysK=(1024, 512), save=True):
    dev = torch.device("cuda"); torch.manual_seed(seed)
    Ytr, Yte, x0tr, x0te, _ = v1.load_windows()
    Yt = torch.as_tensor(Ytr, dtype=torch.float32, device=dev)
    model = CompetitorNile(0.7 if mode == "joint" else 0.84, mode == "joint", Ytr.shape[0]).to(dev)
    with torch.no_grad():
        model.m0.copy_(torch.as_tensor(x0tr, dtype=torch.float32))
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=torch.tensor(1e-3, device=dev), capturable=True)
    out = {"loss": torch.zeros((), device=dev)}

    def step():
        ll, kl = model(Yt, S)
        loss = -(ll - kl).mean(0).sum() / Ytr.shape[0]
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=False)
        out["loss"].copy_(loss.detach())

    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(st)
    gph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gph):
        step()
    t0 = time.time(); hist = []
    for it in range(steps - 3):
        if it % 50 == 0:
            for grp in opt.param_groups:
                grp["lr"].fill_(1e-3 * cosine_factor(it, steps))
        gph.replay()
        if it % 200 == 0:
            hist.append([it, float(out["loss"]), float(model.H())])
    torch.cuda.synchronize(); train_s = time.time() - t0
    Hfit = float(model.H())
    # convert to years: drift / c, sigma * c^(-H) * sqrt(c_H) (their Mandelbrot-Van Ness scale -> standard fBm)
    glob = Globals(1).to(dev)
    with torch.no_grad():
        for a_, b_ in zip(glob.net.W, model.drift.W):
            a_.copy_(b_)
        for a_, b_ in zip(glob.net.b, model.drift.b):
            a_.copy_(b_)
        glob.net.W[-1].div_(CU); glob.net.b[-1].div_(CU)
        sig_years = F.softplus(model.s_raw) * CU ** (-Hfit) * math.sqrt(c_mvn(Hfit))
        glob.s_raw.copy_(torch.log(torch.expm1(sig_years)).reshape(1))
    for p in glob.parameters():
        p.requires_grad_(False)
    ll_ts, bound_ts = v2.yardstick(glob, Yte, x0te, "twoscale", [Hfit], S_OBS, ys[0], ysK[0], dev)
    ll_gr, bound_gr = v2.yardstick(glob, Yte, x0te, "grid", [Hfit], S_OBS, ys[1], ysK[1], dev)
    os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUT, "models", f"n4c_{mode}_seed{seed}.pt"))
    res = {"mode": mode, "seed": seed, "steps": steps, "train_seconds": train_s, "H_fit": Hfit,
           "sigma_years_standard_scale": float(sig_years), "time_unit_years": CU, "dt": TC / NS,
           "test_loglik_exact_law_twoscale_coords_mean": float(ll_ts.mean()), "test_loglik_exact_law_twoscale_coords_per_window": ll_ts[0].tolist(),
           "test_loglik_exact_law_grid_coords_mean": float(ll_gr.mean()), "history": hist,
           "adaptations": "time normalised (50 years -> 2.5); control gets the 50 observations as context; q(x0) per window; drift MLP 1-32-32-1; control 2x200"}
    if save:
        with open(os.path.join(OUT, f"n4c_{mode}_seed{seed}.json"), "w") as fh:
            json.dump(res, fh, indent=1)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()
                      if k in ("mode", "seed", "train_seconds", "H_fit", "sigma_years_standard_scale",
                               "test_loglik_exact_law_twoscale_coords_mean", "test_loglik_exact_law_grid_coords_mean")}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["joint", "oracle"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=5000)
    a = ap.parse_args()
    for s in a.seeds:
        for m in a.modes:
            run(m, s, a.steps)
