"""N1c: the competitor's full system (Daems et al.: type-I Markovian lift + Girsanov control posterior + joint
gradient learning of H) on the N1 data, through the validated PyTorch port, with the same exact yardstick as N1b.

Adaptation (stated in the paper): the control network receives the series' 20 observations as context (their video
model receives an encoder context the same way); q(x0) per series Gaussian with prior N(0, 1) (as in our method);
prior drift = MLP 1-32-32-1 (same as N1b); sigma learned; control MLP 2x200 (their video-model width); K = 5,
gamma_max = 20, omega on horizon 3T; dt = T/400; H learned jointly through omega(H) (finite-difference gradient of the
ported closed form) — their published procedure — and, separately, fixed at H_true (oracle).
Evaluation: drift RMSE (occupation weighted), learned H, sigma, and the held-out log-likelihood of the test series
under the EXACT grid law at the fitted (drift, sigma, H), exactly as in n1b_representations.py.
Out: work/results/n1c/n1c_{mode}_seed{seed}_H{H}.json   (mode = joint or oracle)
Run only after competitor/g0_port_validation.py agrees with the authors' own run.
"""
import os, sys, json, time, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import n1_drift_recovery as n1
import n1b_representations as n1b
from fnvi.mafbm_torch import gamma_by_gamma_max, omega_type1
from fnvi.train_graph import GroupMLP, cosine_factor
from fnvi.train_maps import Globals, LocalsD, train, iwae

OUT = os.path.join(HERE, "..", "..", "results", "n1c"); os.makedirs(OUT, exist_ok=True)
T, S_OBS = n1.T, n1.S_OBS
NS = 400                                          # SDE steps (dt = 0.0125; gamma_max * dt = 0.25 < 0.5)
OBS = np.arange(1, n1.N_OBS + 1) * (NS // n1.N_OBS)
K, GMAX = 5, 20.0


class _Omega(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H, gamma_np, horizon):
        h = float(H); dH = 1e-4
        w = omega_type1(gamma_np, h, horizon)
        dw = (omega_type1(gamma_np, min(h + dH, 0.995), horizon) - omega_type1(gamma_np, max(h - dH, 0.505), horizon)) / (2 * dH)
        ctx.save_for_backward(torch.as_tensor(dw, dtype=H.dtype, device=H.device))
        return torch.as_tensor(w, dtype=H.dtype, device=H.device)

    @staticmethod
    def backward(ctx, g):
        (dw,) = ctx.saved_tensors
        return (g * dw).sum().reshape(1), None, None


class Competitor(nn.Module):
    def __init__(self, H_init, learn_H, M, width=200, ctx_dim=20, config="published"):
        super().__init__()
        self.config = config
        if config == "best":      # 14 Nov fairness: 12 nodes on [1e-6, 30] (stable at dt = T/400), standard-fBm scale
            g = np.geomspace(1e-6, 30.0, 12)
        else:
            g = gamma_by_gamma_max(K, GMAX)
        self.g_np = g; self.K = len(g)
        self.register_buffer("gamma", torch.tensor(g, dtype=torch.float32))
        self.register_buffer("L0", torch.tensor(np.linalg.cholesky(1.0 / (g[:, None] + g[None, :])), dtype=torch.float32))
        self.learn_H = learn_H
        self.h_raw = nn.Parameter(torch.tensor([math.log((H_init - 0.5) / (0.995 - H_init))]), requires_grad=learn_H)
        self.drift = GroupMLP(1, 1, 32, 2, 1)                                    # prior drift b(x)
        self.s_raw = nn.Parameter(torch.tensor(float(np.log(np.expm1(0.3)))))
        self.u1 = nn.Linear(2 + 1 + self.K + ctx_dim, width); self.u2 = nn.Linear(width, width); self.u3 = nn.Linear(width, 1)
        nn.init.zeros_(self.u3.weight); nn.init.zeros_(self.u3.bias)
        self.m0 = nn.Parameter(torch.zeros(M)); self.r0 = nn.Parameter(torch.full((M,), -2.0))
        # omega(H) bank (fix of 14 Nov: computing omega on the host inside a CUDA-graph capture is not allowed);
        # their closed form evaluated on H = 0.5, 0.5025, ..., 0.995 and linearly interpolated on the device
        self.Hb0, self.dHb = 0.5, 0.0025
        Hb = np.round(np.arange(0.5, 0.995 + 1e-9, 0.0025), 6)
        if config == "best":
            from fnvi.ma_fbm import MAFit, c_mvn
            cache = os.path.join(HERE, "..", "..", "results", "cache", f"omega_bank_best_T{T}.npy")
            if os.path.exists(cache):
                bank = np.load(cache)
            else:
                bank = np.stack([MAFit(h, 3 * T, g, "I").w / np.sqrt(c_mvn(h)) for h in Hb]); np.save(cache, bank)
        else:
            bank = np.stack([omega_type1(g, h, 3 * T) for h in Hb])
        self.register_buffer("omega_bank", torch.tensor(bank, dtype=torch.float32))

    def omega(self):
        u = (self.H() - self.Hb0) / self.dHb
        k = torch.clamp(torch.floor(u), 0, self.omega_bank.shape[0] - 2)
        w = u - k; ki = k.long()
        return ((1 - w) * self.omega_bank.index_select(0, ki) + w * self.omega_bank.index_select(0, ki + 1))[0]

    def H(self):
        return 0.5 + 0.495 * torch.sigmoid(self.h_raw)

    def forward(self, Y, S):
        """Y (M, m) observations; returns per-sample (log-lik, KL) arrays of shape (S, M)."""
        M = Y.shape[0]; dev = Y.device; h = T / NS
        w = self.omega()
        sw = w.sum(); sig = F.softplus(self.s_raw)
        ctx = Y.repeat(S, 1)                                                    # (S*M, 20)
        x = (self.m0 + F.softplus(self.r0) * torch.randn(S, M, device=dev)).reshape(S * M, 1)
        v0 = F.softplus(self.r0) ** 2
        kl0 = (0.5 * (v0 + self.m0 ** 2 - 1 - torch.log(v0)))                    # (M,)
        y = torch.randn(S * M, self.K, device=dev) @ self.L0.T
        kl = torch.zeros(S * M, device=dev); out = []; obs_set = set(int(o) for o in OBS)
        for n in range(NS):
            t = n * h
            tt = torch.full((S * M, 1), t, device=dev)
            u = self.u3(torch.tanh(self.u2(torch.tanh(self.u1(torch.cat([torch.sin(tt), torch.cos(tt), x, y, ctx], -1))))))
            dW = math.sqrt(h) * torch.randn(S * M, 1, device=dev)
            dyd = -self.gamma * y + u
            fx = self.drift(x[None])[0] + sig * (dyd * w).sum(-1, keepdim=True)
            x = x + fx * h + sig * sw * dW                                       # additive noise: Euler-Heun = Euler
            y = y + dyd * h + dW
            kl = kl + 0.5 * u[:, 0] ** 2 * h
            if n + 1 in obs_set:
                out.append(x[:, 0])
        X = torch.stack(out, 0).reshape(len(out), S, M)                          # (m, S, M)
        ll = (-0.5 * (Y.T[:, None, :] - X) ** 2 / S_OBS ** 2 - 0.5 * math.log(2 * math.pi * S_OBS ** 2)).sum(0)
        return ll, kl.reshape(S, M) + kl0[None]


def run(mode, H_true, seed, steps=5000, S=4, config="published", lr_h=None):
    suffix = ("" if config == "published" else f"_{config}") + ("" if lr_h is None else f"_lrH{lr_h:g}")
    dev = torch.device("cuda"); torch.manual_seed(seed)
    Ytr, Yte, _, occ = n1.make_data(H_true, seed)
    Yt = torch.as_tensor(Ytr, dtype=torch.float32, device=dev)
    model = Competitor(0.7 if mode == "joint" else H_true, mode == "joint", Ytr.shape[0], config=config).to(dev)
    with torch.no_grad():
        model.m0.copy_(Yt[:, 0])
    others = [p for n_, p in model.named_parameters() if p.requires_grad and n_ != "h_raw"]
    groups = [{"params": others, "lr": torch.tensor(1e-3, device=dev), "base": 1e-3}]
    if model.h_raw.requires_grad:   # their code: one learning rate for everything; lr_h: ablation (14 Nov)
        b = 1e-3 if lr_h is None else lr_h
        groups.append({"params": [model.h_raw], "lr": torch.tensor(b, device=dev), "base": b})
    opt = torch.optim.Adam(groups, capturable=True)
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
                grp["lr"].fill_(grp["base"] * cosine_factor(it, steps))
        gph.replay()
        if it % 200 == 0:
            hist.append([it, float(out["loss"]), float(model.H())])
    torch.cuda.synchronize(); train_s = time.time() - t0
    # ---- exact yardstick (same as N1b): grid law at the fitted H, test posteriors refitted with globals frozen
    Hfit = float(model.H())
    glob = Globals(1).to(dev)
    with torch.no_grad():
        for a_, b_ in zip(glob.net.W, model.drift.W):
            a_.copy_(b_)
        for a_, b_ in zip(glob.net.b, model.drift.b):
            a_.copy_(b_)
        glob.s_raw.copy_(model.s_raw.reshape(1))
    for p in glob.parameters():
        p.requires_grad_(False)
    # the competitor's noise scale is the Mandelbrot-Van Ness scale; express sigma for standard fBm
    from fnvi.ma_fbm import c_mvn
    with torch.no_grad():
        sig_std = F.softplus(model.s_raw) * (math.sqrt(c_mvn(Hfit)) if config == "published" else 1.0)
        glob.s_raw.copy_(torch.log(torch.expm1(sig_std)).reshape(1))
    Mg = n1b.maps_tensor("grid", [Hfit], dev)
    Ye = torch.as_tensor(Yte, dtype=torch.float32, device=dev)
    loc_te = LocalsD(1, Yte.shape[0], n1b.N, Ye[:, 0]).to(dev)
    train(glob, loc_te, Ye, Mg, T, n1b.OBS, S_OBS, 16, 2000, False, n1b.LRS)
    ll_exact = iwae(glob, loc_te, Ye, Mg, T, n1b.OBS, S_OBS)[0]
    Mt = n1b.maps_tensor("twoscale", [Hfit], dev)                       # second yardstick, as in N1b (14 Nov)
    loc_te2 = LocalsD(1, Yte.shape[0], Mt.shape[-1], Ye[:, 0]).to(dev)
    train(glob, loc_te2, Ye, Mt, T, n1b.OBS, S_OBS, 16, 2000, False, n1b.LRS)
    ll_exact_ts = iwae(glob, loc_te2, Ye, Mt, T, n1b.OBS, S_OBS)[0]
    os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUT, "models", f"n1c_{mode}{suffix}_seed{seed}_H{H_true:.2f}.pt"))
    xg = np.linspace(-1.5, 1.5, 301)
    with torch.no_grad():
        fg = model.drift(torch.as_tensor(xg, dtype=torch.float32, device=dev)[None, :, None])[0, :, 0].cpu().numpy()
    ft = n1.f_true(xg)
    w_occ, edges = np.histogram(np.clip(occ, -1.5, 1.5), bins=300, range=(-1.5, 1.5), density=True)
    wm = np.interp(xg, 0.5 * (edges[1:] + edges[:-1]), w_occ); wm = wm / wm.sum()
    res = {"mode": mode, "config": config, "lr_h": lr_h, "H_true": H_true, "seed": seed, "steps": steps, "train_seconds": train_s, "H_fit": Hfit,
           "sigma_their_scale": float(F.softplus(model.s_raw)), "sigma_standard_fbm_scale": float(sig_std),
           "drift_rmse_occupation": float(np.sqrt((((fg - ft) ** 2) * wm).sum())),
           "test_loglik_exact_grid_law_mean": float(ll_exact.mean()), "test_loglik_exact_grid_law_per_series": ll_exact.tolist(),
           "test_loglik_exact_law_twoscale_coords_mean": float(ll_exact_ts.mean()),
           "test_loglik_exact_law_twoscale_coords_per_series": ll_exact_ts.tolist(),
           "drift_curve": {"x": xg[::10].tolist(), "fit": fg[::10].tolist()}, "history": hist,
           "adaptations": "control receives the 20 observations as context; q(x0) per series; drift MLP 1-32-32-1; control 2x200"}
    with open(os.path.join(OUT, f"n1c_{mode}{suffix}_seed{seed}_H{H_true:.2f}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()
                      if k in ("mode", "H_true", "seed", "train_seconds", "H_fit", "sigma_standard_fbm_scale",
                               "drift_rmse_occupation", "test_loglik_exact_grid_law_mean")}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["joint", "oracle"])
    ap.add_argument("--H", type=float, nargs="+", default=[0.6, 0.75, 0.9])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--config", default="published", choices=["published", "best"])
    ap.add_argument("--lr-h", type=float, default=None, help="separate learning rate for H (ablation); default: theirs")
    a = ap.parse_args()
    for s in a.seeds:
        for H in a.H:
            for m in a.modes:
                run(m, H, s, a.steps, config=a.config, lr_h=a.lr_h)
