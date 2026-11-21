r"""N5 (cost per training step) and the GPU cost part of N7 (protocols fixed on 15 Nov 2026,
21:59, before any run).

Time per training step = median of 30 CUDA-graph replays after 5 warm-up replays (forward, backward, Adam update);
peak GPU memory; number of trainable parameters.  The cost does not depend on the values, so data are random, H = 0.75,
and the lift's nodes and weights in the timing harness are benign placeholders (1/K weights, diagonal start law).
  ours  : whitened-coordinate VI with d channels (same map per channel, q full Gaussian per series and channel),
          RK4 with the map's three stage rates; drift MLP d-32-32-d (N5) or fixed linear drift -X (N7)
  port  : the competitor's system as in fnvi/mafbm_torch.py (Euler-Heun, control network 2x1000 tanh with inputs
          [sin t, cos t, x, y] plus the series' observations as context in N5, K OU states per channel)
Sweeps:  --sweep n5 : (i) J, K at d = 1, T = 5; (ii) T in {1, 2, 5, 10, 20}; (iii) d in {1, 2, 4, 8};
                      plus the harness check against train_maps.graph_trainer and mafbm_torch.MAFbmSDE at d = 1.
         --sweep n7 : T in {2, 5, 10, 20, 50}, M = 1, S = 32, step 0.02, linear drift: cosine J in {16..256},
                      two-scale J = 16, grid, port K = 5 and 12.
Run from work\code\experiments:  python n5_scaling.py --sweep n5      (or --sweep n7; --quick for a smoke test)
Out: work/results/n5/n5_scaling.json, work/results/n7/n7_cost.json
"""
import os, sys, json, time, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from fnvi.noise_maps import cosine_map, twoscale_map, grid_map, markov_map
from fnvi.train_maps import LocalsD, Globals, graph_trainer
from fnvi.mafbm_torch import MAFbmSDE

RES = os.path.join(HERE, "..", "..", "results")
H_T, S_OBS = 0.75, 0.05
DEV = torch.device("cuda")


def time_graph(step, warm=5, reps=30):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    torch.cuda.synchronize(); cap = time.perf_counter() - t0
    for _ in range(warm):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t1 = time.perf_counter(); g.replay(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t1)
    return {"step_s": float(np.median(ts)), "step_s_q25_q75": [float(np.percentile(ts, 25)), float(np.percentile(ts, 75))],
            "capture_s": cap, "peak_mem_MB": torch.cuda.max_memory_allocated() / 2 ** 20}


class Drift(nn.Module):
    def __init__(self, d, kind):
        super().__init__()
        self.kind = kind
        if kind == "mlp":
            self.net = nn.Sequential(nn.Linear(d, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, d))

    def forward(self, x):
        return self.net(x) if self.kind == "mlp" else -x


def ours_map(rep, J, K, T, N):
    if rep == "cosine":
        M = cosine_map(H_T, T, N, J)
    elif rep == "twoscale":
        M = twoscale_map(H_T, T, N, 16)
    elif rep == "grid":
        M = grid_map(H_T, T, N)
    elif rep == "markov":
        M = markov_map(H_T, T, N, K, 20.0)
    else:
        raise ValueError(rep)
    return np.stack(M, 0)                                                   # (3, N, D)


def ours_step(rep, J, K, T, N, d, M, S, obs_every, drift_kind):
    Ms = torch.as_tensor(ours_map(rep, J, K, T, N), dtype=torch.float32, device=DEV)
    D = Ms.shape[-1]
    obs = np.arange(obs_every, N + 1, obs_every); obs_set = set(int(o) for o in obs)
    loc = LocalsD(1, M * d, D, torch.zeros(M * d, device=DEV)).to(DEV)
    drift = Drift(d, drift_kind).to(DEV)
    s_raw = nn.Parameter(torch.tensor(-1.0, device=DEV))
    Y = torch.randn(len(obs), M, d, device=DEV)
    params = list(loc.parameters()) + list(drift.parameters()) + [s_raw]
    opt = torch.optim.Adam(params, lr=1e-3, capturable=True)
    h = T / N

    def step():
        z, x0, kl, _, _ = loc.sample(S)                                      # z (S, 1, M*d, D)
        R = torch.einsum("cnd,sjd->cnsj", Ms, z[:, 0]).reshape(3, N, S * M, d)
        sig = F.softplus(s_raw)
        x = x0[:, 0].reshape(S * M, d)
        out = []
        for n in range(N):
            k1 = drift(x) + sig * R[0, n]
            k2 = drift(x + 0.5 * h * k1) + sig * R[1, n]
            k3 = drift(x + 0.5 * h * k2) + sig * R[1, n]
            k4 = drift(x + h * k3) + sig * R[2, n]
            x = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            if n + 1 in obs_set:
                out.append(x)
        X = torch.stack(out, 0).reshape(len(out), S, M, d)
        ll = (-0.5 * (Y[:, None] - X) ** 2 / S_OBS ** 2).sum((0, 3)).mean(0)   # (M,)
        loss = -(ll.sum() - kl.sum()) / M
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=False)
    return step, sum(p.numel() for p in params), int(D)


class PortD(nn.Module):
    """The port's computation (fnvi/mafbm_torch.py) with d channels; placeholder nodes and weights (cost only)."""
    def __init__(self, d, K, ctx_dim, drift_kind):
        super().__init__()
        g = np.geomspace(1 / 20.0, 20.0, K)
        self.register_buffer("gamma", torch.tensor(g, dtype=torch.float32))
        self.register_buffer("omega", torch.full((K,), 1.0 / K))
        self.register_buffer("l0", torch.tensor(1.0 / np.sqrt(2 * g), dtype=torch.float32))
        self.d, self.K = d, K
        self.ctrl = nn.Sequential(nn.Linear(2 + d + d * K + ctx_dim, 1000), nn.Tanh(), nn.Linear(1000, 1000), nn.Tanh(),
                                  nn.Linear(1000, d))
        self.drift = Drift(d, drift_kind)
        self.s_raw = nn.Parameter(torch.tensor(-1.0))

    def loss(self, B, T, N, obs_set, Y, ctx):
        h = T / N; d, K = self.d, self.K
        y = torch.randn(B, d, K, device=DEV) * self.l0
        x = torch.zeros(B, d, device=DEV)
        kl = torch.zeros(B, device=DEV)
        sig = F.softplus(self.s_raw); sw = self.omega.sum()
        out = []
        for n in range(N):
            tt = torch.full((B, 1), n * h, device=DEV)
            inp = [torch.sin(tt), torch.cos(tt), x, y.reshape(B, d * K)] + ([ctx] if ctx is not None else [])
            u = self.ctrl(torch.cat(inp, -1))                                  # (B, d)
            dW = math.sqrt(h) * torch.randn(B, d, device=DEV)
            dy = -self.gamma * y + u[..., None]                                # (B, d, K)
            fx = self.drift(x) + sig * (dy * self.omega).sum(-1)
            gx = sig * sw * torch.ones_like(x)
            x_pred = x + gx * dW
            gx2 = sig * sw * torch.ones_like(x_pred)
            x = x + fx * h + 0.5 * (gx + gx2) * dW
            y = y + dy * h + dW[..., None]
            kl = kl + 0.5 * (u ** 2).sum(-1) * h
            if n + 1 in obs_set:
                out.append(x)
        X = torch.stack(out, 0)                                                # (m, B, d)
        ll = (-0.5 * (Y - X) ** 2 / S_OBS ** 2).sum((0, 2))
        return -(ll - kl).mean()


def port_step(K, T, N, d, M, S, obs_every, drift_kind, context):
    obs = np.arange(obs_every, N + 1, obs_every); obs_set = set(int(o) for o in obs); m = len(obs)
    B = M * S
    model = PortD(d, K, m * d if context else 0, drift_kind).to(DEV)
    Y = torch.randn(m, B, d, device=DEV)
    ctx = torch.randn(B, m * d, device=DEV) if context else None
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, capturable=True)

    def step():
        loss = model.loss(B, T, N, obs_set, Y, ctx)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=False)
    return step, sum(p.numel() for p in model.parameters())


def harness_check():
    """At d = 1, T = 5: the harness beside the trainers used in the experiments."""
    out = {}
    T, N, M, S = 5.0, 80, 16, 16
    obs_steps = np.arange(4, N + 1, 4)
    Ms = torch.as_tensor(np.stack(cosine_map(H_T, T, N, 32), 0)[None], dtype=torch.float32, device=DEV)
    glob = Globals(1).to(DEV)
    Y = torch.randn(M, len(obs_steps), device=DEV)
    loc = LocalsD(1, M, 32, Y[:, 0]).to(DEV)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    replay, _, _ = graph_trainer(glob, loc, Y, Ms, T, obs_steps, S_OBS, S, True, {"local": 2e-2, "net": 3e-3, "sigma": 1e-2})
    for _ in range(5):
        replay()
    torch.cuda.synchronize(); ts = []
    for _ in range(30):
        t1 = time.perf_counter(); replay(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t1)
    out["train_maps_cosine32"] = float(np.median(ts))
    step, _, _ = ours_step("cosine", 32, 0, T, N, 1, M, S, 4, "mlp")
    out["harness_cosine32"] = time_graph(step)["step_s"]
    # the port class used in N1c / N3-port, with a drift MLP and the series' observations as context
    m = len(obs_steps); B = M * S
    drift = Drift(1, "mlp").to(DEV); s_raw = nn.Parameter(torch.tensor(-1.0, device=DEV))
    model = MAFbmSDE(H_T, K=5, b=lambda t, x: drift(x), s=lambda t, x: F.softplus(s_raw) * torch.ones_like(x),
                     ctx_dim=m).to(DEV)
    Yp = torch.randn(m, B, device=DEV); ctx = torch.randn(B, m, device=DEV)
    opt = torch.optim.Adam(list(model.parameters()) + list(drift.parameters()) + [s_raw], lr=1e-3, capturable=True)

    def pstep():
        X, kl = model(B, T, N, obs_steps, ctx)
        loss = -((-0.5 * (Yp - X) ** 2 / S_OBS ** 2).sum(0) - kl).mean()
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=False)
    out["mafbm_torch_K5"] = time_graph(pstep)["step_s"]
    step, _ = port_step(5, T, N, 1, M, S, 4, "mlp", True)
    out["harness_port_K5"] = time_graph(step)["step_s"]
    out["ratio_ours"] = out["harness_cosine32"] / out["train_maps_cosine32"]
    out["ratio_port"] = out["harness_port_K5"] / out["mafbm_torch_K5"]
    out["within_25pct"] = bool(abs(out["ratio_ours"] - 1) <= 0.25 and abs(out["ratio_port"] - 1) <= 0.25)
    return out


def record(rows, tag, **kw):
    t0 = time.time()
    try:
        if kw["method"] == "ours":
            step, npar, D = ours_step(kw["rep"], kw.get("J", 0), kw.get("K", 0), kw["T"], kw["N"], kw["d"], kw["M"],
                                      kw["S"], kw["obs_every"], kw["drift"])
        else:
            step, npar = port_step(kw["K"], kw["T"], kw["N"], kw["d"], kw["M"], kw["S"], kw["obs_every"], kw["drift"],
                                   kw["context"]); D = None
        r = dict(kw, sweep=tag, params=int(npar), D=D, **time_graph(step))
    except torch.cuda.OutOfMemoryError as e:
        r = dict(kw, sweep=tag, error=f"out of memory: {str(e)[:120]}")
    r["wall_s"] = time.time() - t0
    rows.append(r)
    torch.cuda.empty_cache()
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                      if k in ("sweep", "method", "rep", "J", "K", "T", "d", "D", "step_s", "peak_mem_MB", "error")}), flush=True)


def slope(xs, ys):
    x, y = np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float))
    if len(x) < 2:
        return None
    A = np.vstack([x, np.ones_like(x)]).T
    c, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ c; r2 = 1 - ((y - yhat) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-300)
    return {"slope": float(c[0]), "r2": float(r2)}


def sweep_n5(quick):
    rows = []
    base = dict(d=1, M=16, S=16, drift="mlp")
    T5 = dict(T=5.0, N=80, obs_every=4)
    Js = [8, 32] if quick else [8, 16, 32, 64, 128, 256]
    Ks = [5, 12] if quick else [5, 8, 12, 16, 32]
    for J in Js:
        record(rows, "i", method="ours", rep="cosine", J=J, **base, **T5)
    for rep in ["twoscale", "grid"]:
        record(rows, "i", method="ours", rep=rep, **base, **T5)
    for K in Ks:
        record(rows, "i", method="ours", rep="markov", K=K, **base, **T5)
        record(rows, "i", method="port", K=K, context=True, **base, **T5)
    Ts = [1.0, 2.0] if quick else [1.0, 2.0, 5.0, 10.0, 20.0]
    for T in Ts:
        tt = dict(T=T, N=int(16 * T), obs_every=4)
        record(rows, "ii", method="ours", rep="cosine", J=32, **base, **tt)
        for rep in ["twoscale", "grid"]:
            record(rows, "ii", method="ours", rep=rep, **base, **tt)
        record(rows, "ii", method="ours", rep="markov", K=5, **base, **tt)
        for K in [5, 12]:
            record(rows, "ii", method="port", K=K, context=True, **base, **tt)
    ds = [1, 2] if quick else [1, 2, 4, 8]
    for d in ds:
        b = dict(base, d=d)
        record(rows, "iii", method="ours", rep="cosine", J=32, **b, **T5)
        record(rows, "iii", method="ours", rep="twoscale", **b, **T5)
        record(rows, "iii", method="port", K=5, context=True, **b, **T5)
    ok = lambda r: "step_s" in r
    fits = {
        "cosine_vs_J": slope(*zip(*[(r["J"], r["step_s"]) for r in rows if ok(r) and r["sweep"] == "i" and r.get("rep") == "cosine"])),
        "lift_rep_vs_K": slope(*zip(*[(r["K"], r["step_s"]) for r in rows if ok(r) and r["sweep"] == "i" and r.get("rep") == "markov"])),
        "port_vs_K": slope(*zip(*[(r["K"], r["step_s"]) for r in rows if ok(r) and r["sweep"] == "i" and r["method"] == "port"]))}
    for lab, sel in [("cosine32", lambda r: r.get("rep") == "cosine"), ("twoscale", lambda r: r.get("rep") == "twoscale"),
                     ("grid", lambda r: r.get("rep") == "grid"), ("lift_rep_K5", lambda r: r.get("rep") == "markov"),
                     ("port_K5", lambda r: r["method"] == "port" and r["K"] == 5),
                     ("port_K12", lambda r: r["method"] == "port" and r["K"] == 12)]:
        pts = [(r["T"], r["step_s"]) for r in rows if ok(r) and r["sweep"] == "ii" and sel(r)]
        fits[f"{lab}_vs_T"] = slope(*zip(*pts)) if len(pts) > 1 else None
    for lab, sel in [("cosine32", lambda r: r.get("rep") == "cosine"), ("twoscale", lambda r: r.get("rep") == "twoscale"),
                     ("port_K5", lambda r: r["method"] == "port")]:
        pts = [(r["d"], r["step_s"]) for r in rows if ok(r) and r["sweep"] == "iii" and sel(r)]
        fits[f"{lab}_vs_d"] = slope(*zip(*pts)) if len(pts) > 1 else None
    return {"rows": rows, "fits": fits, "harness_check": harness_check()}


def sweep_n7(quick):
    rows = []
    Ts = [2.0, 5.0] if quick else [2.0, 5.0, 10.0, 20.0, 50.0]
    Js = [16, 64] if quick else [16, 32, 64, 128, 256]
    for T in Ts:
        tt = dict(T=T, N=int(round(50 * T)), obs_every=10, d=1, M=1, S=32, drift="linear")
        for J in Js:
            record(rows, "n7", method="ours", rep="cosine", J=J, **tt)
        for rep in ["twoscale", "grid"]:
            record(rows, "n7", method="ours", rep=rep, **tt)
        for K in [5, 12]:
            record(rows, "n7", method="port", K=K, context=False, **tt)
    return {"rows": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", choices=["n5", "n7"], required=True)
    ap.add_argument("--quick", action="store_true", help="smoke test: small sweep, nothing written")
    a = ap.parse_args()
    torch.manual_seed(0)
    info = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "started": time.strftime("%Y-%m-%d %H:%M"),
            "protocol": "fixed 15 Nov 2026 21:59, before the runs"}
    res = sweep_n5(a.quick) if a.sweep == "n5" else sweep_n7(a.quick)
    res["info"] = info
    if a.sweep == "n5":
        print(json.dumps({"fits": res["fits"], "harness_check": res["harness_check"]}, indent=1), flush=True)
    if not a.quick:
        d = os.path.join(RES, "n5" if a.sweep == "n5" else "n7"); os.makedirs(d, exist_ok=True)
        fn = os.path.join(d, "n5_scaling.json" if a.sweep == "n5" else "n7_cost.json")
        with open(fn, "w") as fh:
            json.dump(res, fh, indent=1)
        print("wrote", fn)
