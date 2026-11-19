"""N1 (first trained-network experiment): learn a neural drift, the noise scale and the Hurst index jointly from
sparse noisy observations of a double-well SDE driven by fBm, with cosine-coordinate variational inference.

Truth : dX = (X - X^3) dt + 0.5 dB_H, T = 5, X(0) ~ U(-1.5, 1.5), H_true in {0.6, 0.75, 0.9}
Data  : exact fBm (Davies-Harte, 8192 steps) + Heun; 20 observations per series (every 0.25), noise s = 0.05;
        48 training series, 16 test series; data seeds 0, 1, 2.  No cosine coordinates are used to make data.
Model : f_theta = MLP(1-32-32-1, tanh); sigma learned; q(xi_i) full Gaussian (J = 32) and q(x0_i) Gaussian per
        series; RK4 with 80 steps; S = 16; 6000 Adam steps with cosine decay to 5 %; common initial weights and
        common random numbers across groups (variance reduction for group differences).  Groups trained side by side in one batched run (independent parameters):
        fixed H in {0.5 (Brownian baseline), 0.6, 0.7, 0.8, 0.9}, oracle H_true, and H learned jointly (init 0.7).
Eval  : drift RMSE on [-1.5, 1.5] (uniform and occupation-weighted); sigma; profile-ELBO argmax over the H grid;
        learned H; test ELBO after re-fitting only the local posteriors (globals frozen).
Out   : work/results/n1/n1_seed{seed}_H{H}.json and the data sets work/results/n1/data_*.npz
"""
import os, sys, json, time, argparse
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from fnvi.cvi import rate_table_np
from fnvi.simulate import simulate
from fnvi.train_graph import h_bank, LatentSDE, Locals, graph_trainer, elbo_eval, cosine_factor

OUT = os.path.join(HERE, "..", "..", "results", "n1"); os.makedirs(OUT, exist_ok=True)
T, SIG_TRUE, S_OBS, N_OBS, M_TR, M_TE = 5.0, 0.5, 0.05, 20, 48, 16
J, N_STEPS, S_MC = 32, 80, 16
OBS_STEPS = np.arange(1, N_OBS + 1) * (N_STEPS // N_OBS)
HGRID = [0.5, 0.6, 0.7, 0.8, 0.9]
LRS = {"local": 2e-2, "net": 3e-3, "sigma": 1e-2, "H": 2e-2}


def f_true(x):
    return x - x ** 3


def make_data(H, seed):
    rng = np.random.default_rng(1000 + seed)
    M = M_TR + M_TE; Nf = 8192
    x0 = rng.uniform(-1.5, 1.5, M)
    X = simulate(f_true, lambda x: SIG_TRUE * np.ones_like(x), x0, T, Nf, H, rng)
    idx = np.arange(1, N_OBS + 1) * (Nf // N_OBS)
    Y = X[idx].T + S_OBS * rng.standard_normal((M, N_OBS))
    return Y[:M_TR], Y[M_TR:], x0, X[::8].ravel()


def run(H_true, seed, steps=6000, refit_steps=2000):
    torch.manual_seed(seed); dev = torch.device("cuda")
    Ytr, Yte, x0_true, occ = make_data(H_true, seed)
    np.savez_compressed(os.path.join(OUT, f"data_seed{seed}_H{H_true:.2f}.npz"), Ytr=Ytr, Yte=Yte, x0=x0_true)
    Hs_fixed = HGRID + [H_true]
    G = len(Hs_fixed) + 1
    tabs_fixed = torch.as_tensor(np.stack([rate_table_np(h, T, J, N_STEPS) for h in Hs_fixed]), dtype=torch.float32, device=dev)
    Hs, bank = h_bank(T, J, N_STEPS)
    bank = torch.as_tensor(bank, dtype=torch.float32, device=dev)
    model = LatentSDE(G).to(dev)
    Yt = torch.as_tensor(Ytr, dtype=torch.float32, device=dev)
    loc = Locals(G, M_TR, J, Yt[:, 0]).to(dev)
    t0 = time.time()
    replay, out = graph_trainer(model, loc, Yt, tabs_fixed, bank, Hs, T, N_STEPS, OBS_STEPS, S_OBS, S_MC, True, LRS)
    hist = []
    for it in range(steps - 3):
        if it % 50 == 0:
            out["set_lr_factor"](cosine_factor(it, steps))
        replay()
        if it % 100 == 0:
            hist.append([it] + (out["elbo"] / M_TR).cpu().tolist() + [float(model.H())])
    torch.cuda.synchronize(); train_s = time.time() - t0
    elbo_tr = elbo_eval(model, loc, Yt, tabs_fixed, bank, Hs, T, N_STEPS, OBS_STEPS, S_OBS, S=256, reps=16)
    # held-out: freeze globals, refit locals
    for p in model.parameters():
        p.requires_grad_(False)
    Ye = torch.as_tensor(Yte, dtype=torch.float32, device=dev)
    loc_te = Locals(G, M_TE, J, Ye[:, 0]).to(dev)
    replay_te, out_te = graph_trainer(model, loc_te, Ye, tabs_fixed, bank, Hs, T, N_STEPS, OBS_STEPS, S_OBS, S_MC, False, LRS, learn_H=False)
    for it in range(refit_steps - 3):
        if it % 50 == 0:
            out_te["set_lr_factor"](cosine_factor(it, refit_steps))
        replay_te()
    elbo_te = elbo_eval(model, loc_te, Ye, tabs_fixed, bank, Hs, T, N_STEPS, OBS_STEPS, S_OBS, S=256, reps=16)
    # drift evaluation
    xg = np.linspace(-1.5, 1.5, 301)
    with torch.no_grad():
        fg = model.net(torch.as_tensor(xg, dtype=torch.float32, device=dev)[None, :, None].expand(G, -1, 1))[..., 0].cpu().numpy()
    ft = f_true(xg)
    w_occ, edges = np.histogram(np.clip(occ, -1.5, 1.5), bins=300, range=(-1.5, 1.5), density=True)
    w_mid = np.interp(xg, 0.5 * (edges[1:] + edges[:-1]), w_occ); w_mid = w_mid / w_mid.sum()
    labels = [f"H={h:.2f}" for h in HGRID] + [f"oracle H={H_true:.2f}", "learned H"]
    res = {"H_true": H_true, "seed": seed, "J": J, "n_steps": N_STEPS, "S_mc": S_MC, "train_steps": steps,
           "train_seconds": train_s, "groups": labels, "H_learned": float(model.H()),
           "sigma": model.sigma().detach().cpu().tolist(), "sigma_true": SIG_TRUE,
           "drift_rmse_uniform": np.sqrt(((fg - ft) ** 2).mean(1)).tolist(),
           "drift_rmse_occupation": np.sqrt((((fg - ft) ** 2) * w_mid).sum(1)).tolist(),
           "elbo_train_per_series": elbo_tr.tolist(), "elbo_test_per_series": elbo_te.tolist(),
           "drift_curves": {"x": xg[::10].tolist(), "true": ft[::10].tolist(), "fit": fg[:, ::10].tolist()},
           "history": hist}
    eg = np.array(elbo_tr[:len(HGRID)]); i = int(np.argmax(eg))
    if 0 < i < len(HGRID) - 1:
        a, b, c = eg[i - 1], eg[i], eg[i + 1]
        res["H_profile"] = float(HGRID[i] + 0.5 * (a - c) / (a - 2 * b + c) * (HGRID[1] - HGRID[0]))
    else:
        res["H_profile"] = float(HGRID[i])
    with open(os.path.join(OUT, f"n1_seed{seed}_H{H_true:.2f}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    show = {k: res[k] for k in ("H_true", "seed", "train_seconds", "H_learned", "H_profile")}
    show["sigma"] = [round(v, 3) for v in res["sigma"]]
    show["drift_rmse_occ"] = [round(v, 3) for v in res["drift_rmse_occupation"]]
    show["elbo_test"] = [round(v, 2) for v in res["elbo_test_per_series"]]
    print(json.dumps(show), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=float, nargs="+", default=[0.6, 0.75, 0.9])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--refit", type=int, default=2000)
    a = ap.parse_args()
    for s in a.seeds:
        for H in a.H:
            run(H, s, a.steps, a.refit)
