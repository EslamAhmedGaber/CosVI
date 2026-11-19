"""N4 v2 (real data, Nile minima at the Roda gauge, Cairo, 622-1284), redesigned on 14 Nov after N4 v1.

Why v2 (v1 results kept in results/n4/n4_*_s0.1.json and n4_run.log):
 1. v1 used cosine coordinates with J = 32 on 50-year windows of yearly data, violating our own resolution rule
    (G2b: J must exceed about T/delta = 50): the cosine fit was poor for that reason (own bound about -290 per window
    against about -63 for exact representations). v2 uses J = 64 and N = 200 RK4 steps (h = 0.25 year).
 2. v1's yardstick refitted the test posteriors in grid (Cholesky-innovation) coordinates; at H >= 0.8 the refit failed
    (two-scale: own exact-law training bound -63 per window, but test estimate -390), so the measurement, not the model,
    was at fault. v2 keeps the SAME measured quantity (held-out log-likelihood under the exact grid law at the fitted
    drift, sigma, H) but refits the test posteriors in the two-scale coordinates (same law, better conditioned) with 4000
    steps and K = 1024 importance samples (primary), and still reports the grid-coordinate refit (secondary).
 3. Fitted globals are saved (results/n4/models) so that any later re-evaluation needs no retraining.
Unchanged from v1: data, split (windows 1-10 train, 11-13 test), standardisation on training years, s = 0.1, drift MLP
1-32-32-1, sigma learned, X(0) ~ N(0, 1), full Gaussian q, S = 16, 5000 training steps with cosine decay, H grid
{0.5, ..., 0.9}, seeds 0-2, success criterion (same direction as N1b, or task-specific).
Out: results/n4/n4v2_{rep}_seed{seed}_s{s}.json
"""
import os, sys, json, time, argparse
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from fnvi.noise_maps import build
from fnvi.train_maps import Globals, LocalsD, train, elbo_eval, iwae
import n4_nile as v1

OUT = v1.OUT
T, N = 50.0, 200
OBS = np.arange(1, 51) * (N // 50)                 # every year = every 4 steps
HGRID = [0.5, 0.6, 0.7, 0.8, 0.9]
LRS = {"local": 2e-2, "net": 3e-3, "sigma": 1e-2}
REPS = {"cosine64": ("cosine", {"J": 64}), "twoscale": ("twoscale", {"J": 16}), "grid": ("grid", {}),
        "markov": ("markov", {"K": 5, "gamma_max": 20.0}), "markov_tuned": ("markov", {"K": 8, "gamma_max": 1e4})}


def maps(rep, Hs, dev):
    kind, kw = REPS[rep]
    Ms = [np.stack(build(kind, h, T, N, **kw), 0) for h in Hs]
    return torch.as_tensor(np.stack(Ms, 0), dtype=torch.float32, device=dev)


def yardstick(glob, Yte, x0te, rep_coords, Hs, s_obs, steps, K, dev):
    Mm = maps(rep_coords, Hs, dev)
    Ye = torch.as_tensor(Yte, dtype=torch.float32, device=dev)
    loc = LocalsD(len(Hs), Yte.shape[0], Mm.shape[-1], torch.as_tensor(x0te, dtype=torch.float32, device=dev)).to(dev)
    hist = train(glob, loc, Ye, Mm, T, OBS, s_obs, 16, steps, False, LRS)
    own = elbo_eval(glob, loc, Ye, Mm, T, OBS, s_obs)
    return iwae(glob, loc, Ye, Mm, T, OBS, s_obs, K=K), own


def run(rep, seed, s_obs=0.1, steps=5000, ys_steps=(4000, 2000), ys_K=(1024, 512), save=True):
    dev = torch.device("cuda"); torch.manual_seed(seed)
    Ytr, Yte, x0tr, x0te, (mu, sd) = v1.load_windows()
    Hs = HGRID; G = len(Hs)
    Ms = maps(rep, Hs, dev); D = Ms.shape[-1]
    glob = Globals(G).to(dev)
    Yt = torch.as_tensor(Ytr, dtype=torch.float32, device=dev)
    loc = LocalsD(G, Ytr.shape[0], D, torch.as_tensor(x0tr, dtype=torch.float32, device=dev)).to(dev)
    t0 = time.time()
    hist = train(glob, loc, Yt, Ms, T, OBS, s_obs, 16, steps, True, LRS)
    torch.cuda.synchronize(); train_s = time.time() - t0
    elbo_tr = elbo_eval(glob, loc, Yt, Ms, T, OBS, s_obs)
    for p in glob.parameters():
        p.requires_grad_(False)
    os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
    torch.save(glob.state_dict(), os.path.join(OUT, "models", f"n4v2_{rep}_seed{seed}_s{s_obs}.pt"))
    t1 = time.time()
    ll_ts, refit_bound_ts = yardstick(glob, Yte, x0te, "twoscale", Hs, s_obs, ys_steps[0], ys_K[0], dev)
    ll_gr, refit_bound_gr = yardstick(glob, Yte, x0te, "grid", Hs, s_obs, ys_steps[1], ys_K[1], dev)
    eval_s = time.time() - t1
    eg = np.array(elbo_tr); finite = np.isfinite(eg); i = int(np.nanargmax(np.where(finite, eg, -np.inf)))
    if 0 < i < G - 1 and finite[i - 1] and finite[i + 1]:
        a, b, c = eg[i - 1], eg[i], eg[i + 1]; Hp = float(Hs[i] + 0.5 * (a - c) / (a - 2 * b + c) * (Hs[1] - Hs[0]))
    else:
        Hp = float(Hs[i])
    xg = np.linspace(-3, 3, 121)
    with torch.no_grad():
        fg = glob.net(torch.as_tensor(xg, dtype=torch.float32, device=dev)[None, :, None].expand(G, -1, 1))[..., 0].cpu().numpy()
    res = {"version": "v2", "rep": rep, "rep_args": REPS[rep][1], "D": int(D), "N": N, "seed": seed, "s_obs": s_obs,
           "standardisation": [mu, sd], "train_seconds": train_s, "eval_seconds": eval_s, "groups": [f"H={h:.2f}" for h in Hs],
           "sigma": glob.sigma().detach().cpu().tolist(), "elbo_train_per_window": [float(v) for v in elbo_tr], "H_profile": Hp,
           "test_loglik_exact_law_twoscale_coords_mean": ll_ts.mean(1).tolist(), "test_loglik_exact_law_twoscale_coords_per_window": ll_ts.tolist(),
           "test_refit_bound_twoscale_coords": [float(v) for v in refit_bound_ts],
           "test_loglik_exact_law_grid_coords_mean": ll_gr.mean(1).tolist(), "test_refit_bound_grid_coords": [float(v) for v in refit_bound_gr],
           "drift": {"x": xg.tolist(), "fit": fg.tolist()}, "history": hist}
    if save:
        with open(os.path.join(OUT, f"n4v2_{rep}_seed{seed}_s{s_obs}.json"), "w") as fh:
            json.dump(res, fh, indent=1)
    print(json.dumps({"rep": rep, "seed": seed, "D": int(D), "train_s": round(train_s), "H_profile": round(Hp, 3),
                      "elbo_train": [round(v, 2) for v in res["elbo_train_per_window"]],
                      "test_ts": [round(v, 2) for v in res["test_loglik_exact_law_twoscale_coords_mean"]],
                      "test_grid": [round(v, 2) for v in res["test_loglik_exact_law_grid_coords_mean"]]}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="+", default=list(REPS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--s", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=5000)
    a = ap.parse_args()
    for sd_ in a.seeds:
        for rep in a.reps:
            run(rep, sd_, a.s, a.steps)
