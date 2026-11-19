"""N1b: which noise representation gives the best trained neural SDE, with everything else held fixed?

Data   : the N1 double-well data (fnvi/simulate.py: exact fBm + Heun; T = 5, 20 observations, s = 0.05,
         48 train / 16 test series), H_true in {0.6, 0.75, 0.9}, data seeds 0, 1.
Model  : drift MLP 1-32-32-1 (tanh), sigma learned, X(0) ~ N(0, 1) prior, q full Gaussian over the D coordinates
         of the representation and Gaussian over x0, RK4 with N = 80 steps, S = 16, 6000 Adam steps with cosine
         decay, common random numbers and common initial weights across groups.
Groups : H in {0.5, 0.6, 0.7, 0.8, 0.9} (profile) + H_true (oracle); one representation per run:
         cosine (J=32), twoscale (J=16 + residual), grid (exact, Cholesky), markov (published rule K=5, gamma_max=20,
         weights fitted on [0, 3T]), markov_tuned (K=8, gamma_max=1e4).
Metrics: drift RMSE (occupation weighted) per group; sigma; profile H from the training bound; and the COMMON
         yardstick: held-out log-likelihood of the 16 test series under the EXACT grid law at each group's
         (drift, sigma, H), by importance sampling (K = 512) after re-fitting test posteriors with globals frozen.
Out    : work/results/n1b/n1b_{rep}_seed{seed}_H{H}.json
"""
import os, sys, json, time, argparse
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import n1_drift_recovery as n1
from fnvi.noise_maps import build
from fnvi.train_maps import Globals, LocalsD, train, elbo_eval, iwae

OUT = os.path.join(HERE, "..", "..", "results", "n1b"); os.makedirs(OUT, exist_ok=True)
T, S_OBS, N = n1.T, n1.S_OBS, 80
OBS = np.arange(1, n1.N_OBS + 1) * (N // n1.N_OBS)
HGRID = [0.5, 0.6, 0.7, 0.8, 0.9]
LRS = {"local": 2e-2, "net": 3e-3, "sigma": 1e-2}
REPS = {"cosine": {"J": 32}, "twoscale": {"J": 16}, "grid": {}, "markov": {"K": 5, "gamma_max": 20.0},
        "markov_tuned": {"K": 8, "gamma_max": 1e4},
        # added 14 Nov (fairness): best Markovian configuration found by the exact bridge check (asymmetric range with a
        # long history, gamma_max within the explicit-solver stability limit, standard fBm scale)
        "markov_best": {"K": 12, "gamma_max": 30.0, "gamma_min": 1e-6}}


def maps_tensor(rep, Hs, dev):
    kind = "markov" if rep.startswith("markov") else rep
    Ms = [np.stack(build(kind, h, T, N, **REPS[rep]), 0) for h in Hs]                  # (3, N, D) each
    return torch.as_tensor(np.stack(Ms, 0), dtype=torch.float32, device=dev)          # (G, 3, N, D)


def vertex(xs, ys, i):
    """Abscissa of the parabola through the three grid points around index i (unequal spacing allowed; for equal
    spacing it is the formula used before 16 Nov)."""
    x1, x2, x3 = xs[i - 1], xs[i], xs[i + 1]
    y1, y2, y3 = ys[i - 1], ys[i], ys[i + 1]
    num = (x2 - x1) ** 2 * (y2 - y3) - (x2 - x3) ** 2 * (y2 - y1)
    den = (x2 - x1) * (y2 - y3) - (x2 - x3) * (y2 - y1)
    return float(x2 - 0.5 * num / den)


def run(rep, H_true, seed, steps=6000, refit=2000, tag="", hgrid=None):
    dev = torch.device("cuda"); torch.manual_seed(seed)
    Ytr, Yte, _, occ = n1.make_data(H_true, seed)
    HG = list(hgrid) if hgrid else list(HGRID)
    Hs = HG + [H_true]; G = len(Hs)
    Ms = maps_tensor(rep, Hs, dev); D = Ms.shape[-1]
    glob = Globals(G).to(dev)
    Yt = torch.as_tensor(Ytr, dtype=torch.float32, device=dev)
    loc = LocalsD(G, Ytr.shape[0], D, Yt[:, 0]).to(dev)
    t0 = time.time()
    hist = train(glob, loc, Yt, Ms, T, OBS, S_OBS, 16, steps, True, LRS)
    torch.cuda.synchronize(); train_s = time.time() - t0
    elbo_tr = elbo_eval(glob, loc, Yt, Ms, T, OBS, S_OBS)
    for p in glob.parameters():
        p.requires_grad_(False)
    # common yardstick: exact grid law at each group's H, test posteriors refitted with globals frozen
    Mg = maps_tensor("grid", Hs, dev)
    Ye = torch.as_tensor(Yte, dtype=torch.float32, device=dev)
    loc_te = LocalsD(G, Yte.shape[0], N, Ye[:, 0]).to(dev)
    t1 = time.time()
    train(glob, loc_te, Ye, Mg, T, OBS, S_OBS, 16, refit, False, LRS)
    ll_exact = iwae(glob, loc_te, Ye, Mg, T, OBS, S_OBS)                               # (G, M_te)
    # second yardstick (added 14 Nov): same exact law, test posteriors refitted in the better-conditioned two-scale
    # coordinates (the grid-coordinate refit was found to fail at high H on long windows in N4 v1)
    Mt = maps_tensor("twoscale", Hs, dev)
    loc_te2 = LocalsD(G, Yte.shape[0], Mt.shape[-1], Ye[:, 0]).to(dev)
    train(glob, loc_te2, Ye, Mt, T, OBS, S_OBS, 16, refit, False, LRS)
    ll_exact_ts = iwae(glob, loc_te2, Ye, Mt, T, OBS, S_OBS)
    os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
    torch.save(glob.state_dict(), os.path.join(OUT, "models", f"n1b_{rep}_seed{seed}_H{H_true:.2f}{tag}.pt"))
    eval_s = time.time() - t1
    xg = np.linspace(-1.5, 1.5, 301)
    with torch.no_grad():
        fg = glob.net(torch.as_tensor(xg, dtype=torch.float32, device=dev)[None, :, None].expand(G, -1, 1))[..., 0].cpu().numpy()
    ft = n1.f_true(xg)
    w_occ, edges = np.histogram(np.clip(occ, -1.5, 1.5), bins=300, range=(-1.5, 1.5), density=True)
    w = np.interp(xg, 0.5 * (edges[1:] + edges[:-1]), w_occ); w = w / w.sum()
    eg = np.array(elbo_tr[:len(HG)]); i = int(np.argmax(eg))
    Hp = vertex(HG, eg, i) if 0 < i < len(HG) - 1 else float(HG[i])
    res = {"rep": rep, "rep_args": REPS[rep], "D": int(D), "H_true": H_true, "seed": seed, "steps": steps, "hgrid": HG,
           "H_profile_at_grid_edge": bool(i in (0, len(HG) - 1)),
           "train_seconds": train_s, "eval_seconds": eval_s, "groups": [f"H={h:.2f}" for h in Hs],
           "sigma": glob.sigma().detach().cpu().tolist(),
           "drift_rmse_occupation": np.sqrt((((fg - ft) ** 2) * w).sum(1)).tolist(),
           "elbo_train_own_model": elbo_tr.tolist(), "H_profile_own_model": Hp,
           "test_loglik_exact_grid_law_mean": ll_exact.mean(1).tolist(),
           "test_loglik_exact_grid_law_per_series": ll_exact.tolist(),
           "test_loglik_exact_law_twoscale_coords_mean": ll_exact_ts.mean(1).tolist(),
           "test_loglik_exact_law_twoscale_coords_per_series": ll_exact_ts.tolist(),
           "drift_curves": {"x": xg[::10].tolist(), "true": ft[::10].tolist(), "fit": fg[:, ::10].tolist()},
           "history": hist}
    with open(os.path.join(OUT, f"n1b_{rep}_seed{seed}_H{H_true:.2f}{tag}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({"rep": rep, "H_true": H_true, "seed": seed, "D": int(D), "train_s": round(train_s), "eval_s": round(eval_s),
                      "H_profile": round(Hp, 3), "sigma": [round(v, 3) for v in res["sigma"]],
                      "drift_rmse": [round(v, 3) for v in res["drift_rmse_occupation"]],
                      "test_ll_exact": [round(v, 2) for v in res["test_loglik_exact_grid_law_mean"]]}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="+", default=list(REPS))
    ap.add_argument("--H", type=float, nargs="+", default=[0.6, 0.75, 0.9])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--refit", type=int, default=2000)
    ap.add_argument("--tag", default="", help="suffix of the output file name (e.g. _steps10000 for the budget ablation)")
    ap.add_argument("--hgrid", type=float, nargs="+", default=None,
                    help="profile H grid (default {0.5,...,0.9}); X1 of 16 Nov uses 0.5 0.6 0.7 0.8 0.9 0.95")
    a = ap.parse_args()
    for s in a.seeds:
        for H in a.H:
            for rep in a.reps:
                run(rep, H, s, a.steps, a.refit, a.tag, a.hgrid)
