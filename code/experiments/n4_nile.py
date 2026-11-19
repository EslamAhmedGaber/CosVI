"""N4 (real data): a neural SDE with long-memory noise for the Nile minima at the Roda gauge, Cairo, 622-1284.

PROTOCOL (fixed 13 Nov 2026, before any run):
Data   : 663 yearly minima (longmemo 1.1-4; work/results/n4/data/nile_min.csv), standardised by the mean and sd of the
         TRAINING years only. Split into 13 consecutive windows of 51 years (622-1284 minus the last 0 years):
         windows 1-10 (622-1131) train, windows 11-13 (1132-1284) test. In each window, time runs over [0, 50] years;
         the first year initialises x0's posterior mean, the other 50 years are observations.
Model  : dX = f_theta(X) dt + sigma dB_H, y = X + eps, eps ~ N(0, s^2) with s = 0.1 (standardised units; a
         sensitivity check with s = 0.05 and 0.2 is reported separately); f_theta MLP 1-32-32-1 (tanh); sigma learned;
         X(0) ~ N(0, 1) prior; q full Gaussian over the coordinates of the representation and Gaussian over x0.
Noise  : the five representations of N1b (cosine J = 32; two-scale J = 16; exact grid; Markovian published rule K = 5,
         gamma in [1/20, 20] per year, weights on [0, 3T]; Markovian tuned K = 8, gamma_max = 1e4), N = 100 RK4 steps
         (h = 0.5 year), groups H in {0.5, 0.6, 0.7, 0.8, 0.9}.
Metrics: (1) held-out log-likelihood of the 3 test windows under the EXACT grid law at each group's fitted (drift,
         sigma, H), importance sampling K = 512 after re-fitting the test posteriors with globals frozen (as N1b);
         (2) profile H from the training bound, compared with classical estimates on the same years (Whittle-fGn 0.84,
         aggregated variance 0.85, R/S 0.86 on all 663 years); (3) the learned drift.
Success: same direction as N1b, or the paper says the result is task-specific. Seeds: training seeds 0, 1, 2.
Out    : work/results/n4/n4_{rep}_seed{seed}_s{s}.json
"""
import os, sys, json, time, argparse
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import n1b_representations as n1b
from fnvi.noise_maps import build
from fnvi.train_maps import Globals, LocalsD, train, elbo_eval, iwae

WORK = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(WORK, "results", "n4"); os.makedirs(OUT, exist_ok=True)
T, N, WIN = 50.0, 100, 51
OBS = np.arange(1, 51) * (N // 50)                 # every year = every 2 steps
HGRID = [0.5, 0.6, 0.7, 0.8, 0.9]
LRS = {"local": 2e-2, "net": 3e-3, "sigma": 1e-2}


def load_windows():
    d = np.loadtxt(os.path.join(OUT, "data", "nile_min.csv"), delimiter=",", skiprows=1)
    y = d[:, 1]
    W = [y[i * WIN:(i + 1) * WIN] for i in range(len(y) // WIN)]          # 13 windows of 51
    tr, te = np.array(W[:10]), np.array(W[10:13])
    mu, sd = tr.mean(), tr.std()
    tr, te = (tr - mu) / sd, (te - mu) / sd
    return tr[:, 1:], te[:, 1:], tr[:, 0], te[:, 0], (float(mu), float(sd))


def maps(rep, Hs, dev):
    kind = "markov" if rep.startswith("markov") else rep
    Ms = [np.stack(build(kind, h, T, N, **n1b.REPS[rep]), 0) for h in Hs]
    return torch.as_tensor(np.stack(Ms, 0), dtype=torch.float32, device=dev)


def run(rep, seed, s_obs=0.1, steps=5000, refit=2000):
    dev = torch.device("cuda"); torch.manual_seed(seed)
    Ytr, Yte, x0tr, x0te, (mu, sd) = load_windows()
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
    Mg = maps("grid", Hs, dev)
    Ye = torch.as_tensor(Yte, dtype=torch.float32, device=dev)
    loc_te = LocalsD(G, Yte.shape[0], N, torch.as_tensor(x0te, dtype=torch.float32, device=dev)).to(dev)
    train(glob, loc_te, Ye, Mg, T, OBS, s_obs, 16, refit, False, LRS)
    ll_exact = iwae(glob, loc_te, Ye, Mg, T, OBS, s_obs)
    eg = np.array(elbo_tr); i = int(np.argmax(eg))
    if 0 < i < G - 1:
        a, b, c = eg[i - 1], eg[i], eg[i + 1]; Hp = float(Hs[i] + 0.5 * (a - c) / (a - 2 * b + c) * (Hs[1] - Hs[0]))
    else:
        Hp = float(Hs[i])
    xg = np.linspace(-3, 3, 121)
    with torch.no_grad():
        fg = glob.net(torch.as_tensor(xg, dtype=torch.float32, device=dev)[None, :, None].expand(G, -1, 1))[..., 0].cpu().numpy()
    res = {"rep": rep, "rep_args": n1b.REPS[rep], "D": int(D), "seed": seed, "s_obs": s_obs, "standardisation": [mu, sd],
           "train_seconds": train_s, "groups": [f"H={h:.2f}" for h in Hs], "sigma": glob.sigma().detach().cpu().tolist(),
           "elbo_train_per_window": elbo_tr.tolist(), "H_profile": Hp,
           "test_loglik_exact_grid_law_mean": ll_exact.mean(1).tolist(), "test_loglik_exact_grid_law_per_window": ll_exact.tolist(),
           "drift": {"x": xg.tolist(), "fit": fg.tolist()}, "history": hist}
    with open(os.path.join(OUT, f"n4_{rep}_seed{seed}_s{s_obs}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({"rep": rep, "seed": seed, "s": s_obs, "D": int(D), "train_s": round(train_s), "H_profile": round(Hp, 3),
                      "sigma": [round(v, 3) for v in res["sigma"]], "elbo_train": [round(v, 2) for v in res["elbo_train_per_window"]],
                      "test_ll_exact": [round(v, 2) for v in res["test_loglik_exact_grid_law_mean"]]}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="+", default=list(n1b.REPS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--s", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=5000)
    a = ap.parse_args()
    for sd_ in a.seeds:
        for rep in a.reps:
            run(rep, sd_, a.s, a.steps)
