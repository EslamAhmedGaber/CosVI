r"""X9: the evaluator check on the real series (05_EXPERIMENTS.md section 3b, protocol fixed 18 Nov 2026 15:20).

Saved globals of CosVI, CosVI-64, the exact grid and the MA-fBm laws (published, tuned), one job per series and
training seed, all five models side by side as groups, each at its profile group (grid value nearest its profile H).
Two evaluators with the primary settings of N4 v2 (4000 refit steps, K = 1024):
  smooth : CosVI's coordinates and solver (the N4 v2 / X3 primary yardstick)
  pl     : CosVI-PL coordinates with the exact grid's solver (piecewise-constant forcing within a step)
Per window: held-out log-likelihood estimate, effective sample size and largest normalised importance weight.
Run from work\code\experiments:  python -u x9_real_evaluators.py --series nile --seed 0
Out: work/results/x9/x9_{series}_seed{seed}{tag}.json
"""
import os, sys, json, time, argparse
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from fnvi.noise_maps import build
from fnvi.train_maps import Globals, LocalsD, train, solve, loglik
import n4_nile as v1
import n4v2_nile as v2
import n4t_temperature as t3

RES = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
OUT = os.path.join(RES, "x9"); os.makedirs(OUT, exist_ok=True)
REPS = ["twoscale", "cosine64", "grid", "markov", "markov_tuned"]
EVAL = (("smooth", ("twoscale", {"J": 16})), ("pl", ("twoscale_pl", {"J": 16})))


def series_setup(series):
    if series == "nile":
        hg = [0.5, 0.6, 0.7, 0.8, 0.9]
        Ytr, Yte, x0tr, x0te, _ = v1.load_windows()
        rec = lambda rep, s: os.path.join(RES, "n4", f"n4v2_{rep}_seed{s}_s0.1.json")
        mod = lambda rep, s: os.path.join(RES, "n4", "models", f"n4v2_{rep}_seed{s}_s0.1.pt")
        hkey = "H_profile"
    else:
        hg = list(t3.HGRID_T)
        Ytr, Yte, x0tr, x0te, _ = t3.load_windows_temp()
        rec = lambda rep, s: os.path.join(RES, "n4t", f"n4t_{rep}_seed{s}_s0.1.json")
        mod = lambda rep, s: os.path.join(RES, "n4t", "models", f"n4v2_{rep}_seed{s}_s0.1.pt")
        hkey = "H_profile_general"
    return hg, Yte, x0te, rec, mod, hkey


@torch.no_grad()
def iwae_weights(glob, loc, Y, Ms, T, obs_steps, s_obs, K=1024, chunk=64):
    lw = []
    D = Ms.shape[-1]
    for _ in range(K // chunk):
        z, x0, _, _, _ = loc.sample(chunk, common=False)
        X = solve(glob, Ms, z, x0, T, obs_steps)
        lp = loglik(Y, X, s_obs)
        lprior = -0.5 * (z ** 2).sum(-1) - 0.5 * D * np.log(2 * np.pi) - 0.5 * x0 ** 2 - 0.5 * np.log(2 * np.pi)
        lw.append(lp + lprior - loc.log_q(z, x0))
    lw = torch.cat(lw, 0).double()                                                # (K, G, M)
    ll = torch.logsumexp(lw, 0) - np.log(lw.shape[0])
    w = torch.softmax(lw, 0)
    ess = 1.0 / (w ** 2).sum(0)
    return ll.cpu().numpy(), ess.cpu().numpy(), w.max(0).values.cpu().numpy()


def run(series, seed, steps, K, tag):
    dev = torch.device("cuda")
    hg, Yte, x0te, rec, mod, hkey = series_setup(series)
    G = len(REPS)
    glob = Globals(G).to(dev)
    Hs, info = [], []
    with torch.no_grad():
        for g, rep in enumerate(REPS):
            r = json.load(open(rec(rep, seed)))
            gi = int(np.argmin(np.abs(np.array(hg) - r[hkey])))
            s = torch.load(mod(rep, seed), map_location=dev)
            for k, v in glob.state_dict().items():
                v[g].copy_(s[k][gi])
            Hs.append(hg[gi])
            info.append({"rep": rep, "H_profile": r[hkey], "group_H": hg[gi],
                         "stored_smooth_ll": r["test_loglik_exact_law_twoscale_coords_mean"][gi]})
    for p in glob.parameters():
        p.requires_grad_(False)
    Ye = torch.as_tensor(Yte, dtype=torch.float32, device=dev)
    res = {"rule": "05 3b X9", "series": series, "seed": seed, "steps": steps, "K": K, "models": info, "evaluators": {}}
    t0 = time.time()
    for i, (name, (kind, kw)) in enumerate(EVAL):
        Mm = torch.as_tensor(np.stack([np.stack(build(kind, h, v2.T, v2.N, **kw), 0) for h in Hs], 0), dtype=torch.float32, device=dev)
        torch.manual_seed(104729 * (seed + 1) + i)
        loc = LocalsD(G, Yte.shape[0], Mm.shape[-1], torch.as_tensor(x0te, dtype=torch.float32, device=dev)).to(dev)
        train(glob, loc, Ye, Mm, v2.T, v2.OBS, 0.1, 16, steps, False, v2.LRS)
        ll, ess, wmax = iwae_weights(glob, loc, Ye, Mm, v2.T, v2.OBS, 0.1, K=K)
        res["evaluators"][name] = {"mean": ll.mean(1).tolist(), "per_window": ll.tolist(), "ess": ess.tolist(), "max_weight": wmax.tolist()}
    res["eval_seconds"] = time.time() - t0
    with open(os.path.join(OUT, f"x9_{series}_seed{seed}{tag}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({"series": series, "seed": seed, "reps": REPS,
                      "stored_smooth": [round(m["stored_smooth_ll"], 2) for m in info],
                      "smooth": [round(v, 2) for v in res["evaluators"]["smooth"]["mean"]],
                      "pl": [round(v, 2) for v in res["evaluators"]["pl"]["mean"]],
                      "min_ess": [round(float(np.min(e)), 1) for e in res["evaluators"]["pl"]["ess"]]}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", choices=["nile", "temp"], nargs="+", required=True)
    ap.add_argument("--seed", type=int, nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    for sr in a.series:
        for s in a.seed:
            run(sr, s, a.steps, a.K, a.tag)
