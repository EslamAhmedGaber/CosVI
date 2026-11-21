r"""X8(b, c): coordinates versus solver on the N1b task (protocol fixed 18 Nov 2026,
14:35, before any X8 run).

Two evaluators, used for every model in both parts (oracle group H = H_true; test posteriors refitted with the
globals frozen, 2000 steps, then IWAE K = 512):
  smooth : CosVI's coordinates and CosVI's implemented solver (smooth cosine stage rates + piecewise-constant
           residual) = the two-scale yardstick of N1b
  pl     : CosVI's coordinates with the grid's piecewise-constant forcing (map twoscale_pl) = the exact grid law with
           the grid's likelihood map, refitted in well-conditioned coordinates
(b) --part b : saved N1b-protocol models (5000 steps) of the six representations at (H, seed): the base N1b model if
               saved, else the X1 model (_hg95); their oracle groups are evaluated side by side (no training).
(c) --part c : CosVI, CosVI-PL and the exact grid trained side by side at H = H_true (N1b settings, common initial
               network and random numbers); the grid's 80 coordinates are padded with 15 zero columns whose variational
               parameters get zero gradient (they stay at the prior; checked and recorded).
Run from work\code\experiments, e.g.  python -u x8_common_interp.py --part c --H 0.75 --seed 0
Out: work/results/x8/x8{b,c}_seed{seed}_H{H}{tag}.json (+ models for part c)
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

RES = os.path.join(HERE, "..", "..", "results")
OUT = os.path.join(RES, "x8"); os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
MODELS = os.path.join(RES, "n1b", "models")
T, S_OBS, N, J = n1.T, n1.S_OBS, 80, 16
D = J + N - 1
OBS = np.arange(1, n1.N_OBS + 1) * (N // n1.N_OBS)
LRS = {"local": 2e-2, "net": 3e-3, "sigma": 1e-2}
REPS_B = ["twoscale", "cosine", "grid", "markov", "markov_tuned", "markov_best"]
EVALUATORS = (("smooth", "twoscale"), ("pl", "twoscale_pl"))


def map_np(rep, H):
    return np.stack(build(rep, H, T, N, **({"J": J} if rep.startswith("twoscale") else {})), 0)   # (3, N, D_rep)


def pad(Mx, width):
    out = np.zeros(Mx.shape[:-1] + (width,)); out[..., :Mx.shape[-1]] = Mx
    return out


def evaluate(glob, Ye, H, seed, refit, K, dev):
    G = glob.sigma().shape[0]
    res = {}
    for i, (name, rep) in enumerate(EVALUATORS):
        Mt = torch.as_tensor(np.stack([map_np(rep, H)] * G, 0), dtype=torch.float32, device=dev)
        torch.manual_seed(7919 * (seed + 1) + i)
        loc = LocalsD(G, Ye.shape[0], Mt.shape[-1], Ye[:, 0]).to(dev)
        train(glob, loc, Ye, Mt, T, OBS, S_OBS, 16, refit, False, LRS)
        ll = iwae(glob, loc, Ye, Mt, T, OBS, S_OBS, K=K)                                  # (G, M_te)
        res[name] = {"mean": ll.mean(1).tolist(), "per_series": ll.tolist()}
    return res


@torch.no_grad()
def drift_rmse(glob, occ, dev):
    G = glob.sigma().shape[0]
    xg = np.linspace(-1.5, 1.5, 301)
    fg = glob.net(torch.as_tensor(xg, dtype=torch.float32, device=dev)[None, :, None].expand(G, -1, 1))[..., 0].cpu().numpy()
    ft = n1.f_true(xg)
    w_occ, edges = np.histogram(np.clip(occ, -1.5, 1.5), bins=300, range=(-1.5, 1.5), density=True)
    w = np.interp(xg, 0.5 * (edges[1:] + edges[:-1]), w_occ); w = w / w.sum()
    return np.sqrt((((fg - ft) ** 2) * w).sum(1)).tolist()


def model_file(rep, H, seed):
    for src in ("", "_hg95"):
        fn = os.path.join(MODELS, f"n1b_{rep}_seed{seed}_H{H:.2f}{src}.pt")
        if os.path.exists(fn):
            return fn, src or "base"
    return None, None


def part_b(H, seed, refit, K, tag):
    dev = torch.device("cuda")
    _, Yte, _, occ = n1.make_data(H, seed)
    found = [(rep,) + model_file(rep, H, seed) for rep in REPS_B]
    found = [f for f in found if f[1]]
    if not found:
        print("no saved models for", H, seed); return
    G = len(found)
    glob = Globals(G).to(dev)
    with torch.no_grad():
        for g, (rep, fn, src) in enumerate(found):
            s = torch.load(fn, map_location=dev)
            for k, v in glob.state_dict().items():
                v[g].copy_(s[k][-1])                                                     # oracle group = last group
    for p in glob.parameters():
        p.requires_grad_(False)
    Ye = torch.as_tensor(Yte, dtype=torch.float32, device=dev)
    t0 = time.time()
    ev = evaluate(glob, Ye, H, seed, refit, K, dev)
    res = {"part": "b", "rule": "X8(b)", "H_true": H, "seed": seed, "refit": refit, "K": K,
           "models": [{"rep": r, "file": os.path.basename(f), "source": s} for r, f, s in found],
           "sigma": glob.sigma().cpu().tolist(), "drift_rmse_occupation": drift_rmse(glob, occ, dev),
           "test_loglik": ev, "eval_seconds": time.time() - t0}
    with open(os.path.join(OUT, f"x8b_seed{seed}_H{H:.2f}{tag}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({"part": "b", "H": H, "seed": seed, "reps": [r for r, _, _ in found],
                      "smooth": [round(v, 2) for v in ev["smooth"]["mean"]], "pl": [round(v, 2) for v in ev["pl"]["mean"]],
                      "drift": [round(v, 3) for v in res["drift_rmse_occupation"]]}), flush=True)


def part_c(H, seed, steps, refit, K, tag):
    dev = torch.device("cuda"); torch.manual_seed(seed)
    Ytr, Yte, _, occ = n1.make_data(H, seed)
    arms = ["twoscale", "twoscale_pl", "grid"]
    Ms = torch.as_tensor(np.stack([map_np("twoscale", H), map_np("twoscale_pl", H), pad(map_np("grid", H), D)], 0),
                         dtype=torch.float32, device=dev)                                 # (3, 3, N, D)
    glob = Globals(3).to(dev)
    Yt = torch.as_tensor(Ytr, dtype=torch.float32, device=dev)
    loc = LocalsD(3, Ytr.shape[0], D, Yt[:, 0]).to(dev)
    mv = torch.ones(3, 1, D, device=dev); mv[2, :, N:] = 0                                # inert block of the grid group
    mo = torch.ones(3, 1, D, D, device=dev); mo[2, :, N:, :] = 0; mo[2, :, :, N:] = 0
    loc.mu.register_hook(lambda g: g * mv)
    loc.rho.register_hook(lambda g: g * mv)
    loc.off.register_hook(lambda g: g * mo)
    rho0 = float(loc.rho[2, 0, N].item())
    t0 = time.time()
    hist = train(glob, loc, Yt, Ms, T, OBS, S_OBS, 16, steps, True, LRS)
    torch.cuda.synchronize(); train_s = time.time() - t0
    elbo_tr = elbo_eval(glob, loc, Yt, Ms, T, OBS, S_OBS)
    with torch.no_grad():
        inert = {"max_abs_mu": float(loc.mu[2, :, N:].abs().max()), "max_abs_off": float(torch.cat(
                     [loc.off[2, :, N:, :].reshape(-1), loc.off[2, :, :, N:].reshape(-1)]).abs().max()),
                 "max_abs_rho_change": float((loc.rho[2, :, N:] - rho0).abs().max())}
    for p in glob.parameters():
        p.requires_grad_(False)
    torch.save(glob.state_dict(), os.path.join(OUT, "models", f"x8c_seed{seed}_H{H:.2f}{tag}.pt"))
    Ye = torch.as_tensor(Yte, dtype=torch.float32, device=dev)
    t1 = time.time()
    ev = evaluate(glob, Ye, H, seed, refit, K, dev)
    res = {"part": "c", "rule": "X8(c)", "H_true": H, "seed": seed, "arms": arms, "names": ["CosVI", "CosVI-PL", "exact grid"],
           "D": D, "steps": steps, "refit": refit, "K": K, "train_seconds": train_s, "eval_seconds": time.time() - t1,
           "sigma": glob.sigma().cpu().tolist(), "drift_rmse_occupation": drift_rmse(glob, occ, dev),
           "elbo_train_own_model": elbo_tr.tolist(), "inert_block_check": inert, "test_loglik": ev, "history": hist}
    with open(os.path.join(OUT, f"x8c_seed{seed}_H{H:.2f}{tag}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({"part": "c", "H": H, "seed": seed, "train_s": round(train_s), "inert": inert,
                      "smooth": [round(v, 2) for v in ev["smooth"]["mean"]], "pl": [round(v, 2) for v in ev["pl"]["mean"]],
                      "drift": [round(v, 3) for v in res["drift_rmse_occupation"]],
                      "elbo": [round(float(v), 2) for v in elbo_tr]}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["b", "c"], required=True)
    ap.add_argument("--H", type=float, nargs="+", required=True)
    ap.add_argument("--seed", type=int, nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--refit", type=int, default=2000)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    for s in a.seed:
        for H in a.H:
            if a.part == "b":
                part_b(H, s, a.refit, a.K, a.tag)
            else:
                part_c(H, s, a.steps, a.refit, a.K, a.tag)
