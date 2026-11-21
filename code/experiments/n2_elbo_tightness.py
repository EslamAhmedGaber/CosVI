"""N2: how tight is each representation's bound with respect to the SAME exact evidence? (protocol fixed on
15 Nov 2026, before the run)

Model : CosVI's trained N1b model (drift MLP 1-32-32-1 and sigma; all six groups H = 0.5..0.9 and H_true) of one
        data seed, frozen.  Data: the 16 N1b test series of that seed (T = 5, 20 observations, s = 0.05).
For each representation the variational posteriors are fitted with the model frozen (refit steps, S = 16); recorded:
  elbo_own   : the representation's bound (S = 256), per series, mean over series per group
  iwae_own   : IWAE (K = 512) under the representation's own law (its own model evidence)
Reference log p(y) of the exact model on the solver grid: IWAE in two-scale coordinates (exact law) at K = 512, 4096,
32768 (convergence check) and in grid coordinates at K = 32768.   gap = reference(K max, two-scale) - elbo_own.
Run from work\\code\\experiments:  python n2_elbo_tightness.py --H 0.75 --seed 1
Out: work/results/n2/n2_seed{seed}_H{H}.json
"""
import os, sys, json, time, argparse
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import n1_drift_recovery as n1
import n1b_representations as nb
from fnvi.noise_maps import build
from fnvi.train_maps import Globals, LocalsD, train, elbo_eval, iwae

OUT = os.path.join(HERE, "..", "..", "results", "n2"); os.makedirs(OUT, exist_ok=True)
MODELS = os.path.join(HERE, "..", "..", "results", "n1b", "models")
REPS = {"cosine16": ("cosine", {"J": 16}), "cosine32": ("cosine", {"J": 32}), "cosine64": ("cosine", {"J": 64}),
        "twoscale": ("twoscale", {"J": 16}), "grid": ("grid", {}),
        "markov": ("markov", nb.REPS["markov"]), "markov_tuned": ("markov", nb.REPS["markov_tuned"]),
        "markov_best": ("markov", nb.REPS["markov_best"])}


def maps(kind, args, Hs, dev):
    Ms = [np.stack(build(kind, h, nb.T, nb.N, **args), 0) for h in Hs]
    return torch.as_tensor(np.stack(Ms, 0), dtype=torch.float32, device=dev)


def fit(glob, Ye, Ms, refit, dev):
    loc = LocalsD(Ms.shape[0], Ye.shape[0], Ms.shape[-1], Ye[:, 0]).to(dev)
    train(glob, loc, Ye, Ms, nb.T, nb.OBS, nb.S_OBS, 16, refit, False, nb.LRS)
    return loc


def run(H_true, seed, refit=4000, Ks=(512, 4096, 32768), reps=None, save=True, model_rep="twoscale"):
    # model_rep (added 15 Nov 2026 21:29 for N2-x, protocol fixed before its run): which trained N1b model is frozen;
    # "twoscale" (CosVI) is the N2 protocol, "grid" the cross-check with the model trained under exact grid whitening.
    dev = torch.device("cuda"); torch.manual_seed(seed)
    _, Yte, _, _ = n1.make_data(H_true, seed)
    Hs = nb.HGRID + [H_true]; G = len(Hs)
    glob = Globals(G).to(dev)
    src = os.path.join(MODELS, f"n1b_{model_rep}_seed{seed}_H{H_true:.2f}.pt")
    glob.load_state_dict(torch.load(src, map_location=dev))
    for p in glob.parameters():
        p.requires_grad_(False)
    Ye = torch.as_tensor(Yte, dtype=torch.float32, device=dev)
    t0 = time.time(); per = {}; loc_ts = None
    for r in (reps or list(REPS)):
        kind, args = REPS[r]
        Ms = maps(kind, args, Hs, dev)
        t1 = time.time()
        loc = fit(glob, Ye, Ms, refit, dev)
        el = elbo_eval(glob, loc, Ye, Ms, nb.T, nb.OBS, nb.S_OBS)                       # (G,) per series
        iw = iwae(glob, loc, Ye, Ms, nb.T, nb.OBS, nb.S_OBS, K=512).mean(1)              # (G,)
        per[r] = {"D": int(Ms.shape[-1]), "elbo_own": el.tolist(), "iwae_own_K512": iw.tolist(),
                  "seconds": time.time() - t1}
        if r == "twoscale":
            loc_ts, M_ts = loc, Ms
        print(json.dumps({"rep": r, "H_true": H_true, "seed": seed, "elbo_own": [round(float(v), 3) for v in el],
                          "iwae_own": [round(float(v), 3) for v in iw], "s": round(time.time() - t1)}), flush=True)
    if loc_ts is None:
        M_ts = maps("twoscale", {"J": 16}, Hs, dev); loc_ts = fit(glob, Ye, M_ts, refit, dev)
    ref_ts = {str(K): iwae(glob, loc_ts, Ye, M_ts, nb.T, nb.OBS, nb.S_OBS, K=K).mean(1).tolist() for K in Ks}
    Mg = maps("grid", {}, Hs, dev)
    loc_g = fit(glob, Ye, Mg, refit, dev)
    ref_grid = iwae(glob, loc_g, Ye, Mg, nb.T, nb.OBS, nb.S_OBS, K=Ks[-1]).mean(1).tolist()
    ref = np.array(ref_ts[str(Ks[-1])])
    for r in per:
        per[r]["gap_to_reference"] = (ref - np.array(per[r]["elbo_own"])).tolist()
        per[r]["own_evidence_minus_reference"] = (np.array(per[r]["iwae_own_K512"]) - ref).tolist()
    res = {"H_true": H_true, "seed": seed, "groups": [f"H={h:.2f}" for h in Hs], "model_source": os.path.basename(src),
           "model_rep": model_rep,
           "refit_steps": refit, "reference_twoscale_iwae": ref_ts, "reference_grid_iwae_Kmax": ref_grid,
           "representations": per, "seconds": time.time() - t0}
    if save:
        tag = "" if model_rep == "twoscale" else f"_model-{model_rep}"
        with open(os.path.join(OUT, f"n2_seed{seed}_H{H_true:.2f}{tag}.json"), "w") as fh:
            json.dump(res, fh, indent=1)
    print(json.dumps({"H_true": H_true, "seed": seed, "model_rep": model_rep,
                      "reference": {k: [round(v, 3) for v in x] for k, x in ref_ts.items()},
                      "reference_grid": [round(v, 3) for v in ref_grid],
                      "gap_oracle_group": {r: round(per[r]["gap_to_reference"][-1], 3) for r in per}}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=float, nargs="+", default=[0.6, 0.75, 0.9])
    ap.add_argument("--seed", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--refit", type=int, default=4000)
    ap.add_argument("--Ks", type=int, nargs="+", default=[512, 4096, 32768])
    ap.add_argument("--reps", nargs="+", default=None)
    ap.add_argument("--no-save", action="store_true", help="smoke test: do not write the result file")
    ap.add_argument("--model-rep", default="twoscale", choices=sorted(nb.REPS),
                    help="which trained N1b model to freeze (twoscale = N2 protocol; grid = N2-x cross-check)")
    a = ap.parse_args()
    for s in a.seed:
        for H in a.H:
            run(H, s, a.refit, tuple(a.Ks), a.reps, not a.no_save, a.model_rep)
