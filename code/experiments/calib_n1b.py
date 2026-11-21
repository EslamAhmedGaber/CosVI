r"""X4 (16 Nov 2026): calibration of the latent-path uncertainty of the trained N1b models (no new training).

Protocol fixed before any run.  For a saved N1b model (globals of every group;
the oracle group, H = H_true, is the last one) and its data seed:
  1. regenerate the data exactly as n1_drift_recovery.make_data (same random stream), keeping the true latent paths;
     the regenerated observations are checked against make_data;
  2. refit the 16 test posteriors in the representation's OWN coordinates (2000 steps, globals frozen, as the
     yardstick refit of N1b);
  3. draw 1024 posterior samples of the latent path on the 80-step grid and compare them with the true path
     (8192-step simulation interpolated linearly onto the grid), at grid times t > 0.
Metrics (oracle group): coverage of the central 50 % and 90 % intervals, interval score at 90 % (Gneiting and
Raftery 2007), CRPS from the samples, the PIT histogram (20 bins) for reliability diagrams, and coverage of central
intervals at levels 0.1, ..., 0.9.
Run from work\code\experiments:  python calib_n1b.py --reps twoscale --seeds 1 2 --H 0.6 0.75 0.9 [--tag _hg95]
Out: work/results/calib/calib_{rep}_seed{seed}_H{H}{tag}.json
"""
import os, sys, json, time, argparse
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import n1_drift_recovery as n1
import n1b_representations as nb
from fnvi.simulate import simulate
from fnvi.train_maps import Globals, LocalsD, train, solve

OUT = os.path.join(HERE, "..", "..", "results", "calib"); os.makedirs(OUT, exist_ok=True)
MODELS = os.path.join(HERE, "..", "..", "results", "n1b", "models")
LEVELS = np.round(np.arange(0.1, 1.0, 0.1), 2)


def data_with_paths(H, seed):
    """Same random stream as n1.make_data; also returns the true latent paths of the test series on the model grid."""
    rng = np.random.default_rng(1000 + seed)
    M = n1.M_TR + n1.M_TE; Nf = 8192
    x0 = rng.uniform(-1.5, 1.5, M)
    X = simulate(n1.f_true, lambda x: n1.SIG_TRUE * np.ones_like(x), x0, n1.T, Nf, H, rng)       # (Nf+1, M)
    idx = np.arange(1, n1.N_OBS + 1) * (Nf // n1.N_OBS)
    Y = X[idx].T + n1.S_OBS * rng.standard_normal((M, n1.N_OBS))
    tf = np.linspace(0, n1.T, Nf + 1)
    tg = np.arange(1, nb.N + 1) * (n1.T / nb.N)
    Xg = np.stack([np.interp(tg, tf, X[:, i]) for i in range(M)], 0)                               # (M, N)
    return Y[:n1.M_TR], Y[n1.M_TR:], Xg[n1.M_TR:]


def crps_samples(s, x):
    """CRPS estimate from samples s (n, ...) for observations x (...): E|S - x| - 0.5 E|S - S'| (sorted form)."""
    n = s.shape[0]
    t1 = np.abs(s - x[None]).mean(0)
    ss = np.sort(s, 0)
    w = (2 * np.arange(1, n + 1) - n - 1).reshape((n,) + (1,) * (s.ndim - 1))
    t2 = (2.0 / n ** 2) * (w * ss).sum(0)
    return t1 - 0.5 * t2


def run(rep, H_true, seed, tag="", S_total=1024, refit=2000, out_tag=""):
    dev = torch.device("cuda"); torch.manual_seed(10_000 + seed)
    fn_model = os.path.join(MODELS, f"n1b_{rep}_seed{seed}_H{H_true:.2f}{tag}.pt")
    if not os.path.exists(fn_model):
        print(json.dumps({"rep": rep, "H_true": H_true, "seed": seed, "skipped": "no saved model"}), flush=True)
        return
    state = torch.load(fn_model, map_location=dev)
    G = state["s_raw"].shape[0]
    hgrid = nb.HGRID if G == len(nb.HGRID) + 1 else [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    Hs = list(hgrid) + [H_true]
    assert len(Hs) == G, (len(Hs), G)
    glob = Globals(G).to(dev); glob.load_state_dict(state)
    for p in glob.parameters():
        p.requires_grad_(False)
    Ytr, Yte, Xte = data_with_paths(H_true, seed)
    Ytr0, Yte0, _, _ = n1.make_data(H_true, seed)
    assert np.allclose(Yte, Yte0) and np.allclose(Ytr, Ytr0), "regenerated data differ from make_data"
    Ms = nb.maps_tensor(rep, Hs, dev); D = Ms.shape[-1]
    Ye = torch.as_tensor(Yte, dtype=torch.float32, device=dev)
    loc = LocalsD(G, Yte.shape[0], D, Ye[:, 0]).to(dev)
    t0 = time.time()
    train(glob, loc, Ye, Ms, nb.T, nb.OBS, nb.S_OBS, 16, refit, False, nb.LRS)
    refit_s = time.time() - t0
    all_steps = np.arange(1, nb.N + 1)
    g = G - 1                                                              # oracle group (H = H_true)
    samp = []
    with torch.no_grad():
        for _ in range(S_total // 128):
            z, x0, _, _, _ = loc.sample(128, common=False)
            X = solve(glob, Ms, z, x0, nb.T, all_steps)                     # (N, G, S, M)
            samp.append(X[:, g].permute(1, 2, 0).cpu().numpy())           # (S, M, N)
    s = np.concatenate(samp, 0)                                            # (S, M, N)
    x = Xte                                                                # (M, N)
    q = lambda p: np.quantile(s, p, axis=0)
    cover = {}
    for lev in LEVELS:
        lo, hi = q(0.5 - lev / 2), q(0.5 + lev / 2)
        cover[f"{lev:.1f}"] = float(((x >= lo) & (x <= hi)).mean())
    lo90, hi90 = q(0.05), q(0.95)
    a = 0.1
    is90 = (hi90 - lo90) + (2 / a) * (lo90 - x) * (x < lo90) + (2 / a) * (x - hi90) * (x > hi90)
    crps = crps_samples(s, x)
    pit = (s <= x[None]).mean(0)
    hist = np.histogram(pit.ravel(), bins=20, range=(0, 1))[0] / pit.size
    res = {"rep": rep, "H_true": H_true, "seed": seed, "tag": tag, "D": int(D), "group": "oracle", "samples": int(s.shape[0]),
           "refit_steps": refit, "refit_seconds": refit_s,
           "coverage_central": cover, "coverage50": cover["0.5"], "coverage90": cover["0.9"],
           "calibration_error": float(0.5 * (abs(cover["0.5"] - 0.5) + abs(cover["0.9"] - 0.9))),
           "interval_score90_mean": float(is90.mean()), "crps_mean": float(crps.mean()),
           "coverage90_by_time": ((x >= lo90) & (x <= hi90)).mean(0).tolist(),
           "crps_by_time": crps.mean(0).tolist(), "pit_hist20": hist.tolist(),
           "posterior_sd_mean": float(s.std(0).mean())}
    with open(os.path.join(OUT, f"calib_{rep}_seed{seed}_H{H_true:.2f}{tag}{out_tag}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()
                      if k in ("rep", "H_true", "seed", "coverage50", "coverage90", "calibration_error",
                               "interval_score90_mean", "crps_mean", "refit_seconds")}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="+", default=list(nb.REPS))
    ap.add_argument("--H", type=float, nargs="+", default=[0.6, 0.75, 0.9])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--tag", default="")
    ap.add_argument("--refit", type=int, default=2000)
    ap.add_argument("--samples", type=int, default=1024)
    ap.add_argument("--out_tag", default="", help="suffix of the output name only (smoke tests)")
    a = ap.parse_args()
    for s in a.seeds:
        for H in a.H:
            for rep in a.reps:
                run(rep, H, s, a.tag, a.samples, a.refit, a.out_tag)
