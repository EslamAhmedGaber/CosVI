r"""X10(c): the proposal check on the real series (05_EXPERIMENTS.md section 3b; protocol fixed 18 Nov 2026, 19:46,
before any X10(c) run).  CPU only.

Frozen saved globals of CosVI, CosVI-64, the exact grid and the MA-fBm laws (published, tuned), training seeds 0-2, each
at its profile group (as X9), scored under the smooth-stage evaluator (CosVI's map at the group H; T = 50, N = 200,
observations every 4 steps, s = 0.1) with the joint Gauss-Newton proposal of X10 (two independent sets of 512 draws per
test window; common random numbers across the models of one series and seed).  The solver is the X10 frozen-model
solver (x10_proposal_sensitivity.Frozen.paths) with T, N, s and the group as arguments; the readout is
x10_proposal_sensitivity.joint_readout with the observation noise as an argument.
Run from work\code\experiments:  python -X utf8 x10c_real_series.py --series nile --seed 0
Out: work/results/x10/x10c_{series}_{rep}_seed{seed}.json  (skipped if present)
"""
import os, sys, json, time, argparse
import numpy as np
from scipy.linalg import cholesky, cho_solve
from scipy.optimize import least_squares
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from x10_proposal_sensitivity import Frozen
RES = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
OUT = os.path.join(RES, "x10")
REPS = ["twoscale", "cosine64", "grid", "markov", "markov_tuned"]
T_REAL, N_REAL, S_OBS = 50.0, 200, 0.1


class FrozenReal(Frozen):
    """Same solver as Frozen (RK4, observations every 4 steps); real-series horizon, group and evaluation map."""

    def __init__(self, state_path, group, H, Yte):
        import torch
        from fnvi.noise_maps import build
        torch.set_num_threads(1)
        self.model_path = os.path.basename(state_path)
        self.state = torch.load(state_path, map_location="cpu", weights_only=True)
        g = lambda k: self.state[k][group].double().numpy()
        self.W = [g(f"net.W.{j}") for j in range(3)]
        self.bias = [g(f"net.b.{j}").reshape(-1) for j in range(3)]
        self.sigma = float(np.logaddexp(0.0, g("s_raw")))
        self.map = np.stack(build("twoscale", H, T_REAL, N_REAL, J=16), 0)
        self.n = self.map.shape[1]
        self.step = T_REAL / self.n
        self.dim = self.map.shape[2] + 1
        self.rate_jac = np.pad(self.sigma * self.map, ((0, 0), (0, 0), (0, 1)))
        self.data = np.asarray(Yte, dtype=float)


def joint_readout(model, y, idx, noise, batches=4, rng_offset=0):
    """x10_proposal_sensitivity.joint_readout (independent check, 18 Nov 2026) with the observation noise as an argument."""
    d = model.dim
    start = np.zeros(d); start[-1] = y[0]
    prev, cached = None, None

    def local(w):
        nonlocal prev, cached
        if prev is None or not np.array_equal(w, prev):
            v, G = model.paths(w[None, :], True)
            cached = (np.r_[(v.ravel() - y) / noise, w], np.vstack([G / noise, np.eye(d)]))
            prev = w.copy()
        return cached
    fit = least_squares(lambda w: local(w)[0], start, jac=lambda w: local(w)[1], method="lm",
                        ftol=1e-11, xtol=1e-11, gtol=1e-9, max_nfev=100)
    if not fit.success:
        raise RuntimeError(f"Series {idx}: {fit.message}")
    mean = fit.x
    P = fit.jac.T @ fit.jac
    Pz, a, c = P[:-1, :-1], P[:-1, -1], P[-1, -1]
    L = cholesky(Pz, lower=True)
    B = np.linalg.solve(L.T, np.eye(d - 1))
    beta = -cho_solve((L, True), a)
    vj = 1 / (c + a @ beta)
    logdet_joint = -2 * np.log(L.diagonal()).sum() + np.log(vj)
    sets = []
    for rep in range(2):
        independent = np.random.default_rng(8873111 + rng_offset + 100 * idx + rep)
        weights = []
        for _ in range(batches):
            e = independent.normal(size=(128, d))
            dx = np.sqrt(vj) * e[:, -1]
            z = mean[:-1] + e[:, :-1] @ B.T + dx[:, None] * beta
            draws = np.column_stack((z, mean[-1] + dx))
            pred, _ = model.paths(draws)
            ll = -0.5 * (((pred - y) / noise) ** 2).sum(1) - 0.5 * len(y) * np.log(2 * np.pi * noise ** 2)
            weights.extend(ll + 0.5 * ((e * e).sum(1) - (draws * draws).sum(1) + logdet_joint))
        lw = np.asarray(weights)
        total = logsumexp(lw)
        norm = np.exp(lw - total)
        sets.append({"log_score": float(total - np.log(len(lw))), "ess": float(1 / (norm @ norm)),
                     "max_weight": float(norm.max()), "draws": int(len(lw))})
    return {"series": idx, "mode_evaluations": int(fit.nfev), "mode_gradient_inf": float(fit.optimality),
            "linearised_gap": float(0.5 * np.log(vj * c)), "sets": sets}


def run(series, seed):
    from x9_real_evaluators import series_setup
    os.makedirs(OUT, exist_ok=True)
    hg, Yte, x0te, rec, mod, hkey = series_setup(series)
    offset = 100000 * (100 * seed + (1 if series == "nile" else 2))      # common random numbers within (series, seed)
    for rep in REPS:
        f = os.path.join(OUT, f"x10c_{series}_{rep}_seed{seed}.json")
        if os.path.exists(f):
            continue
        t0 = time.time()
        r = json.load(open(rec(rep, seed)))
        gi = int(np.argmin(np.abs(np.array(hg) - r[hkey])))
        model = FrozenReal(mod(rep, seed), gi, hg[gi], Yte)
        windows, failures = [], []
        for i, y in enumerate(model.data):
            try:
                windows.append(joint_readout(model, y, i, S_OBS, rng_offset=offset))
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as err:
                failures.append({"series": i, "error": str(err)})
        scores = [float(np.mean([w["sets"][k]["log_score"] for w in windows])) for k in range(2)] if windows else []
        out = {"part": "c", "series": series, "rep": rep, "seed": seed, "group_H": hg[gi], "H_profile": r[hkey],
               "model_file": model.model_path, "windows_ok": len(windows), "windows_total": len(model.data),
               "failures": failures, "score_sets": scores, "score": float(np.mean(scores)) if scores else None,
               "set_difference": float(abs(scores[0] - scores[1])) if scores else None,
               "archived_smooth_ll": r["test_loglik_exact_law_twoscale_coords_mean"][gi],
               "min_ess": min(s["ess"] for w in windows for s in w["sets"]) if windows else None,
               "max_weight": max(s["max_weight"] for w in windows for s in w["sets"]) if windows else None,
               "flagged_windows_ess_below_5pct": sorted({w["series"] for w in windows for s in w["sets"]
                                                         if s["ess"] < 0.05 * 512}),
               "seconds": time.time() - t0, "windows": windows}
        json.dump(out, open(f, "w"), indent=1)
        print(f"c {series} {rep:12s} seed={seed} H={hg[gi]}: joint {out['score']} archived {out['archived_smooth_ll']:.3f} "
              f"min ESS {out['min_ess']}, max w {out['max_weight']}, fails {len(failures)}, {out['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", choices=["nile", "temp"], required=True)
    ap.add_argument("--seed", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    with threadpool_limits(1):
        for s in a.seed:
            run(a.series, s)
