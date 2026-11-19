r"""X10: do the importance scores depend on the proposal? (05_EXPERIMENTS.md section 3b; protocol fixed 18 Nov 2026,
19:28, before any X10 run.)  CPU only.

The frozen-model RK4 solver (`Frozen.drift`, `Frozen.paths`) and the joint Gauss-Newton importance readout
(`joint_readout`) are copied from an independent check of 18 Nov 2026 (check_proofs.py:
FrozenModel and candidate_test(check_evidence=True)), unchanged except that the model file and the evaluation map are
arguments.  For every test window: Levenberg-Marquardt mode from z = 0, x0 = y_1; Gauss-Newton precision; joint
Gaussian proposal (coupling between x0 and the noise kept); two independent sets of 4 x 128 = 512 draws; log score,
ESS and largest normalised weight per set.  Common random numbers across the models of one (H, seed).

  --part a : the six N2 cases (CosVI-trained oracle-H model, seeds 1-2) under the maps twoscale, grid, markov (K = 5)
  --part b : the saved X8(b) models (twoscale, grid, markov, markov_best) under the grid map (primary evaluator)
Run from work\code\experiments:  python -X utf8 x10_proposal_sensitivity.py --part a [--H 0.75] [--smoke]
Out: work/results/x10/x10{part}_{model}_on-{map}_seed{s}_H{H}.json  (skipped if present)
"""
import os, sys, json, time, argparse
import numpy as np
from scipy.linalg import cholesky, cho_solve
from scipy.optimize import least_squares
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
RES = os.path.join(WORK, "results")
OUT = os.path.join(RES, "x10")
MAPS = {"twoscale": "map_twoscale_H{H:.4f}_T5.0_N80_J16.npz", "grid": "map_grid_H{H:.4f}_T5.0_N80_.npz",
        "markov": "map_markov_H{H:.4f}_T5.0_N80_K5_gamma_max20.0.npz"}
NOISE = 0.05


def model_file(rep, H, seed):
    base = os.path.join(RES, "n1b", "models", f"n1b_{rep}_seed{seed}_H{H:.2f}")
    for tag in ("", "_hg95"):
        if os.path.exists(base + tag + ".pt"):
            return base + tag + ".pt", tag
    return None, None


class Frozen:
    """Frozen double-well model (drift MLP 1-32-32-1, sigma) of one saved run, oracle group = last group."""

    def __init__(self, rep, H, seed, eval_map):
        import torch
        torch.set_num_threads(1)
        path, self.tag = model_file(rep, H, seed)
        if path is None:
            raise FileNotFoundError(f"no saved model for {rep} seed {seed} H {H}")
        self.model_path = os.path.basename(path)
        self.state = torch.load(path, map_location="cpu", weights_only=True)
        last = lambda k: self.state[k][-1].double().numpy()
        self.W = [last(f"net.W.{j}") for j in range(3)]
        self.bias = [last(f"net.b.{j}").reshape(-1) for j in range(3)]
        self.sigma = float(np.logaddexp(0.0, last("s_raw")))
        with np.load(os.path.join(RES, "cache", MAPS[eval_map].format(H=H))) as z:
            self.map = np.stack([z[k] for k in ("left", "mid", "right")])
        self.n = self.map.shape[1]
        self.step = 5.0 / self.n
        self.dim = self.map.shape[2] + 1
        self.rate_jac = np.pad(self.sigma * self.map, ((0, 0), (0, 0), (0, 1)))
        from n1_drift_recovery import make_data
        self.data = make_data(H, seed)[1]
        saved = os.path.join(RES, "n1", f"data_seed{seed}_H{H:.2f}.npz")
        if os.path.exists(saved):
            with np.load(saved) as z:
                if not np.array_equal(z["Yte"], self.data):
                    raise ValueError("test data differ from the saved series")

    # ---- copied from check_proofs.FrozenModel (independent check, 18 Nov 2026) ----
    def drift(self, x, tangent=False):
        a = np.tanh(x[:, None] * self.W[0] + self.bias[0])
        b = np.tanh(a @ self.W[1] + self.bias[1])
        val = (b @ self.W[2]).ravel() + self.bias[2]
        if not tangent:
            return val, None
        da = (1 - a * a) * self.W[0]
        deriv = (((da @ self.W[1]) * (1 - b * b)) @ self.W[2]).ravel()
        return val, deriv

    def paths(self, samples, tangent=False):
        rates = (samples[:, :-1] @ self.map.reshape(-1, self.dim - 1).T).reshape(-1, 3, self.n) * self.sigma
        x = samples[:, -1].copy()
        if tangent:
            assert len(x) == 1
            dx = np.eye(1, self.dim, k=self.dim - 1).ravel()
        values, derivs = [], []
        h = self.step
        for n in range(self.n):
            stages, slopes = [], []
            for stage, half in ((0, 0.), (1, h / 2), (1, h / 2), (2, h)):
                previous = x if not stages else x + half * stages[-1]
                val, dval = self.drift(previous, tangent)
                if tangent:
                    dtan = dx if not slopes else dx + half * slopes[-1]
                    slopes.append(dval[0] * dtan + self.rate_jac[stage, n])
                stages.append(val + rates[:, stage, n])
            x = x + h / 6 * (stages[0] + 2 * stages[1] + 2 * stages[2] + stages[3])
            if tangent:
                dx = dx + h / 6 * (slopes[0] + 2 * slopes[1] + 2 * slopes[2] + slopes[3])
            if (n + 1) % 4 == 0:
                values.append(x.copy())
                if tangent:
                    derivs.append(dx.copy())
        return np.stack(values, axis=1), np.asarray(derivs)


def joint_readout(model, y, idx, batches=4, rng_offset=0):
    """Copied from check_proofs.candidate_test(check_evidence=True) (independent check, 18 Nov 2026)."""
    d = model.dim
    start = np.zeros(d); start[-1] = y[0]
    prev, cached = None, None

    def local(w):
        nonlocal prev, cached
        if prev is None or not np.array_equal(w, prev):
            v, G = model.paths(w[None, :], True)
            cached = (np.r_[(v.ravel() - y) / NOISE, w], np.vstack([G / NOISE, np.eye(d)]))
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
            ll = -0.5 * (((pred - y) / NOISE) ** 2).sum(1) - 0.5 * len(y) * np.log(2 * np.pi * NOISE ** 2)
            weights.extend(ll + 0.5 * ((e * e).sum(1) - (draws * draws).sum(1) + logdet_joint))
        lw = np.asarray(weights)
        total = logsumexp(lw)
        norm = np.exp(lw - total)
        sets.append({"log_score": float(total - np.log(len(lw))), "ess": float(1 / (norm @ norm)),
                     "max_weight": float(norm.max()), "draws": int(len(lw))})
    return {"series": idx, "mode_evaluations": int(fit.nfev), "mode_gradient_inf": float(fit.optimality),
            "linearised_gap": float(0.5 * np.log(vj * c)), "sets": sets}


def run_one(part, rep, eval_map, H, seed, smoke=False):
    os.makedirs(OUT, exist_ok=True)
    f = os.path.join(OUT, f"x10{part}_{rep}_on-{eval_map}_seed{seed}_H{H:.2f}{'_smoke' if smoke else ''}.json")
    if os.path.exists(f):
        return json.load(open(f))
    t0 = time.time()
    model = Frozen(rep, H, seed, eval_map)
    offset = 100000 * (100 * seed + round(H * 100))          # depends on (H, seed) only: common random numbers
    windows, failures = [], []
    data = model.data[:2] if smoke else model.data
    for i, y in enumerate(data):
        try:
            windows.append(joint_readout(model, y, i, rng_offset=offset))
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as err:
            failures.append({"series": i, "error": str(err)})
    scores = [float(np.mean([w["sets"][k]["log_score"] for w in windows])) for k in range(2)] if windows else []
    rec = {"part": part, "rep": rep, "eval_map": eval_map, "H": H, "seed": seed, "model_file": model.model_path,
           "windows_ok": len(windows), "windows_total": len(data), "failures": failures,
           "score_sets": scores, "score": float(np.mean(scores)) if scores else None,
           "set_difference": float(abs(scores[0] - scores[1])) if scores else None,
           "min_ess": min(s["ess"] for w in windows for s in w["sets"]) if windows else None,
           "max_weight": max(s["max_weight"] for w in windows for s in w["sets"]) if windows else None,
           "draws_per_set": 512, "flagged_windows_ess_below_5pct": sorted({w["series"] for w in windows
                                                                       for s in w["sets"] if s["ess"] < 0.05 * 512}),
           "seconds": time.time() - t0, "windows": windows}
    json.dump(rec, open(f, "w"), indent=1)
    print(f"{part} {rep:12s} on {eval_map:8s} H={H} seed={seed}: score {rec['score']} (sets {scores}), "
          f"min ESS {rec['min_ess']}, max w {rec['max_weight']}, fails {len(failures)}, {rec['seconds']:.0f}s", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "b"], required=True)
    ap.add_argument("--H", type=float, nargs="*", default=[0.6, 0.75, 0.9])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    with threadpool_limits(1):
        for H in a.H:
            if a.part == "a":
                for seed in (1, 2):
                    for m in ("twoscale", "grid", "markov"):
                        run_one("a", "twoscale", m, H, seed, a.smoke)
                        if a.smoke:
                            return
            else:
                seeds = (0, 1, 2, 3, 4) if H == 0.9 else (1, 2, 3, 4)
                for seed in seeds:
                    for rep in ("twoscale", "grid", "markov", "markov_best"):
                        run_one("b", rep, "grid", H, seed, a.smoke)
                        if a.smoke:
                            return


if __name__ == "__main__":
    main()
