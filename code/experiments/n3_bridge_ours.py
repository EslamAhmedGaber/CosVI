"""N3: our whitened-coordinate VI on the competitor's own benchmark (fOU bridge of Daems et al. 2024, Sec. 5).

Task (theirs): dX = -theta X dt + dB_H, X(0) = 0, T = 2, one observation y = X(2) + eps, eps ~ N(0, 0.1^2), y = 0;
theta in {0, 1}; H in {0.6, 0.7, 0.8, 0.9}; their step dt = 0.01 (200 steps), batch 32, 2000 Adam steps.
Ours: representations cosine (J = 32), two-scale (J = 16 + residual), exact grid (Cholesky), each with a full
Gaussian q over its coordinates; RK4 with N = 200 steps; S = 32 samples; Adam lr 0.05 with cosine decay to 1e-3;
2000 steps; seeds 0, 1, 2. The model is linear in z, so after training the posterior std on the grid is computed
exactly from q (no sampling noise; their protocol uses 256 samples, Monte Carlo s.e. about 4.4 % per point).
References (exact): the true posterior (fOU started at 0, standard fBm) and their analytic reference (the
stationary fOU conditioned on X(0) = 0 for theta = 1; the standard fBm bridge for theta = 0).
Out: work/results/n3/n3_{rep}_H{H}_theta{theta}_seed{seed}.json
"""
import os, sys, json, time, argparse
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from fnvi.noise_maps import build
from fnvi.linear_exact import exact_cov, gp_posterior

OUT = os.path.join(HERE, "..", "..", "results", "n3"); os.makedirs(OUT, exist_ok=True)
T, N, S_OBS = 2.0, 200, 0.1
TIMES = np.arange(1, N + 1) * T / N
REPS = {"cosine": ("cosine", {"J": 32}), "twoscale": ("twoscale", {"J": 16}), "grid": ("grid", {})}
_ref_cache = {}


def references(H, theta):
    key = (H, theta)
    if key in _ref_cache:
        return _ref_cache[key]
    if theta == 0:
        s = TIMES[:, None]; t = TIMES[None, :]
        C = 0.5 * (s ** (2 * H) + t ** (2 * H) - np.abs(s - t) ** (2 * H)); Cst = C
    else:
        C = exact_cov(TIMES, H, theta, 1.0, levels=24, n=10)
        Lb = 40.0
        Cs = exact_cov(np.concatenate([[Lb], Lb + TIMES]), H, theta, 1.0, levels=24, n=10)
        k0 = Cs[0, 1:]; Cst = Cs[1:, 1:] - np.outer(k0, k0) / Cs[0, 0]
    out = {}
    for lab, K in (("exact_true", C), ("their_reference", Cst)):
        _, v = gp_posterior(K[:, -1:], K[-1:, -1:], np.diag(K), np.zeros(1), S_OBS ** 2)
        out[lab] = np.sqrt(np.clip(v, 1e-300, None))
    _ref_cache[key] = out
    return out


def linear_solution_map(Ms, theta):
    """X at all steps as a linear map of z for the RK4 solve of dx/dt = -theta x + r(t); returns (N, D)."""
    Ml, Mm, Mr = Ms; h = T / N; D = Ml.shape[1]
    x = np.zeros(D); out = np.zeros((N, D))
    for n in range(N):
        k1 = -theta * x + Ml[n]
        k2 = -theta * (x + 0.5 * h * k1) + Mm[n]
        k3 = -theta * (x + 0.5 * h * k2) + Mm[n]
        k4 = -theta * (x + h * k3) + Mr[n]
        x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[n] = x
    return out


def run(rep, H, theta, seed, steps=2000, S=32):
    kind, kw = REPS[rep]
    Ms = build(kind, H, T, N, **kw)
    Psi = linear_solution_map(Ms, theta)                                   # (N, D)
    D = Psi.shape[1]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.manual_seed(seed)
    P = torch.as_tensor(Psi, dtype=torch.float64, device=dev); pT = P[-1]
    mu = torch.zeros(D, dtype=torch.float64, device=dev, requires_grad=True)
    rho = torch.full((D,), 0.5413, dtype=torch.float64, device=dev, requires_grad=True)
    off = torch.zeros(D, D, dtype=torch.float64, device=dev, requires_grad=True)
    opt = torch.optim.Adam([mu, rho, off], lr=0.05)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-3)
    t0 = time.time()
    for it in range(steps):
        d = torch.nn.functional.softplus(rho); L = torch.tril(off, -1) + torch.diag(d)
        z = mu + torch.randn(S, D, dtype=torch.float64, device=dev) @ L.T
        xT = z @ pT
        ll = (-0.5 * xT ** 2 / S_OBS ** 2).mean()
        kl = 0.5 * ((L ** 2).sum() + (mu ** 2).sum() - D - 2 * torch.log(d).sum())
        loss = -(ll - kl)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    train_s = time.time() - t0
    with torch.no_grad():
        d = torch.nn.functional.softplus(rho); L = torch.tril(off, -1) + torch.diag(d)
        Sig = L @ L.T
        std = torch.sqrt(torch.einsum("nd,de,ne->n", P, Sig, P)).cpu().numpy()
        mean = (P @ mu).cpu().numpy()
    refs = references(H, theta)
    interior = (TIMES >= 0.5) & (TIMES <= 1.9)
    res = {"rep": rep, "D": int(D), "H": H, "theta": theta, "seed": seed, "steps": steps, "train_seconds": train_s,
           "times": TIMES.tolist(), "std": std.tolist(), "max_abs_mean": float(np.abs(mean).max()), "metrics": {}}
    for lab, r in refs.items():
        res["metrics"][lab] = {"std_relerr_integrated": float(np.sum(np.abs(std - r)) / np.sum(r)),
                               "std_relerr_max_interior": float(np.max(np.abs(std[interior] / r[interior] - 1)))}
        res[f"std_{lab}"] = r.tolist()
    with open(os.path.join(OUT, f"n3_{rep}_H{H:.2f}_theta{theta:.1f}_seed{seed}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({"rep": rep, "H": H, "theta": theta, "seed": seed, "s": round(train_s),
                      "vs_true": round(res["metrics"]["exact_true"]["std_relerr_integrated"], 4),
                      "vs_their_ref": round(res["metrics"]["their_reference"]["std_relerr_integrated"], 4)}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="+", default=list(REPS))
    ap.add_argument("--H", type=float, nargs="+", default=[0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--theta", type=float, nargs="+", default=[1.0, 0.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    for th in a.theta:
        for H in a.H:
            for rep in a.reps:
                for s in a.seeds:
                    run(rep, H, th, s)
