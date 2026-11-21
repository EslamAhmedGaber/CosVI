"""X11b (Tier 3): Joint q(z, x0) & Block-Two-Scale O(J^2 + N) Joint Posterior in full training,
plus long-horizon GPU memory/parameter scaling (N = 80..1600).

Tests five paired variants on the 1D double well (H in {0.6, 0.75, 0.9}, 3 seeds = 9 batched groups
per job, identical initial weights and data, 5000 Adam steps):
  1. pl_prod_full   : CosVI-PL with baseline product full q(z) q(x0)         (4,657 local params/series)
  2. pl_joint_full  : CosVI-PL with joint full q(z, x0)                      (4,752 local params/series)
  3. pl_block_joint : CosVI-PL with two-scale block-joint q(xi,x0) q_diag(eta) (328 local params/series, 14x smaller!)
  4. simp_joint_full: Simpson-exact smooth-stage CosVI with joint full q(z, x0)
  5. grid_joint_full: Exact grid with joint full q(z, x0) (fairness control)

Evaluated under BOTH:
  - Joint-family exact PL evaluator (LocalsJointFull refit, 2000 steps, IWAE N_IS = 512, high ESS!)
  - Product-family exact PL evaluator (LocalsD refit, 2000 steps, IWAE N_IS = 512)
Plus: exact training ELBO, drift RMSE, sigma, wall-clock, parameter count, and long-horizon memory
scaling benchmark (N = 80, 200, 400, 800, 1600) comparing Full vs BlockJoint!

Run from code/experiments:  python -X utf8 x11b_structured_q.py
Out: results/x11/x11b_structured_q.json
"""
import os, sys, json, time, itertools
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import n1_drift_recovery as n1
from fnvi.noise_maps import twoscale_pl_map, twoscale_exact_map, grid_map
from fnvi.train_maps import Globals, LocalsD, solve as solve_rk4
from fnvi.families import LocalsJointFull, LocalsBlockJoint, LocalsPrecLR
from fnvi.train_graph import cosine_factor

OUT = os.path.join(HERE, "..", "..", "results", "x11")
os.makedirs(OUT, exist_ok=True)

T, S_OBS, N, J = n1.T, n1.S_OBS, 80, 16
OBS = np.arange(1, n1.N_OBS + 1) * (N // n1.N_OBS)
LRS = {"local": 2e-2, "net": 3e-3, "sigma": 1e-2}
HS = [0.6, 0.75, 0.9]
SEEDS = [0, 1, 2]
LOG2PI = float(np.log(2.0 * np.pi))


def loglik_g(Y, X):
    Yb = Y.permute(2, 0, 1)[:, :, None, :]
    return (-0.5 * (Yb - X) ** 2 / S_OBS ** 2 - 0.5 * np.log(2 * np.pi * S_OBS ** 2)).sum(0).permute(1, 0, 2)


def train_batched(glob, loc, Y, Ms, S, steps, train_globals=True):
    dev = Y.device
    T_ = lambda v: torch.tensor(float(v), device=dev)
    groups = [{"params": list(loc.parameters()), "lr": T_(LRS["local"]), "base": LRS["local"]}]
    if train_globals:
        groups += [{"params": list(glob.net.parameters()), "lr": T_(LRS["net"]), "base": LRS["net"]},
                   {"params": [glob.s_raw], "lr": T_(LRS["sigma"]), "base": LRS["sigma"]}]
    opt = torch.optim.Adam(groups, capturable=True)
    out = {"elbo": torch.zeros(Ms.shape[0], device=dev)}

    def step():
        z, x0, kl, _, _ = loc.sample(S, common=False)
        X = solve_rk4(glob, Ms, z, x0, T, OBS)
        elbo = (loglik_g(Y, X).mean(0) - kl).sum(-1)
        (-elbo.sum() / Y.shape[1]).backward()
        opt.step()
        opt.zero_grad(set_to_none=False)
        out["elbo"].copy_(elbo.detach())

    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(st)
    gph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gph):
        step()
    hist = []
    for it in range(steps - 3):
        if it % 50 == 0:
            fac = cosine_factor(it, steps)
            for grp in opt.param_groups:
                grp["lr"].fill_(grp["base"] * fac)
        gph.replay()
        if it % 200 == 0:
            hist.append([it] + (out["elbo"] / Y.shape[1]).cpu().tolist())
    return hist


@torch.no_grad()
def eval_elbo_batched(glob, loc, Y, Ms, S_tot=256, chunk=16):
    acc = torch.zeros(Ms.shape[0], device=Y.device)
    reps = S_tot // chunk
    for _ in range(reps):
        z, x0, kl, _, _ = loc.sample(chunk, common=False)
        X = solve_rk4(glob, Ms, z, x0, T, OBS)
        acc += (loglik_g(Y, X).mean(0) - kl).sum(-1)
    return (acc / reps / Y.shape[1]).cpu().numpy()


@torch.no_grad()
def iwae_batched(glob, loc, Y, Ms, K_is=512, chunk=64):
    logw = []
    D = Ms.shape[-1]
    for _ in range(K_is // chunk):
        z, x0, _, _, _ = loc.sample(chunk, common=False)
        X = solve_rk4(glob, Ms, z, x0, T, OBS)
        lp = loglik_g(Y, X)
        lpr = -0.5 * (z ** 2).sum(-1) - 0.5 * D * LOG2PI - 0.5 * x0 ** 2 - 0.5 * LOG2PI
        logw.append(lp + lpr - loc.log_q(z, x0))
    lw = torch.cat(logw, 0)                                     # (K_is, G, M)
    lse = torch.logsumexp(lw, 0)                                # (G, M)
    w = torch.exp(lw - lse.unsqueeze(0))
    ess = (1.0 / (w ** 2).sum(0)).cpu().numpy()                 # (G, M)
    return (lse - np.log(lw.shape[0])).cpu().numpy(), ess


def build_variant_maps(rep, Hs, dev):
    Ms = []
    for H in Hs:
        if rep == "twoscale_pl":
            M = np.stack(twoscale_pl_map(H, T, N, J), 0)
        elif rep == "twoscale_exact":
            M = np.stack(twoscale_exact_map(H, T, N, J), 0)
        elif rep == "grid":
            M = np.stack(grid_map(H, T, N), 0)
        else:
            raise ValueError(rep)
        Ms.append(M)
    return torch.as_tensor(np.stack(Ms, 0), dtype=torch.float32, device=dev)


def make_locals(fam, G, M, D, init_x0, dev):
    if fam == "prod_full":
        loc = LocalsD(G, M, D, torch.zeros(M, device=dev)).to(dev)
        with torch.no_grad():
            loc.m0.copy_(init_x0)
        npar = D + D * (D + 1) // 2 + 2
        return loc, npar
    elif fam == "joint_full":
        loc = LocalsJointFull(G, M, D, init_x0).to(dev)
        return loc, loc.num_local_params_per_series()
    elif fam == "block_joint":
        loc = LocalsBlockJoint(G, M, J, D, init_x0).to(dev)
        return loc, loc.num_local_params_per_series()
    elif fam.startswith("prec_lr"):
        rk = int(fam[len("prec_lr"):])
        loc = LocalsPrecLR(G, M, D, init_x0, rank=rk).to(dev)
        return loc, loc.num_local_params_per_series()
    raise ValueError(fam)


def run_variant(vname, rep, fam, steps=5000, refit=2000, K_is=512):
    dev = torch.device("cuda")
    torch.manual_seed(12345)
    combos = list(itertools.product(HS, SEEDS)); G = len(combos)
    data = [n1.make_data(H, s) for H, s in combos]
    Ytr = torch.as_tensor(np.stack([d[0] for d in data]), dtype=torch.float32, device=dev)
    Yte = torch.as_tensor(np.stack([d[1] for d in data]), dtype=torch.float32, device=dev)
    Hs = [H for H, _ in combos]
    Ms = build_variant_maps(rep, Hs, dev); D = Ms.shape[-1]

    glob = Globals(G).to(dev)   # created first -> identical initial weights across all variants!
    loc, npar_per_series = make_locals(fam, G, Ytr.shape[1], D, Ytr[:, :, 0], dev)

    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    hist = train_batched(glob, loc, Ytr, Ms, 16, steps, train_globals=True)
    torch.cuda.synchronize(); train_s = time.time() - t0
    peak_mb = torch.cuda.max_memory_allocated() / (2 ** 20)
    elbo_tr = eval_elbo_batched(glob, loc, Ytr, Ms)

    for p in glob.parameters():
        p.requires_grad_(False)
    Mpl = build_variant_maps("twoscale_pl", Hs, dev)
    Dpl = Mpl.shape[-1]

    # 1. Joint-family exact PL evaluator (captures x0-noise coupling -> high ESS!)
    torch.manual_seed(77777)
    loc_j = LocalsJointFull(G, Yte.shape[1], Dpl, Yte[:, :, 0]).to(dev)
    train_batched(glob, loc_j, Yte, Mpl, 16, refit, train_globals=False)
    elbo_te_j = eval_elbo_batched(glob, loc_j, Yte, Mpl)
    ll_j, ess_j = iwae_batched(glob, loc_j, Yte, Mpl, K_is=K_is)

    # 2. Product-family exact PL evaluator (direct comparability with Table 4/X8)
    torch.manual_seed(88888)
    loc_p, _ = make_locals("prod_full", G, Yte.shape[1], Dpl, Yte[:, :, 0], dev)
    train_batched(glob, loc_p, Yte, Mpl, 16, refit, train_globals=False)
    elbo_te_p = eval_elbo_batched(glob, loc_p, Yte, Mpl)
    ll_p, ess_p = iwae_batched(glob, loc_p, Yte, Mpl, K_is=K_is)

    xg = np.linspace(-1.5, 1.5, 301)
    with torch.no_grad():
        fg = glob.net(torch.as_tensor(xg, dtype=torch.float32, device=dev)[None, :, None].expand(G, -1, 1))[..., 0].cpu().numpy()
    ft = n1.f_true(xg)
    sig = glob.sigma().cpu().numpy()

    rows = []
    for gi, (H, s) in enumerate(combos):
        w_occ, edges = np.histogram(np.clip(data[gi][3], -1.5, 1.5), bins=300, range=(-1.5, 1.5), density=True)
        w = np.interp(xg, 0.5 * (edges[1:] + edges[:-1]), w_occ); w = w / w.sum()
        rows.append({
            "variant": vname, "rep": rep, "family": fam, "H": H, "seed": s,
            "D": int(D), "local_params_per_series": int(npar_per_series),
            "train_seconds_job": train_s, "peak_mem_MB": peak_mb,
            "sigma": float(sig[gi]),
            "drift_rmse": float(np.sqrt((((fg[gi] - ft) ** 2) * w).sum())),
            "elbo_train": float(elbo_tr[gi]),
            "elbo_test_joint": float(elbo_te_j[gi]),
            "ll_test_joint": float(ll_j[gi].mean()),
            "ll_test_joint_min_ess": float(ess_j[gi].min()),
            "elbo_test_prod": float(elbo_te_p[gi]),
            "ll_test_prod": float(ll_p[gi].mean()),
            "ll_test_prod_min_ess": float(ess_p[gi].min()),
        })
    print(f"  {vname:16s} (params/ser={npar_per_series:5d}, {train_s:5.1f}s, {peak_mb:5.1f}MB):", flush=True)
    for H in HS:
        sub = [r for r in rows if r["H"] == H]
        print(f"    H={H:.2f}: trainELBO={np.mean([r['elbo_train'] for r in sub]):+6.2f} | "
              f"LL_joint={np.mean([r['ll_test_joint'] for r in sub]):+6.2f} (minESS={min(r['ll_test_joint_min_ess'] for r in sub):5.1f}) | "
              f"LL_prod={np.mean([r['ll_test_prod'] for r in sub]):+6.2f} | drift={np.mean([r['drift_rmse'] for r in sub]):.3f}", flush=True)
    return rows


def scaling_memory_benchmark():
    """Measure local parameter count, peak GPU memory, and step time vs N for Full vs BlockJoint."""
    dev = torch.device("cuda")
    M, S, H = 16, 16, 0.75
    out = []
    for N_s in (80, 200, 400, 800, 1600):
        T_s = 5.0 * (N_s / 80.0)
        obs_s = np.arange(4, N_s + 1, 4)
        M_np = np.stack(twoscale_pl_map(H, T_s, N_s, J), 0)[None]
        Ms = torch.as_tensor(M_np, dtype=torch.float32, device=dev)
        D_s = Ms.shape[-1]
        Y = torch.randn(1, M, len(obs_s), device=dev)
        for fam in ("joint_full", "block_joint", "prec_lr8", "prec_lr16"):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            glob = loc = opt = None
            Dp_s = int(D_s) + 1
            if fam == "joint_full":
                npar = Dp_s + Dp_s * (Dp_s + 1) // 2
            elif fam == "block_joint":
                npar = (J + 1) + (J + 1) * (J + 2) // 2 + 2 * (int(D_s) - J)
            else:
                rk = int(fam[len("prec_lr"):])
                npar = 2 * Dp_s + Dp_s * rk
            try:
                glob = Globals(1).to(dev)
                loc, npar = make_locals(fam, 1, M, D_s, Y[:, :, 0], dev)
                opt = torch.optim.Adam(list(loc.parameters()) + list(glob.parameters()), lr=1e-3)
                # Eager + sync timing (median of 10 steps after 3 warmups)
                def one_step():
                    z, x0, kl, _, _ = loc.sample(S, common=False)
                    X = solve_rk4(glob, Ms, z, x0, T_s, obs_s)
                    elbo = (loglik_g(Y, X).mean(0) - kl).sum(-1)
                    (-elbo.sum() / M).backward(); opt.step(); opt.zero_grad(set_to_none=True)
                for _ in range(3):
                    one_step()
                torch.cuda.synchronize()
                ts = []
                for _ in range(8):
                    t0 = time.perf_counter(); one_step(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
                peak = torch.cuda.max_memory_allocated() / (2 ** 20)
                rec = {"N": N_s, "D": int(D_s), "family": fam, "local_params_per_series": int(npar),
                       "total_local_params_M16": int(npar * M), "peak_mem_MB": float(peak), "step_ms_eager": 1e3 * float(np.median(ts))}
            except torch.cuda.OutOfMemoryError as e:
                rec = {"N": N_s, "D": int(D_s), "family": fam, "local_params_per_series": int(npar), "error": "OOM"}
            out.append(rec)
            print(f"  Scaling N={N_s:4d} D={D_s:4d} {fam:11s}: params/ser={rec['local_params_per_series']:7d} | "
                  f"mem={rec.get('peak_mem_MB', -1):7.1f} MB | step={rec.get('step_ms_eager', -1):6.1f} ms {rec.get('error', '')}", flush=True)
            del glob, loc, opt
            torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    variants = [
        ("pl_prod_full", "twoscale_pl", "prod_full"),
        ("pl_joint_full", "twoscale_pl", "joint_full"),
        ("pl_block_joint", "twoscale_pl", "block_joint"),
        ("pl_prec_lr16", "twoscale_pl", "prec_lr16"),
        ("grid_joint_full", "grid", "joint_full"),
    ]
    all_rows = []
    print("=== X11b: Training 5 variants on 1D double well (3 seeds x 3 H) ===", flush=True)
    for vname, rep, fam in variants:
        all_rows.extend(run_variant(vname, rep, fam))
        json.dump({"rows": all_rows}, open(os.path.join(OUT, "x11b_structured_q.json"), "w"), indent=1)
    print("\n=== X11b: Long-horizon memory & parameter scaling (N=80..1600) ===", flush=True)
    sc = scaling_memory_benchmark()
    # Compute paired summary against pl_prod_full and grid_joint_full
    summary = {}
    for H in HS:
        Hk = f"{H}"
        summary[Hk] = {}
        by_v = {v[0]: {r["seed"]: r for r in all_rows if r["variant"] == v[0] and r["H"] == H} for v in variants}
        for vname, _, _ in variants:
            rs = [by_v[vname][s] for s in SEEDS]
            summary[Hk][vname] = {
                "local_params_per_series": rs[0]["local_params_per_series"],
                "train_s": rs[0]["train_seconds_job"],
                "elbo_train": [float(np.mean([r["elbo_train"] for r in rs])), float(np.std([r["elbo_train"] for r in rs], ddof=1))],
                "ll_joint": [float(np.mean([r["ll_test_joint"] for r in rs])), float(np.std([r["ll_test_joint"] for r in rs], ddof=1))],
                "ll_prod": [float(np.mean([r["ll_test_prod"] for r in rs])), float(np.std([r["ll_test_prod"] for r in rs], ddof=1))],
                "drift": [float(np.mean([r["drift_rmse"] for r in rs])), float(np.std([r["drift_rmse"] for r in rs], ddof=1))],
                "min_ess_joint": float(min(r["ll_test_joint_min_ess"] for r in rs)),
            }
        # Key paired contrasts
        def paired(va, vb, field):
            d = np.array([by_v[va][s][field] - by_v[vb][s][field] for s in SEEDS])
            return [float(d.mean()), float(d.std(ddof=1)), d.tolist()]
        summary[Hk]["paired"] = {
            "joint_minus_prod_elbo": paired("pl_joint_full", "pl_prod_full", "elbo_train"),
            "joint_minus_prod_ll": paired("pl_joint_full", "pl_prod_full", "ll_test_joint"),
            "block_minus_joint_elbo": paired("pl_block_joint", "pl_joint_full", "elbo_train"),
            "block_minus_joint_ll": paired("pl_block_joint", "pl_joint_full", "ll_test_joint"),
            "block_minus_prod_ll": paired("pl_block_joint", "pl_prod_full", "ll_test_joint"),
            "preclr16_minus_joint_ll": paired("pl_prec_lr16", "pl_joint_full", "ll_test_joint"),
            "preclr16_minus_prod_ll": paired("pl_prec_lr16", "pl_prod_full", "ll_test_joint"),
            "pl_joint_minus_grid_joint_ll": paired("pl_joint_full", "grid_joint_full", "ll_test_joint"),
            "block_minus_grid_joint_ll": paired("pl_block_joint", "grid_joint_full", "ll_test_joint"),
        }
    json.dump({"rows": all_rows, "summary": summary, "scaling": sc}, open(os.path.join(OUT, "x11b_structured_q.json"), "w"), indent=1)
    print("DONE X11b.")
