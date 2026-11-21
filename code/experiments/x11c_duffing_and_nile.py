"""X11c (Tier 2 & 3):
  Part 1 — 2D damped Duffing system (3 seeds x 3 H = 9 groups):
    Compare CosVI-PL (full prod, 18,339 local params/series), CosVI-PL (2D BlockJoint, 915 params/series,
    20x smaller!), exact grid, and MA-fBm published law, ALL evaluated under the well-conditioned
    2D PL evaluator (fixing the 2D grid-coordinate refit failure at H=0.9!) AND grid evaluator.
  Part 2 — Nile river minima (3 seeds, N=200, T=50, H grid {0.5..0.9}):
    Compare CosVI-PL (prod full, 23,437 local params/window), CosVI-PL (joint full, 23,652),
    CosVI-PL (BlockJoint, 568 params/window — 41x fewer!), and CosVI-PL (PrecLR r=16, 3,888 — 6x fewer!),
    evaluated under the joint PL evaluator + product PL evaluator.

Run from code/experiments:  python -X utf8 x11c_duffing_and_nile.py
Out: results/x11/x11c_duffing_and_nile.json
"""
import os, sys, json, time, itertools
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import n1_drift_recovery as n1
import n1b_representations as nb
import n8_two_dim as n8
import n4_nile as n4v1
from fnvi.noise_maps import twoscale_pl_map, grid_map, markov_map
from fnvi.train_maps import Globals, LocalsD, solve as solve_rk4, loglik as loglik_1d
from fnvi.families import LocalsJointFull, LocalsBlockJoint, Locals2BlockJoint, LocalsPrecLR
from fnvi.train_graph import cosine_factor

OUT = os.path.join(HERE, "..", "..", "results", "x11")
os.makedirs(OUT, exist_ok=True)
LRS = {"local": 2e-2, "net": 3e-3, "sigma": 1e-2}
LOG2PI = float(np.log(2.0 * np.pi))


# ============================ PART 1: 2D DUFFING ============================
def maps_2d(rep, Hs, dev):
    T, N, J = nb.T, nb.N, 16
    Ms = []
    for H in Hs:
        if rep == "twoscale_pl":
            M = np.stack(twoscale_pl_map(H, T, N, J), 0)
        elif rep == "grid":
            M = np.stack(grid_map(H, T, N), 0)
        elif rep == "markov":
            M = np.stack(markov_map(H, T, N, K=5, gamma_max=20.0), 0)
        else:
            raise ValueError(rep)
        Ms.append(M)
    return torch.as_tensor(np.stack(Ms, 0), dtype=torch.float32, device=dev)


def run_duffing(steps=5000, refit=2000, K_is=512):
    dev = torch.device("cuda")
    Hs_true = [0.6, 0.75, 0.9]
    seeds = [0, 1, 2]
    combos = list(itertools.product(Hs_true, seeds)); G = len(combos)
    Hs = [H for H, _ in combos]
    data = [n8.make_data2(H, s) for H, s in combos]
    Ytr = torch.as_tensor(np.stack([d[0] for d in data]), dtype=torch.float32, device=dev)
    Yte = torch.as_tensor(np.stack([d[1] for d in data]), dtype=torch.float32, device=dev)

    Mpl = maps_2d("twoscale_pl", Hs, dev)
    Mg = maps_2d("grid", Hs, dev)

    configs = [
        ("cosvi_pl_full", "twoscale_pl", "full2"),
        ("cosvi_pl_block", "twoscale_pl", "block2"),
        ("grid_full", "grid", "full2"),
        ("markov_pub_full", "markov", "full2"),
    ]
    rows = []
    xs = np.linspace(-2, 2, 41)
    XX, YY = np.meshgrid(xs, xs)
    Pt = np.stack([XX.ravel(), YY.ravel()], -1)
    Ft = n8.f2(Pt)

    for cname, rep, fam in configs:
        torch.manual_seed(12345)
        Ms = maps_2d(rep, Hs, dev); D = Ms.shape[-1]
        glob = n8.Globals2(G).to(dev)   # identical init across all configs
        if fam == "full2":
            loc = n8.Locals2(G, Ytr.shape[1], D, Ytr[:, :, 0, :]).to(dev)
            npar = (2 * D) + (2 * D) * (2 * D + 1) // 2 + 4
        else:
            loc = Locals2BlockJoint(G, Ytr.shape[1], 16, D, Ytr[:, :, 0, :]).to(dev)
            npar = loc.num_local_params_per_series()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        hist = n8.train2(glob, loc, Ytr, Ms, 16, steps, train_globals=True, lrs=LRS)
        torch.cuda.synchronize(); train_s = time.time() - t0
        peak_mb = torch.cuda.max_memory_allocated() / (2 ** 20)
        for p in glob.parameters():
            p.requires_grad_(False)

        # Evaluate under well-conditioned 2D PL evaluator (Locals2 + Locals2BlockJoint check) AND grid evaluator
        torch.manual_seed(55555)
        loc_pl = n8.Locals2(G, Yte.shape[1], Mpl.shape[-1], Yte[:, :, 0, :]).to(dev)
        n8.train2(glob, loc_pl, Yte, Mpl, 16, refit, train_globals=False, lrs=LRS)
        ll_pl = n8.iwae2(glob, loc_pl, Yte, Mpl, K_is)

        torch.manual_seed(66666)
        loc_gr = n8.Locals2(G, Yte.shape[1], Mg.shape[-1], Yte[:, :, 0, :]).to(dev)
        n8.train2(glob, loc_gr, Yte, Mg, 16, refit, train_globals=False, lrs=LRS)
        ll_gr = n8.iwae2(glob, loc_gr, Yte, Mg, K_is)

        with torch.no_grad():
            Ff = glob.net(torch.as_tensor(Pt, dtype=torch.float32, device=dev)[None].expand(G, -1, -1).contiguous()).cpu().numpy()
        sig = glob.sigma().cpu().numpy()

        for gi, (H, s) in enumerate(combos):
            occ = data[gi][2]
            Hh, _, _ = np.histogram2d(occ[:, 0], occ[:, 1], bins=41, range=[[-2.05, 2.05], [-2.05, 2.05]])
            w = Hh.T.ravel(); w = w / w.sum()
            err2 = ((Ff[gi] - Ft) ** 2).sum(-1)
            rows.append({
                "config": cname, "rep": rep, "family": fam, "H": H, "seed": s,
                "local_params_per_series": int(npar), "train_s": train_s, "peak_mem_MB": peak_mb,
                "sigma": sig[gi].tolist(), "drift_rmse": float(np.sqrt((err2 * w).sum())),
                "elbo_end": float(hist[-1][1 + gi]),
                "ll_pl": float(ll_pl[gi].mean()), "ll_grid": float(ll_gr[gi].mean()),
            })
        print(f"  2D {cname:16s} (params/ser={npar:5d}, {train_s:5.1f}s, {peak_mb:5.1f}MB):", flush=True)
        for H in Hs_true:
            sub = [r for r in rows if r["config"] == cname and r["H"] == H]
            print(f"    H={H:.2f}: ELBO={np.mean([r['elbo_end'] for r in sub]):+6.2f} | "
                  f"LL_PL={np.mean([r['ll_pl'] for r in sub]):+6.2f} | LL_grid={np.mean([r['ll_grid'] for r in sub]):+6.2f} | "
                  f"drift={np.mean([r['drift_rmse'] for r in sub]):.3f} | sig=({np.mean([r['sigma'][0] for r in sub]):.3f},{np.mean([r['sigma'][1] for r in sub]):.3f})", flush=True)
    return rows


# ============================ PART 2: NILE MINIMA ============================
def run_nile_families(steps=5000, refit=4000, K_is=1024):
    dev = torch.device("cuda")
    T, N, J, S_OBS = 50.0, 200, 16, 0.1
    OBS = np.arange(1, 51) * (N // 50)
    HGRID = [0.5, 0.6, 0.7, 0.8, 0.9]
    SEEDS = [0, 1, 2]
    nH = len(HGRID); nS = len(SEEDS); G = nS * nH   # 15 groups batched in one CUDA graph
    Ytr_np, Yte_np, x0tr_np, x0te_np, _ = n4v1.load_windows()
    Yt = torch.as_tensor(Ytr_np, dtype=torch.float32, device=dev)
    Ye = torch.as_tensor(Yte_np, dtype=torch.float32, device=dev)
    x0tr = torch.as_tensor(x0tr_np, dtype=torch.float32, device=dev)
    x0te = torch.as_tensor(x0te_np, dtype=torch.float32, device=dev)
    Ms_base = np.stack([np.stack(twoscale_pl_map(h, T, N, J), 0) for h in HGRID], 0)   # (5, 3, N, D)
    Ms = torch.as_tensor(np.concatenate([Ms_base] * nS, axis=0), dtype=torch.float32, device=dev)  # (15, 3, N, D)
    D = Ms.shape[-1]  # 215

    def train_nile(glob, loc, Y, nsteps, train_globals):
        T_ = lambda v: torch.tensor(float(v), device=dev)
        grps = [{"params": list(loc.parameters()), "lr": T_(LRS["local"]), "base": LRS["local"]}]
        if train_globals:
            grps += [{"params": list(glob.net.parameters()), "lr": T_(LRS["net"]), "base": LRS["net"]},
                     {"params": [glob.s_raw], "lr": T_(LRS["sigma"]), "base": LRS["sigma"]}]
        opt = torch.optim.Adam(grps, capturable=True)
        def step():
            z, x0, kl, _, _ = loc.sample(16, common=False)
            X = solve_rk4(glob, Ms, z, x0, T, OBS)
            elbo = (loglik_1d(Y, X, S_OBS).mean(0) - kl).sum(-1)
            (-elbo.sum() / Y.shape[0]).backward(); opt.step(); opt.zero_grad(set_to_none=False)
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(st)
        gph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gph):
            step()
        for it in range(nsteps - 3):
            if it % 50 == 0:
                fac = cosine_factor(it, nsteps)
                for grp in opt.param_groups:
                    grp["lr"].fill_(grp["base"] * fac)
            gph.replay()

    @torch.no_grad()
    def eval_nile(glob, loc, Y, mode="elbo"):
        if mode == "elbo":
            acc = torch.zeros(G, device=dev)
            for _ in range(16):
                z, x0, kl, _, _ = loc.sample(16, common=False)
                X = solve_rk4(glob, Ms, z, x0, T, OBS)
                acc += (loglik_1d(Y, X, S_OBS).mean(0) - kl).sum(-1)
            return (acc / 16.0 / Y.shape[0]).cpu().numpy()
        logw = []
        for _ in range(K_is // 64):
            z, x0, _, _, _ = loc.sample(64, common=False)
            X = solve_rk4(glob, Ms, z, x0, T, OBS)
            lp = loglik_1d(Y, X, S_OBS)
            lpr = -0.5 * (z ** 2).sum(-1) - 0.5 * D * LOG2PI - 0.5 * x0 ** 2 - 0.5 * LOG2PI
            logw.append(lp + lpr - loc.log_q(z, x0))
        lw = torch.cat(logw, 0)
        lse = torch.logsumexp(lw, 0)
        w = torch.exp(lw - lse.unsqueeze(0))
        ess = (1.0 / (w ** 2).sum(0)).cpu().numpy()
        return (lse - np.log(lw.shape[0])).cpu().numpy(), ess

    fams = ["prod_full", "joint_full", "block_joint", "prec_lr16"]
    rows = []
    for fam in fams:
        torch.manual_seed(12345)
        glob = Globals(G).to(dev)
        # Ensure each seed s shares initial weights across its 5 H values (matches n4_nile protocol)
        with torch.no_grad():
            for si in range(nS):
                g0 = si * nH
                for hi in range(1, nH):
                    g = g0 + hi
                    for w in glob.net.W:
                        w[g].copy_(w[g0])
                    for b in glob.net.b:
                        b[g].copy_(b[g0])
                    glob.s_raw[g].copy_(glob.s_raw[g0])
        if fam == "prod_full":
            loc = LocalsD(G, Yt.shape[0], D, x0tr).to(dev)
            npar = D + D * (D + 1) // 2 + 2
        elif fam == "joint_full":
            loc = LocalsJointFull(G, Yt.shape[0], D, x0tr[None].expand(G, -1)).to(dev)
            npar = loc.num_local_params_per_series()
        elif fam == "block_joint":
            loc = LocalsBlockJoint(G, Yt.shape[0], J, D, x0tr[None].expand(G, -1)).to(dev)
            npar = loc.num_local_params_per_series()
        else:
            loc = LocalsPrecLR(G, Yt.shape[0], D, x0tr[None].expand(G, -1), rank=16).to(dev)
            npar = loc.num_local_params_per_series()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        train_nile(glob, loc, Yt, steps, True)
        torch.cuda.synchronize(); train_s = time.time() - t0
        peak_mb = torch.cuda.max_memory_allocated() / (2 ** 20)
        elbo_tr_all = eval_nile(glob, loc, Yt, "elbo").reshape(nS, nH)
        for p in glob.parameters():
            p.requires_grad_(False)
        torch.manual_seed(9000)
        loc_te = LocalsJointFull(G, Ye.shape[0], D, x0te[None].expand(G, -1)).to(dev)
        train_nile(glob, loc_te, Ye, refit, False)
        ll_te_all, ess_te_all = eval_nile(glob, loc_te, Ye, "iwae")
        ll_te_all = ll_te_all.reshape(nS, nH, -1)
        ess_te_all = ess_te_all.reshape(nS, nH, -1)
        for si, seed in enumerate(SEEDS):
            elbo_tr = elbo_tr_all[si]
            i = int(np.argmax(elbo_tr))
            if 0 < i < nH - 1:
                a, b, c = elbo_tr[i - 1], elbo_tr[i], elbo_tr[i + 1]
                Hp = float(HGRID[i] + 0.5 * (a - c) / (a - 2 * b + c) * (HGRID[1] - HGRID[0]))
            else:
                Hp = float(HGRID[i])
            rows.append({
                "family": fam, "seed": seed, "local_params_per_window": int(npar),
                "train_s": train_s / nS, "train_s_batched3": train_s, "peak_mem_MB": peak_mb,
                "H_profile": Hp, "best_group_idx": i,
                "elbo_train_per_H": elbo_tr.tolist(),
                "elbo_at_best_H": float(elbo_tr[i]),
                "ll_test_joint_pl_per_H": ll_te_all[si].mean(1).tolist(),
                "ll_test_at_profile_group": float(ll_te_all[si, i].mean()),
                "ll_test_per_window_at_profile": ll_te_all[si, i].tolist(),
                "min_ess_at_profile": float(ess_te_all[si, i].min()),
            })
        sub = [r for r in rows if r["family"] == fam]
        print(f"  Nile {fam:12s} (params/win={sub[0]['local_params_per_window']:5d}, {train_s:5.1f}s/3seeds, {sub[0]['peak_mem_MB']:5.1f}MB): "
              f"H_prof={np.mean([r['H_profile'] for r in sub]):.3f} | "
              f"trainELBO={np.mean([r['elbo_at_best_H'] for r in sub]):+.2f} | "
              f"LL_PL={np.mean([r['ll_test_at_profile_group'] for r in sub]):+.2f}+-{np.std([r['ll_test_at_profile_group'] for r in sub], ddof=1):.2f} "
              f"(minESS={min(r['min_ess_at_profile'] for r in sub):.1f})", flush=True)
    return rows


if __name__ == "__main__":
    print("=== X11c Part 1: 2D Duffing with PL evaluator & 2D BlockJoint ===", flush=True)
    r_duff = run_duffing()
    json.dump({"duffing": r_duff}, open(os.path.join(OUT, "x11c_duffing_and_nile.json"), "w"), indent=1)
    print("\n=== X11c Part 2: Nile minima with JointFull, BlockJoint, PrecLR16 ===", flush=True)
    r_nile = run_nile_families()
    json.dump({"duffing": r_duff, "nile": r_nile}, open(os.path.join(OUT, "x11c_duffing_and_nile.json"), "w"), indent=1)
    print("DONE X11c.")
