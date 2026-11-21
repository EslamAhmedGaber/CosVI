"""X11a (Tier 2): Rotated-coordinate causal control for Section 5.3.

Tests the exact mechanism proposed in Section 5.3 ("a control that trains the grid in randomly
rotated coordinates would test this mechanism"):
All four arms share the EXACT SAME discrete grid likelihood and posterior precision spectrum:
  0. twoscale_pl : CosVI-PL (aligned 16 cosine + 79 residual, D = 95)
  1. grid        : exact grid Cholesky coordinates (D = 80, padded with 15 inert zero cols)
  2. rot_grid    : exact grid right-multiplied by Haar-random Q in O(80) (padded with 15 inert cols)
  3. rot_pl      : CosVI-PL right-multiplied by Haar-random Q in O(95) (destroys cosine alignment
                   while keeping D = 95 and the identical grid covariance)

Paired design: for each (H, seed) with H in {0.6, 0.75, 0.9} and seed in 0..4, all 4 arms start
from identical network weights, identical initial sigma, and share common Monte Carlo noise during
the 5000 Adam steps (identical N1b/X8 hyperparameters). Evaluated under the exact PL evaluator
(2000 steps, IWAE N_IS = 512) + drift RMSE + final training ELBO + linear sensitivity shares.

Run from code/experiments:  python -X utf8 x11a_rotated_grid.py
Out: results/x11/x11a_rotated_grid.json
"""
import os, sys, json, time
import numpy as np
from scipy.stats import ortho_group
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import n1_drift_recovery as n1
from fnvi.noise_maps import twoscale_pl_map, grid_map, path_cov_on_grid
from fnvi.train_maps import Globals, LocalsD, solve as solve_rk4
from fnvi.train_graph import cosine_factor
from x8_linear_check import obs_matrix

OUT = os.path.join(HERE, "..", "..", "results", "x11")
os.makedirs(OUT, exist_ok=True)

T, S_OBS, N, J = n1.T, n1.S_OBS, 80, 16
D = J + N - 1  # 95
OBS = np.arange(1, n1.N_OBS + 1) * (N // n1.N_OBS)
LRS = {"local": 2e-2, "net": 3e-3, "sigma": 1e-2}
ARMS = ["twoscale_pl", "grid", "rot_grid", "rot_pl"]
HS = [0.6, 0.75, 0.9]
SEEDS = [0, 1, 2, 3, 4]


def pad95(Mx):
    out = np.zeros(Mx.shape[:-1] + (D,), dtype=np.float64)
    out[..., :Mx.shape[-1]] = Mx
    return out


def make_arm_maps(H, seed):
    M_pl = np.stack(twoscale_pl_map(H, T, N, J), 0)       # (3, N, 95)
    M_gr = np.stack(grid_map(H, T, N), 0)                 # (3, N, 80)
    rng = np.random.default_rng(20261121 + 1000 * seed + int(round(100 * H)))
    Q80 = ortho_group.rvs(N, random_state=rng)
    Q95 = ortho_group.rvs(D, random_state=rng)
    M_rgr = M_gr @ Q80
    M_rpl = M_pl @ Q95
    # Verify exact law invariance to < 1e-13
    K_gr = path_cov_on_grid(*M_gr, T)
    errs = {
        "pl_vs_grid": float(np.linalg.norm(path_cov_on_grid(*M_pl, T) - K_gr) / np.linalg.norm(K_gr)),
        "rot_grid_vs_grid": float(np.linalg.norm(path_cov_on_grid(*M_rgr, T) - K_gr) / np.linalg.norm(K_gr)),
        "rot_pl_vs_grid": float(np.linalg.norm(path_cov_on_grid(*M_rpl, T) - K_gr) / np.linalg.norm(K_gr)),
    }
    assert max(errs.values()) < 1e-12, errs
    return [M_pl, pad95(M_gr), pad95(M_rgr), M_rpl], errs


def linear_sensitivity():
    res = {}
    for H in HS:
        maps, _ = make_arm_maps(H, seed=0)
        row = {}
        for name, Mx in zip(ARMS, maps):
            # Active columns only for grid / rot_grid
            act = N if name in ("grid", "rot_grid") else D
            F = obs_matrix((Mx[0, :, :act], Mx[1, :, :act], Mx[2, :, :act]))[:, :-1]
            cs = (F ** 2).sum(0)
            s_sorted = np.sort(cs)[::-1]
            c = np.cumsum(s_sorted) / cs.sum()
            ev = np.linalg.eigvalsh(np.eye(act) + F.T @ F / (S_OBS ** 2))
            row[name] = {
                "share_first_16": float(cs[:J].sum() / cs.sum()),
                "share_best_16": float(s_sorted[:J].sum() / cs.sum()),
                "n90": int(np.searchsorted(c, 0.90) + 1),
                "n99": int(np.searchsorted(c, 0.99) + 1),
                "precision_cond": float(ev[-1] / ev[0]),
            }
        res[f"{H}"] = row
    return res


def loglik_grouped(Y, X):
    # Y: (G, M, m), X: (m, G, S, M) -> (S, G, M)
    Yb = Y.permute(2, 0, 1)[:, :, None, :]
    return (-0.5 * (Yb - X) ** 2 / S_OBS ** 2 - 0.5 * np.log(2 * np.pi * S_OBS ** 2)).sum(0).permute(1, 0, 2)


def sample_crn_by_block(loc, S, n_blocks=3, block_size=4):
    """Common random numbers shared across the 4 arms of each H-block."""
    L, d = loc.scale()
    G, M, D_ = loc.mu.shape
    dev = loc.mu.device
    e_blk = torch.randn((S, n_blocks, 1, M, D_), device=dev).expand(S, n_blocks, block_size, M, D_).reshape(S, G, M, D_)
    z = loc.mu + torch.einsum("gmij,sgmj->sgmi", L, e_blk)
    e0_blk = torch.randn((S, n_blocks, 1, M), device=dev).expand(S, n_blocks, block_size, M).reshape(S, G, M)
    x0 = loc.m0 + torch.nn.functional.softplus(loc.r0) * e0_blk
    kl_z = 0.5 * ((L ** 2).sum((-1, -2)) + (loc.mu ** 2).sum(-1) - D_ - 2 * torch.log(d).sum(-1))
    v0 = torch.nn.functional.softplus(loc.r0) ** 2
    kl_x0 = 0.5 * (v0 + loc.m0 ** 2 - 1 - torch.log(v0))
    return z, x0, kl_z + kl_x0


def run_seed(seed, steps=5000, refit=2000, K_is=512, S=16):
    dev = torch.device("cuda")
    torch.manual_seed(seed)
    n_H = len(HS); n_A = len(ARMS); G = n_H * n_A  # 12 groups: [H0*(4 arms), H1*(4 arms), H2*(4 arms)]

    Ytr_list, Yte_list, occ_list, Ms_list, Mpl_eval_list = [], [], [], [], []
    law_errs = {}
    for H in HS:
        Ytr, Yte, _, occ = n1.make_data(H, seed)
        arms_maps, errs = make_arm_maps(H, seed)
        law_errs[f"{H}"] = errs
        for a_idx in range(n_A):
            Ytr_list.append(Ytr)
            Yte_list.append(Yte)
            occ_list.append(occ)
            Ms_list.append(arms_maps[a_idx])
            Mpl_eval_list.append(arms_maps[0])  # PL evaluator for every arm

    Yt = torch.as_tensor(np.stack(Ytr_list, 0), dtype=torch.float32, device=dev)   # (12, 48, 20)
    Ye = torch.as_tensor(np.stack(Yte_list, 0), dtype=torch.float32, device=dev)   # (12, 16, 20)
    Ms = torch.as_tensor(np.stack(Ms_list, 0), dtype=torch.float32, device=dev)    # (12, 3, 80, 95)
    Me = torch.as_tensor(np.stack(Mpl_eval_list, 0), dtype=torch.float32, device=dev)

    glob = Globals(G).to(dev)
    # Enforce identical initial weights across all 4 arms of each H
    with torch.no_grad():
        for hi in range(n_H):
            base_g = hi * n_A
            for a_idx in range(1, n_A):
                g = base_g + a_idx
                for w in glob.net.W:
                    w[g].copy_(w[base_g])
                for b in glob.net.b:
                    b[g].copy_(b[base_g])
                glob.s_raw[g].copy_(glob.s_raw[base_g])

    loc = LocalsD(G, Yt.shape[1], D, torch.zeros(Yt.shape[1], device=dev)).to(dev)
    with torch.no_grad():
        loc.m0.copy_(Yt[:, :, 0])

    # Inert mask for grid (a_idx=1) and rot_grid (a_idx=2)
    mv = torch.ones(G, 1, D, device=dev)
    mo = torch.ones(G, 1, D, D, device=dev)
    inert_gs = [hi * n_A + a_idx for hi in range(n_H) for a_idx in (1, 2)]
    for g in inert_gs:
        mv[g, :, N:] = 0.0
        mo[g, :, N:, :] = 0.0
        mo[g, :, :, N:] = 0.0
    loc.mu.register_hook(lambda gr: gr * mv)
    loc.rho.register_hook(lambda gr: gr * mv)
    loc.off.register_hook(lambda gr: gr * mo)
    rho0 = float(loc.rho[inert_gs[0], 0, N].item())

    T_ = lambda v: torch.tensor(float(v), device=dev)
    opt = torch.optim.Adam([
        {"params": list(loc.parameters()), "lr": T_(LRS["local"]), "base": LRS["local"]},
        {"params": list(glob.net.parameters()), "lr": T_(LRS["net"]), "base": LRS["net"]},
        {"params": [glob.s_raw], "lr": T_(LRS["sigma"]), "base": LRS["sigma"]},
    ], capturable=True)
    out_buf = {"elbo": torch.zeros(G, device=dev)}

    def step():
        z, x0, kl = sample_crn_by_block(loc, S, n_H, n_A)
        X = solve_rk4(glob, Ms, z, x0, T, OBS)
        elbo = (loglik_grouped(Yt, X).mean(0) - kl).sum(-1)
        (-elbo.sum() / Yt.shape[1]).backward()
        opt.step()
        opt.zero_grad(set_to_none=False)
        out_buf["elbo"].copy_(elbo.detach())

    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(st)
    gph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gph):
        step()

    t0 = time.time()
    hist = []
    for it in range(steps - 3):
        if it % 50 == 0:
            fac = cosine_factor(it, steps)
            for grp in opt.param_groups:
                grp["lr"].fill_(grp["base"] * fac)
        gph.replay()
        if it % 200 == 0:
            hist.append([it] + (out_buf["elbo"] / Yt.shape[1]).cpu().tolist())
    torch.cuda.synchronize()
    train_s = time.time() - t0

    # Check inert block
    with torch.no_grad():
        inert_max = max(
            float(loc.mu[inert_gs, :, N:].abs().max()),
            float((loc.rho[inert_gs, :, N:] - rho0).abs().max()),
            float(loc.off[inert_gs, :, N:, :].abs().max()),
            float(loc.off[inert_gs, :, :, N:].abs().max()),
        )
        # Final training ELBO (256 samples)
        elbo_tr = torch.zeros(G, device=dev)
        for _ in range(16):
            z, x0, kl = sample_crn_by_block(loc, 16, n_H, n_A)
            X = solve_rk4(glob, Ms, z, x0, T, OBS)
            elbo_tr += (loglik_grouped(Yt, X).mean(0) - kl).sum(-1)
        elbo_tr = (elbo_tr / 16.0 / Yt.shape[1]).cpu().numpy()

    # Freeze globals and evaluate under PL evaluator
    for p in glob.parameters():
        p.requires_grad_(False)
    torch.manual_seed(7919 * (seed + 1))
    loc_te = LocalsD(G, Ye.shape[1], D, torch.zeros(Ye.shape[1], device=dev)).to(dev)
    with torch.no_grad():
        loc_te.m0.copy_(Ye[:, :, 0])
    opt_te = torch.optim.Adam([{"params": list(loc_te.parameters()), "lr": T_(LRS["local"]), "base": LRS["local"]}], capturable=True)

    def step_te():
        z, x0, kl = sample_crn_by_block(loc_te, S, n_H, n_A)
        X = solve_rk4(glob, Me, z, x0, T, OBS)
        elbo = (loglik_grouped(Ye, X).mean(0) - kl).sum(-1)
        (-elbo.sum() / Ye.shape[1]).backward()
        opt_te.step()
        opt_te.zero_grad(set_to_none=False)

    st2 = torch.cuda.Stream(); st2.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st2):
        for _ in range(3):
            step_te()
    torch.cuda.current_stream().wait_stream(st2)
    gph_te = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gph_te):
        step_te()
    t1 = time.time()
    for it in range(refit - 3):
        if it % 50 == 0:
            fac = cosine_factor(it, refit)
            for grp in opt_te.param_groups:
                grp["lr"].fill_(grp["base"] * fac)
        gph_te.replay()
    torch.cuda.synchronize()

    # IWAE K_is = 512
    logw = []
    with torch.no_grad():
        for _ in range(K_is // 64):
            z, x0, _, _, _ = loc_te.sample(64, common=False)
            X = solve_rk4(glob, Me, z, x0, T, OBS)
            lp = loglik_grouped(Ye, X)
            lpr = -0.5 * (z ** 2).sum(-1) - 0.5 * D * np.log(2 * np.pi) - 0.5 * x0 ** 2 - 0.5 * np.log(2 * np.pi)
            logw.append(lp + lpr - loc_te.log_q(z, x0))
        lw = torch.cat(logw, 0)
        ll_pl = (torch.logsumexp(lw, 0) - np.log(lw.shape[0])).cpu().numpy()  # (12, 16)
    eval_s = time.time() - t1

    # Drift RMSE
    xg = np.linspace(-1.5, 1.5, 301)
    with torch.no_grad():
        fg = glob.net(torch.as_tensor(xg, dtype=torch.float32, device=dev)[None, :, None].expand(G, -1, 1))[..., 0].cpu().numpy()
    ft = n1.f_true(xg)
    drifts = []
    for g in range(G):
        w_occ, edges = np.histogram(np.clip(occ_list[g], -1.5, 1.5), bins=300, range=(-1.5, 1.5), density=True)
        w = np.interp(xg, 0.5 * (edges[1:] + edges[:-1]), w_occ); w = w / w.sum()
        drifts.append(float(np.sqrt((((fg[g] - ft) ** 2) * w).sum())))

    sig = glob.sigma().cpu().tolist()
    per_H = {}
    for hi, H in enumerate(HS):
        per_H[f"{H}"] = {}
        for a_idx, name in enumerate(ARMS):
            g = hi * n_A + a_idx
            per_H[f"{H}"][name] = {
                "ll_pl_mean": float(ll_pl[g].mean()),
                "ll_pl_per_series": ll_pl[g].tolist(),
                "elbo_train": float(elbo_tr[g]),
                "drift_rmse": drifts[g],
                "sigma": sig[g],
            }
        d_pl = per_H[f"{H}"]["twoscale_pl"]["ll_pl_mean"]
        print(f"  seed {seed} H={H:.2f}: PL={d_pl:+.2f} | grid={per_H[f'{H}']['grid']['ll_pl_mean']:+.2f} "
              f"| rot_grid={per_H[f'{H}']['rot_grid']['ll_pl_mean']:+.2f} | rot_pl={per_H[f'{H}']['rot_pl']['ll_pl_mean']:+.2f} "
              f"| train={train_s:.0f}s eval={eval_s:.0f}s inert={inert_max:.1e}", flush=True)
    return {"seed": seed, "train_s": train_s, "eval_s": eval_s, "inert_max": inert_max,
            "law_errs": law_errs, "by_H": per_H}


def summarize(seed_recs, sens):
    summary = {}
    for H in HS:
        Hk = f"{H}"
        row = {"arms": {}, "contrasts_vs_pl": {}}
        for a in ARMS:
            lls = np.array([r["by_H"][Hk][a]["ll_pl_mean"] for r in seed_recs])
            els = np.array([r["by_H"][Hk][a]["elbo_train"] for r in seed_recs])
            drs = np.array([r["by_H"][Hk][a]["drift_rmse"] for r in seed_recs])
            row["arms"][a] = {
                "ll_mean": float(lls.mean()), "ll_sd": float(lls.std(ddof=1)), "ll_se": float(lls.std(ddof=1) / np.sqrt(len(lls))),
                "elbo_mean": float(els.mean()), "drift_mean": float(drs.mean()), "drift_sd": float(drs.std(ddof=1)),
            }
        pl_ll = np.array([r["by_H"][Hk]["twoscale_pl"]["ll_pl_mean"] for r in seed_recs])
        pl_el = np.array([r["by_H"][Hk]["twoscale_pl"]["elbo_train"] for r in seed_recs])
        pl_dr = np.array([r["by_H"][Hk]["twoscale_pl"]["drift_rmse"] for r in seed_recs])
        for a in ("grid", "rot_grid", "rot_pl"):
            diff_ll = pl_ll - np.array([r["by_H"][Hk][a]["ll_pl_mean"] for r in seed_recs])
            diff_el = pl_el - np.array([r["by_H"][Hk][a]["elbo_train"] for r in seed_recs])
            diff_dr = np.array([r["by_H"][Hk][a]["drift_rmse"] for r in seed_recs]) - pl_dr
            se_ll = float(diff_ll.std(ddof=1) / np.sqrt(len(diff_ll)))
            se_el = float(diff_el.std(ddof=1) / np.sqrt(len(diff_el)))
            row["contrasts_vs_pl"][a] = {
                "ll_gain_mean": float(diff_ll.mean()), "ll_gain_se": se_ll, "ll_wins": int((diff_ll > 0).sum()),
                "ll_verdict_2se": "present, +" if diff_ll.mean() > 2 * se_ll else ("present, -" if diff_ll.mean() < -2 * se_ll else "n.d."),
                "elbo_gain_mean": float(diff_el.mean()), "elbo_gain_se": se_el, "elbo_wins": int((diff_el > 0).sum()),
                "drift_red_mean": float(diff_dr.mean()), "drift_red_se": float(diff_dr.std(ddof=1) / np.sqrt(len(diff_dr))),
                "per_seed_ll_gain": diff_ll.tolist(),
            }
        summary[Hk] = row
    return {"rule": "X11a rotated-coordinate causal control (5 seeds, |mean| > 2 SE)",
            "linear_sensitivity": sens, "summary": summary, "seeds": seed_recs}


if __name__ == "__main__":
    print("=== X11a: Computing linear sensitivity shares ===", flush=True)
    sens = linear_sensitivity()
    for Hk, v in sens.items():
        print(f"  H={Hk}: " + " | ".join(f"{a}: top16={100*v[a]['share_best_16']:.1f}% n90={v[a]['n90']}" for a in ARMS), flush=True)
    print("=== X11a: Running 5-seed paired GPU control ===", flush=True)
    recs = []
    for s in SEEDS:
        recs.append(run_seed(s))
        out = summarize(recs, sens)
        json.dump(out, open(os.path.join(OUT, "x11a_rotated_grid.json"), "w"), indent=1)
    print("\n=== X11a SUMMARY (5 seeds, PL minus comparator) ===", flush=True)
    for Hk, row in out["summary"].items():
        print(f"  H={Hk}:")
        for a, c in row["contrasts_vs_pl"].items():
            print(f"    PL - {a:8s}: LL {c['ll_gain_mean']:+.2f} (SE {c['ll_gain_se']:.2f}) [{c['ll_wins']}/5, {c['ll_verdict_2se']}] | ELBO {c['elbo_gain_mean']:+.2f} ({c['elbo_wins']}/5) | drift_red {c['drift_red_mean']:+.3f}")
    print("DONE X11a.")
