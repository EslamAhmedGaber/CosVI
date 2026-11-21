"""Figure 1 — CosVI at a glance.

(a) The two scales of CosVI on one exact fBm path (H = 0.75, T = 1, 4096 steps, Davies-Harte): the projection on
    J = 16 cosine coordinates (smooth, carries the long memory) and the residual (short range, independent of the
    coordinates, zero at both ends; whitened on the grid in CosVI).  Illustration, seed fixed.
(b) Estimated Hurst index against the truth on the double-well data of Section 5: CosVI (profile over a grid of fixed H,
    mean and sd over seeds 0-2; H_true = 0.9 from the grid extended to 0.95, X1), CosVI-32 with a joint gradient (N1,
    seeds 0-1), the competitor's system learning H jointly (validated port, published and best nodes, seeds 0-2), and the
    competitor's own code on one path per H_true (G0-H).  Band: +-0.05 around the truth.
(c) The published bridge benchmark: error of the posterior std against the exact posterior of the simulated model,
    all eight configurations; their code (seed 0; seed 1 where it was run), CosVI-32 and CosVI (means of three seeds).
    Revised 18 Nov 2026: full CosVI shown beside CosVI-32; the 5 % line is our reproduction tolerance.
Sources: fnvi (coordinates), results/summary_trained.json, results/n1b/x1_summary.json, results/n1/, results/g0/, results/n3/.
"""
import os, sys, json, glob
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S
sys.path.insert(0, os.path.join(S.WORK, "code"))
from fnvi.coords import CosineCoords
from fnvi.simulate import fbm_increments


def decomposition(H=0.75, T=1.0, J=16, N=4096, seed=7):
    rng = np.random.default_rng(seed)
    dB = fbm_increments(T, N, H, 1, rng)[0]
    t = np.linspace(0, T, N + 1); B = np.concatenate([[0.0], np.cumsum(dB)])
    tm = 0.5 * (t[1:] + t[:-1])
    hk = np.stack([np.full_like(tm, T ** -0.5)] + [np.sqrt(2 / T) * np.cos(k * np.pi * tm / T) for k in range(1, J)], 0)
    Z = hk @ dB
    cc = CosineCoords(H, T, J)
    xi = np.linalg.solve(cc.L, Z)
    A = cc.path_funcs(t[1:], "projection")
    BJ = np.concatenate([[0.0], A @ xi])
    return t, B, BJ, B - BJ


def h_points():
    s = json.load(open(os.path.join(S.RES, "summary_trained.json")))
    x1 = json.load(open(os.path.join(S.RES, "n1b", "x1_summary.json")))["part_i"]
    cosvi = {0.6: s["twoscale|0.6"]["H_profile"][:2], 0.75: s["twoscale|0.75"]["H_profile"][:2],
             0.9: [x1["twoscale|0.9"]["H_profile_mean"], x1["twoscale|0.9"]["H_profile_sd"]]}
    port = {H: s[f"competitor_joint_published|{H}"]["H_profile"][:2] for H in (0.6, 0.75, 0.9)}
    portb = {H: s[f"competitor_joint_best|{H}"]["H_profile"][:2] for H in (0.6, 0.75, 0.9)}
    code = {}
    for H in (0.6, 0.75, 0.9):
        d = json.load(open(glob.glob(os.path.join(S.RES, "g0", f"learnH_H{H:.2f}_theta0_seed0.json"))[0]))
        code[H] = d["H_fit"]
    joint = {}
    for H in (0.6, 0.75, 0.9):
        v = [json.load(open(f))["H_learned"] for f in glob.glob(os.path.join(S.RES, "n1", f"n1_seed*_H{H:.2f}.json"))]
        joint[H] = [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0]
    return cosvi, port, portb, code, joint


def draw(layout):
    S.setup()
    W = S.WIDE[layout]
    fig = plt.figure(figsize=(W, 2.6))
    outer = fig.add_gridspec(1, 3, width_ratios=[1.12, 1.0, 1.0], wspace=0.36, left=0.055, right=0.99, top=0.86, bottom=0.17)
    # (a) two scales
    inner = outer[0, 0].subgridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.08)
    ax1 = fig.add_subplot(inner[0]); ax2 = fig.add_subplot(inner[1], sharex=ax1)
    t, B, BJ, R = decomposition()
    ax1.plot(t, B, color=S.C["ref"], lw=0.55, label=r"fBm, $H=0.75$")
    ax1.plot(t, BJ, color=S.C["cosvi"], lw=2.0, label=r"cosine part, $J=16$")
    ax1.set_ylabel(r"$B_H(t)$")
    ax1.legend(loc="lower right", fontsize=6.4, handlelength=1.4, labelspacing=0.2, borderaxespad=0.1)
    ax1.tick_params(labelbottom=False)
    ax2.plot(t, R, color="#4D4D4D", lw=0.5)
    ax2.axhline(0, color="#BBBBBB", lw=0.5, zorder=0)
    ax2.set_ylabel(r"residual", fontsize=7.5)
    ax2.set_xlabel(r"time $t$")
    ax2.set_xlim(0, 1)
    for a in (ax1, ax2):
        S.clean(a)
    ax1.set_title(r"the two scales of \CosVI")
    S.panel_label(ax1, "a", x=-0.25, y=1.04)
    # (b) learned H
    ax = fig.add_subplot(outer[0, 1])
    cosvi, port, portb, code, joint = h_points()
    Hs = np.array([0.6, 0.75, 0.9])
    ax.fill_between([0.55, 0.915], [0.50, 0.865], [0.60, 0.965], color="#EEEEEE", lw=0, zorder=0)
    ax.plot([0.55, 0.915], [0.55, 0.915], color=S.C["ref"], lw=0.8, zorder=1)
    ax.errorbar(Hs - 0.008, [portb[H][0] for H in Hs], yerr=[portb[H][1] for H in Hs], color=S.C["their"], ls=":", lw=1.0,
                marker="D", ms=3.4, mfc="white", mec=S.C["their"], capsize=1.5, zorder=3, label="MA-fBm system, best nodes")
    ax.errorbar(Hs, [port[H][0] for H in Hs], yerr=[port[H][1] for H in Hs], color=S.C["their"], ls="-", lw=1.2,
                marker="^", ms=4.2, mfc=S.C["their"], mec=S.C["their"], capsize=1.5, zorder=3, label="MA-fBm system (port)")
    ax.plot(Hs, [code[H] for H in Hs], color=S.C["their"], ls="none", marker="*", ms=6.0, zorder=4,
            label="MA-fBm public code, one path")
    ax.errorbar(Hs, [joint[H][0] for H in Hs], yerr=[joint[H][1] for H in Hs], color=S.C["cosvi_j"], ls="--", lw=1.0,
                marker="o", ms=3.6, mfc="white", mec=S.C["cosvi_j"], capsize=1.5, zorder=4, label=r"\CosVI-32, joint gradient")
    ax.errorbar(Hs + 0.005, [cosvi[H][0] for H in Hs], yerr=[cosvi[H][1] for H in Hs], color=S.C["cosvi"], ls="-", lw=1.9,
                marker="o", ms=4.6, mfc=S.C["cosvi"], mec=S.C["cosvi"], capsize=1.8, zorder=6, label=r"\CosVI, profile")
    ax.set_xlim(0.54, 1.13); ax.set_ylim(0.48, 0.98)
    ax.set_xticks(Hs); ax.set_xlabel(r"true Hurst index $H$")
    ax.set_ylabel(r"estimated $\widehat H$")
    ax.set_title(r"learning the Hurst index")
    lab = dict(fontsize=6.3, va="center", ha="left")
    ax.text(0.922, cosvi[0.9][0], r"\CosVI", color=S.C["cosvi"], **lab)
    ax.text(0.922, joint[0.9][0] + 0.015, r"\CosVI-32,", color="#3A93C9", **lab)
    ax.text(0.922, joint[0.9][0] - 0.011, r"joint", color="#3A93C9", **lab)
    ax.text(0.922, port[0.9][0] + 0.011, r"MA-fBm sys.", color=S.C["their"], **lab)
    ax.text(0.922, code[0.9] - 0.015, r"MA-fBm code", color=S.C["their"], **lab)
    ax.text(0.560, portb[0.6][0] + 0.068, r"MA-fBm sys.,", color=S.C["their"], **lab)
    ax.text(0.560, portb[0.6][0] + 0.038, r"best nodes", color=S.C["their"], **lab)
    S.clean(ax); S.panel_label(ax, "b", x=-0.22)
    # (c) benchmark
    ax = fig.add_subplot(outer[0, 2])
    cfg = [(th, H) for th in (1.0, 0.0) for H in (0.6, 0.7, 0.8, 0.9)]
    xs = np.array([k + (0.6 if th == 0.0 else 0.0) for k, (th, H) in enumerate(cfg)])
    et = [100 * S.load(os.path.join("g0", f"bridge_H{H:.2f}_theta{th:.1f}_seed0_steps2000.json"))["metrics"]["ref_true"]["std_relerr_integrated"] for th, H in cfg]
    e1 = {k: 100 * S.load(os.path.join("g0", f"bridge_H{H:.2f}_theta{th:.1f}_seed1_steps2000.json"))["metrics"]["ref_true"]["std_relerr_integrated"]
          for k, (th, H) in enumerate(cfg) if os.path.exists(os.path.join(S.RES, "g0", f"bridge_H{H:.2f}_theta{th:.1f}_seed1_steps2000.json"))}
    n3m = lambda rep, th, H: 100 * np.mean([json.load(open(f))["metrics"]["exact_true"]["std_relerr_integrated"]
                                             for f in glob.glob(os.path.join(S.RES, "n3", f"n3_{rep}_H{H:.2f}_theta{th:.1f}_seed*.json"))])
    ec = [n3m("cosine", th, H) for th, H in cfg]
    ev = [n3m("twoscale", th, H) for th, H in cfg]
    for x, a, b in zip(xs, et, ev):
        ax.plot([x, x], [b, a], color="#DDDDDD", lw=0.8, zorder=1)
    ax.plot(xs, et, ls="none", marker="^", ms=4.4, color=S.C["their"], label="MA-fBm code", zorder=3)
    ax.plot([xs[k] + 0.18 for k in e1], list(e1.values()), ls="none", marker="^", ms=3.8, mfc="white", mec=S.C["their"],
            color=S.C["their"], label="MA-fBm, seed 1", zorder=3)
    ax.plot(xs, ec, ls="none", marker="o", ms=3.8, mfc="white", mec=S.C["cosvi_j"], mew=1.0, color=S.C["cosvi_j"],
            label=r"\CosVI-32", zorder=4)
    ax.plot(xs, ev, ls="none", marker="o", ms=4.4, color=S.C["cosvi"], label=r"\CosVI", zorder=5)
    ax.axhline(5, color=S.C["grey"], ls="--", lw=0.7)
    ax.text(xs[0] - 0.12, 4.3, r"5\,\% tolerance", fontsize=6.1, color="#666666", ha="left", va="top")
    ax.set_yscale("log"); ax.set_ylim(0.005, 900)
    ax.set_yticks([0.01, 0.1, 1, 10]); ax.set_yticklabels([r"$0.01$", r"$0.1$", r"$1$", r"$10$"])
    ax.set_xticks(xs); ax.set_xticklabels([rf"${H:.1f}$" for th, H in cfg], fontsize=6.6)
    ax.axvline((xs[3] + xs[4]) / 2, color="#E3E3E3", lw=0.6, zorder=0)
    ax.text(xs[:4].mean(), 0.0068, r"$\theta_{\rm OU}=1$", ha="center", fontsize=6.5, color="#555555")
    ax.text(xs[4:].mean(), 0.0068, r"$\theta_{\rm OU}=0$", ha="center", fontsize=6.5, color="#555555")
    ax.set_xlabel(r"Hurst index $H$")
    ax.set_ylabel(r"posterior std error (\%)")
    ax.set_title(r"published bridge benchmark")
    ax.legend(loc="upper center", fontsize=6.0, handletextpad=0.2, borderaxespad=0.15, labelspacing=0.2, ncol=2,
              columnspacing=0.8)
    S.clean(ax); S.panel_label(ax, "c", x=-0.22)
    S.save(fig, "fig01_hero", layout)


if __name__ == "__main__":
    for lay in S.LAYOUTS:
        draw(lay)
    print("ok fig01")
