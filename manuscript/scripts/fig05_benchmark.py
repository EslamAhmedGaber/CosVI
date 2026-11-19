"""Figure 5 — the published bridge benchmark, run with the authors' own code (G0), scored against three references.

(a) Posterior standard deviation of the fOU bridge (theta = 1, H = 0.9, T = 2, X(0) = 0, one noisy observation at T):
    the exact posterior of the simulated model (standard fBm started at zero; black), the analytic curve the authors'
    code compares against (stationary process conditioned at zero; grey dotted), the exact posterior of the lift as
    coded (vermilion dashed), the authors' fitted posterior (seed 0, 256 paths, band +-2 Monte Carlo standard errors),
    CosVI and CosVI-32 (mean of three seeds; analytic standard deviation of the fitted Gaussian).
(b) Integrated relative error of the authors' fitted posterior SD against each reference, for the four runs with two
    seeds (theta = 1, H = 0.7 and 0.9), with the central 95 % range of the same error for a PERFECT fit read out from
    256 paths of the exact own-lift posterior (read-out control), our 5 % reproduction tolerance, and CosVI / CosVI-32
    against the standard reference (log scale).
Sources: results/g0/bridge_*_theta1.0_seed{0,1}_steps2000.json, results/g0/readout_control.json, results/n3/.
"""
import os, sys, json, glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S


def ierr(s, ref):
    return 100 * np.abs(np.asarray(s) - np.asarray(ref)).sum() / np.asarray(ref).sum()


def ours(rep, H):
    rs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(S.RES, "n3", f"n3_{rep}_H{H:.2f}_theta1.0_seed*.json")))]
    return np.array(rs[0]["times"]), np.mean([r["std"] for r in rs], 0), np.mean([100 * r["metrics"]["exact_true"]["std_relerr_integrated"] for r in rs])


def draw(layout):
    S.setup()
    W = S.WIDE[layout]
    fig = plt.figure(figsize=(W, 2.75))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.08], wspace=0.30, left=0.075, right=0.985, top=0.80, bottom=0.16)
    # (a) curves
    ax = fig.add_subplot(gs[0, 0])
    d = S.load(os.path.join("g0", "bridge_H0.90_theta1.0_seed0_steps2000.json"))
    t = np.array(d["times"]); ex = np.array(d["std_ref_true"]); th = np.array(d["std_empirical"])
    au = np.array(d["std_ref_authors"]); own = np.array(d["std_ref_ownmodel"]); se = d["mc_std_relative_se"]
    ax.fill_between(t, th * (1 - 2 * se), th * (1 + 2 * se), color=S.C["their"], alpha=0.16, lw=0, zorder=1)
    ax.plot(t, au, color=S.C["grey"], ls=":", lw=1.2, zorder=2, label="authors' analytic curve")
    ax.plot(t, own, color=S.C["ma"], ls="--", lw=1.1, zorder=2, label="exact posterior of the lift as coded")
    ax.plot(t, th, color=S.C["their"], ls="-", lw=1.4, zorder=3, label="authors' code, fitted (seed 0)")
    ax.plot(t, ex, **S.st("ref", label="exact posterior, standard model", lw=1.1))
    tc, cj, ecj = ours("cosine", 0.9); tv, cv, ecv = ours("twoscale", 0.9)
    ax.plot(tc, cj, color=S.C["cosvi_j"], ls=(0, (4, 2)), lw=1.6, zorder=8, label=r"\CosVI-32")
    ax.plot(tv, cv, color=S.C["cosvi"], ls=(0, (1.2, 1.6)), lw=2.0, zorder=9, label=r"\CosVI")
    ax.set_xlim(0, 2.0); ax.set_ylim(0, None)
    ax.set_xlabel(r"time $t$"); ax.set_ylabel(r"posterior std of $X_t$")
    ax.text(0.98, 0.97, r"$H=0.9$, $\theta=1$", transform=ax.transAxes, ha="right", va="top", fontsize=7.4)
    S.clean(ax)
    ax.legend(loc="lower center", fontsize=6.0, handlelength=2.1, labelspacing=0.18, borderaxespad=0.1, ncol=2,
              bbox_to_anchor=(0.5, 1.01), columnspacing=0.8)
    S.panel_label(ax, "a", x=-0.16, y=1.20)
    # (b) three references, four runs
    ax = fig.add_subplot(gs[0, 1])
    rc = S.load(os.path.join("g0", "readout_control.json"))
    runs = [(0.7, 0), (0.7, 1), (0.9, 0), (0.9, 1)]
    xs = np.array([0.0, 0.55, 1.6, 2.15])
    lo = min(v["readout_error_percentiles_2.5_50_97.5"]["256"][0] for v in rc["by_H"].values())
    hi = max(v["readout_error_percentiles_2.5_50_97.5"]["256"][2] for v in rc["by_H"].values())
    for H, x0 in ((0.7, xs[:2].mean()), (0.9, xs[2:].mean())):
        p = rc["by_H"][str(H)]["readout_error_percentiles_2.5_50_97.5"]["256"]
        ax.fill_between([x0 - 0.52, x0 + 0.52], [p[0]] * 2, [p[2]] * 2, color=S.C["warm"], lw=0, zorder=0)
        ax.plot([x0 - 0.52, x0 + 0.52], [p[1]] * 2, color="#E7B9A0", lw=0.8, zorder=0)
    ax.axhline(5, color=S.C["grey"], ls="--", lw=0.7, zorder=1)
    mk = {"authors_curve": dict(marker="^", ms=5.0, mfc="white", mec="#666666", color="#666666"),
          "standard_fou": dict(marker="o", ms=4.8, mfc=S.C["ref"], mec=S.C["ref"], color=S.C["ref"]),
          "own_lift": dict(marker="s", ms=4.4, mfc=S.C["ma"], mec=S.C["ma"], color=S.C["ma"])}
    off = {"authors_curve": -0.12, "standard_fou": 0.0, "own_lift": 0.12}
    for (H, sd), x in zip(runs, xs):
        r = rc["runs"][f"{H}|{sd}"]
        vals = [r[k] for k in mk]
        ax.plot([x + off[k] for k in mk], vals, color="#D0D0D0", lw=0.7, zorder=1)
        for k in mk:
            ax.plot([x + off[k]], [r[k]], ls="none", zorder=3, **mk[k])
    for H, x0 in ((0.7, xs[:2].mean()), (0.9, xs[2:].mean())):
        _, _, e32 = ours("cosine", H); _, _, ecv = ours("twoscale", H)
        ax.plot([x0 - 0.07], [e32], ls="none", marker="o", ms=4.0, mfc="white", mec=S.C["cosvi_j"], mew=1.1, zorder=4)
        ax.plot([x0 + 0.07], [ecv], ls="none", marker="o", ms=4.4, mfc=S.C["cosvi"], mec=S.C["cosvi"], zorder=5)
    ax.set_yscale("log"); ax.set_ylim(0.05, 45)
    ax.set_yticks([0.1, 1, 10]); ax.set_yticklabels([r"$0.1$", r"$1$", r"$10$"])
    ax.set_xticks(list(xs)); ax.set_xticklabels([r"seed 0", r"seed 1", r"seed 0", r"seed 1"], fontsize=6.6)
    ax.set_xlim(-0.45, 2.75)
    for H, x0 in ((0.7, xs[:2].mean()), (0.9, xs[2:].mean())):
        ax.text(x0, 33, rf"$H={H}$", ha="center", fontsize=7.0, color="#333333")
    ax.set_ylabel(r"integrated std error (\%)")
    handles = [Line2D([], [], ls="none", label="vs authors' curve", **mk["authors_curve"]),
               Line2D([], [], ls="none", label="vs standard model", **mk["standard_fou"]),
               Line2D([], [], ls="none", label="vs lift as coded", **mk["own_lift"]),
               Patch(color=S.C["warm"], label=r"perfect fit, 256 paths (95\,\%)"),
               Line2D([], [], ls="none", marker="o", ms=4.4, mfc=S.C["cosvi"], mec=S.C["cosvi"], label=r"\CosVI"),
               Line2D([], [], ls="none", marker="o", ms=4.0, mfc="white", mec=S.C["cosvi_j"], label=r"\CosVI-32"),
               Line2D([], [], color=S.C["grey"], ls="--", lw=0.7, label=r"5\,\%, our tolerance")]
    ax.legend(handles=handles, loc="lower center", fontsize=6.0, handletextpad=0.3, labelspacing=0.25, borderaxespad=0.1,
              bbox_to_anchor=(0.5, 1.01), ncol=3, columnspacing=0.9)
    S.clean(ax)
    S.panel_label(ax, "b", x=-0.14, y=1.20)
    S.save(fig, "fig05_benchmark", layout)


if __name__ == "__main__":
    for lay in S.LAYOUTS:
        draw(lay)
    print("ok fig05")
