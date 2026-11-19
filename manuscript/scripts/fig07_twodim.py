"""Figure 7 — the two-dimensional damped Duffing system (X2; gate fixed 16 Nov 07:03, FAILED: 12 of 18 drift wins
against the MA-fBm laws where 14 were required; no representation ahead of CosVI in held-out likelihood).

(a) True vector field (grey streamlines) and the field learned with CosVI (blue arrows) at H = 0.75, data seed 0, on
    the region visited by the data (shading: occupation density of the training paths).
(b) Drift error (occupation-weighted RMSE) per representation and H: one dot per data seed, bar = mean.
(c) Paired difference of held-out log-likelihood per test series, CosVI minus comparator (exact grid law, grid-
    coordinate refit), mean +- sd over three seeds; above zero favours CosVI.
Sources: results/n8/n8_{rep}_seed{s}_H{H}.json, results/n8/x2_summary.json.
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S

REPS = [("twoscale", "cosvi", r"\CosVI"), ("cosine", "cosvi_j", r"\CosVI-32"), ("grid", "grid", "exact grid"),
        ("markov", "ma", r"MA-fBm law, publ."), ("markov_best", "ma_best", r"MA-fBm law, best")]
HS = (0.6, 0.75, 0.9)


def run(rep, H, s):
    return json.load(open(os.path.join(S.RES, "n8", f"n8_{rep}_seed{s}_H{H:.2f}.json")))


def draw(layout):
    S.setup()
    W = S.WIDE[layout]
    if layout == "two_columns":
        fig = plt.figure(figsize=(W, 2.45))
        gs = fig.add_gridspec(1, 3, width_ratios=[0.82, 1.15, 1.0], wspace=0.36, left=0.06, right=0.99, top=0.87, bottom=0.17)
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    else:
        fig = plt.figure(figsize=(W, 4.7))
        gs = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.45, left=0.09, right=0.99, top=0.95, bottom=0.10)
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    # (a) field
    ax = axes[0]
    d = run("twoscale", 0.75, 0)
    x = np.array(d["field"]["x"]); Ft = np.array(d["field"]["true"]); Ff = np.array(d["field"]["fit"]); occ = np.array(d["field"]["occupation"])
    X, Y = np.meshgrid(x, x, indexing="xy")
    ax.contourf(X, Y, occ, levels=np.quantile(occ[occ > 0], [0.0, 0.5, 0.8, 0.95, 1.0]) if (occ > 0).sum() > 10 else 4,
                colors=["#F7F7F7", "#ECECEC", "#DEDEDE", "#CFCFCF"], zorder=0)
    ax.streamplot(x, x, Ft[..., 0], Ft[..., 1], color="#9A9A9A", linewidth=0.6, density=0.75, arrowsize=0.6, zorder=1)
    m = occ > np.quantile(occ, 0.55)
    ax.quiver(X[m], Y[m], Ff[..., 0][m], Ff[..., 1][m], color=S.C["cosvi"], angles="xy", scale_units="xy", scale=6.0,
              width=0.006, headwidth=3.5, headlength=4, zorder=3)
    ax.set_xlim(x.min(), x.max()); ax.set_ylim(x.min(), x.max()); ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$", labelpad=1); ax.set_ylabel(r"$x_2$", labelpad=1)
    ax.set_title(r"learned field, $H=0.75$")
    S.clean(ax); S.panel_label(ax, "a", x=-0.30)
    # (b) drift RMSE
    ax = axes[1]
    for j, H in enumerate(HS):
        for k, (rep, key, name) in enumerate(REPS):
            v = np.array([run(rep, H, s)["drift_rmse_occupation"] for s in range(3)])
            xx = j * 1.0 - 0.32 + 0.16 * k
            mfc = S.C[key] if key in ("cosvi", "grid", "ma_best") else "white"
            ax.plot([xx - 0.06, xx + 0.06], [v.mean()] * 2, color=S.C[key], lw=1.8, zorder=3, solid_capstyle="butt")
            ax.plot([xx] * 3, v, ls="none", marker="o", ms=2.9, mfc=mfc, mec=S.C[key], mew=0.8, zorder=4,
                    label=name if j == 0 else None)
    x2 = json.load(open(os.path.join(S.RES, "n8", "x2_summary.json")))
    for j, H in enumerate(HS):
        w = sum(int(x2["by_H"][str(H)]["paired_vs_cosvi"][r]["drift_wins"].split("/")[0]) for r in ("markov", "markov_best"))
        ax.text(j, 1.14, rf"{w}/6", ha="center", fontsize=6.4, color="#444444")
    ax.text(-0.5, 1.205, r"\CosVI{} lower than both lifts:", ha="left", fontsize=6.0, color="#444444")
    ax.set_xticks(range(3)); ax.set_xticklabels([rf"$H={H}$" for H in HS])
    ax.set_ylim(0.38, 1.25); ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel(r"drift RMSE", labelpad=1)
    handles, labels = ax.get_legend_handles_labels()
    S.clean(ax, grid_y=True); S.panel_label(ax, "b", x=-0.20)
    # (c) paired held-out difference
    ax = axes[2]
    comps = [r for r in REPS if r[0] != "twoscale"]
    for j, H in enumerate(HS):
        for k, (rep, key, name) in enumerate(comps):
            p = x2["by_H"][str(H)]["paired_vs_cosvi"][rep]["grid_law"]
            xx = j * 1.0 - 0.24 + 0.16 * k
            mfc = S.C[key] if key in ("grid", "ma_best") else "white"
            ax.errorbar([xx], [p["mean"]], yerr=[p["sd"]], color=S.C[key], lw=1.0, capsize=1.6, marker="o", ms=3.6, mfc=mfc,
                        mec=S.C[key], zorder=3)
    ax.axhline(0, color="#888888", lw=0.7, zorder=1)
    ax.set_xticks(range(3)); ax.set_xticklabels([rf"$H={H}$" for H in HS])
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel(r"$\ell(\mathrm{CosVI})-\ell(r)$, nats", labelpad=1)
    ax.text(2.45, ax.get_ylim()[1] * 0.96, r"\CosVI{} better $\uparrow$", ha="right", va="top", fontsize=6.0, color="#555555")
    ax.legend(handles, labels, loc="lower left", fontsize=5.8, ncol=2, handletextpad=0.1, columnspacing=0.6, labelspacing=0.2,
              borderaxespad=0.1)
    S.clean(ax, grid_y=True); S.panel_label(ax, "c", x=-0.20)
    S.save(fig, "fig07_twodim", layout)


if __name__ == "__main__":
    for lay in S.LAYOUTS:
        draw(lay)
    print("ok fig07")
