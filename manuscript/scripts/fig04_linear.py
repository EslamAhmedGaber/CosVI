"""Figure 4 — exact inference in the linear model (G2, G2b), where every posterior and evidence is known exactly.

fOU dX = -theta X dt + dB_H (theta = 1, sigma = 1), T = 2, observation noise s = 0.1.
(a) Posterior variance of the bridge (one observation at T) at H = 0.75: exact; cosine coordinates (CosVI-J) with
    J = 8, 16, 32; the MA-fBm lift with K = 5 (published node rule) and as coded by its authors.
(b) Absolute value of the MEAN SIGNED log-evidence difference log p_model(y) - log p(y) over 20 data sets with ten
    observations (not a bound on individual data sets), against the dimension: cosine
    projection (J) and the lift with published and tuned nodes (K), H = 0.75.
(c) Integrated posterior-variance error (max over data sets) against J and the number m of evenly spaced observations,
    H = 0.75; cells within 2 % outlined.  Integrated over the whole path, the error needs J >= 64 almost independently
    of m: resolving the posterior between observations needs fine time scales, which the residual of CosVI supplies.
Sources: work/results/g2/g2_linear_exact.json, g2b_posterior_resolution.json.
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S

G2 = S.load(os.path.join("g2", "g2_linear_exact.json"))
G2B = S.load(os.path.join("g2", "g2b_posterior_resolution.json"))
H = "0.75"


def draw(layout):
    S.setup()
    W = S.WIDE[layout]
    fig = plt.figure(figsize=(W, 2.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.42, left=0.07, right=0.93, top=0.84, bottom=0.18)
    r = G2["results"][H]
    cur = r["bridge_posterior_var_curves"]
    t = np.array(cur["times"])
    # (a)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t, cur["ma_authors_as_coded_K5"], color=S.C["their"], ls=":", lw=1.4, label=r"MA-fBm, as coded")
    ax.plot(t, cur["ma_default_K5"], color=S.C["ma"], ls="--", lw=1.3, label=r"MA-fBm, $K=5$")
    for J, a in ((8, 0.45), (16, 0.7), (32, 1.0)):
        ax.plot(t, cur[f"cosine_projection_J{J}"], color=S.C["cosvi"], lw=1.0 + 0.3 * (J // 16), alpha=a,
                label=r"cosine, $J{=}8,16,32$" if J == 32 else None)
    ax.plot(t, cur["exact"], color=S.C["ref"], lw=0.9, ls=(0, (1, 1)), label="exact", zorder=9)
    ax.set_xlabel(r"time $t$"); ax.set_ylabel(r"posterior variance of $X_t$")
    ax.set_xlim(0, 2); ax.set_ylim(0, None)
    ax.set_title(r"bridge posterior, $H=0.75$")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=1, fontsize=6.2, labelspacing=0.15, handlelength=1.4)
    S.clean(ax); S.panel_label(ax, "a", x=-0.24)
    # (b)
    ax = fig.add_subplot(gs[0, 1])
    Js = [4, 6, 8, 12, 16, 24, 32, 48, 64]
    cj = [abs(r["cosine"][f"projection_J{J}"]["B"]["dlogp_mean"]) for J in Js]
    Ks = sorted({int(k.split("K")[1]) for k in r["ma_typeI"]})
    md = [abs(r["ma_typeI"][f"default_K{K}"]["B"]["dlogp_mean"]) for K in Ks]
    mt = [abs(r["ma_typeI"][f"tuned_K{K}"]["B"]["dlogp_mean"]) for K in Ks]
    ax.plot(Ks, md, **S.st("ma", label="MA-fBm, published"))    # short labels: the legend fits the empty
    ax.plot(Ks, mt, **S.st("ma_best", label="MA-fBm, tuned"))   # upper right (J, K >= 16)
    ax.plot(Js, cj, **S.st("cosvi_j", label=r"\CosVI-$J$", lw=1.6))
    ax.set_xscale("log", base=2); ax.set_xticks([2, 4, 8, 16, 32, 64]); ax.set_xticklabels(["2", "4", "8", "16", "32", "64"])
    ax.minorticks_off()
    S.logy(ax)
    ax.set_ylim(1.5e-3, 22.0)
    ax.set_xlabel(r"dimension ($J$ or $K$)"); ax.set_ylabel(r"$|\mathrm{mean}\ \Delta \log p(\mathbf{y})|$ (nats)")
    ax.set_title(r"mean evidence error, $m=10$")
    ax.legend(loc="upper right", fontsize=6.1, labelspacing=0.18, handlelength=1.5, borderaxespad=0.1,
              bbox_to_anchor=(1.02, 1.00))
    S.clean(ax); S.panel_label(ax, "b", x=-0.27)
    # (c) heatmap
    ax = fig.add_subplot(gs[0, 2])
    Jg, mg = G2B["J"], G2B["m"]
    E = np.array([[G2B["results"][H][f"m{m}_J{J}"]["post_var_relerr_integrated_max"] for J in Jg] for m in mg]) * 100
    im = ax.imshow(E, origin="lower", aspect="auto", cmap="Blues_r", norm=LogNorm(vmin=0.1, vmax=30))
    for i in range(len(mg)):
        for j in range(len(Jg)):
            ok = E[i, j] <= 2.0
            ax.text(j, i, f"{E[i, j]:.1f}" if E[i, j] < 10 else f"{E[i, j]:.0f}", ha="center", va="center", fontsize=5.6,
                    color="black" if E[i, j] > 1.0 else "white", fontweight="bold" if ok else "normal")
            if ok:
                ax.add_patch(plt.Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, ec="#111111", lw=0.9))
    ax.set_xticks(range(len(Jg))); ax.set_xticklabels([str(J) for J in Jg], fontsize=6.8)
    ax.set_yticks(range(len(mg))); ax.set_yticklabels([str(m) for m in mg])
    ax.set_xlabel(r"cosine coordinates $J$"); ax.set_ylabel(r"observations $m$")
    ax.set_title(r"posterior-variance error (\%)")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.ax.tick_params(labelsize=6.5)
    S.panel_label(ax, "c", x=-0.30)
    S.save(fig, "fig04_linear", layout)


if __name__ == "__main__":
    for lay in S.LAYOUTS:
        draw(lay)
    print("ok fig04")
