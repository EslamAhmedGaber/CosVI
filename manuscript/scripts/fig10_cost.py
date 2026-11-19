"""Figure 10 — what it costs, and where it stops paying (N5, N7).

(a) Time per training step (median of 30 CUDA-graph replays, RTX 2060) against the coordinate dimension at T = 5, d = 1:
    CosVI-J (cosine J = 8..256), CosVI (two-scale, D = 95), exact grid (D = 80), the lift as a representation under our
    inference (K = 5..32, D = K + N) and the competitor's full system (port, K = 5..32).
(b) Time per step against the horizon T (N = 16 T steps): CosVI-32, CosVI, exact grid, the competitor's system (K = 5).
(c) Cost at matched accuracy (posterior std within 2 % at the observation times; time per step x 2000) against the
    horizon, for the cheapest qualifying CosVI configuration and the cheapest qualifying lift, at H = 0.6, 0.75, 0.9;
    labels: the CosVI configuration used (cosine J*, or two-scale when cheaper or when no J* <= 256 exists).
Sources: work/results/n5/n5_scaling.json, work/results/n7/n7_summary.json.
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S

N5 = S.load(os.path.join("n5", "n5_scaling.json"))["rows"]
N7 = S.load(os.path.join("n7", "n7_summary.json"))["by_H"]


def rows(**k):
    return [r for r in N5 if all(r.get(a) == b for a, b in k.items())]


def draw(layout):
    S.setup()
    W = S.WIDE[layout]
    fig = plt.figure(figsize=(W, 2.45))
    gs = fig.add_gridspec(1, 3, wspace=0.36, left=0.07, right=0.99, top=0.84, bottom=0.18)
    # (a)
    ax = fig.add_subplot(gs[0, 0])
    c = sorted(rows(method="ours", rep="cosine", sweep="i"), key=lambda r: r["D"])
    ax.plot([r["D"] for r in c], [1e3 * r["step_s"] for r in c], **S.st("cosvi_j", label=r"\CosVI-$J$"))
    m = sorted(rows(method="ours", rep="markov", sweep="i"), key=lambda r: r["D"])
    ax.plot([r["D"] for r in m], [1e3 * r["step_s"] for r in m], **S.st("ma", label="lift as representation"))
    p = [1e3 * r["step_s"] for r in rows(method="port", sweep="i")]
    ax.axhspan(min(p), max(p), color=S.C["their"], alpha=0.25, lw=0, zorder=1)
    ax.axhline(np.mean(p), color=S.C["their"], ls=":", lw=1.4, zorder=2, label=r"their system ($K=5$--$32$)")
    for rep, key, lab in (("grid", "grid", "exact grid"), ("twoscale", "cosvi", r"\CosVI")):
        r = rows(method="ours", rep=rep, sweep="i")[0]
        ax.plot([r["D"]], [1e3 * r["step_s"]], **S.st(key, label=lab))
    ax.set_xscale("log", base=2); ax.set_xticks([8, 16, 32, 64, 128, 256]); ax.set_xticklabels(["8", "16", "32", "64", "128", "256"])
    ax.minorticks_off()
    ax.set_ylim(0, 120)
    ax.set_xlabel(r"coordinates $D$"); ax.set_ylabel(r"ms per training step")
    ax.set_title(r"cost against dimension ($T=5$)")
    ax.legend(loc="lower right", fontsize=6.0, ncol=1, labelspacing=0.15, handlelength=1.8)
    S.clean(ax, grid_y=True); S.panel_label(ax, "a", x=-0.24)
    # (b)
    ax = fig.add_subplot(gs[0, 1])
    for sel, key, lab in ((dict(method="ours", rep="cosine", sweep="ii"), "cosvi_j", r"\CosVI-32"),
                          (dict(method="ours", rep="twoscale", sweep="ii"), "cosvi", r"\CosVI"),
                          (dict(method="ours", rep="grid", sweep="ii"), "grid", "exact grid"),
                          (dict(method="port", K=5, sweep="ii"), "their", "their system")):
        rr = sorted(rows(**sel), key=lambda r: r["T"])
        ax.plot([r["T"] for r in rr], [1e3 * r["step_s"] for r in rr], **S.st(key, label=lab))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([1, 2, 5, 10, 20]); ax.set_xticklabels(["1", "2", "5", "10", "20"]); ax.minorticks_off()
    ax.set_xlabel(r"horizon $T$"); ax.set_ylabel(r"ms per training step")
    ax.set_title(r"cost against horizon")
    ax.legend(loc="upper left", fontsize=6.1, labelspacing=0.15, handlelength=1.8)
    S.clean(ax); S.panel_label(ax, "b", x=-0.24)
    # (c)
    ax = fig.add_subplot(gs[0, 2])
    first = True
    for H, v in N7.items():
        Ts = [r["T"] for r in v["rows"]]
        ax.plot(Ts, [r["lift_cheapest_qualifying"][1] for r in v["rows"]], color=S.C["ma"], ls="--", marker="s", ms=3.6,
                lw=1.1, mfc="white", mec=S.C["ma"], label=r"lift (cheapest qualifying)" if first else None, zorder=3)
        ax.plot(Ts, [r["cosvi_cheapest"][1] for r in v["rows"]], color=S.C["cosvi"], marker="o", ms=3.8, lw=1.7,
                mfc=S.C["cosvi"], mec=S.C["cosvi"], label=r"cheapest \CosVI{} configuration" if first else None, zorder=5)
        first = False
    v = N7["0.75"]["rows"]
    for r in v:
        name = r["cosvi_cheapest"][0]
        lab = rf"$J^*\!=\!{r['J_star']}$" if name.startswith("cosine") else r"two-scale"
        ax.annotate(lab, (r["T"], r["cosvi_cheapest"][1]), xytext=(5, -2), textcoords="offset points", ha="left",
                    va="top", fontsize=5.8, color=S.C["cosvi"],
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([2, 5, 10, 20, 50]); ax.set_xticklabels(["2", "5", "10", "20", "50"]); ax.minorticks_off()
    ax.set_xlabel(r"horizon $T$"); ax.set_ylabel(r"projected seconds, 2000 steps")
    ax.set_title(r"projected cost, 2\,\% accuracy")
    ax.legend(loc="upper left", fontsize=6.1, labelspacing=0.15, handlelength=1.8)
    S.clean(ax); S.panel_label(ax, "c", x=-0.24)
    S.save(fig, "fig10_cost", layout)


if __name__ == "__main__":
    for lay in S.LAYOUTS:
        draw(lay)
    print("ok fig10")
