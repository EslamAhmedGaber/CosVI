"""Figure 11 — computational cost, memory scaling, and matched-accuracy projection (N5, X11b, X11d).

2x2 layout (clean, spacious, zero clipped labels):
  (a) Step time vs coordinate dimension D at T = 5 (replaces 'their system' with 'MA-fBm system').
  (b) Step time vs horizon T (log-log).
  (c) Long-horizon GPU peak VRAM scaling N = 80..1600 (X11b: JointFull O(N^2) vs BlockJoint & PrecLR O(N)),
      plus fast_hfc Chebyshev-asymptotic precomputation callout (up to 170x speedup).
  (d) Projected 2000-step cost at 2% accuracy using standard-scale qualifying lifts only (X11d),
      fixing right-edge label clipping at T = 50.
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S

N5 = S.load(os.path.join("n5", "n5_scaling.json"))["rows"]
N7C = S.load(os.path.join("x11", "n7_clean_summary.json"))["by_H"]
FH = S.load(os.path.join("x11", "fast_hfc_benchmark.json"))


def rows(**k):
    return [r for r in N5 if all(r.get(a) == b for a, b in k.items())]


def draw(layout):
    S.setup()
    W = S.WIDE[layout]
    fig = plt.figure(figsize=(W, 4.65 if layout == "one_column" else 4.2))
    gs = fig.add_gridspec(2, 2, wspace=0.34, hspace=0.44, left=0.09, right=0.985, top=0.93, bottom=0.10)
    # (a) cost vs D
    ax = fig.add_subplot(gs[0, 0])
    c = sorted(rows(method="ours", rep="cosine", sweep="i"), key=lambda r: r["D"])
    ax.plot([r["D"] for r in c], [1e3 * r["step_s"] for r in c], **S.st("cosvi_j", label=r"\CosVI-$J$"))
    m = sorted(rows(method="ours", rep="markov", sweep="i"), key=lambda r: r["D"])
    ax.plot([r["D"] for r in m], [1e3 * r["step_s"] for r in m], **S.st("ma", label="lift as representation"))
    p = [1e3 * r["step_s"] for r in rows(method="port", sweep="i")]
    ax.axhspan(min(p), max(p), color=S.C["their"], alpha=0.22, lw=0, zorder=1)
    ax.axhline(np.mean(p), color=S.C["their"], ls=":", lw=1.4, zorder=2, label=r"MA-fBm system ($K=5$--$32$)")
    for rep, key, lab in (("grid", "grid", "exact grid"), ("twoscale", "cosvi", r"\CosVI")):
        r = rows(method="ours", rep=rep, sweep="i")[0]
        ax.plot([r["D"]], [1e3 * r["step_s"]], **S.st(key, label=lab))
    ax.set_xscale("log", base=2); ax.set_xticks([8, 16, 32, 64, 128, 256]); ax.set_xticklabels(["8", "16", "32", "64", "128", "256"])
    ax.minorticks_off(); ax.set_ylim(0, 120)
    ax.set_xlabel(r"coordinates $D$"); ax.set_ylabel(r"ms per training step")
    ax.set_title(r"step cost vs.\ dimension ($T=5$)")
    ax.legend(loc="lower right", fontsize=6.0, ncol=1, labelspacing=0.14, handlelength=1.7)
    S.clean(ax, grid_y=True); S.panel_label(ax, "a", x=-0.20)
    # (b) cost vs T
    ax = fig.add_subplot(gs[0, 1])
    for sel, key, lab in ((dict(method="ours", rep="cosine", sweep="ii"), "cosvi_j", r"\CosVI-32"),
                          (dict(method="ours", rep="twoscale", sweep="ii"), "cosvi", r"\CosVI"),
                          (dict(method="ours", rep="grid", sweep="ii"), "grid", "exact grid"),
                          (dict(method="port", K=5, sweep="ii"), "their", "MA-fBm system")):
        rr = sorted(rows(**sel), key=lambda r: r["T"])
        ax.plot([r["T"] for r in rr], [1e3 * r["step_s"] for r in rr], **S.st(key, label=lab))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([1, 2, 5, 10, 20]); ax.set_xticklabels(["1", "2", "5", "10", "20"]); ax.minorticks_off()
    ax.set_xlabel(r"horizon $T$"); ax.set_ylabel(r"ms per training step")
    ax.set_title(r"step cost vs.\ horizon")
    ax.legend(loc="upper left", fontsize=6.1, labelspacing=0.15, handlelength=1.7)
    S.clean(ax); S.panel_label(ax, "b", x=-0.20)
    # (c) Long-horizon GPU VRAM scaling (X11b)
    ax = fig.add_subplot(gs[1, 0])
    xb_p = os.path.join(S.RES, "x11", "x11b_structured_q.json")
    xb = json.load(open(xb_p)) if os.path.exists(xb_p) else None
    if xb and "scaling" in xb:
        sc = xb["scaling"]
        fams = [("joint_full", "ma", r"JointFull $\mathcal{O}(N^2)$"),
                ("prec_lr16", "cosvi_j", r"PrecLR $r{=}16$ $\mathcal{O}(rN)$"),
                ("block_joint", "cosvi", r"BlockJoint $\mathcal{O}(J^2{+}N)$")]
        for fk, colk, lab in fams:
            pts = [r for r in sc if r["family"] == fk and "peak_mem_MB" in r]
            ax.plot([r["N"] for r in pts], [r["peak_mem_MB"] for r in pts], **S.st(colk, label=lab))
            ooms = [r for r in sc if r["family"] == fk and "error" in r]
            for om in ooms:
                ax.annotate("OOM (6 GB)", xy=(om["N"], 4800), xytext=(om["N"] * 0.55, 3600),
                            fontsize=6.0, color=S.C["ma"], arrowprops=dict(arrowstyle="-|>", color=S.C["ma"], lw=0.7))
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks([80, 200, 400, 800, 1600]); ax.set_xticklabels(["80", "200", "400", "800", "1600"])
        ax.set_ylim(220, 2200)
        ax.minorticks_off()
        jf1600 = next(r["peak_mem_MB"] for r in sc if r["family"] == "joint_full" and r["N"] == 1600)
        bj1600 = next(r["peak_mem_MB"] for r in sc if r["family"] == "block_joint" and r["N"] == 1600)
        ax.annotate("", xy=(1600, bj1600 * 1.05), xytext=(1600, jf1600 * 0.96),
                    arrowprops=dict(arrowstyle="<->", color="#444444", lw=0.8))
        ax.text(1460, np.sqrt(jf1600 * bj1600), rf"${jf1600/bj1600:.1f}\times$", ha="right", va="center",
                fontsize=6.2, color="#333333", fontweight="bold")
        ax.set_xlabel(r"grid steps $N$"); ax.set_ylabel(r"peak GPU VRAM (MiB)")
        ax.set_title(r"long-horizon memory scaling ($M=16$)")
        ax.legend(loc="upper left", fontsize=5.8, labelspacing=0.14, handlelength=1.5)
        ax.text(0.97, 0.05, r"precomp.\ speedup: $11$--$170\times$", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=5.8, color="#444444", bbox=dict(boxstyle="round,pad=0.2", fc="#F7F9FC", ec="#CCCCCC", lw=0.5))
    else:
        ax.text(0.5, 0.5, "X11b scaling pending", transform=ax.transAxes, ha="center", va="center", color="#999999")
        ax.set_title(r"long-horizon memory scaling")
    S.clean(ax); S.panel_label(ax, "c", x=-0.20)
    # (d) Matched-accuracy 2% projection (standard-scale qualifying lifts only, X11d)
    ax = fig.add_subplot(gs[1, 1])
    for Hk, ls, lab_l in (("0.60", ":", r"std.\ lift ($H=0.6$)"), ("0.75", "--", r"std.\ lift ($H=0.75, 0.9$)")):
        rs = N7C[Hk]
        Ts = [r["T"] for r in rs]
        ax.plot(Ts, [r["lift_std_cheapest"][1] for r in rs], color=S.C["ma"], ls=ls, marker="s", ms=3.6,
                lw=1.2, mfc="white", mec=S.C["ma"], label=lab_l, zorder=3)
    rs75 = N7C["0.75"]
    Ts = [r["T"] for r in rs75]
    ax.plot(Ts, [r["cosvi_cheapest"][1] for r in rs75], color=S.C["cosvi"], marker="o", ms=4.0, lw=1.8,
            mfc=S.C["cosvi"], mec=S.C["cosvi"], label=r"\CosVI{} (all $H$)", zorder=5)
    for r in rs75:
        name = r["cosvi_cheapest"][0]
        lab = rf"$J^*\!=\!{r['J_star']}$" if name.startswith("cosine") else r"two-scale"
        if r["T"] >= 50:
            xytext, ha, va = (-5, -6), "right", "top"
        else:
            xytext, ha, va = (4, -4), "left", "top"
        ax.annotate(lab, (r["T"], r["cosvi_cheapest"][1]), xytext=xytext, textcoords="offset points", ha=ha, va=va,
                    fontsize=5.8, color=S.C["cosvi"],
                    bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.85))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(25, 4500); ax.set_xlim(1.6, 68)
    ax.set_xticks([2, 5, 10, 20, 50]); ax.set_xticklabels(["2", "5", "10", "20", "50"]); ax.minorticks_off()
    ax.set_xlabel(r"horizon $T$"); ax.set_ylabel(r"projected seconds (2000 steps)")
    ax.set_title(r"projected cost at $2\,\%$ accuracy")
    ax.legend(loc="upper left", fontsize=5.9, labelspacing=0.14, handlelength=1.6)
    S.clean(ax); S.panel_label(ax, "d", x=-0.20)
    S.save(fig, "fig10_cost", layout)


if __name__ == "__main__":
    for lay in S.LAYOUTS:
        draw(lay)
    print("ok fig10")
