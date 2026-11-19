"""Figure 9 — is the uncertainty right, and which components of CosVI matter? (X4, N6)

(a) Reliability of the latent-path posterior at H = 0.75: empirical coverage of the central intervals of each nominal
    level against the level (test series, all grid times; mean over seeds), per representation; the diagonal is
    perfect calibration, points below it are overconfident bands.
(b) Calibration error mean(|cov50 - 0.5|, |cov90 - 0.9|) against H_true, mean +- sd over seeds.
(c) Ablation (N6): paired Delta = ll(base) - ll(variant), nats per test series, mean +- sd over three seeds, per
    component and H_true; positive = the component of CosVI helps; right: training time of the variant / base.
Sources: work/results/calib/x4_summary.json, work/results/n6/n6_summary.json.
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S

X4 = S.load(os.path.join("calib", "x4_summary.json"))["by"]
N6 = S.load(os.path.join("n6", "n6_summary.json"))["rows"]
REPS = [("markov", "ma", "MA-fBm law, published"), ("markov_best", "ma_best", "MA-fBm law, best"),
        ("grid", "grid", "exact grid"), ("cosine", "cosvi_j", r"\CosVI-32"), ("twoscale", "cosvi", r"\CosVI")]
HM = {0.6: "o", 0.75: "s", 0.9: "D"}


def draw(layout):
    S.setup()
    W = S.WIDE[layout]
    fig = plt.figure(figsize=(W, 2.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 0.9, 1.2], wspace=0.62, left=0.065, right=0.985, top=0.84, bottom=0.17)
    # (a) reliability at H = 0.75
    ax = fig.add_subplot(gs[0, 0])
    ax.plot([0, 1], [0, 1], color=S.C["ref"], lw=0.8, zorder=1)
    for rep, key, lab in REPS:
        v = X4.get(f"{rep}|0.75")
        if not v:
            continue
        lev = sorted(float(k) for k in v["coverage_curve"])
        cov = [v["coverage_curve"][f"{l:.1f}"] for l in lev]
        st = S.st(key, label=lab); st["ms"] = 3.2
        ax.plot(lev, cov, **st)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel(r"nominal level"); ax.set_ylabel(r"empirical coverage")
    ax.set_title(r"reliability, $H=0.75$")
    ax.text(0.62, 0.30, r"overconfident", fontsize=6.4, color="#777777", rotation=0)
    ax.legend(loc="upper left", fontsize=5.9, labelspacing=0.12, handlelength=1.6, borderaxespad=0.1)
    S.clean(ax); S.panel_label(ax, "a", x=-0.30)
    # (b) calibration error vs H
    ax = fig.add_subplot(gs[0, 1])
    Hs = [0.6, 0.75, 0.9]
    for rep, key, lab in REPS:
        m = [X4[f"{rep}|{H}"]["cal_error"][0] for H in Hs]
        sd = [X4[f"{rep}|{H}"]["cal_error"][1] or 0 for H in Hs]
        st = S.st(key, label=lab)
        ax.errorbar(Hs, 100 * np.array(m), yerr=100 * np.array(sd), capsize=1.5, elinewidth=0.7,
                    **{k: v for k, v in st.items() if k != "label"})
    ax.set_xticks(Hs); ax.set_xlabel(r"true Hurst index $H$")
    ax.set_ylabel(r"calibration error (\%)"); ax.set_ylim(0, None)
    ax.set_title(r"calibration")
    S.clean(ax, grid_y=True); S.panel_label(ax, "b", x=-0.30)
    # (c) ablation
    ax = fig.add_subplot(gs[0, 2])
    comp = [("raw", "whitened / raw"), ("classical", "projection / classical"),
            ("mf", r"full / mean-field $q$"), ("lr2", r"full / rank-2 $q$"), ("lr8", r"full / rank-8 $q$"),
            ("heun", "RK4 / Heun"), ("s4", r"$S=16$ / $S=4$"), ("s8", r"$S=16$ / $S=8$")]
    for i, (k, lab) in enumerate(comp):
        r = N6[k]["by_H"]
        for j, H in enumerate(("0.60", "0.75", "0.90")):
            c = r[H]; y = i + (j - 1) * 0.22
            col = S.C["cosvi"] if c["component_verdict"] == "helps" else ("#B03A2E" if c["component_verdict"] == "hurts" else "#9A9A9A")
            ax.errorbar([c["delta_grid_mean"]], [y], xerr=[[c["delta_grid_sd"]], [c["delta_grid_sd"]]], fmt=HM[float(H)],
                        color=col, mfc=col if j != 1 else "white", mec=col, ms=3.5, lw=0.8, capsize=1.2, zorder=4)
        ax.text(1.62, i, rf"$\times{r['0.60']['wallclock_ratio']:.2f}$", fontsize=6.4, va="center", ha="left", color="#555555")
    ax.axvline(0, color="#333333", lw=0.7, zorder=1)
    ax.set_yticks(range(len(comp))); ax.set_yticklabels([c[1] for c in comp], fontsize=6.8)
    ax.invert_yaxis(); ax.set_xlim(-0.8, 1.6)
    ax.set_xlabel(r"$\Delta$ held-out log-lik. (nats)")
    ax.set_title(r"what matters inside \CosVI", fontsize=8.2)
    ax.text(1.62, -0.85, r"time", fontsize=6.4, ha="left", color="#555555")
    ax.tick_params(axis="y", length=0); ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#EEEEEE", lw=0.5, zorder=0)
    S.panel_label(ax, "c", x=-0.50)
    hh = [plt.Line2D([], [], ls="none", marker=HM[H], color="#666666", mfc="white" if H == 0.75 else "#666666", ms=3.6) for H in Hs]
    hh += [plt.Line2D([], [], ls="none", marker="s", color=S.C["cosvi"], ms=4), plt.Line2D([], [], ls="none", marker="s", color="#9A9A9A", ms=4),
           plt.Line2D([], [], ls="none", marker="s", color="#B03A2E", ms=4)]
    fig.legend(hh, [r"$H=0.6$", r"$H=0.75$", r"$H=0.9$", "helps", "no detectable effect", "hurts"], loc="upper center", ncol=6,
               bbox_to_anchor=(0.5, 1.0), fontsize=6.8, handletextpad=0.2, columnspacing=1.0)
    S.save(fig, "fig09_calib_ablation", layout)


if __name__ == "__main__":
    for lay in S.LAYOUTS:
        draw(lay)
    print("ok fig09")
