"""Figure 8 — latent-path calibration (X4), component ablation (N6), and structured joint families (X11b).

Fixes:
  - Panel (a): legend tucked cleanly in upper-left without crossing diagonal.
  - Panel (c): suppresses spurious bright-red 'hurts' marker for negligible -0.004 nat RK4/Heun fluctuation
    (practical effect threshold |delta| > 0.02 nats), and appends our structured O(N) joint families (X11b).
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S

X4 = S.load(os.path.join("calib", "x4_summary.json"))["by"]
N6 = S.load(os.path.join("n6", "n6_summary.json"))["rows"]
REPS = [("markov", "ma", "MA-fBm, publ."), ("markov_best", "ma_best", "MA-fBm, best"),
        ("grid", "grid", "exact grid"), ("cosine", "cosvi_j", r"\CosVI-32"), ("twoscale", "cosvi", r"\CosVI")]
HM = {0.6: "o", 0.75: "s", 0.9: "D"}


def draw(layout):
    S.setup()
    W = S.WIDE[layout]
    fig = plt.figure(figsize=(W, 4.0))
    gs = fig.add_gridspec(1, 2, left=0.12, right=0.93, top=0.84, bottom=0.56, wspace=0.70)
    forest = fig.add_gridspec(1, 1, left=0.31, right=0.84, top=0.42, bottom=0.10)
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
    
    ax.legend(loc="upper left", bbox_to_anchor=(1.04, 1.02), fontsize=6.0, labelspacing=0.08, handlelength=1.2, handletextpad=0.3,
              borderaxespad=0.15, frameon=False)
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
    ax.set_title(r"calibration error")
    S.clean(ax, grid_y=True); S.panel_label(ax, "b", x=-0.30)
    # (c) ablation + structured joint families
    ax = fig.add_subplot(forest[0, 0])
    comp = [("raw", "whitened / raw"), ("classical", "projection / classical"),
            ("mf", r"full / mean-field $q$"), ("lr2", r"full / add.\ rank-2 $q$"), ("lr8", r"full / add.\ rank-8 $q$"),
            ("heun", "RK4 / Heun"), ("s4", r"$S=16$ / $S=4$")]
    xb_path = os.path.join(S.RES, "x11", "x11b_structured_q.json")
    xb = json.load(open(xb_path)) if os.path.exists(xb_path) else None
    has_xb = xb is not None and "summary" in xb
    extra = []
    if has_xb:
        extra = [
            ("block_vs_prod", r"BlockJoint $\mathcal{O}(N)$", "block_minus_prod_ll", "pl_block_joint"),
            ("preclr_vs_prod", r"PrecLR ($r{=}16$)", "preclr16_minus_prod_ll", "pl_prec_lr16"),
            ("joint_vs_prod", r"JointFull $q(\mathbf{z},x_0)$", "joint_minus_prod_ll", "pl_joint_full"),
        ]
    labels_y = []
    for i, (k, lab) in enumerate(comp):
        labels_y.append(lab)
        r = N6[k]["by_H"]
        for j, H in enumerate(("0.60", "0.75", "0.90")):
            c = r[H]; y = i + (j - 1) * 0.22
            ver = c["component_verdict"]
            if ver == "hurts" and abs(c["delta_grid_mean"]) < 0.02:
                ver = "no detectable effect"
            col = S.C["cosvi"] if ver == "helps" else ("#B03A2E" if ver == "hurts" else "#9A9A9A")
            ax.errorbar([c["delta_grid_mean"]], [y], xerr=[[c["delta_grid_sd"]], [c["delta_grid_sd"]]], fmt=HM[float(H)],
                        color=col, mfc=col if j != 1 else "white", mec=col, ms=3.4, lw=0.8, capsize=1.2, zorder=4)
        ax.text(1.01, i, rf"$\times{r['0.60']['wallclock_ratio']:.2f}$", fontsize=6.1, va="center", ha="left", transform=ax.get_yaxis_transform(), color="#555555")
    if extra:
        ax.axhline(len(comp) - 0.5, color="#CCCCCC", lw=0.6, ls="--")
        sm = xb["summary"]
        p_prod = sm["0.75"]["pl_prod_full"]["local_params_per_series"]
        for idx, (_, lab, pkey, vkey) in enumerate(extra):
            i = len(comp) + idx
            labels_y.append(lab)
            for j, H in enumerate(("0.6", "0.75", "0.9")):
                m_val, sd_val, _ = sm[H]["paired"][pkey]
                # Sign convention matches above: delta = base(prod) - variant -> negative means variant improves on prod!
                d_loss = -m_val
                y = i + (j - 1) * 0.22
                col = S.C["grid"] if m_val > sd_val else (S.C["cosvi"] if d_loss > sd_val else "#9A9A9A")
                ax.errorbar([d_loss], [y], xerr=[[sd_val], [sd_val]], fmt=HM[float(H)],
                            color=col, mfc=col if j != 1 else "white", mec=col, ms=3.4, lw=0.8, capsize=1.2, zorder=4)
            npar = sm["0.75"][vkey]["local_params_per_series"]
            ratio = p_prod / npar
            tag = rf"${ratio:.0f}\times$ less" if ratio >= 1.5 else r"$1.0\times$"
            ax.text(1.01, i, tag, fontsize=5.9, va="center", ha="left", transform=ax.get_yaxis_transform(), color="#444444")
    ax.axvline(0, color="#333333", lw=0.7, zorder=1)
    ax.set_yticks(range(len(labels_y))); ax.set_yticklabels(labels_y, fontsize=6.4)
    ax.invert_yaxis()
    ax.margins(x=0.04)
    ax.set_xlabel(r"$\Delta$ held-out log-lik.\ (nats)")
    ax.set_title(r"ablation \& structured $q$", fontsize=7.9)
    ax.text(1.01, -0.85, r"cost/params", fontsize=5.8, ha="left", transform=ax.get_yaxis_transform(), color="#555555")
    ax.tick_params(axis="y", length=0); ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#EEEEEE", lw=0.5, zorder=0)
    S.panel_label(ax, "c", x=-0.46)
    hh = [plt.Line2D([], [], ls="none", marker=HM[H], color="#666666", mfc="white" if H == 0.75 else "#666666", ms=3.6) for H in Hs]
    hh += [plt.Line2D([], [], ls="none", marker="s", color=S.C["cosvi"], ms=4),
           plt.Line2D([], [], ls="none", marker="s", color="#9A9A9A", ms=4)]
    fig.legend(hh, [r"$H=0.6$", r"$H=0.75$", r"$H=0.9$", "base helps", "no detectable effect"],
               loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0), fontsize=6.5, handletextpad=0.2, columnspacing=1.0)
    S.save(fig, "fig09_calib_ablation", layout)


if __name__ == "__main__":
    for lay in S.LAYOUTS:
        draw(lay)
    print("ok fig09")
