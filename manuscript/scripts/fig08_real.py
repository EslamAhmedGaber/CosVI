"""Figure 10 — real long-memory series (N4 v2 Nile, X5 sensitivity, X3 temperature).

Clean, fair like-for-like comparison across the five representations under identical inference
(removes the rescaled MA-fBm system straw-man arrow; uses collision-free sigma_obs notation; adds
shared top legend for panel d).
"""
import os, sys, json, glob
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S

HG = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
REPS = [("twoscale", "cosvi", r"\CosVI"), ("cosine64", "cosvi_j", r"\CosVI-64"), ("grid", "grid", "exact grid"),
        ("markov", "ma", r"MA-fBm law, publ."), ("markov_tuned", "ma_best", r"MA-fBm law, tuned")]


def nile(rep, s="0.1"):
    out = []
    for f in sorted(glob.glob(os.path.join(S.RES, "n4", f"n4v2_{rep}_seed*_s{s}.json"))):
        d = json.load(open(f)); gi = int(np.argmin(np.abs(HG - d["H_profile"])))
        out.append((d["seed"], d["test_loglik_exact_law_twoscale_coords_mean"][gi], d["H_profile"]))
    return out


def temp(rep):
    out = []
    for f in sorted(glob.glob(os.path.join(S.RES, "n4t", f"n4t_{rep}_seed*_s0.1.json"))):
        d = json.load(open(f)); hg = np.array(d["hgrid"]); ll = np.array(d["test_loglik_exact_law_twoscale_coords_mean"][:len(hg)])
        i = int(np.argmin(np.abs(hg - d["H_profile_general"])))
        out.append((d["seed"], float(ll[i]), d["H_profile_general"]))
    return out


def strip(ax, data, ylim, title):
    for k, (rep, key, name) in enumerate(REPS):
        v = np.array([x[1] for x in data[rep]])
        col = S.C[key]
        ax.plot([k - 0.28, k + 0.28], [v.mean()] * 2, color=col, lw=2.2 if rep == "twoscale" else 1.5, zorder=3, solid_capstyle="butt")
        jit = np.linspace(-0.12, 0.12, len(v))
        mfc = col if key in ("cosvi", "grid", "ma_best") else "white"
        ax.plot(k + jit, v, ls="none", marker="o", ms=3.8, mfc=mfc, mec=col, mew=0.9, zorder=4)
    ax.set_xticks(range(len(REPS)))
    ax.set_xticklabels([n for _, _, n in REPS], rotation=28, ha="right", fontsize=6.4)
    ax.set_ylim(*ylim); ax.set_xlim(-0.6, len(REPS) - 0.4)
    ax.set_title(title)
    S.clean(ax, grid_y=True)


def draw(layout):
    S.setup()
    W = S.WIDE[layout]
    if layout == "two_columns":
        fig = plt.figure(figsize=(W, 2.75))
        gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 0.85, 0.95], wspace=0.42, left=0.065, right=0.99, top=0.79, bottom=0.26)
        axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    else:
        fig = plt.figure(figsize=(W, 5.15))
        gs = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.58, left=0.09, right=0.99, top=0.87, bottom=0.10)
        axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    nd = {r: nile(r) for r, _, _ in REPS}
    td = {r: temp(r) for r, _, _ in REPS}
    strip(axes[0], nd, (-59.0, -45.5), r"Nile minima (like-for-like)")
    axes[0].set_ylabel(r"held-out log-lik.\ per window")
    S.panel_label(axes[0], "a", x=-0.26)
    strip(axes[1], td, (-68.5, -52.5), r"NH temperature (like-for-like)")
    S.panel_label(axes[1], "b", x=-0.22)
    # (c) sensitivity to sigma_obs
    ax = axes[2]
    for k, s in enumerate(("0.05", "0.1", "0.2")):
        g = {x[0]: x[1] for x in nile("grid", s)}; c = {x[0]: x[1] for x in nile("twoscale", s)}
        dd = np.array([g[i] - c[i] for i in sorted(set(g) & set(c))])
        ax.errorbar([k], [dd.mean()], yerr=[dd.std(ddof=1)], color="#444444", lw=1.0, capsize=2.4, marker="_", ms=9, zorder=3)
        ax.plot(k + np.linspace(-0.13, 0.13, len(dd)), dd, ls="none", marker="o", ms=3.8, mfc=S.C["grid"], mec=S.C["grid"], zorder=4)
    ax.axhline(0, color="#999999", lw=0.7, zorder=1)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels([r"$0.05$", r"$0.1$", r"$0.2$"])
    ax.set_xlabel(r"observation noise $\sigma_{\mathrm{obs}}$", labelpad=1)
    ax.set_ylabel(r"grid $-$ \CosVI{} (nats)")
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(-17, 7)
    ax.set_title(r"Nile sensitivity to $\sigma_{\mathrm{obs}}$")
    ax.text(2.44, 6.2, r"grid better", ha="right", va="top", fontsize=6.1, color="#555555")
    ax.text(2.44, -16.2, r"\CosVI{} better", ha="right", va="bottom", fontsize=6.1, color="#555555")
    S.clean(ax, grid_y=True); S.panel_label(ax, "c", x=-0.26)
    # (d) H estimates
    ax = axes[3]
    prov = json.load(open(os.path.join(S.RES, "n4", "data", "provenance.json")))["series"]
    cls = {"Nile": [prov["nile_min"][k] for k in ("H_whittle_fgn", "H_aggregated_variance", "H_RS")],
           "temp.": [prov["nhemi_temp"][k] for k in ("H_whittle_fgn", "H_aggregated_variance", "H_RS")]}
    handles, labels = [], []
    for j, (lab, data) in enumerate((("Nile", nd), ("temp.", td))):
        x0 = j * 1.0
        ax.fill_between([x0 - 0.42, x0 + 0.42], [min(cls[lab])] * 2, [max(cls[lab])] * 2, color="#E6E6E6", lw=0, zorder=0)
        for k, (rep, key, name) in enumerate(REPS):
            h = np.array([x[2] for x in data[rep]])
            xx = x0 - 0.30 + 0.15 * k
            mfc = S.C[key] if key in ("cosvi", "grid", "ma_best") else "white"
            eb = ax.errorbar([xx], [h.mean()], yerr=[h.std(ddof=1)], color=S.C[key], lw=1.0, capsize=1.6, marker="o", ms=4.0,
                             mfc=mfc, mec=S.C[key], zorder=3 + (rep == "twoscale"), label=name if j == 0 else None)
            if j == 0:
                handles.append(eb); labels.append(name)
    ax.set_xticks([0, 1]); ax.set_xticklabels([r"Nile", r"NH temperature"])
    ax.set_ylim(0.5, 0.96); ax.set_xlim(-0.55, 1.55)
    ax.set_ylabel(r"profile $\widehat H$", labelpad=1)
    ax.set_title(r"Hurst-index estimates")
    ax.text(0.5, 0.93, r"shaded band: classical estimators", ha="center", fontsize=6.0, color="#666666")
    S.clean(ax, grid_y=True); S.panel_label(ax, "d", x=-0.22)
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=6.4, bbox_to_anchor=(0.53, 0.99),
               handletextpad=0.3, columnspacing=0.9)
    S.save(fig, "fig08_real", layout)


if __name__ == "__main__":
    for lay in S.LAYOUTS:
        draw(lay)
    print("ok fig08")
