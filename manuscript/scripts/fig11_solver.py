"""Figure 7 — coordinates versus solver (X8c) and four-arm Haar-random rotated-coordinate causal control (X11a).

(a) X8c paired contrasts under PL evaluator after 5000 Adam steps (mean +- 2 SE, 5 seeds): coordinates, solver, total.
(b) Training trajectory: CosVI-PL minus exact grid in training ELBO per series (median over 5 seeds).
(c) X11a causal rotation control: aligned CosVI-PL minus exact grid, minus Haar-rotated grid, and minus Haar-rotated
    CosVI-PL (identical grid covariance and precision spectrum; destroying alignment destroys the Adam gain).
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S

CON = [("coordinate", "coordinates", S.C["cosvi_j"], "o"),
       ("solver", "solver", S.C["grid"], "D"),
       ("total", "total", S.C["cosvi"], "s")]
ROT = [("grid", r"PL $-$ grid", S.C["grid"], "o"),
       ("rot_grid", r"PL $-$ rot.\ grid", S.C["ma_best"], "^"),
       ("rot_pl", r"PL $-$ rot.\ PL", S.C["ma"], "s")]
H_LS = {"0.6": ":", "0.75": "--", "0.9": "-"}


def contrasts_x8(ax, cc):
    for j, H in enumerate(("0.6", "0.75", "0.9")):
        for k, (con, lab, col, m) in enumerate(CON):
            v = cc[H]["contrasts"][con]["pl"]
            x = j - 0.22 + 0.22 * k
            present = v["verdict"].startswith("present")
            ax.errorbar([x], [v["mean"]], yerr=[2 * v["se"]], color=col, lw=1.1, capsize=1.8, marker=m, ms=4.0,
                        mfc=col if present else "white", mec=col, zorder=3)
    ax.axhline(0, color="#888888", lw=0.7, zorder=1)
    ax.set_xticks(range(3)); ax.set_xticklabels([r"$H=0.6$", r"$H=0.75$", r"$H=0.9$"])
    ax.set_xlim(-0.5, 2.5)


def path(ax):
    for H, ls in H_LS.items():
        runs = [json.load(open(os.path.join(S.RES, "x8", f"x8c_seed{s}_H{float(H):.2f}.json"))) for s in range(5)]
        names = runs[0]["names"]
        ip, ig = names.index("CosVI-PL"), names.index("exact grid")
        steps = np.array([h[0] for h in runs[0]["history"]])
        diff = np.array([[h[1 + ip] - h[1 + ig] for h in r["history"]] for r in runs])
        keep = steps >= 200
        ax.plot(steps[keep], np.median(diff, axis=0)[keep], color=S.C["cosvi_j"], ls=ls, lw=1.4, zorder=3,
                label=rf"$H={H}$")
    ax.axhline(0, color="#888888", lw=0.7, zorder=1)
    ax.set_xlabel(r"Adam step", labelpad=1)
    ax.set_xlim(0, 5000); ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
    ax.set_xticklabels(["0", "", "2000", "", "4000", ""])
    ax.legend(loc="lower right", fontsize=6.4, handlelength=2.2, borderaxespad=0.2)


def contrasts_rot(ax, xa):
    sm = xa["summary"]
    for j, H in enumerate(("0.6", "0.75", "0.9")):
        for k, (comp, lab, col, m) in enumerate(ROT):
            c = sm[H]["contrasts_vs_pl"][comp]
            x = j - 0.22 + 0.22 * k
            present = c["ll_verdict_2se"].startswith("present")
            ax.errorbar([x], [c["ll_gain_mean"]], yerr=[2 * c["ll_gain_se"]], color=col, lw=1.1, capsize=1.8, marker=m,
                        ms=4.0, mfc=col if present else "white", mec=col, zorder=3, label=lab if j == 0 else None)
    ax.axhline(0, color="#888888", lw=0.7, zorder=1)
    ax.set_xticks(range(3)); ax.set_xticklabels([r"$H=0.6$", r"$H=0.75$", r"$H=0.9$"])
    ax.set_xlim(-0.5, 2.5); ax.set_ylim(-0.5, 5.1)
    ax.legend(loc="upper right", fontsize=5.8, handletextpad=0.25, labelspacing=0.14, borderaxespad=0.15, frameon=False)


def draw(layout):
    S.setup()
    W = S.WIDE[layout]
    fig = plt.figure(figsize=(W, 2.55 if layout == "two_columns" else 2.75))
    gs = fig.add_gridspec(1, 3, wspace=0.38, left=0.07, right=0.99, top=0.78, bottom=0.16)
    f8 = os.path.join(S.RES, "x8", "x8_summary.json")
    cc = json.load(open(f8)).get("c", {}).get("by_H", {}) if os.path.exists(f8) else {}
    fa = os.path.join(S.RES, "x11", "x11a_rotated_grid.json")
    xa = json.load(open(fa)) if os.path.exists(fa) else None
    titles = [r"coordinates vs.\ solver (5000 steps)", r"training ELBO, \CosVI-PL $-$ grid", r"Haar-rotation causal control"]
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        if i == 0:
            contrasts_x8(ax, cc)
        elif i == 1:
            path(ax)
        else:
            if xa and "summary" in xa:
                contrasts_rot(ax, xa)
            else:
                ax.text(0.5, 0.5, "X11a pending", transform=ax.transAxes, ha="center", va="center", color="#999999")
        ax.set_title(titles[i], fontsize=7.6)
        ax.set_ylabel(r"nats per series", labelpad=1)
        S.clean(ax, grid_y=True); S.panel_label(ax, "abc"[i], x=-0.26, y=1.04)
    handles = [Line2D([], [], color=col, marker=m, ms=4.0, lw=1.1, label=lab) for _, lab, col, m in
               [(c[0], {"coordinates": r"coordinates (\CosVI-PL $-$ grid)", "solver": r"solver (\CosVI{} $-$ \CosVI-PL)",
                        "total": r"total (\CosVI{} $-$ grid)"}[c[1]], c[2], c[3]) for c in CON]]
    handles.append(Line2D([], [], color="#666666", marker="o", ms=4.0, mfc="white", lw=0, label=r"open: $|\text{mean}|\le 2\,\text{SE}$"))
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=6.2, bbox_to_anchor=(0.52, 1.0), columnspacing=1.0,
               handletextpad=0.35)
    S.save(fig, "fig11_solver", layout)


if __name__ == "__main__":
    for lay in S.LAYOUTS:
        draw(lay)
    print("ok fig11")
