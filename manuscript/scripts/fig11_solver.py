"""Figure 11 — coordinates or solver? (X8 c; protocol fixed 18 Nov 2026 14:35).

Paired contrasts on the double well over five data seeds (mean +- 2 SE; the rule calls an effect present when
|mean| > 2 SE): coordinate effect = CosVI-PL - exact grid (same law and solver), solver effect = CosVI - CosVI-PL
(same coordinates, different within-step forcing), total = CosVI - exact grid.
(a) held-out log-likelihood per test series under the grid's solver (PL evaluator, primary);
(b) under CosVI's solver (smooth-stage evaluator); (c) reduction of the occupation-weighted drift RMSE.
Positive values favour the first-named arm.  Source: work/results/x8/x8_summary.json.
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S

CON = [("coordinate", "coordinates", S.C["cosvi_j"], "o"), ("solver", "solver", S.C["grid"], "D"),
       ("total", "total", S.C["cosvi"], "s")]
METRIC = [("pl", r"held-out log-lik., grid's solver"), ("smooth", r"held-out log-lik., \CosVI's solver"),
          ("drift", r"drift RMSE reduction")]


def draw(layout):
    S.setup()
    W = S.WIDE[layout]
    fig = plt.figure(figsize=(W, 2.35 if layout == "two_columns" else 2.6))
    gs = fig.add_gridspec(1, 3, wspace=0.38, left=0.07, right=0.99, top=0.80, bottom=0.15)
    f = os.path.join(S.RES, "x8", "x8_summary.json")
    cc = json.load(open(f)).get("c", {}).get("by_H", {}) if os.path.exists(f) else {}
    for i, (mk, title) in enumerate(METRIC):
        ax = fig.add_subplot(gs[0, i])
        if len(cc) < 3:
            ax.text(0.5, 0.5, "results pending", transform=ax.transAxes, ha="center", va="center", color="#999999")
        else:
            for j, H in enumerate(("0.6", "0.75", "0.9")):
                for k, (con, lab, col, m) in enumerate(CON):
                    v = cc[H]["contrasts"][con][mk]
                    x = j - 0.2 + 0.2 * k
                    present = v["verdict"].startswith("present")
                    ax.errorbar([x], [v["mean"]], yerr=[2 * v["se"]], color=col, lw=1.1, capsize=1.8, marker=m, ms=4.0,
                                mfc=col if present else "white", mec=col, zorder=3)
            ax.axhline(0, color="#888888", lw=0.7, zorder=1)
            ax.set_xticks(range(3)); ax.set_xticklabels([r"$H=0.6$", r"$H=0.75$", r"$H=0.9$"])
            ax.set_xlim(-0.5, 2.5)
        ax.set_title(title, fontsize=7.8)
        ax.set_ylabel(r"nats per series" if mk != "drift" else r"RMSE reduction", labelpad=1)
        S.clean(ax, grid_y=True); S.panel_label(ax, "abc"[i], x=-0.26, y=1.04)
    handles = [Line2D([], [], color=col, marker=m, ms=4.0, lw=1.1, label=lab) for _, lab, col, m in
               [(c[0], {"coordinates": r"coordinates (\CosVI-PL $-$ grid)", "solver": r"solver (\CosVI{} $-$ \CosVI-PL)",
                        "total": r"total (\CosVI{} $-$ grid)"}[c[1]], c[2], c[3]) for c in CON]]
    handles.append(Line2D([], [], color="#666666", marker="o", ms=4.0, mfc="white", lw=0, label=r"open: not distinguishable"))
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=6.4, bbox_to_anchor=(0.53, 1.0), columnspacing=1.2,
               handletextpad=0.4)
    S.save(fig, "fig11_solver", layout)


if __name__ == "__main__":
    for lay in S.LAYOUTS:
        draw(lay)
    print("ok fig11")
