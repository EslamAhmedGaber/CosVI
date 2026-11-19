"""Figure 2 — the CosVI procedure (replaces the earlier TikZ box diagram).

Built after the scientific-writing-studio figure kit (procedure diagram, family MF28) and its recipe S01 "Explain a
method": three stages read left to right in quiet dashed containers, one accent per stage, short labels, one straight
labelled arrow between consecutive stages; the loops are stated in each stage's subtitle instead of crossing arrows.
No results are drawn.
"""
import os, sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S

STAGES = [
    (r"Once per value of $H$", r"for every $H$ of the grid", "#3A93C9", "#EEF6FC",
     ["Gram matrix $G$,\nCholesky factor $L$",
      "residual covariance $C_R$,\nfactor $L_R$",
      "noise map $\\Phi$\n(exact law on the grid)"]),
    (r"Every training step", r"repeated 5000 times", S.C["cosvi"], "#EAF1F8",
     ["draw $\\boldsymbol z=\\mu_j+U_j\\boldsymbol\\epsilon$ and $x_0$",
      "noise rates $\\Phi\\boldsymbol z$",
      "solve $\\dot X=f_\\theta(X)+\\sigma\\,\\Phi\\boldsymbol z$\nby Runge--Kutta",
      "ELBO with closed-form KL;\nAdam step"]),
    (r"Evaluation", r"of any trained model", "#4D4D4D", "#F3F3F3",
     ["select $\\hat H$:\nbest training bound",
      "freeze $(f_\\theta,\\sigma,\\hat H)$;\nrefit $q$ on held-out series",
      "importance sampling\nunder a common evaluator"]),
]


def draw(layout):
    S.setup()
    W = S.WIDE[layout]
    H = 2.45 if layout == "two_columns" else 2.6
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
    fs = 7.5 if layout == "two_columns" else 7.3
    xs = [0.155, 0.5, 0.845]; cw = 0.296; bw = 0.264
    top, bottom = 0.755, 0.045                       # extent of the boxes
    frames = []
    for (head, sub, col, fill, items), cx in zip(STAGES, xs):
        x0f, y0f, hf = cx - cw / 2, bottom - 0.03, top - bottom + 0.215
        ax.add_patch(FancyBboxPatch((x0f, y0f), cw, hf, boxstyle="round,pad=0.0,rounding_size=0.02", fc=fill, ec=col,
                                    lw=0.7, linestyle=(0, (3, 2)), zorder=1))
        frames.append((x0f, y0f, cw, hf))
        ax.text(cx, top + 0.135, rf"\textbf{{{head}}}", ha="center", va="center", color=col, fontsize=fs + 0.9, zorder=5)
        ax.text(cx, top + 0.068, rf"\textit{{{sub}}}", ha="center", va="center", color=col, fontsize=fs - 0.4, zorder=5)
        n = len(items); gap = 0.04
        bh = (top - bottom - gap * (n - 1)) / n
        prev = None
        for i, txt in enumerate(items):
            y1 = top - i * (bh + gap); y0 = y1 - bh
            ax.add_patch(FancyBboxPatch((cx - bw / 2, y0), bw, bh, boxstyle="round,pad=0.0,rounding_size=0.012",
                                        fc="white", ec=col, lw=0.9, zorder=2))
            ax.text(cx, (y0 + y1) / 2, txt, ha="center", va="center", fontsize=fs, color="#111111", zorder=5,
                    linespacing=1.05)
            if prev is not None:
                ax.add_patch(FancyArrowPatch((cx, prev), (cx, y1), arrowstyle="-|>", mutation_scale=6.5, lw=0.75,
                                             color=col, shrinkA=0.5, shrinkB=0.5, zorder=4))
            prev = y0
    # one straight arrow between consecutive stages, at mid-height, labelled with what is handed over
    ym = (top + bottom) / 2
    for j, lab in ((0, r"$\Phi$"), (1, r"$f_\theta,\ \sigma$")):
        xa = frames[j][0] + frames[j][2]; xb = frames[j + 1][0]
        ax.add_patch(FancyArrowPatch((xa + 0.002, ym), (xb - 0.002, ym), arrowstyle="-|>", mutation_scale=9, lw=1.1,
                                     color="#222222", shrinkA=0, shrinkB=0, zorder=6))
        ax.text((xa + xb) / 2, ym + 0.035, lab, ha="center", va="bottom", fontsize=fs, color="#222222", zorder=6)
    S.save(fig, "fig02_procedure", layout)


if __name__ == "__main__":
    for lay in S.LAYOUTS:
        draw(lay)
    print("ok fig02")
