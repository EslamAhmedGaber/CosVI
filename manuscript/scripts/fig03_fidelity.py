"""Figure 3 — noise fidelity at equal dimension (G1).

Relative integrated mean-square error of the noise path on [0, 1] against the dimension D, one panel per H:
Karhunen-Loeve optimum (black), cosine coordinates (the noise law of CosVI-J), and the MA-fBm lift with the published
node rule and with tuned nodes (K = D).  A double arrow at D = 64 gives the factor between the cosine coordinates and
the tuned lift, in the colour of the more accurate one.  CosVI itself is exact on the time grid (zero error): stated in
the caption, not plotted.  Source: work/pilot_checks/results/g1_pilot.json (deterministic quadrature).
"""
import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S

G1 = json.load(open(os.path.join(S.WORK, "pilot_checks", "results", "g1_pilot.json")))
HS = ["0.60", "0.75", "0.90"]
DIMS = [int(d) for d in G1["dims"]]


def series(H):
    kl = np.array([G1["kl"][H][str(d)] for d in DIMS])
    cos = np.array([G1["cosine"][H][str(d)]["projection"]["rel_int_mse"] for d in DIMS])
    pub = np.array([G1["ma_I_default"][H][str(d)]["rel_int_mse"] for d in DIMS])
    tun = np.array([G1["ma_I_tuned"][H][str(d)]["rel_int_mse"] for d in DIMS])
    return kl, cos, pub, tun


def factor_arrow(ax, x, y_cos, y_lift):
    better_cos = y_cos < y_lift
    f = (y_lift / y_cos) if better_cos else (y_cos / y_lift)
    col = S.C["cosvi"] if better_cos else S.C["ma_best"]
    ax.annotate("", xy=(x, y_cos), xytext=(x, y_lift),
                arrowprops=dict(arrowstyle="<|-|>", lw=0.7, color=col, shrinkA=1.5, shrinkB=1.5, mutation_scale=5))
    ym = np.sqrt(y_cos * y_lift)
    fs = float(f"{f:.2g}")
    num = f"{fs:,.0f}".replace(",", "{,}") if fs >= 10 else f"{fs:.1f}"
    txt = (rf"at $D={x}$: cosine ${num}\times$ smaller" if better_cos
           else rf"at $D={x}$: tuned lift ${num}\times$ smaller")
    ax.text(0.04, 0.05, txt, transform=ax.transAxes, ha="left", va="bottom", fontsize=6.8, color=col)


def draw(layout):
    S.setup()
    W = S.WIDE[layout]
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.65), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.74, bottom=0.19, wspace=0.10)
    for i, H in enumerate(HS):
        ax = axes[i]
        kl, cos, pub, tun = series(H)
        ax.plot(DIMS, pub, **S.st("ma", label=r"MA-fBm lift, published nodes"))
        ax.plot(DIMS, tun, **S.st("ma_best", label=r"MA-fBm lift, tuned nodes"))
        ax.plot(DIMS, kl, **S.st("ref", label=r"Karhunen--Lo\`eve optimum", lw=0.9))
        ax.plot(DIMS, cos, **S.st("cosvi_j", label=r"cosine coordinates (\CosVI-$J$)", lw=1.7, ms=4.0))
        ax.set_xscale("log", base=2)
        ax.set_xticks([4, 8, 16, 32, 64]); ax.set_xticklabels(["4", "8", "16", "32", "64"])
        ax.minorticks_off()
        S.logy(ax)
        ax.set_ylim(3e-9, 2.0)
        ax.set_xlim(3.3, 90)
        ax.set_title(rf"$H={float(H):.2f}$")
        if i == 0:
            ax.set_ylabel(r"relative integrated MSE")
        S.clean(ax)
        S.panel_label(ax, "abc"[i], x=-0.20 if i == 0 else -0.06)
        factor_arrow(ax, 64, cos[-1], tun[-1])
    fig.text(0.54, 0.03, r"dimension $D$ (cosine coordinates $J$, or OU states $K$ of the lift)", ha="center", fontsize=8.5)
    h, l = axes[0].get_legend_handles_labels()
    order = [3, 2, 0, 1]
    fig.legend([h[k] for k in order], [l[k] for k in order], loc="upper center", ncol=2, bbox_to_anchor=(0.54, 1.005),
               fontsize=7.3, handlelength=2.6, columnspacing=1.6, labelspacing=0.3)
    S.save(fig, "fig03_fidelity", layout)


if __name__ == "__main__":
    for lay in S.LAYOUTS:
        draw(lay)
    print("ok fig03")
