"""One visual identity for every figure of the CosVI paper (both Elsevier layouts), in the family look of CosChaos.

Text is set by LaTeX with Latin Modern (the body font of the manuscript), so labels and mathematics in the figures
match the text exactly.  Sizes are print sizes: one-column layout 6.30 in wide (text width 6.48 in); two-column layout
7.00 in for full-width figures (text width 7.22 in) and 3.40 in for single-column figures (column width 3.49 in).

Colours (validated 16 Nov 2026 with the dataviz checker, light mode on white, legend order below: lightness band,
chroma, colour-vision-deficiency and normal-vision separation all pass; sky blue and orange are below 3:1 contrast, so
they always carry markers and direct labels):
  CosVI #2166AC (deep blue, solid, filled circles, thick)    CosVI-J #56B4E9 (sky blue, dashed, open circles)
  exact grid #009E73 (bluish green, dash-dot, diamonds)       MA-fBm law, published nodes #D55E00 (vermilion, dashed,
  open squares)   MA-fBm law, best or tuned nodes #E69F00 (orange, dotted, squares)   MA-fBm, their full system
  #CC79A7 (reddish purple, dotted, triangles)   exact reference / truth #111111 (black)   Brownian / other #7F7F7F.
A colour always follows its method in every figure; CosVI is drawn last (on top) and thickest.
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
MAN = os.path.dirname(HERE)
WORK = os.path.dirname(MAN)
RES = os.path.join(WORK, "results")
FIGS = {"one_column": os.path.join(MAN, "source", "figures", "one_column"),
        "two_columns": os.path.join(MAN, "source", "figures", "two_columns")}
WIDE = {"one_column": 6.3, "two_columns": 7.0}
SINGLE = {"one_column": 4.6, "two_columns": 3.4}
LAYOUTS = ("one_column", "two_columns")

C = {"cosvi": "#2166AC", "cosvi_j": "#56B4E9", "grid": "#009E73", "ma": "#D55E00", "ma_best": "#E69F00",
     "their": "#CC79A7", "ref": "#111111", "grey": "#7F7F7F", "light": "#D9D9D9", "band": "#EEF3F9", "warm": "#FBEDE5"}

STYLE = {
    "cosvi":   dict(color=C["cosvi"], ls="-", marker="o", mfc=C["cosvi"], lw=1.9, ms=4.2, zorder=6, label=r"\CosVI"),
    "cosvi_j": dict(color=C["cosvi_j"], ls="--", marker="o", mfc="white", lw=1.3, ms=3.8, zorder=5, label=r"\CosVI-$J$"),
    "grid":    dict(color=C["grid"], ls="-.", marker="D", mfc=C["grid"], lw=1.3, ms=3.4, zorder=4, label="exact grid"),
    "ma":      dict(color=C["ma"], ls="--", marker="s", mfc="white", lw=1.3, ms=3.8, zorder=3, label="MA-fBm law, published nodes"),
    "ma_best": dict(color=C["ma_best"], ls=":", marker="s", mfc=C["ma_best"], lw=1.4, ms=3.6, zorder=3, label="MA-fBm law, best nodes"),
    "their":   dict(color=C["their"], ls=":", marker="^", mfc=C["their"], lw=1.4, ms=4.2, zorder=3, label="MA-fBm system (Daems et al.)"),
    "ref":     dict(color=C["ref"], ls="-", marker=None, lw=1.0, zorder=7, label="exact"),
    "grey":    dict(color=C["grey"], ls=":", marker=None, lw=1.0, zorder=1, label=""),
}
REP_KEY = {"twoscale": "cosvi", "cosine": "cosvi_j", "cosine64": "cosvi_j", "grid": "grid", "markov": "ma",
           "markov_best": "ma_best", "markov_tuned": "ma_best"}

PREAMBLE = (r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath,amssymb}"
            r"\newcommand{\CosVI}{CosVI}")


def setup():
    plt.rcParams.update({
        "text.usetex": True, "text.latex.preamble": PREAMBLE, "font.family": "serif", "font.serif": ["Latin Modern Roman"],
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5, "legend.fontsize": 7.4,
        "xtick.labelsize": 7.6, "ytick.labelsize": 7.6, "axes.linewidth": 0.6, "lines.linewidth": 1.3,
        "lines.markersize": 4, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4, "xtick.major.size": 3.0, "ytick.major.size": 3.0, "xtick.minor.size": 1.8,
        "ytick.minor.size": 1.8, "xtick.direction": "out", "ytick.direction": "out", "legend.frameon": False,
        "legend.handlelength": 2.2, "legend.borderaxespad": 0.3, "axes.titlepad": 4.0, "savefig.dpi": 300,
        "pdf.fonttype": 42, "axes.grid": False, "figure.constrained_layout.use": False})


def st(key, **over):
    """Plot keyword arguments for a method key, with overrides (e.g. label=...)."""
    s = dict(STYLE[key]); s.update(over)
    out = dict(color=s["color"], ls=s["ls"], lw=s["lw"], zorder=s["zorder"], label=s.get("label"))
    if s.get("marker"):
        out.update(marker=s["marker"], ms=s.get("ms", 4), mfc=s.get("mfc", s["color"]), mec=s["color"], mew=0.9)
    return out


def panel_label(ax, s, x=-0.14, y=1.03):
    ax.text(x, y, rf"\textbf{{({s})}}", transform=ax.transAxes, fontsize=9, va="bottom", ha="left")


def clean(ax, grid_y=False):
    ax.spines[["top", "right"]].set_visible(False)
    if grid_y:
        ax.grid(axis="y", color="#E6E6E6", lw=0.5, zorder=0)


def logy(ax, numticks=12):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=numticks))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=numticks))
    ax.yaxis.set_minor_formatter(NullFormatter())


def direct_label(ax, x, y, text, color, dx=4, dy=0, ha="left", va="center", size=7.2, **kw):
    """Label a curve at its end in its own colour (relief for light colours; legend kept too)."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", color=color, fontsize=size,
                ha=ha, va=va, **kw)


def note(ax, x, y, text, xy_text, color="#333333", size=7.0, **kw):
    """An insight call-out with a thin arrow (text in data or axes coordinates via kw)."""
    ax.annotate(text, (x, y), xytext=xy_text, color=color, fontsize=size,
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color=color, shrinkA=0, shrinkB=2, mutation_scale=6), **kw)


def save(fig, name, layout):
    os.makedirs(FIGS[layout], exist_ok=True)
    fig.savefig(os.path.join(FIGS[layout], name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(FIGS[layout], name + ".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def load(path):
    with open(os.path.join(RES, path)) as fh:
        return json.load(fh)


def files(pattern):
    return sorted(glob.glob(os.path.join(RES, pattern)))
