"""Figure 6 — trained neural SDEs on the double well (N1b, N1c): which noise representation gives the better model?

Data: dX = (X - X^3) dt + 0.5 dB_H, T = 5, 20 noisy observations per series, 48 training / 16 test series, data seeds
0-2.  Every model is scored by the exact-law yardstick (held-out log-likelihood per test series, oracle-H group).
(a) Same inference, different noise law: paired difference ll(r) - ll(CosVI) per data seed, mean +- sd over seeds,
    for each representation r and H_true (marker shape).  Zero = CosVI.
(b) The competitor's full system (validated port: lift + Girsanov control network), H learned jointly or fixed at the
    truth, published and best nodes: the same paired difference (note the wider scale).
(c) Learned drift at H_true = 0.75 (mean over seeds 0-2) against the truth; grey: where the data are (occupation).
Sources: work/results/n1b/n1b_{rep}_seed{s}_H{H}.json, work/results/n1c/n1c_{mode}_seed{s}_H{H}.json.
"""
import os, sys, json, glob
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S

HS = (0.6, 0.75, 0.9)
HM = {0.6: "o", 0.75: "s", 0.9: "D"}
REPS = [("cosine", "cosvi_j", r"\CosVI-32"), ("grid", "grid", "exact grid"), ("markov", "ma", "MA-fBm law, published"),
        ("markov_tuned", "ma_best", "MA-fBm law, tuned"), ("markov_best", "ma_best", "MA-fBm law, best")]
SYS = [("oracle", r"$H$ fixed"), ("joint", r"$H$ learned"),
       ("oracle_best", r"best, $H$ fixed"), ("joint_best", r"best, $H$ learned")]


def nb(rep, H, s):
    f = os.path.join(S.RES, "n1b", f"n1b_{rep}_seed{s}_H{H:.2f}.json")
    return json.load(open(f)) if os.path.exists(f) else None


def nc(mode, H, s):
    f = os.path.join(S.RES, "n1c", f"n1c_{mode}_seed{s}_H{H:.2f}.json")
    return json.load(open(f)) if os.path.exists(f) else None


def ll_cosvi(H, s):
    return nb("twoscale", H, s)["test_loglik_exact_grid_law_mean"][-1]


def dot_rows(ax, rows, xlim, xlabel):
    for i, (vals, key, label) in enumerate(rows):
        col = S.STYLE[key]["color"]
        for j, H in enumerate(HS):
            m, sd = vals[H]
            y = i + (j - 1) * 0.22
            ax.errorbar([m], [y], xerr=[[sd], [sd]], fmt=HM[H], color=col, mfc=col if j != 1 else "white", mec=col,
                        ms=3.8, lw=0.9, capsize=1.4, zorder=4)
    ax.axvline(0, color=S.C["cosvi"], lw=1.6, zorder=2)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[2] for r in rows], fontsize=7.0)
    ax.invert_yaxis(); ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#EEEEEE", lw=0.5, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)


def draw(layout):
    S.setup()
    W = S.WIDE[layout]
    fig = plt.figure(figsize=(W, 2.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.95, 1.0], wspace=0.55, left=0.135, right=0.99, top=0.82, bottom=0.17)
    # (a)
    rows = []
    for rep, key, label in REPS:
        vals = {}
        for H in HS:
            d = [nb(rep, H, s)["test_loglik_exact_grid_law_mean"][-1] - ll_cosvi(H, s) for s in (0, 1, 2) if nb(rep, H, s)]
            vals[H] = (float(np.mean(d)), float(np.std(d, ddof=1)))
        rows.append((vals, key, label))
    ax = fig.add_subplot(gs[0, 0])
    dot_rows(ax, rows, (-2.6, 1.0), r"$\Delta$ held-out log-lik. vs \CosVI{} (nats)")
    ax.set_title(r"other noise law, same inference", fontsize=8.2)
    S.panel_label(ax, "a", x=-0.78)
    # (b)
    rows = []
    for mode, label in SYS:
        vals = {}
        for H in HS:
            d = [nc(mode, H, s)["test_loglik_exact_grid_law_mean"] - ll_cosvi(H, s) for s in (0, 1, 2) if nc(mode, H, s)]
            vals[H] = (float(np.mean(d)), float(np.std(d, ddof=1)))
        rows.append((vals, "their", label))
    ax = fig.add_subplot(gs[0, 1])
    dot_rows(ax, rows, (-30, 2), r"$\Delta$ held-out log-lik. vs \CosVI{} (nats)")
    ax.set_title(r"MA-fBm system (Daems et al.)", fontsize=8.2)
    S.panel_label(ax, "b", x=-0.40)
    hh = [ax.plot([], [], HM[H], color="#666666", mfc="white" if H == 0.75 else "#666666", ms=3.6)[0] for H in HS]
    fig.legend(hh, [r"$H_{\rm true}=%s$" % H for H in HS], loc="upper center", ncol=3, bbox_to_anchor=(0.36, 1.005),
               fontsize=7.0, handletextpad=0.25, columnspacing=1.2)
    # (c) drift curves at H = 0.75
    ax = fig.add_subplot(gs[0, 2])
    H = 0.75
    d0 = nb("twoscale", H, 0); x = np.array(d0["drift_curves"]["x"]); ft = np.array(d0["drift_curves"]["true"])
    occ = []
    for rep, key, lab in (("markov", "ma", "MA-fBm law"), ("grid", "grid", "exact grid"), ("twoscale", "cosvi", r"\CosVI")):
        fits = np.array([nb(rep, H, s)["drift_curves"]["fit"][-1] for s in (0, 1, 2)])
        st = S.st(key, label=lab); st.pop("marker", None); st.pop("ms", None); st.pop("mfc", None); st.pop("mec", None); st.pop("mew", None)
        ax.plot(x, fits.mean(0), **st)
    fits = np.array([nc("joint", H, s)["drift_curve"]["fit"] for s in (0, 1, 2)])
    ax.plot(np.array(nc("joint", H, 0)["drift_curve"]["x"]), fits.mean(0), color=S.C["their"], ls=":", lw=1.4, label="MA-fBm system")
    ax.plot(x, ft, color=S.C["ref"], lw=0.9, label="truth", zorder=9)
    sys.path.insert(0, os.path.join(S.WORK, "code", "experiments")); sys.path.insert(0, os.path.join(S.WORK, "code"))
    import n1_drift_recovery as n1
    occ = np.concatenate([n1.make_data(H, s)[3] for s in (0, 1, 2)])
    hist, edges = np.histogram(np.clip(occ, -1.5, 1.5), bins=60, range=(-1.5, 1.5), density=True)
    xc = 0.5 * (edges[1:] + edges[:-1])
    ax.fill_between(xc, -2.05, -2.05 + 0.45 * hist / hist.max(), color="#DDDDDD", lw=0, zorder=0, step="mid")
    ax.text(0.0, -1.90, r"empirical occupation", fontsize=5.8, color="#666666", ha="center", va="bottom")
    ax.axhline(0, color="#DDDDDD", lw=0.5, zorder=0)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-2.05, 2.05)
    ax.set_xlabel(r"state $x$"); ax.set_ylabel(r"drift $f_\theta(x)$")
    ax.set_title(r"learned drift, $H=0.75$", fontsize=8.2)
    ax.legend(loc="upper right", fontsize=5.9, handlelength=1.6, labelspacing=0.14, borderaxespad=0.25)
    S.clean(ax); S.panel_label(ax, "c", x=-0.30)
    S.save(fig, "fig06_doublewell", layout)


if __name__ == "__main__":
    for lay in S.LAYOUTS:
        draw(lay)
    print("ok fig06")
