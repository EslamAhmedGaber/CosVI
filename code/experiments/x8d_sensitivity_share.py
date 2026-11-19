r"""X8(d), descriptive and post hoc (19 Nov 2026; suggested by an independent review of 18 Nov, recomputed here): how
concentrated is the observations' sensitivity to the noise coordinates?

In the linear analogue of X8(a) (drift -x, sigma = 0.5, T = 5, N = 80, J = 16, 20 observations; x8_linear_check.py)
the observed states are X(t_i) = F z + b x0 under the exact RK4 recursion. Under the prior z ~ N(0, I) the variance of
the observations due to the noise is ||F||_F^2 = sum_i Var(X(t_i) | x0), and a set S of coordinates carries the share
||F_S||_F^2 / ||F||_F^2 of it.  Reported for the CosVI-PL coordinates (the first J are the cosine block) and the exact
grid (Cholesky coordinates): the share of the cosine block, the share of the best J grid columns (largest column norms),
and the number of coordinates needed for 90 % and 99 % in each system (best columns first).  No decision rule: this is
an interpretation aid for Section 5.3, not a test.
Run from work\code\experiments:  python -X utf8 x8d_sensitivity_share.py      Out: work/results/x8/x8d_sensitivity_share.json
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
from x8_linear_check import T, N, J, obs_matrix, grid_map, twoscale_pl_map

OUT = os.path.join(HERE, "..", "..", "results", "x8", "x8d_sensitivity_share.json")


def need(colsq, frac):
    c = np.cumsum(np.sort(colsq)[::-1]) / colsq.sum()
    return int(np.searchsorted(c, frac) + 1)


def main():
    res = {"settings": {"T": T, "N": N, "J": J}, "by_H": {}}
    for H in (0.6, 0.75, 0.9):
        F_pl = obs_matrix(twoscale_pl_map(H, T, N, J))[:, :-1]
        F_gr = obs_matrix(grid_map(H, T, N))[:, :-1]
        cs_pl = (F_pl ** 2).sum(0); cs_gr = (F_gr ** 2).sum(0)
        row = {"total_pl": float(cs_pl.sum()), "total_grid": float(cs_gr.sum()),
               "share_cosine_block": float(cs_pl[:J].sum() / cs_pl.sum()),
               "share_best_J_grid": float(np.sort(cs_gr)[::-1][:J].sum() / cs_gr.sum()),
               "share_best_J_pl": float(np.sort(cs_pl)[::-1][:J].sum() / cs_pl.sum()),
               "n90_pl": need(cs_pl, 0.90), "n99_pl": need(cs_pl, 0.99),
               "n90_grid": need(cs_gr, 0.90), "n99_grid": need(cs_gr, 0.99)}
        res["by_H"][str(H)] = row
        print(f"H={H}: total {row['total_pl']:.6f} vs {row['total_grid']:.6f}; cosine block {100*row['share_cosine_block']:.1f} %, "
              f"best {J} grid columns {100*row['share_best_J_grid']:.1f} %; coordinates for 90/99 %: "
              f"CosVI-PL {row['n90_pl']}/{row['n99_pl']}, grid {row['n90_grid']}/{row['n99_grid']}")
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
