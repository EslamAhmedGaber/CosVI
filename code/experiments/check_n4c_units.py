r"""E-f check (16 Nov 2026): numerical verification of the unit conversion used to evaluate the competitor's Nile
models (N4c) under our exact-law yardstick.

N4c fits their system in normalised time tau = t / c (c = 20 years per unit, T_c = 2.5) with noise of the
Mandelbrot-Van Ness (MVN) scale of their code, and passes it to the yardstick in years as
    f_years = f_c / c,     sigma_years = sigma_c * c^(-H) * sqrt(c_H),     c_H = Var B_MVN(1) = 1 / (Gamma(2H+1) sin(pi H)).
The drift factor is exact algebra (last linear layer divided by c).  This script checks the two numerical
ingredients of the noise factor:
  (1) c_H against direct quadrature of the MVN kernel with prefactor 1/Gamma(H + 1/2);
  (2) that their type-I lift (published nodes K = 5, gamma in [1/20, 20], weights fitted on [0, 3 T_c] by
      omega_type1, as in N4c) has Var B_lift(tau) close to c_H tau^(2H) on [0, T_c] (closed form: for stationary OU
      states, Cov(Y_k(t) - Y_k(0), Y_l(t) - Y_l(0)) = (2 - e^{-g_k t} - e^{-g_l t}) / (g_k + g_l)), i.e. that the
      lift's noise is on the MVN scale, so sqrt(c_H) maps it to standard fBm;
  (3) the self-similarity step: Var of the converted noise sigma_years * B_std(t) at t = c tau equals Var of
      sigma_c * B_MVN(tau) (exact); the published lift's own variance at tau = 1 is recorded beside it.
Run from work\code\experiments:  python check_n4c_units.py     Out: work/results/n4/n4c_unit_check.json
"""
import os, sys, json
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma as Gamma
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from fnvi.ma_fbm import c_mvn
from fnvi.mafbm_torch import gamma_by_gamma_max, omega_type1

OUT = os.path.join(HERE, "..", "..", "results", "n4", "n4c_unit_check.json")
TC, CU = 2.5, 20.0


def c_quadrature(H):
    a = H - 0.5
    i1 = quad(lambda u: ((1 + u) ** a - u ** a) ** 2, 0, 1, limit=200)[0] + \
        quad(lambda u: ((1 + u) ** a - u ** a) ** 2, 1, np.inf, limit=200)[0]
    i2 = 1.0 / (2 * H)                                     # int_0^1 (1 - s)^{2H - 1} ds
    return (i1 + i2) / Gamma(H + 0.5) ** 2


def lift_variance(H, t):
    g = np.asarray(gamma_by_gamma_max(5, 20.0), float)
    w = omega_type1(g, H, 3 * TC)
    gi, gj = g[None, :], g[:, None]
    C = (2 - np.exp(-gi * t) - np.exp(-gj * t)) / (gi + gj)
    return float(w @ C @ w)


def main():
    res = {"note": __doc__.split("\n")[0], "c_H": {}, "lift_scale": {}, "self_similarity": {}}
    for H in (0.58, 0.6, 0.7, 0.75, 0.8, 0.84, 0.9):
        cf, cq = c_mvn(H), c_quadrature(H)
        res["c_H"][f"{H:.2f}"] = {"formula": cf, "quadrature": cq, "rel_diff": abs(cf - cq) / cq}
        ts = np.linspace(0.1, TC, 25)
        ratio = np.array([lift_variance(H, t) / (cf * t ** (2 * H)) for t in ts])
        res["lift_scale"][f"{H:.2f}"] = {"t": ts.tolist(), "var_lift_over_cH_t2H": ratio.tolist(),
                                         "max_abs_dev": float(np.abs(ratio - 1).max()),
                                         "max_abs_dev_t_ge_0.5": float(np.abs(ratio[ts >= 0.5] - 1).max())}
        # self-similarity: sigma_c = 1, tau = 1  ->  t = c years
        sig_years = CU ** (-H) * np.sqrt(cf)
        v_years = sig_years ** 2 * (CU * 1.0) ** (2 * H)         # Var of sigma_years * B_std(t) at t = c
        v_mvn = cf * 1.0 ** (2 * H)                               # Var of sigma_c * B_MVN(tau) at tau = 1 (sigma_c = 1)
        v_lift = lift_variance(H, 1.0)                            # Var of sigma_c * B_lift(tau): the published lift
        res["self_similarity"][f"{H:.2f}"] = {"var_years_at_t_eq_c": v_years, "var_mvn_at_tau_1": v_mvn,
                                              "rel_diff": abs(v_years - v_mvn) / v_mvn,
                                              "var_published_lift_at_tau_1": v_lift,
                                              "lift_over_mvn_at_tau_1": v_lift / v_mvn}
    worst_c = max(v["rel_diff"] for v in res["c_H"].values())
    worst_l = max(v["max_abs_dev_t_ge_0.5"] for v in res["lift_scale"].values())
    worst_s = max(v["rel_diff"] for v in res["self_similarity"].values())
    res["verdict"] = {"c_H_formula_vs_quadrature_max_rel_diff": worst_c,
                      "lift_variance_vs_cH_t2H_max_dev_t_ge_0.5": worst_l,
                      "self_similarity_max_rel_diff": worst_s,
                      "conversion_confirmed": bool(worst_c < 1e-6 and worst_s < 1e-9),
                      "finding": "the conversion is exact for MVN fBm; the published lift itself (K = 5, [1/20, 20], "
                                 "fitted on [0, 3 T_c]) deviates from the MVN variance (lift_scale), so part of the "
                                 "competitor's exact-law gap on the Nile is the lift's approximation error, which the "
                                 "yardstick charges to every representation alike"}
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res["verdict"], indent=1))


if __name__ == "__main__":
    main()
