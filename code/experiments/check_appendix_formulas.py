r"""Check the closed forms printed in Appendix A (cosine Gram matrix G, covariances q_k(t) and rates q_k'(t)) against
direct quadrature of their defining integrals, independently of the implementation (19 Nov 2026).

Definitions (standard fBm, 1/2 < H < 1, h_0 = T^{-1/2}, h_k = sqrt(2/T) cos(k pi t/T)):
  q_k(t)  = E[B_H(t) Z_k] = H int_0^T h_k(v) [v^{2H-1} + sgn(t-v)|t-v|^{2H-1}] dv          (regular integrand)
  G_kl    = E[Z_k Z_l]    = int_0^T h_k(t) q_l'(t) dt
Closed forms (as printed):  E(s, a) = int_0^s u^{2H-1} e^{iau} du = s^{2H} 1F1(2H; 2H+1; ias) / (2H),
  q_k(t)  = H Re{ c_k [ E(T,a_k) + e^{i a_k t} ( conj E(t,a_k) - E(T-t,a_k) ) ] },  a_k = k pi / T,
  q_k'(t) = H Re{ c_k e^{i a_k t} [ t^{2H-1} conj M(a_k t) + (T-t)^{2H-1} M(a_k (T-t)) ] },  M(x) = 1F1(2H-1; 2H; ix),
  G from CosChaos Proposition 1 with p = Re M(k pi), r = Im M(k pi), C = 2H T^{2H-1}.
The hypergeometric functions are evaluated with mpmath here (the code uses Gauss-Jacobi quadrature).
Run from work\code\experiments:  python -X utf8 check_appendix_formulas.py
"""
import os, sys, json
import numpy as np
import mpmath as mp
from scipy.integrate import quad
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from fnvi.vendor import hfc

mp.mp.dps = 30


def E(H, s, a):
    return complex(mp.mpf(s) ** (2 * H) * mp.hyp1f1(2 * H, 2 * H + 1, 1j * a * s) / (2 * H))


def Mf(H, x):
    return complex(mp.hyp1f1(2 * H - 1, 2 * H, 1j * x))


def ck(k, T):
    return 1 / np.sqrt(T) if k == 0 else np.sqrt(2 / T)


def q_closed(H, T, k, t):
    a = k * np.pi / T
    return float(np.real(H * ck(k, T) * (E(H, T, a) + np.exp(1j * a * t) * (np.conj(E(H, t, a)) - E(H, T - t, a)))))


def dq_closed(H, T, k, t):
    a = k * np.pi / T
    return float(np.real(H * ck(k, T) * np.exp(1j * a * t) * (t ** (2 * H - 1) * np.conj(Mf(H, a * t))
                                                              + (T - t) ** (2 * H - 1) * Mf(H, a * (T - t)))))


def q_quad(H, T, k, t):
    hk = lambda v: ck(k, T) * np.cos(k * np.pi * v / T)
    f = lambda v: hk(v) * (v ** (2 * H - 1) + np.sign(t - v) * abs(t - v) ** (2 * H - 1))
    return H * (quad(f, 0, t, limit=400, epsabs=1e-13, epsrel=1e-12)[0] + quad(f, t, T, limit=400, epsabs=1e-13, epsrel=1e-12)[0])


def G_closed(H, T, J):
    C = 2 * H * T ** (2 * H - 1)
    m = [Mf(H, k * np.pi) for k in range(J)]
    p = np.array([z.real for z in m]); r = np.array([z.imag for z in m])
    G = np.zeros((J, J)); G[0, 0] = T ** (2 * H - 1)
    for k in range(1, J):
        G[k, k] = C * (p[k] - 2 * (1 - H) * r[k] / (k * np.pi))
        if k % 2 == 0:
            G[0, k] = G[k, 0] = -np.sqrt(2) * C * r[k] / (np.pi * k)
        for l in range(1, J):
            if l != k and (k + l) % 2 == 0:
                G[k, l] = 2 * C / np.pi * (l * r[l] - k * r[k]) / (k ** 2 - l ** 2)
    return G


def G_quad(H, T, k, l):
    hk = lambda t: ck(k, T) * np.cos(k * np.pi * t / T)
    return quad(lambda t: hk(t) * dq_closed(H, T, l, t), 0, T, limit=600, epsabs=1e-12, epsrel=1e-11)[0]


def main():
    out = {}
    for H, T in ((0.6, 5.0), (0.75, 2.0), (0.9, 5.0)):
        J = 6
        qerr = max(abs(q_closed(H, T, k, t) - q_quad(H, T, k, t)) for k in range(J) for t in (0.3, 1.1, 0.5 * T, 0.93 * T))
        Gc = G_closed(H, T, J)
        Gq = np.array([[G_quad(H, T, k, l) for l in range(J)] for k in range(J)])
        gerr = float(np.abs(Gc - Gq).max())
        code_G = float(np.abs(Gc - hfc.gram(H, T, "cos", J)).max())
        tt = np.array([0.3, 1.1, 0.5 * T, 0.93 * T])
        code_q = float(np.abs(np.array([[q_closed(H, T, k, t) for k in range(J)] for t in tt]) - hfc.Q(H, T, "cos", J, tt)).max())
        out[f"H={H},T={T}"] = {"q_closed_vs_quadrature": qerr, "G_closed_vs_quadrature": gerr,
                               "G_closed_vs_code": code_G, "q_closed_vs_code": code_q}
        print(f"H={H} T={T}: |q closed - quadrature| {qerr:.1e}; |G closed - quadrature| {gerr:.1e}; "
              f"vs code: G {code_G:.1e}, q {code_q:.1e}")
    # H = 1/2: G = I and q_k(t) = int_0^t h_k
    H, T = 0.5, 3.0
    out["H=0.5"] = {"q0": abs(hfc.Q(H, T, "cos", 3, np.array([1.2]))[0, 0] - 1.2 / np.sqrt(T)),
                    "q2": abs(hfc.Q(H, T, "cos", 3, np.array([1.2]))[0, 2] - np.sqrt(2 * T) / (2 * np.pi) * np.sin(2 * np.pi * 1.2 / T))}
    print("H = 1/2:", out["H=0.5"])
    json.dump(out, open(os.path.join(HERE, "..", "..", "results", "x8", "appendix_formulas_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
