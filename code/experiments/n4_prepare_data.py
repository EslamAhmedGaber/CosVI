"""N4 data preparation: parse the Nile minima (and the Northern-hemisphere temperature series) from the R *text*
files of CRAN package longmemo 1.1-4 (downloaded 13 Nov 2026 with the author's permission; tarball SHA-256
390849c9…34a0f9; package licence GPL-2 | GPL-3). Nothing from the package is executed: the numbers are read with a
regular expression. Outputs: work/results/n4/data/nile_min.csv, nhemi_temp.csv, provenance.json, plus classical
Hurst estimates (aggregated variance, R/S, Whittle for fractional Gaussian noise) as reference values.
"""
import os, re, json, hashlib
import numpy as np
from scipy.optimize import minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.abspath(os.path.join(HERE, "..", ".."))
RAW = os.path.join(WORK, "external", "data_raw", "longmemo", "longmemo")
OUT = os.path.join(WORK, "results", "n4", "data"); os.makedirs(OUT, exist_ok=True)


def numbers_in_c(text):
    m = re.search(r"c\((.*?)\)\s*[,)]", text, re.S)
    return np.array([float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", m.group(1))])


def fgn_whittle(x):
    """Whittle estimate of H for fractional Gaussian noise (spectral density by the standard series, truncated)."""
    x = np.asarray(x, float) - np.mean(x); n = len(x)
    I = np.abs(np.fft.fft(x)) ** 2 / (2 * np.pi * n)
    lam = 2 * np.pi * np.arange(1, (n - 1) // 2 + 1) / n; I = I[1:len(lam) + 1]
    from math import gamma
    def spec(H):
        j = np.arange(-200, 201)[:, None]
        s = (np.abs(2 * np.pi * j + lam[None, :]) ** (-2 * H - 1)).sum(0)
        return 2 * np.sin(np.pi * H) * gamma(2 * H + 1) * (1 - np.cos(lam)) * s
    def obj(H):
        g = spec(H); c = np.mean(I / g)
        return np.log(c) + np.mean(np.log(g))
    r = minimize_scalar(obj, bounds=(0.51, 0.99), method="bounded")
    return float(r.x)


def aggvar(x):
    x = np.asarray(x, float); n = len(x); ms = np.unique(np.floor(np.logspace(0.3, np.log10(n / 8), 15)).astype(int))
    v = [np.var([x[i:i + m].mean() for i in range(0, n - m + 1, m)], ddof=1) for m in ms]
    slope = np.polyfit(np.log(ms), np.log(v), 1)[0]
    return float(1 + slope / 2)


def rs(x):
    x = np.asarray(x, float); n = len(x); ms = np.unique(np.floor(np.logspace(1, np.log10(n / 4), 12)).astype(int))
    vals = []
    for m in ms:
        r_ = []
        for i in range(0, n - m + 1, m):
            s = x[i:i + m]; z = np.cumsum(s - s.mean()); sd = s.std()
            if sd > 0:
                r_.append((z.max() - z.min()) / sd)
        vals.append(np.mean(r_))
    return float(np.polyfit(np.log(ms), np.log(vals), 1)[0])


def main():
    prov = {"source": "CRAN package longmemo 1.1-4 (https://cran.r-project.org/src/contrib/longmemo_1.1-4.tar.gz)",
            "downloaded": "2026-11-13", "tarball_sha256": hashlib.sha256(open(os.path.join(RAW, "..", "longmemo_1.1-4.tar.gz"), "rb").read()).hexdigest(),
            "package_licence": "GPL-2 | GPL-3", "series": {}}
    t = open(os.path.join(RAW, "data", "NileMin.R"), encoding="latin-1").read()
    nile = 1000 + numbers_in_c(t)
    assert len(nile) == 663, len(nile)
    years = np.arange(622, 622 + 663)
    np.savetxt(os.path.join(OUT, "nile_min.csv"), np.c_[years, nile], delimiter=",", header="year,minimum_level", comments="", fmt=["%d", "%.1f"])
    t2 = open(os.path.join(RAW, "data", "NhemiTemp.R"), encoding="latin-1").read()
    temp = numbers_in_c(t2)
    np.savetxt(os.path.join(OUT, "nhemi_temp.csv"), temp, delimiter=",", header="anomaly", comments="", fmt="%.2f")
    for name, x in (("nile_min", nile), ("nhemi_temp", temp)):
        prov["series"][name] = {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
                                "H_whittle_fgn": fgn_whittle(x), "H_aggregated_variance": aggvar(x), "H_RS": rs(x)}
    prov["series"]["nile_min"].update({"years": [622, 1284], "description": "yearly minimal water levels, Roda gauge near Cairo (Tousson 1925; Beran 1994 dataset no. 1)"})
    prov["series"]["nhemi_temp"].update({"description": "Northern hemisphere temperature series as shipped in longmemo (see man/NhemiTemp.Rd)"})
    with open(os.path.join(OUT, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=1)
    print(json.dumps(prov["series"], indent=1))


if __name__ == "__main__":
    main()
