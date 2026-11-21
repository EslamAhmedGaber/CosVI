"""Every number quoted in the manuscript, computed from the stored result files and written as LaTeX macros
(source/tables/numbers.tex).  No number in the text is typed by hand; re-run after any new result.
Macro names are letters only (LaTeX); each block names its source.  A result that does not exist yet is written as a
visible placeholder (bold ??) and listed at the end, so an incomplete build cannot pass unnoticed.
Revised 18 Nov 2026 (three bridge references, CosVI vs CosVI-32, read-out control, G1/G2
wording, N2 own-model gaps, X2, X3, X5, X8, the implemented solver's covariance error).  Run: python make_macros.py
"""
import os, sys, json, glob, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S

OUT = os.path.join(S.MAN, "source", "tables", "numbers.tex")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
M, MISSING = {}, []


def put(name, value, fmt=None, note=""):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    if fmt is not None:
        value = format(value, fmt)
    value = str(value)
    if value[:1] in "-+" and value[1:2].isdigit():
        value = r"\ensuremath{" + value + "}"         # a true minus or plus sign in text
    M[name] = (value, note)


def missing(name, note):
    MISSING.append(name)
    M[name] = (r"\textbf{??}", "MISSING: " + note)


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}"


def J(path):
    with open(os.path.join(S.RES, path)) as fh:
        return json.load(fh)


def exists(path):
    return os.path.exists(os.path.join(S.RES, path))


def sgn(x, nd=2):
    return f"{x:+.{nd}f}"


def ierr(s, ref):
    return 100 * np.abs(np.asarray(s) - np.asarray(ref)).sum() / np.asarray(ref).sum()


def num(x):
    """Thousands with a thin space in text."""
    return f"{x:,.0f}".replace(",", "{,}")


# ------------------------------------------------------------------ G1 noise fidelity (results/g1/g1_pilot.json)
g1 = J(os.path.join("g1", "g1_pilot.json"))
dims = g1["dims"]
c = lambda H, d: g1["cosine"][H][str(d)]["projection"]["rel_int_mse"]
t = lambda H, d: g1["ma_I_tuned"][H][str(d)]["rel_int_mse"]
p = lambda H, d: g1["ma_I_default"][H][str(d)]["rel_int_mse"]
ratios = [c(H, d) / g1["kl"][H][str(d)] for H in g1["cosine"] for d in dims]
put("GoneCosOptMin", min(ratios), ".3f", "cosine / numerical KL optimum, all H and sizes")
put("GoneCosOptMax", max(ratios), ".3f")
pub = [p(H, d) / c(H, d) for H in g1["cosine"] for d in dims]
put("GonePubOverCosMin", f"{min(pub):.2f}", note="published lift MSE / cosine MSE, over the sweep")
put("GonePubOverCosMax", num(max(pub)))
put("GonePubOverCosNine", num(p("0.90", 64) / c("0.90", 64)), note="published lift / cosine at H 0.9, size 64")
put("GoneTunedOverCosNineMin", num(min(t("0.90", d) / c("0.90", d) for d in dims)), note="tuned / cosine at H 0.9, min over sizes")
put("GoneCosOverTunedSix", num(c("0.60", 64) / t("0.60", 64)), note="cosine / tuned lift at H 0.6, size 64")
cross = [d for d in dims if t("0.75", d) < c("0.75", d)]
assert cross == [d for d in dims if d >= cross[0]], "tuned lift does not win monotonically at H 0.75"
put("GoneCrossSevenFive", str(cross[0]), note="smallest size from which the tuned lift beats cosine at H 0.75")
put("GoneCosWinsSevenFiveMax", str(max(d for d in dims if d < cross[0])))
put("GoneTunedGainSevenFiveMax", f"{max(c('0.75', d) / t('0.75', d) for d in cross):.0f}", note="max cosine/tuned at H 0.75 beyond the crossover")
put("GoneSizeMin", str(min(dims))); put("GoneSizeMax", str(max(dims)))
tim = g1["timing"]
sp_ma = [tim[H]["ma_typeI_exact_recursion"] / tim[H]["cosine_projection"] for H in tim]
sp_dh = [tim[H]["exact_fbm_davies_harte"] / tim[H]["cosine_projection"] for H in tim]
put("GonePathSpeedMin", min(sp_ma), ".0f"); put("GonePathSpeedMax", max(sp_ma), ".0f")
put("GoneDHSpeedMin", min(sp_dh), ".0f"); put("GoneDHSpeedMax", max(sp_dh), ".0f")

# ------------------------------------------------------------------ G2 linear exact (signed evidence errors), V1
g2 = J("g2/g2_linear_exact.json")
sig = {H: g2["results"][H]["cosine"]["projection_J16"]["B"] for H in g2["results"]}
put("GtwoEvidMeanMax", f"{max(abs(v['dlogp_mean']) for v in sig.values()):.3f}",
    note="largest |mean signed log-evidence difference| over H, cosine J=16, 20 data sets")
put("GtwoEvidSdSix", f"{sig['0.60']['dlogp_sd']:.3f}", note="sd across data sets at H 0.6")
cv = g2["results"]["0.75"]["bridge_posterior_var_curves"]
tt = np.array(cv["times"]); ex = np.array(cv["exact"]); win = slice(5, 35)
put("GtwoLiftVarOff", f"{100 * np.max(np.abs(np.array(cv['ma_default_K5']) / ex - 1)[win]):.0f}",
    note="published lift (K=5, standard scale) bridge posterior variance, max relative error at 0.3<=t<=1.75, H 0.75")
put("GtwoLiftVarOffCoded", f"{100 * np.max(np.abs(np.array(cv['ma_authors_as_coded_K5']) / ex - 1)[win]):.0f}")
put("GtwoWinLo", f"{tt[5]:.2f}"); put("GtwoWinHi", f"{tt[34]:.2f}")
v1 = [r for r in J("v1/v1_trained_vi_linear.json") if r["kind"] == "full"]
put("VoneGapMin", f"{min(r['gap_elbo_to_logpJ'] for r in v1):.3f}"); put("VoneGapMax", f"{max(r['gap_elbo_to_logpJ'] for r in v1):.3f}")

# ------------------------------------------------------------------ G0 their code; three references (curves); read-out control
cfg = [(H, th) for th in (1.0, 0.0) for H in (0.6, 0.7, 0.8, 0.9)]
e_std, e_auth, e_own, hours = [], [], [], []
for H, th in cfg:
    d = J(f"g0/bridge_H{H:.2f}_theta{th:.1f}_seed0_steps2000.json")
    e_std.append(ierr(d["std_empirical"], d["std_ref_true"])); e_auth.append(ierr(d["std_empirical"], d["std_ref_authors"]))
    e_own.append(ierr(d["std_empirical"], d["std_ref_ownmodel"])); hours.append(d["train_seconds"] / 3600)
put("GzeroErrMin", f"{min(e_std):.1f}", note="their code seed 0, vs the standard fOU started at zero, 8 configurations")
put("GzeroErrMax", f"{max(e_std):.1f}")
put("GzeroAuthMin", f"{min(e_auth):.1f}"); put("GzeroAuthMax", f"{max(e_auth):.1f}")
put("GzeroOwnMin", f"{min(e_own):.1f}"); put("GzeroOwnMax", f"{max(e_own):.1f}")
put("GzeroOwnBelowFive", str(sum(e < 5 for e in e_own)))
put("GzeroHoursMin", f"{min(hours):.1f}"); put("GzeroHoursMax", f"{max(hours):.1f}")
rc = J("g0/readout_control.json")
lo = min(v["readout_error_percentiles_2.5_50_97.5"]["256"][0] for v in rc["by_H"].values())
hi = max(v["readout_error_percentiles_2.5_50_97.5"]["256"][2] for v in rc["by_H"].values())
med = [v["readout_error_percentiles_2.5_50_97.5"]["256"][1] for v in rc["by_H"].values()]
put("ReadoutLo", f"{lo:.1f}", note="2.5th percentile, 256-path read-out of the exact own-lift posterior (theta 1, H 0.7/0.9)")
put("ReadoutHi", f"{hi:.1f}"); put("ReadoutMedMin", f"{min(med):.1f}"); put("ReadoutMedMax", f"{max(med):.1f}")
lo4 = min(v["readout_error_percentiles_2.5_50_97.5"]["4096"][0] for v in rc["by_H"].values())
hi4 = max(v["readout_error_percentiles_2.5_50_97.5"]["4096"][2] for v in rc["by_H"].values())
put("ReadoutBigLo", f"{lo4:.1f}"); put("ReadoutBigHi", f"{hi4:.1f}")
disc = [v["own_vs_standard_reference_discrepancy"] for v in rc["by_H"].values()]
put("RefDiscMin", f"{min(disc):.1f}", note="integrated SD discrepancy own-lift vs standard reference, theta 1, H 0.7 / 0.9")
put("RefDiscMax", f"{max(disc):.1f}")
put("RefDiscCheck", f"{max(v['recomputed_own_lift_max_abs_sd_diff'] for v in rc['by_H'].values()):.0e}".replace("e-0", r"\times10^{-").replace("e-", r"\times10^{-") + "}",
    note="independent recomputation of the own-lift posterior SD vs the stored curve (max abs)")
for H, nm in ((0.7, "Seven"), (0.9, "Nine")):
    for sd_, sn in ((0, "Zero"), (1, "One")):
        r = rc["runs"][f"{H}|{sd_}"]
        put(f"GzeroStd{nm}{sn}", f"{r['standard_fou']:.1f}"); put(f"GzeroAuth{nm}{sn}", f"{r['authors_curve']:.1f}")
        put(f"GzeroOwn{nm}{sn}", f"{r['own_lift']:.1f}")
d = J("g0/bridge_H0.90_theta1.0_seed0_steps2000.json")
tt = np.array(d["times"]); th_ = np.array(d["std_empirical"]); ex = np.array(d["std_ref_true"])
m = (tt >= 0.5) & (tt <= 1.9)
put("GzeroWorstOff", f"{100 * np.max(np.abs(th_[m] / ex[m] - 1)):.0f}", note="their posterior std off at H0.9 theta1 (0.5<=t<=1.9)")

# ------------------------------------------------------------------ N3 our bridge fits: CosVI, CosVI-32, exact grid (24 fits each)
def n3(rep):
    rs = [json.load(open(f)) for f in glob.glob(os.path.join(S.RES, "n3", f"n3_{rep}_H*_theta*_seed*.json"))]
    e = [100 * r["metrics"]["exact_true"]["std_relerr_integrated"] for r in rs]
    return e, [r["train_seconds"] for r in rs]
for rep, nm in (("twoscale", "Cosvi"), ("cosine", "CosJ"), ("grid", "Grid")):
    e, s = n3(rep)
    assert len(e) == 24, (rep, len(e))
    put(f"Nthree{nm}Min", f"{min(e):.3f}"); put(f"Nthree{nm}Max", f"{max(e):.3f}")
    put(f"Nthree{nm}Sec", f"{np.median(s):.0f}", note=f"median timed training loop, s, {rep}")
put("NthreeFits", "24")

# ------------------------------------------------------------------ G0-H their code learning H (one path, one seed per H)
for H, nm in ((0.6, "Six"), (0.75, "SevenFive"), (0.9, "Nine")):
    f = glob.glob(os.path.join(S.RES, "g0", f"learnH_H{H:.2f}_theta0_seed0.json"))[0]
    d = json.load(open(f))
    put(f"GzeroHFit{nm}", f"{d['H_fit']:.3f}"); put(f"GzeroHMle{nm}", f"{d['H_true_mle']:.3f}")
    put(f"GzeroHMleSe{nm}", f"{d['H_true_mle_se_curvature']:.3f}", note="curvature SE of the exact-law MLE")
    put(f"GzeroHMaMle{nm}", f"{d['H_ma_mle']:.3f}", note="MLE of H under the lift's own law (exact lift covariance)")
    put(f"GzeroHHours{nm}", f"{d['train_seconds'] / 3600:.1f}")

# ------------------------------------------------------------------ N1b trained double well (per-run files, seeds 0-2)
def n1b(rep, H, s, tag=""):
    f = os.path.join(S.RES, "n1b", f"n1b_{rep}_seed{s}_H{H:.2f}{tag}.json")
    return json.load(open(f)) if os.path.exists(f) else None


def oracle(d, key):
    return d[key][-1]


pairs, drift_red = {}, []
for H in (0.6, 0.75, 0.9):
    for r in ("markov", "grid", "markov_best", "markov_tuned", "cosine"):
        dd = [oracle(n1b("twoscale", H, s), "test_loglik_exact_grid_law_mean") - oracle(n1b(r, H, s), "test_loglik_exact_grid_law_mean")
              for s in (0, 1, 2) if n1b("twoscale", H, s) and n1b(r, H, s)]
        pairs[(r, H)] = (np.mean(dd), np.std(dd, ddof=1), len(dd))
    for s in (0, 1, 2):
        a = n1b("twoscale", H, s)
        best = min(oracle(x, "drift_rmse_occupation") for x in [n1b(r, H, s) for r in ("markov", "markov_tuned", "markov_best")] if x)
        drift_red.append(1 - oracle(a, "drift_rmse_occupation") / best)
for H, nm in ((0.6, "Six"), (0.75, "SevenFive"), (0.9, "Nine")):
    mu, sd, n = pairs[("markov", H)]
    put(f"NoneGainMa{nm}", sgn(mu)); put(f"NoneGainMaSd{nm}", f"{sd:.2f}")
    mu, sd, n = pairs[("grid", H)]
    put(f"NoneGainGrid{nm}", sgn(mu)); put(f"NoneGainGridSd{nm}", f"{sd:.2f}")
put("NoneDriftRedMin", pct(min(drift_red), 0)); put("NoneDriftRedMax", pct(max(drift_red), 0))
put("NoneDriftPairs", str(len(drift_red)))
gaps = []
for mode in ("oracle", "joint", "oracle_best", "joint_best"):
    for H in (0.6, 0.75, 0.9):
        for s in (0, 1, 2):
            f = os.path.join(S.RES, "n1c", f"n1c_{mode}_seed{s}_H{H:.2f}.json")
            if os.path.exists(f):
                gaps.append(oracle(n1b("twoscale", H, s), "test_loglik_exact_grid_law_mean") - json.load(open(f))["test_loglik_exact_grid_law_mean"])
put("NoneSysGapMin", f"{min(gaps):.1f}"); put("NoneSysGapMax", f"{max(gaps):.0f}")
red = []
for H in (0.6, 0.75, 0.9):
    a5, g5 = n1b("twoscale", H, 0), n1b("grid", H, 0)
    a10, g10 = n1b("twoscale", H, 0, "_steps10000"), n1b("grid", H, 0, "_steps10000")
    d5 = oracle(a5, "test_loglik_exact_grid_law_mean") - oracle(g5, "test_loglik_exact_grid_law_mean")
    d10 = oracle(a10, "test_loglik_exact_grid_law_mean") - oracle(g10, "test_loglik_exact_grid_law_mean")
    red.append(1 - d10 / d5)
put("BudgetRedMin", pct(min(red), 0)); put("BudgetRedMax", pct(max(red), 0))

# ------------------------------------------------------------------ X1 (ii) five seeds (seeds 0-2 N1b, 3-4 X1)
x1 = J("n1b/x1_summary.json")
p2 = x1.get("part_ii", {})
for r, rn in (("grid", "Grid"), ("markov", "Ma"), ("markov_best", "MaBest")):
    for H, nm in ((0.6, "Six"), (0.75, "SevenFive"), (0.9, "Nine")):
        v = p2.get(f"{r}|{H}")
        if v and v["n"] == 5:
            put(f"FiveSeed{rn}{nm}", sgn(v["delta_mean"])); put(f"FiveSeed{rn}SE{nm}", f"{v['se']:.2f}")
            put(f"FiveSeed{rn}Verdict{nm}", {"gain": "gain", "loss": "loss", "not distinguishable": "not distinguishable"}[v["verdict_n5_rule"]])
        else:
            missing(f"FiveSeed{rn}{nm}", f"X1(ii) {r} H={H} n<5"); missing(f"FiveSeed{rn}SE{nm}", "X1(ii)")
            missing(f"FiveSeed{rn}Verdict{nm}", "X1(ii)")
if p2 and all(v["n"] == 5 for v in p2.values()):
    put("FiveSeedGains", str(sum(v["verdict_n5_rule"] == "gain" for v in p2.values())))
    put("FiveSeedLosses", str(sum(v["verdict_n5_rule"] == "loss" for v in p2.values())))
    put("FiveSeedComparisons", str(len(p2)))
else:
    for k in ("FiveSeedGains", "FiveSeedLosses", "FiveSeedComparisons"):
        missing(k, "X1(ii) incomplete")

# ------------------------------------------------------------------ H estimates (N1b profile; X1 extended grid; their system)
st_ = J("summary_trained.json")
x1i = x1["part_i"]
put("HcosviSix", f"{st_['twoscale|0.6']['H_profile'][0]:.3f}"); put("HcosviSixSd", f"{st_['twoscale|0.6']['H_profile'][1]:.3f}")
put("HcosviSevenFive", f"{st_['twoscale|0.75']['H_profile'][0]:.3f}"); put("HcosviSevenFiveSd", f"{st_['twoscale|0.75']['H_profile'][1]:.3f}")
put("HcosviNine", f"{x1i['twoscale|0.9']['H_profile_mean']:.3f}"); put("HcosviNineSd", f"{x1i['twoscale|0.9']['H_profile_sd']:.3f}")
put("HgridNine", f"{x1i['grid|0.9']['H_profile_mean']:.3f}")
put("HmaPubNine", f"{x1i['markov|0.9']['H_profile_mean']:.3f}")
put("HportMin", f"{min(st_[f'competitor_joint_published|{h}']['H_profile'][0] for h in (0.6, 0.75, 0.9)):.2f}")
put("HportMax", f"{max(st_[f'competitor_joint_published|{h}']['H_profile'][0] for h in (0.6, 0.75, 0.9)):.2f}")
hv = [x1i[f"{r}|0.9"]["H_val_sd"] for r in ("twoscale", "cosine", "grid", "markov", "markov_tuned", "markov_best")]
put("HvalSdMin", f"{min(hv):.2f}"); put("HvalSdMax", f"{max(hv):.2f}")

# ------------------------------------------------------------------ N2: difference to the reference score, split into the own-model
# variational gap (own evidence - ELBO) and the model difference (own evidence - reference); model trained with CosVI
n2 = J("n2/n2_summary.json")["by_model"]["twoscale"]["by_H"]
vg = lambda H, r: n2[H]["reps"][r]["gap_oracle"][0] + n2[H]["reps"][r]["own_minus_ref_oracle"][0]
md = lambda H, r: n2[H]["reps"][r]["own_minus_ref_oracle"][0]
Hs2 = ("0.60", "0.75", "0.90")
exact_reps = ("twoscale", "grid")
put("NtwoVgCosviMin", f"{min(vg(H, 'twoscale') for H in Hs2):.1f}"); put("NtwoVgCosviMax", f"{max(vg(H, 'twoscale') for H in Hs2):.1f}")
put("NtwoVgGridMin", f"{min(vg(H, 'grid') for H in Hs2):.1f}"); put("NtwoVgGridMax", f"{max(vg(H, 'grid') for H in Hs2):.1f}")
put("NtwoVgPubMin", f"{min(vg(H, 'markov') for H in Hs2):.1f}"); put("NtwoVgPubMax", f"{max(vg(H, 'markov') for H in Hs2):.1f}")
put("NtwoMdPubMin", f"{min(md(H, 'markov') for H in Hs2):.1f}"); put("NtwoMdPubMax", f"{max(md(H, 'markov') for H in Hs2):.1f}")
put("NtwoMdExactMin", f"{min(md(H, r) for H in Hs2 for r in exact_reps):.2f}"); put("NtwoMdExactMax", f"{max(md(H, r) for H in Hs2 for r in exact_reps):.2f}")
put("NtwoSeeds", str(len(n2["0.75"]["seeds"])))

# ------------------------------------------------------------------ Nile port settings against CosVI (Appendix C): initial and final sigma in years
from scipy.special import gamma as _gamma
_cmvn = lambda h: 1.0 / (_gamma(2 * h + 1) * np.sin(np.pi * h))
put("PortSigmaInitYears", f"{0.3 * 20.0 ** (-0.7) * np.sqrt(_cmvn(0.7)):.3f}",
    note="0.3 in the port's unit (50 years -> 2.5) at its initial H = 0.7 (n4c_competitor_nile.py), converted to years")
_pj = [json.load(open(f))["sigma_years_standard_scale"] for f in sorted(glob.glob(os.path.join(S.RES, "n4", "n4c_joint_seed*.json")))]
put("PortSigmaFinalMin", f"{min(_pj):.3f}"); put("PortSigmaFinalMax", f"{max(_pj):.3f}")
_cs = []
for f in sorted(glob.glob(os.path.join(S.RES, "n4", "n4v2_twoscale_seed*_s0.1.json"))):
    _d = json.load(open(f)); _gi = int(np.argmin(np.abs(np.array([0.5, 0.6, 0.7, 0.8, 0.9]) - _d["H_profile"])))
    _cs.append(_d["sigma"][_gi])
put("CosviSigmaNileMin", f"{min(_cs):.2f}"); put("CosviSigmaNileMax", f"{max(_cs):.2f}")

# ------------------------------------------------------------------ N7(b): CosVI's discrete posterior at the N7 configuration
n7b = J("n7/n7_cosvi_accuracy.json")
assert n7b["all_qualify"], "N7(b): a case misses the 2 % criterion"
put("NsevenCosviErrMax", f"{100 * n7b['max_err']:.4f}"); put("NsevenCosviCases", str(len(n7b["rows"])))

# ------------------------------------------------------------------ X10 (a): N2 under a proposal that keeps the x0-noise coupling
# (post hoc, CPU; results/x10/x10_summary.json) and the candidate contrasts of the six N2 cases (stored in results/x10)
x10p = os.path.join(S.RES, "x10", "x10_summary.json")
x10 = json.load(open(x10p)) if os.path.exists(x10p) else None
if x10 and x10["a"]["verdict"] in ("holds", "qualified"):
    ca = x10["a"]["by_case"]
    byH = lambda f: {H: np.mean([f(ca[f"{H}|{s}"]) for s in (1, 2)]) for H in ("0.6", "0.75", "0.9")}
    sh = [v["shift_of_reference"] for v in ca.values()]
    put("XtenRefShiftMin", f"{min(sh):.2f}"); put("XtenRefShiftMax", f"{max(sh):.2f}")
    pub = byH(lambda v: v["md_joint"]["markov"])
    put("XtenMdPubMin", sgn(min(pub.values()), 1)); put("XtenMdPubMax", sgn(max(pub.values()), 1))
    put("XtenMdGridMaxAbs", f"{max(abs(v['md_joint']['grid']) for v in ca.values()):.2f}",
        note="largest |ev_J(grid) - ev_J(CosVI)| over the six cases")
    arch = [v["md_archived"][m] for v in ca.values() for m in ("twoscale", "grid")]
    put("XtenMdExactArchMin", sgn(min(arch), 2)); put("XtenMdExactArchMax", sgn(max(arch), 2))
    put("XtenMdExactArchAbs", f"{max(abs(x) for x in arch):.2f}",
        note="largest |product-proposal score - reference| of the two exact representations, per case")
    for m, nm in (("twoscale", "Cosvi"), ("grid", "Grid"), ("markov", "Pub")):
        g = byH(lambda v, m=m: v["gap_joint"][m])
        put(f"XtenGap{nm}Min", f"{min(g.values()):.1f}"); put(f"XtenGap{nm}Max", f"{max(g.values()):.1f}")
    put("XtenMinEss", f"{min(min(v['min_ess'].values()) for v in ca.values()):.0f}")
    put("XtenCasesHold", f"{x10['a']['cases_holding']}"); put("XtenCases", f"{x10['a']['cases_complete']}")
else:
    for k in ("XtenRefShiftMin", "XtenRefShiftMax", "XtenMdPubMin", "XtenMdPubMax", "XtenMdGridMaxAbs", "XtenMdExactArchMin",
              "XtenMdExactArchMax", "XtenGapCosviMin", "XtenGapCosviMax", "XtenGapGridMin", "XtenGapGridMax", "XtenGapPubMin",
              "XtenGapPubMax", "XtenMinEss", "XtenCasesHold", "XtenCases"):
        missing(k, "X10(a) pending")
if x10 and x10["b"]["robust"] is not None:
    b10 = x10["b"]
    put("XtenBChanged", str(len(b10["changed"]))); put("XtenBRows", str(sum(len(v["rows"]) for v in b10["by_H"].values())))
    shf = b10["score_shift_joint_minus_archived"]
    put("XtenShiftMin", f"{min(v['mean_shift'] for v in shf.values()):.1f}")
    put("XtenShiftMax", f"{max(v['mean_shift'] for v in shf.values()):.1f}")
    put("XtenShiftCosviMin", f"{min(v['mean_shift'] for k, v in shf.items() if k.startswith('twoscale|')):.2f}")
    put("XtenShiftCosviMax", f"{max(v['mean_shift'] for k, v in shf.items() if k.startswith('twoscale|')):.2f}")
    put("XtenShiftOtherMin", f"{min(v['mean_shift'] for k, v in shf.items() if not k.startswith('twoscale|')):.2f}")
    put("XtenShiftOtherMax", f"{max(v['mean_shift'] for k, v in shf.items() if not k.startswith('twoscale|')):.2f}")
    for H, nm in (("0.6", "Six"), ("0.75", "SevenFive"), ("0.9", "Nine")):
        put(f"XtenBGrid{nm}", sgn(b10["by_H"][H]["rows"]["grid"]["joint"]["mean"]))
        put(f"XtenBGridArch{nm}", sgn(b10["by_H"][H]["rows"]["grid"]["archived"]["mean"]))
    put("XtenBMinEss", f"{min(v['min_ess'] for v in b10['by_H'].values()):.0f}")
    put("XtenBFlagged", str(sum(v["flagged_windows"] for v in b10["by_H"].values())))
else:
    for k in ("XtenBChanged", "XtenBRows", "XtenShiftMin", "XtenShiftMax", "XtenShiftCosviMin", "XtenShiftCosviMax",
              "XtenShiftOtherMin", "XtenShiftOtherMax", "XtenBGridSix", "XtenBGridSevenFive", "XtenBGridNine",
              "XtenBGridArchSix", "XtenBGridArchSevenFive", "XtenBGridArchNine", "XtenBMinEss", "XtenBFlagged"):
        missing(k, "X10(b) pending")
if x10 and x10.get("c") and x10["c"]["robust"] is not None:
    c10 = x10["c"]["by_series"]
    shc = [v for sr in c10.values() for v in sr["shift_joint_minus_archived"].values()]
    put("XtenCShiftMin", sgn(min(shc), 2)); put("XtenCShiftMax", sgn(max(shc), 2))
    put("XtenCChanged", str(len(x10["c"]["changed"])))
    put("XtenCRows", str(sum(len(v["rows"]) for v in c10.values())))
    put("XtenCMinEssTemp", f"{c10['temp']['min_ess']:.0f}"); put("XtenCMinEssNile", f"{c10['nile']['min_ess']:.0f}")
else:
    for k in ("XtenCShiftMin", "XtenCShiftMax", "XtenCChanged", "XtenCRows", "XtenCMinEssTemp", "XtenCMinEssNile"):
        missing(k, "X10(c) pending")
rvp = os.path.join(S.RES, "x10", "n2_joint_candidates.json")   # joint-minus-product candidate contrasts, six N2 cases
if os.path.exists(rvp):
    rv = [c for c in json.load(open(rvp)) if "mean_joint_minus_product" in c]
    put("RevCandMin", f"{min(c['mean_joint_minus_product'] for c in rv):.2f}")
    put("RevCandMax", f"{max(c['mean_joint_minus_product'] for c in rv):.2f}")
    put("RevCandWindows", str(sum(c["positive_windows"] for c in rv))); put("RevCandWindowsTotal", str(sum(c["windows"] for c in rv)))

# ------------------------------------------------------------------ N4 Nile (summary_n4_s*.json) and N4c; X5 three seeds per s
s4 = J("n4/summary_n4_s0.1.json")
put("NfourCosviLL", f"{s4['twoscale']['heldout_at_profile'][0]:.1f}", note="CosVI held-out per window at profile H, s 0.1")
put("NfourCosviSd", f"{s4['twoscale']['heldout_at_profile'][1]:.2f}")
put("NfourCosviH", f"{s4['twoscale']['H_profile'][0]:.2f}"); put("NfourCosviHSd", f"{s4['twoscale']['H_profile'][1]:.2f}")
prov = J("n4/data/provenance.json")["series"]
hc = [prov["nile_min"][k] for k in ("H_whittle_fgn", "H_aggregated_variance", "H_RS")]
put("NfourClassMin", f"{min(hc):.2f}"); put("NfourClassMax", f"{max(hc):.2f}")
cj = [J(f"n4/n4c_joint_seed{s}.json") for s in (0, 1, 2)]
put("NfourTheirLL", f"{np.mean([d['test_loglik_exact_law_twoscale_coords_mean'] for d in cj]):.0f}")
put("NfourTheirH", f"{np.mean([d['H_fit'] for d in cj]):.2f}")
co = [J(f"n4/n4c_oracle_seed{s}.json") for s in (0, 1, 2)]
put("NfourTheirOracleLL", f"{np.mean([d['test_loglik_exact_law_twoscale_coords_mean'] for d in co]):.0f}")
for rep, nm in (("cosine64", "CosJ"), ("grid", "Grid"), ("markov", "Ma"), ("markov_tuned", "MaTuned")):
    put(f"Nfour{nm}LL", f"{s4[rep]['heldout_at_profile'][0]:.1f}"); put(f"Nfour{nm}H", f"{s4[rep]['H_profile'][0]:.2f}")
    put(f"Nfour{nm}Sd", f"{s4[rep]['heldout_at_profile'][1]:.2f}")


def x5(s):
    out = []
    for seed in (0, 1, 2):
        sc = {}
        for rep in ("grid", "twoscale"):
            run = J(f"n4/n4v2_{rep}_seed{seed}_s{s}.json")
            gi = int(np.argmin(np.abs(np.array([0.5, 0.6, 0.7, 0.8, 0.9]) - run["H_profile"])))
            sc[rep] = run["test_loglik_exact_law_twoscale_coords_mean"][gi]
        out.append(sc["grid"] - sc["twoscale"])
    return np.array(out)
for s, nm in (("0.05", "Zfive"), ("0.2", "Ztwo")):
    dd = x5(s)
    put(f"Xfive{nm}Mean", sgn(dd.mean())); put(f"Xfive{nm}Sd", f"{dd.std(ddof=1):.2f}")
    put(f"Xfive{nm}Seeds", ", ".join(f"{v:+.2f}" for v in dd).replace("+", r"$+$").replace("-", r"$-$"))
    sx = J(f"n4/summary_n4_s{s}.json")
    put(f"Xfive{nm}Cosvi", f"{sx['twoscale']['heldout_at_profile'][0]:.1f}"); put(f"Xfive{nm}Grid", f"{sx['grid']['heldout_at_profile'][0]:.1f}")
    put(f"Xfive{nm}CosviSd", f"{sx['twoscale']['heldout_at_profile'][1]:.2f}"); put(f"Xfive{nm}GridSd", f"{sx['grid']['heldout_at_profile'][1]:.2f}")
d0 = x5("0.2")[0]
put("XfiveSeedZeroReversal", sgn(d0), note="grid - CosVI at s 0.2, seed 0 (the reversal reported on 16 Nov)")

# ------------------------------------------------------------------ X3 temperature (n4t/x3_summary.json)
x3 = J("n4t/x3_summary.json")
for rep, nm in (("twoscale", "Cosvi"), ("cosine64", "CosJ"), ("grid", "Grid"), ("markov", "Ma"), ("markov_tuned", "MaTuned"),
                ("competitor_joint", "TheirJoint"), ("competitor_oracle", "TheirOracle")):
    v = x3["by"][rep]
    put(f"Xthree{nm}LL", f"{v['ll'][0]:.2f}" if abs(v["ll"][0]) < 100 else f"{v['ll'][0]:.0f}")
    put(f"Xthree{nm}Sd", f"{v['ll'][1]:.2f}"); put(f"Xthree{nm}H", f"{v['H'][0]:.2f}"); put(f"Xthree{nm}HSd", f"{v['H'][1]:.2f}")
pv = x3["paired_vs_cosvi"]
put("XthreeCosJGain", sgn(-pv["cosine64"]["mean"])); put("XthreeCosJGainSd", f"{pv['cosine64']['sd']:.2f}")
put("XthreeGridGap", sgn(pv["grid"]["mean"])); put("XthreeGridGapSd", f"{pv['grid']['sd']:.2f}")
put("XthreeMaGap", sgn(pv["markov"]["mean"])); put("XthreeMaGapSd", f"{pv['markov']['sd']:.2f}")
hcT = list(x3["classical_H"].values())
put("XthreeClassMin", f"{min(hcT):.2f}"); put("XthreeClassMax", f"{max(hcT):.2f}")

# ------------------------------------------------------------------ X2 two-dimensional Duffing (n8/x2_summary.json)
x2 = J("n8/x2_summary.json")
put("XtwoDriftWins", str(x2["gate"]["drift_wins_vs_lifts"])); put("XtwoDriftReq", str(x2["gate"]["drift_wins_required"]))
put("XtwoDriftOf", str(x2["gate"]["of"]))
put("XtwoNoneAhead", "no" if x2["gate"]["no_representation_ahead_of_cosvi"] else "a")
r9 = x2["by_H"]["0.9"]["reps"]
put("XtwoNineDriftCosvi", f"{r9['twoscale']['drift_rmse'][0]:.2f}"); put("XtwoNineDriftMa", f"{r9['markov']['drift_rmse'][0]:.2f}")
put("XtwoNineDriftGrid", f"{r9['grid']['drift_rmse'][0]:.2f}"); put("XtwoNineDriftMaBest", f"{r9['markov_best']['drift_rmse'][0]:.2f}")
wins = {H: sum(int(x2["by_H"][H]["paired_vs_cosvi"][r]["drift_wins"].split("/")[0]) for r in ("markov", "markov_best")) for H in x2["by_H"]}
put("XtwoWinsSix", str(wins["0.6"])); put("XtwoWinsSevenFive", str(wins["0.75"])); put("XtwoWinsNine", str(wins["0.9"]))

# ------------------------------------------------------------------ X4 calibration of the latent-path bands (calib/x4_summary.json)
x4 = J("calib/x4_summary.json")["by"]
put("CalCosviFiftySevenFive", f"{100 * x4['twoscale|0.75']['cov50'][0]:.0f}")
put("CalCosviNinetySevenFive", f"{100 * x4['twoscale|0.75']['cov90'][0]:.0f}")
put("CalCosviFiftySix", f"{100 * x4['twoscale|0.6']['cov50'][0]:.0f}"); put("CalCosviNinetySix", f"{100 * x4['twoscale|0.6']['cov90'][0]:.0f}")
put("CalGridFiftySevenFive", f"{100 * x4['grid|0.75']['cov50'][0]:.0f}"); put("CalGridNinetySevenFive", f"{100 * x4['grid|0.75']['cov90'][0]:.0f}")
put("CalPubFiftySevenFive", f"{100 * x4['markov|0.75']['cov50'][0]:.0f}"); put("CalPubNinetySevenFive", f"{100 * x4['markov|0.75']['cov90'][0]:.0f}")
lifts = [x4[f"{r}|{H}"]["cov90"][0] for r in ("markov_tuned", "markov_best") for H in ("0.6", "0.75")]
put("CalLiftNinetyMin", f"{100 * min(lifts):.0f}"); put("CalLiftNinetyMax", f"{100 * max(lifts):.0f}")
put("CalSeeds", str(len(x4["twoscale|0.75"]["seeds"]))); put("CalSeedsLight", str(len(x4["cosine|0.75"]["seeds"])))
for r, rn in (("twoscale", "Cosvi"), ("grid", "Grid"), ("markov", "Pub"), ("markov_best", "Best"), ("cosine", "CosJ")):
    put(f"Cal{rn}FiftyNine", f"{100 * x4[f'{r}|0.9']['cov50'][0]:.0f}"); put(f"Cal{rn}NinetyNine", f"{100 * x4[f'{r}|0.9']['cov90'][0]:.0f}")
crps_pub = [H for H in ("0.6", "0.75", "0.9") if x4[f"markov|{H}"]["vs_cosvi_crps"] == "CosVI better"]
put("CalCrpsPubBetterAt", ", ".join(f"$H={H}$" for H in crps_pub) if crps_pub else "no $H$",
    note="H values at which CosVI's CRPS is better than the published lift law's (paired, > sd)")

# ------------------------------------------------------------------ N5 cost per step; N7 projected fixed-budget ratio
n5 = J("n5/n5_scaling.json")
rows = n5["rows"]
ours = [r["step_s"] for r in rows if r["method"] == "ours" and r["sweep"] == "i"]
port = [r["step_s"] for r in rows if r["method"] == "port" and r["sweep"] == "i"]
put("NfiveOursMin", f"{min(ours):.3f}"); put("NfiveOursMax", f"{max(ours):.3f}")
put("NfivePortMin", f"{min(port):.3f}"); put("NfivePortMax", f"{max(port):.3f}")
put("NfiveRatio", f"{np.mean(port) / np.mean(ours):.1f}")
n7 = J("n7/n7_summary.json")
rat = [r["lift_cheapest_qualifying"][1] / r["cosvi_cheapest"][1] for v in n7["by_H"].values() for r in v["rows"]]
put("NsevenCostRatioMin", f"{min(rat):.1f}"); put("NsevenCostRatioMax", f"{max(rat):.1f}")

# ------------------------------------------------------------------ N6 ablation (n6_summary.json)
n6 = J("n6/n6_summary.json")["rows"]
fq = [n6[v]["by_H"][h]["delta_grid_mean"] for v in ("mf", "lr2", "lr4", "lr8") for h in n6[v]["by_H"]]
put("NsixFullQMin", f"{min(fq):.2f}"); put("NsixFullQMax", f"{max(fq):.2f}")
put("NsixHeunTime", f"{n6['heun']['by_H']['0.60']['wallclock_ratio']:.2f}")
put("NsixRawSix", f"{n6['raw']['by_H']['0.60']['delta_grid_mean']:.2f}"); put("NsixRawNine", f"{n6['raw']['by_H']['0.90']['delta_grid_mean']:.2f}")
put("NsixSfourNine", f"{n6['s4']['by_H']['0.90']['delta_grid_mean']:.2f}"); put("NsixSfourTime", f"{n6['s4']['by_H']['0.60']['wallclock_ratio']:.2f}")

# ------------------------------------------------------------------ the competitor's lift variance (check_n4c_units.py)
u = J("n4/n4c_unit_check.json")
r84 = u["lift_scale"]["0.84"]["var_lift_over_cH_t2H"]; t84 = u["lift_scale"]["0.84"]["t"]
put("LiftVarMinEightFour", f"{100 * min(r84):.0f}"); put("LiftVarMaxEightFour", f"{100 * max(r84):.0f}")
put("LiftVarTLo", f"{min(t84):.1f}"); put("LiftVarTHi", f"{max(t84):.1f}")

# ------------------------------------------------------------------ X8 coordinates versus solver (x8/x8_summary.json)
x8 = J("x8/x8_summary.json") if exists("x8/x8_summary.json") else {}
a = x8.get("a")
if a:
    ce = a["cov_relerr_grid_values"]
    fmt_e = lambda v: (f"{v:.1e}".replace("e-0", r"\times10^{-").replace("e-", r"\times10^{-") + "}")
    put("CovErrSix", fmt_e(ce["0.6"]["twoscale"]), note="implemented CosVI, rel. Frobenius error of grid-value covariance, T5 N80 J16")
    put("CovErrSevenFive", fmt_e(ce["0.75"]["twoscale"])); put("CovErrNine", fmt_e(ce["0.9"]["twoscale"]))
    put("CovErrPlMax", fmt_e(max(ce[H]["twoscale_pl"] for H in ce)))
    put("XeightAgree", fmt_e(a["max_abs_diff_pl_vs_grid"]), note="max |CosVI-PL - grid| in evidence and optimal ELBO (nats)")
    sm = a["max_abs_diff_smooth_vs_grid"]
    put("XeightSmoothEvidMax", f"{max(v['logev'] for v in sm.values()):.3f}")
    put("XeightSmoothEvidMin", f"{min(v['logev'] for v in sm.values()):.4f}")
    gp = a["product_family_gap_mean"]
    put("XeightGapMin", f"{min(gp.values()):.1f}"); put("XeightGapMax", f"{max(gp.values()):.1f}")
else:
    for k in ("CovErrSix", "CovErrSevenFive", "CovErrNine", "CovErrPlMax", "XeightAgree", "XeightSmoothEvidMax",
              "XeightSmoothEvidMin", "XeightGapMin", "XeightGapMax"):
        missing(k, "X8(a)")
b = x8.get("b", {}).get("by_H", {})
nb = sum(len(r) for r in b.values())
if nb and all(len(b.get(H, {})) >= 5 for H in ("0.6", "0.75", "0.9")):
    put("XeightRobust", "robust" if x8["b"]["robust_to_evaluation_map"] else "not robust")
    put("XeightChanged", str(len(x8["b"]["changed_rows"])))
    put("XeightRows", str(nb))
    for H, nm in (("0.6", "Six"), ("0.75", "SevenFive"), ("0.9", "Nine")):
        g = b[H]["grid"]
        put(f"XeightBGridSmooth{nm}", sgn(g["smooth"]["mean"])); put(f"XeightBGridPl{nm}", sgn(g["pl"]["mean"]))
        put(f"XeightBGridN{nm}", str(len(g["seeds"])))
else:
    for k in ("XeightRobust", "XeightChanged", "XeightRows") + tuple(f"XeightBGrid{e}{nm}" for e in ("Smooth", "Pl", "N") for nm in ("Six", "SevenFive", "Nine")):
        missing(k, "X8(b)")
cc = x8.get("c", {}).get("by_H", {})
if len(cc) == 3 and all(len(cc[H]["seeds"]) == 5 for H in cc):
    for H, nm in (("0.6", "Six"), ("0.75", "SevenFive"), ("0.9", "Nine")):
        for con, cn in (("coordinate", "Coord"), ("solver", "Solver"), ("total", "Total")):
            v = cc[H]["contrasts"][con]
            put(f"Xeight{cn}{nm}", sgn(v["pl"]["mean"])); put(f"Xeight{cn}SE{nm}", f"{v['pl']['se']:.2f}")
            put(f"Xeight{cn}Smooth{nm}", sgn(v["smooth"]["mean"]))
            put(f"Xeight{cn}Drift{nm}", sgn(v["drift"]["mean"], 3))
        put(f"XeightDecision{nm}", cc[H]["decision"])
    dec = x8["c"]["overall"]["decision"]
    put("XeightDecision", dec)
    put("XeightTrainSec", f"{np.mean([cc[H]['train_seconds'] for H in cc]) / 60:.0f}")
    # descriptive, not a criterion: paired training-bound gap CosVI-PL - grid (same optimum by Proposition 3.3)
    eg, pos = {}, 0
    for H, nm in (("0.60", "Six"), ("0.75", "SevenFive"), ("0.90", "Nine")):
        v = []
        for s in range(5):
            r = json.load(open(os.path.join(S.RES, "x8", f"x8c_seed{s}_H{H}.json")))
            e = dict(zip(r["arms"], r["elbo_train_own_model"]))
            v.append(e["twoscale_pl"] - e["grid"])
        eg[nm] = (np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v))); pos += sum(x > 0 for x in v)
        put(f"XeightElboGap{nm}", sgn(eg[nm][0])); put(f"XeightElboGapSE{nm}", f"{eg[nm][1]:.2f}")
    put("XeightElboGapMin", f"{min(m for m, _ in eg.values()):.1f}"); put("XeightElboGapMax", f"{max(m for m, _ in eg.values()):.1f}")
    put("XeightElboPositive", str(pos))
    # training path (descriptive): paired training objective CosVI-PL - grid along training, 15 runs
    path = {}
    for H in ("0.60", "0.75", "0.90"):
        for s in range(5):
            for st, _c, pl, gr in json.load(open(os.path.join(S.RES, "x8", f"x8c_seed{s}_H{H}.json")))["history"]:
                path.setdefault(st, []).append(pl - gr)
    early = 200
    late = min(st for st in path if st > early and all(min(path[u]) > 0 for u in path if u >= st))
    put("XeightPathEarlyStep", str(early)); put("XeightPathEarlyGrid", str(sum(v < 0 for v in path[early])))
    put("XeightPathLateStep", str(late)); put("XeightPathLateRuns", str(len(path[late])))
    peak = max(path, key=lambda st: np.median(path[st]))
    put("XeightPathPeakStep", str(peak)); put("XeightPathPeakMedian", f"{np.median(path[peak]):.0f}")
    lastst = max(path); put("XeightPathLastStep", str(lastst)); put("XeightPathLastMedian", f"{np.median(path[lastst]):.1f}")
    # sensitivity concentration in the linear analogue (x8d, descriptive)
    sd = J("x8/x8d_sensitivity_share.json")["by_H"]
    put("SensCosSevenFive", f"{100 * sd['0.75']['share_cosine_block']:.1f}"); put("SensGridSevenFive", f"{100 * sd['0.75']['share_best_J_grid']:.1f}")
    put("SensCosMin", f"{100 * min(v['share_cosine_block'] for v in sd.values()):.1f}")
    put("SensGridMax", f"{100 * max(v['share_best_J_grid'] for v in sd.values()):.1f}")
    put("SensNinetyCosMin", str(min(v["n90_pl"] for v in sd.values()))); put("SensNinetyCosMax", str(max(v["n90_pl"] for v in sd.values())))
    put("SensNinetyGridMin", str(min(v["n90_grid"] for v in sd.values()))); put("SensNinetyGridMax", str(max(v["n90_grid"] for v in sd.values())))
    cm = [cc[H]["contrasts"]["coordinate"]["pl"]["mean"] for H in cc]
    put("XeightCoordAbsMin", f"{min(cm):.2f}"); put("XeightCoordAbsMax", f"{max(cm):.2f}")
    coords = [cc[H]["contrasts"]["coordinate"]["pl"]["per_seed"] for H in cc]
    put("XeightCoordPositive", str(sum(x > 0 for c in coords for x in c))); put("XeightCoordRuns", str(sum(len(c) for c in coords)))
    # the sentences are written for each possible outcome of the rule fixed before the runs (X8)
    ABS = {"coordinates": "a controlled comparison with a common solver attributes its advantage over the exact grid, which has the same law on the grid, to the coordinates",
           "solver": "a controlled comparison with a common solver attributes its advantage over the exact grid, which has the same law on the grid, to the smoother within-step noise of its solver rather than to the coordinates",
           "both": "a controlled comparison with a common solver attributes its advantage over the exact grid, which has the same law on the grid, to both the coordinates and the smoother within-step noise of its solver",
           "neither": "a controlled comparison with a common solver does not separate a distinguishable coordinate or solver effect in its advantage over the exact grid"}
    put("XeightAbstract", ABS[dec])
    put("XeightIntro", "A controlled comparison with a common solver " + ABS[dec].split("common solver ", 1)[1] + r" (Section~\ref{sec:solver}).")
else:
    for H, nm in (("0.6", "Six"), ("0.75", "SevenFive"), ("0.9", "Nine")):
        for cn in ("Coord", "Solver", "Total"):
            for suf in ("", "SE", "Smooth", "Drift"):
                missing(f"Xeight{cn}{suf}{nm}", "X8(c)")
        missing(f"XeightDecision{nm}", "X8(c)")
    missing("XeightDecision", "X8(c)"); missing("XeightTrainSec", "X8(c)")
    missing("XeightAbstract", "X8(c)"); missing("XeightIntro", "X8(c)")

# ------------------------------------------------------------------ X9 evaluator check on the real series (x9/x9_summary.json)
x9 = J("x9/x9_summary.json") if exists("x9/x9_summary.json") else {}
bs = x9.get("by_series", {})
if len(bs) == 2 and all(len(v["seeds"]) == 3 for v in bs.values()):
    nm = {"nile": "Nile", "temp": "temperature"}
    if x9["robust_all"]:
        txt = (r"Scored instead with the PL evaluator, which uses the grid's solver, every paired classification and the "
               r"leading representation are unchanged on both series (\ref{app:results}).")
    else:
        ch = []
        for s, v in bs.items():
            rows = [r for r, p in v["paired_vs_cosvi"].items() if not p["same_reading"]]
            lead = v["evaluators"]["smooth"]["leading"] != v["evaluators"]["pl"]["leading"]
            if rows or lead:
                ch.append(f"{nm[s]} ({len(rows)} paired classification{'s' if len(rows) != 1 else ''}"
                          + (", and the leading representation" if lead else "") + ")")
        txt = (r"Scored instead with the PL evaluator, which uses the grid's solver, the readings change on the "
               + " and ".join(ch) + r"; \ref{app:results} gives both evaluators.")
    put("XnineSentence", txt)
    for s, v in bs.items():
        e = v["evaluators"]
        put(f"XnineMinEss{nm[s].capitalize()[:4]}", f"{min(e['smooth']['min_ess'], e['pl']['min_ess']):.0f}")
    pm = bs["nile"]["paired_vs_cosvi"]["markov"]
    put("XnineNileMaSmooth", f"{pm['smooth']['mean']:.1f}"); put("XnineNileMaPl", f"{pm['pl']['mean']:.1f}")
    # abstract clause on the real data, by the X9 rule fixed before the runs (fixed 18 Nov 2026 15:20): the Nile claim is stated
    # for both evaluators only if the Nile reading is unchanged; otherwise it names the evaluator under which it holds
    nile = bs["nile"]
    nile_same = nile["robust_to_evaluation_map"]
    lead_pl = nile["evaluators"]["pl"]["leading"]
    lead_pl = "twoscale" if lead_pl in ("twoscale", "CosVI") else lead_pl      # the summary stores display names
    if nile_same and lead_pl == "twoscale":
        put("XnineAbstract", r"On the Nile minima \CosVI{} fits best under both evaluators.")
    elif lead_pl == "twoscale":
        put("XnineAbstract", r"On the Nile minima \CosVI{} fits best, although some paired readings change with the evaluator.")
    else:
        put("XnineAbstract", r"On the Nile minima \CosVI{} fits best under its own solver but not under the grid's.")
else:
    missing("XnineSentence", "X9 incomplete"); missing("XnineAbstract", "X9 incomplete")

# ------------------------------------------------------------------ 19 Nov 2026: mechanism, real-data score stability, timings
# (i) linear analogue: the two coordinate systems give the observations the same covariance, so the posterior precision
#     of the noise coordinates has the same spectrum (x8d_sensitivity_share.py)
xd = J(os.path.join("x8", "x8d_sensitivity_share.json"))["by_H"]
for H, nmH in (("0.6", "Six"), ("0.75", "SevenFive"), ("0.9", "Nine")):
    r = xd[H]
    assert abs(r["precision_condition_pl"] / r["precision_condition_grid"] - 1) < 1e-9, "condition numbers differ"
    put(f"SensCond{nmH}", f"{r['precision_condition_pl']:.2f}")
put("SensNinetyCosSevenFive", str(xd["0.75"]["n90_pl"])); put("SensNinetyGridSevenFive", str(xd["0.75"]["n90_grid"]))
# (ii) real series: five readouts of the same models, window influence (x9_window_audit.py)
au = J(os.path.join("x9", "x9_window_audit.json"))["series"]
spread = max(max(c["gap_seed_mean"].values()) - min(c["gap_seed_mean"].values())
             for s in au.values() for c in s["by_comparison"].values())
repl = max(abs(x) for s in au.values() for c in s["by_comparison"].values() for x in c["replicate_difference_A_minus_B"])
put("AuditReadoutSpread", f"{np.ceil(spread * 10) / 10:.1f}", note="largest spread of a seed-mean paired gap over 5 readouts")
put("AuditReplicateMax", f"{np.ceil(repl * 10) / 10:.1f}", note="largest per-seed difference of two primary-evaluator runs")
assert all(c["stable_sign"] for c in au["nile"]["by_comparison"].values()), "a Nile comparison is not stable"
tc = au["temp"]["by_comparison"]
assert not tc["cosine64"]["stable_sign"] and not tc["markov"]["stable_sign"], "temperature stability changed: revise the text"
assert tc["grid"]["stable_sign"] and tc["markov_tuned"]["stable_sign"], "temperature stability changed: revise the text"
wc, wm = tc["cosine64"]["window_gap_seed_mean"], tc["markov"]["window_gap_seed_mean"]
nwin = len(wc["B"])
put("AuditEarlyCosJ", f"{abs(np.mean([v[0] for v in wc.values()])):.0f}")
put("AuditEarlyMa", f"{abs(np.mean([v[0] for v in wm.values()])):.0f}")
put("AuditLateMa", f"{np.mean([v[-1] for v in wm.values()]):.0f}")
assert all(v[0] < 0 for v in wc.values()) and all(v[0] < 0 for v in wm.values()) and all(v[-1] > 0 for v in wm.values())
woc = [float(np.mean(v[1:])) for v in wc.values()]; wol = [float(np.mean(v[:-1])) for v in wm.values()]
assert min(woc) > 0 and max(wol) < 0, "single-window reversal changed: revise the text"
put("AuditWithoutEarlyMin", f"{min(woc):.1f}"); put("AuditWithoutEarlyMax", f"{max(woc):.1f}")
put("AuditWithoutLateMin", f"{abs(max(wol)):.1f}"); put("AuditWithoutLateMax", f"{abs(min(wol)):.1f}")
TEST_T = list(range(3, 32, 4))                         # temperature test windows k (n4t_temperature.TEST), 51 months each
yr = lambda m: 1854 + m // 12
put("AuditEarlyDates", f"{yr(51 * TEST_T[0])}--{yr(51 * TEST_T[0] + 50)}")
put("AuditLateDates", f"{yr(51 * TEST_T[-1])}--{yr(51 * TEST_T[-1] + 50)}")
assert nwin == len(TEST_T)
ess_low, ess_tot, wmax = {}, {}, 0.0
for s in ("nile", "temp"):
    lo = tot = 0
    for k in range(3):
        e = J(os.path.join("x9", f"x9_{s}_seed{k}.json"))["evaluators"]["smooth"]
        a = np.array(e["ess"]); lo += int((a < 10).sum()); tot += a.size
        if s == "temp":
            wmax = max(wmax, float(np.max(e["max_weight"])))
    ess_low[s], ess_tot[s] = lo, tot
put("AuditEssLowNile", str(ess_low["nile"])); put("AuditEssTotNile", str(ess_tot["nile"]))
put("AuditEssLowTemp", str(ess_low["temp"])); put("AuditEssTotTemp", str(ess_tot["temp"]))
put("AuditWmaxTemp", f"{100 * wmax:.0f}")
# the X9 smooth column is a fresh refit of the Table 5 evaluator: largest change of a paired seed mean
diffs = []
for s in ("nile", "temp"):
    new, old = [], []
    for k in range(3):
        d = J(os.path.join("x9", f"x9_{s}_seed{k}.json"))
        new.append(d["evaluators"]["smooth"]["mean"]); old.append([m["stored_smooth_ll"] for m in d["models"]])
    new, old = np.array(new), np.array(old)
    diffs += list(np.abs((new[:, :1] - new[:, 1:]).mean(0) - (old[:, :1] - old[:, 1:]).mean(0)))
put("XnineRefitMax", f"{np.ceil(max(diffs) * 10) / 10:.1f}")
# (iii) end-to-end time of one profile run (all candidate H trained side by side in one GPU job) and its held-out refits
def minutes(pattern, key):
    v = [json.load(open(f))[key] for f in glob.glob(os.path.join(S.RES, pattern))]
    assert v, pattern
    return f"{np.median(v) / 60:.0f}"
put("TimeDwTrain", minutes(os.path.join("n1b", "n1b_twoscale_seed[0-2]_H0.??.json"), "train_seconds"))
put("TimeDwEval", minutes(os.path.join("n1b", "n1b_twoscale_seed[0-2]_H0.??.json"), "eval_seconds"))
put("TimeNileTrain", minutes(os.path.join("n4", "n4v2_twoscale_seed*_s0.1.json"), "train_seconds"))
put("TimeNileEval", minutes(os.path.join("n4", "n4v2_twoscale_seed*_s0.1.json"), "eval_seconds"))
put("TimeTempTrain", minutes(os.path.join("n4t", "n4t_twoscale_seed*_s0.1.json"), "train_seconds"))
put("TimeTempEval", minutes(os.path.join("n4t", "n4t_twoscale_seed*_s0.1.json"), "eval_seconds"))

# ------------------------------------------------------------------ X11: fast_hfc, simpson_exact, n7_clean, x11a, x11b, x11c
def _fmt_sci(v):
    s = f"{v:.1e}"
    mant, exp = s.split("e")
    return rf"{mant}\times 10^{{{int(exp)}}}"

if exists("x11/fast_hfc_benchmark.json"):
    fh = J("x11/fast_hfc_benchmark.json")
    ac = fh["accuracy"]
    put("FastHfcErrN", _fmt_sci(ac["worst_N"]))
    put("FastHfcErrM", _fmt_sci(ac["worst_M"]))
    put("FastHfcErrQ", _fmt_sci(ac["worst_Q"]))
    put("FastHfcErrDq", _fmt_sci(ac["worst_dQ"]))
    put("FastHfcErrG", _fmt_sci(ac["worst_G"]))
    brows = fh["benchmark"]["rows"]
    _by = {(r["kind"], r["N"], r["J"], r["H"]): r for r in brows}
    put("FastHfcSpeedTwoscaleSmall", f"{_by[('twoscale', 80, 16, 0.75)]['speedup']:.1f}")
    put("FastHfcSpeedTwoscaleMed", f"{_by[('twoscale', 200, 16, 0.75)]['speedup']:.1f}")
    put("FastHfcSpeedCosineMed", f"{_by[('cosine', 200, 64, 0.75)]['speedup']:.1f}")
    put("FastHfcSpeedCosineBig", f"{_by[('cosine', 200, 256, 0.75)]['speedup']:.0f}")
    put("FastHfcSpeedCosineLong", f"{_by[('cosine', 2500, 64, 0.75)]['speedup']:.1f}")
    put("FastHfcSlowCosineBigMs", num(_by[('cosine', 200, 256, 0.75)]['slow_ms']))
    put("FastHfcFastCosineBigMs", f"{_by[('cosine', 200, 256, 0.75)]['fast_ms']:.1f}")
    put("FastHfcBankSec", f"{fh['benchmark']['h_bank_99H_N200_J64_fast_s']:.2f}")
else:
    for k in ("FastHfcErrN", "FastHfcErrM", "FastHfcErrQ", "FastHfcErrDq", "FastHfcErrG",
              "FastHfcSpeedTwoscaleSmall", "FastHfcSpeedTwoscaleMed", "FastHfcSpeedCosineMed",
              "FastHfcSpeedCosineBig", "FastHfcSpeedCosineLong", "FastHfcSlowCosineBigMs",
              "FastHfcFastCosineBigMs", "FastHfcBankSec"):
        missing(k, "x11/fast_hfc_benchmark.json")

if exists("x11/simpson_exact_check.json"):
    sx = J("x11/simpson_exact_check.json")["by_H"]
    put("SimpCovErrSix", _fmt_sci(sx["0.60"]["cov_relerr"]["simpson_exact"]))
    put("SimpCovErrSevenFive", _fmt_sci(sx["0.75"]["cov_relerr"]["simpson_exact"]))
    put("SimpCovErrNine", _fmt_sci(sx["0.90"]["cov_relerr"]["simpson_exact"]))
    put("SimpCovErrMax", _fmt_sci(max(v["cov_relerr"]["simpson_exact"] for v in sx.values())))
    put("SimpLinEvidSix", f"{sx['0.60']['linear_check']['max_logev_diff_simpson_exact_vs_grid']:.4f}")
    put("SimpLinEvidSevenFive", f"{sx['0.75']['linear_check']['max_logev_diff_simpson_exact_vs_grid']:.4f}")
    put("SimpLinEvidNine", f"{sx['0.90']['linear_check']['max_logev_diff_simpson_exact_vs_grid']:.4f}")
    for Hk, nm in (("0.60", "Six"), ("0.75", "SevenFive"), ("0.90", "Nine")):
        lc = sx[Hk]["linear_check"]
        put(f"LinGapMfGrid{nm}", f"{lc['exact_gap_mf_grid']:.1f}")
        put(f"LinGapMfCosvi{nm}", f"{lc['exact_gap_mf_cosvi']:.1f}")
        put(f"LinGapBlock{nm}", f"{lc['exact_gap_block_joint_J16']:.2f}")
        put(f"LinGapProd{nm}", f"{lc['exact_gap_product_full']:.2f}")
        put(f"CondCr{nm}", f"{sx[Hk]['cond_CR']:.1f}")
else:
    for k in ("SimpCovErrSix", "SimpCovErrSevenFive", "SimpCovErrNine", "SimpCovErrMax",
              "SimpLinEvidSix", "SimpLinEvidSevenFive", "SimpLinEvidNine",
              "LinGapMfGridSix", "LinGapMfGridSevenFive", "LinGapMfGridNine",
              "LinGapMfCosviSix", "LinGapMfCosviSevenFive", "LinGapMfCosviNine",
              "LinGapBlockSix", "LinGapBlockSevenFive", "LinGapBlockNine",
              "LinGapProdSix", "LinGapProdSevenFive", "LinGapProdNine",
              "CondCrSix", "CondCrSevenFive", "CondCrNine"):
        missing(k, "x11/simpson_exact_check.json")

if exists("x11/n7_clean_summary.json"):
    nc = J("x11/n7_clean_summary.json")["by_H"]
    rats = [r["ratio_std_lift_over_cosvi"] for v in nc.values() for r in v]
    put("NsevenStdRatioMin", f"{min(rats):.1f}")
    put("NsevenStdRatioMax", f"{max(rats):.1f}")
    put("NsevenPubRescaledNineErr", f"{100 * nc['0.90'][0]['err_lift_published_rescaled']:.1f}")
    put("NsevenPubCodedNineErr", f"{100 * nc['0.90'][0]['err_lift_published_coded']:.1f}")
else:
    for k in ("NsevenStdRatioMin", "NsevenStdRatioMax", "NsevenPubRescaledNineErr", "NsevenPubCodedNineErr"):
        missing(k, "x11/n7_clean_summary.json")

if exists("x11/x11a_rotated_grid.json"):
    xa = J("x11/x11a_rotated_grid.json")
    ls = xa["linear_sensitivity"]
    sm = xa["summary"]
    for Hk, nm in (("0.6", "Six"), ("0.75", "SevenFive"), ("0.9", "Nine")):
        put(f"XelevenASensTopPl{nm}", f"{100 * ls[Hk]['twoscale_pl']['share_best_16']:.1f}")
        put(f"XelevenASensTopRotPl{nm}", f"{100 * ls[Hk]['rot_pl']['share_best_16']:.1f}")
        for comp, cn in (("rot_pl", "RotPl"), ("rot_grid", "RotGrid"), ("grid", "Grid")):
            c = sm[Hk]["contrasts_vs_pl"][comp]
            put(f"XelevenAPlVs{cn}{nm}", sgn(c["ll_gain_mean"]))
            put(f"XelevenAPlVs{cn}SE{nm}", f"{c['ll_gain_se']:.2f}")
    put("XelevenAPlWinsRotPl", str(sum(sm[H]["contrasts_vs_pl"]["rot_pl"]["ll_wins"] for H in ("0.6", "0.75", "0.9"))))
    put("XelevenAPlWinsRotGrid", str(sum(sm[H]["contrasts_vs_pl"]["rot_grid"]["ll_wins"] for H in ("0.6", "0.75", "0.9"))))
    put("XelevenATotalRuns", str(3 * len(xa["seeds"])))
else:
    for Hk, nm in (("0.6", "Six"), ("0.75", "SevenFive"), ("0.9", "Nine")):
        for pfx in ("XelevenASensTopPl", "XelevenASensTopRotPl",
                    "XelevenAPlVsRotPl", "XelevenAPlVsRotPlSE",
                    "XelevenAPlVsRotGrid", "XelevenAPlVsRotGridSE",
                    "XelevenAPlVsGrid", "XelevenAPlVsGridSE"):
            missing(f"{pfx}{nm}", "x11a pending")
    for k in ("XelevenAPlWinsRotPl", "XelevenAPlWinsRotGrid", "XelevenATotalRuns"):
        missing(k, "x11a pending")

if exists("x11/x11b_structured_q.json") and "summary" in J("x11/x11b_structured_q.json"):
    xb = J("x11/x11b_structured_q.json")
    sxb = xb["summary"]
    put("XelevenBParamsProd", num(sxb["0.75"]["pl_prod_full"]["local_params_per_series"]))
    put("XelevenBParamsJoint", num(sxb["0.75"]["pl_joint_full"]["local_params_per_series"]))
    put("XelevenBParamsBlock", num(sxb["0.75"]["pl_block_joint"]["local_params_per_series"]))
    put("XelevenBParamsPrecSixteen", num(sxb["0.75"]["pl_prec_lr16"]["local_params_per_series"]))
    eg = [sxb[H]["paired"]["joint_minus_prod_elbo"][0] for H in ("0.6", "0.75", "0.9")]
    put("XelevenBJointElboGainMin", f"{min(eg):.2f}")
    put("XelevenBJointElboGainMax", f"{max(eg):.2f}")
    put("XelevenBJointMinEssMin", f"{min(sxb[H]['pl_joint_full']['min_ess_joint'] for H in ('0.6', '0.75', '0.9')):.0f}")
    put("XelevenBProdMinEssMin", f"{min(r['ll_test_prod_min_ess'] for r in xb['rows'] if r['variant'] == 'pl_prod_full'):.0f}")
    sc = xb["scaling"]
    sc_big = {r["family"]: r for r in sc if r["N"] == 1600}
    put("XelevenBParamsFullBig", num(sc_big["joint_full"]["local_params_per_series"]))
    put("XelevenBParamsBlockBig", num(sc_big["block_joint"]["local_params_per_series"]))
    put("XelevenBMemFullBig", "OOM" if "error" in sc_big["joint_full"] else f"{sc_big['joint_full']['peak_mem_MB']:.0f}")
    put("XelevenBMemBlockBig", f"{sc_big['block_joint']['peak_mem_MB']:.0f}")
    put("XelevenBMemPrecBig", f"{sc_big['prec_lr16']['peak_mem_MB']:.0f}")
else:
    for k in ("XelevenBParamsProd", "XelevenBParamsJoint", "XelevenBParamsBlock", "XelevenBParamsPrecSixteen",
              "XelevenBJointElboGainMin", "XelevenBJointElboGainMax", "XelevenBJointMinEssMin",
              "XelevenBProdMinEssMin", "XelevenBParamsFullBig", "XelevenBParamsBlockBig",
              "XelevenBMemFullBig", "XelevenBMemBlockBig", "XelevenBMemPrecBig"):
        missing(k, "x11b pending")

if exists("x11/x11c_duffing_and_nile.json") and "nile" in J("x11/x11c_duffing_and_nile.json"):
    xc = J("x11/x11c_duffing_and_nile.json")
    df_ = xc["duffing"]
    nl_ = xc["nile"]
    put("XelevenCDuffParamsFull", num(next(r["local_params_per_series"] for r in df_ if r["config"] == "cosvi_pl_full")))
    put("XelevenCDuffParamsBlock", num(next(r["local_params_per_series"] for r in df_ if r["config"] == "cosvi_pl_block")))
    n_by = {f: [r for r in nl_ if r["family"] == f] for f in ("prod_full", "joint_full", "block_joint", "prec_lr16")}
    put("XelevenCNileParamsFull", num(n_by["joint_full"][0]["local_params_per_window"]))
    put("XelevenCNileParamsBlock", num(n_by["block_joint"][0]["local_params_per_window"]))
    put("XelevenCNileParamsPrec", num(n_by["prec_lr16"][0]["local_params_per_window"]))
    eg_n = np.mean([a["elbo_at_best_H"] - b["elbo_at_best_H"] for a, b in zip(n_by["joint_full"], n_by["prod_full"])])
    put("XelevenCNileJointElboGain", f"{eg_n:+.2f}")
    put("XelevenCNileJointMinEss", f"{min(r['min_ess_at_profile'] for r in n_by['joint_full']):.0f}")
else:
    for k in ("XelevenCDuffParamsFull", "XelevenCDuffParamsBlock", "XelevenCNileParamsFull",
              "XelevenCNileParamsBlock", "XelevenCNileParamsPrec", "XelevenCNileJointElboGain",
              "XelevenCNileJointMinEss"):
        missing(k, "x11c pending")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("% Generated by scripts/make_macros.py from the stored results. Do not edit by hand.\n")
    for k in sorted(M):
        v, note = M[k]
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}" + (f"  % {note}" if note else "") + "\n")
for k in sorted(M):
    print(f"{k:28s} {M[k][0]}")
print(len(M), "macros ->", OUT)
if MISSING:
    print(f"WARNING: {len(MISSING)} placeholder macros (results pending):", ", ".join(MISSING[:12]), "..." if len(MISSING) > 12 else "")
