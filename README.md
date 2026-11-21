# CosVI: code, unit tests, and results archive

Supplementary software and reproducibility archive for the manuscript
**"Exact whitened coordinates for fractional noise in neural stochastic differential equations"**
by Eslam Ahmed, Mohamed A. El-Beltagy, Ahmed Etman, and Sroor M. Elnady
(prepared for *Computational Statistics & Data Analysis*).

Public repository: <https://github.com/EslamAhmedGaber/CosVI>

## Quick verification (clean-room)

Run all 28 automated unit tests, regenerate all LaTeX numeric macros, and rebuild all 12 tables from the stored JSON records:

```bash
python -m pytest code/tests/test_cosvi.py
python -X utf8 manuscript/scripts/make_macros.py
python -X utf8 manuscript/scripts/make_tables.py
```

## Core library (`code/fnvi/`)

- `fast_hfc.py` — pure NumPy/SciPy piecewise degree-17 Chebyshev ($|x|\le 56$) + 18-term asymptotic ($|x|>56$) engine for $N_H(x)={}_1F_1(2H;2H+1;ix)$ and $M_H(x)={}_1F_1(2H-1;2H;ix)$ (worst error $\le 3.4\times 10^{-13}$ vs. 30-digit `mpmath`; $2.7\times$–$170\times$ single-thread build speedup across tested settings).
- `coords.py` — exact closed-form cosine Gram matrix $G(H)$, nested Cholesky factor $L$, covariance curves $Q(t)$ and rates $\dot Q(t)$, and strictly positive-definite interior residual covariance $C_R\succ 0$ (`fast=True` by default; `fast=False` retained for benchmarking).
- `noise_maps.py` — linear stage-rate maps for `twoscale` (smooth CosVI), `twoscale_exact_map` (Simpson-exact CosVI-EX, covariance error $\le 4.5\times 10^{-16}$, zero jitter), `twoscale_pl_map` (CosVI-PL solver-matched control), `cosine`, `grid`, and `markov`.
- `families.py` — CUDA-graph-capturable structured joint Gaussian posteriors:
  - `LocalsJointFull`: joint full $q(\mathbf{z}, x_0)\in\mathbb{R}^{D+1}$,
  - `LocalsBlockJoint` & `Locals2BlockJoint`: two-scale $\mathcal{O}(J^2+N)$ joint posteriors,
  - `LocalsPrecLR`: Matheron-sampled precision-low-rank Woodbury posterior $\Lambda=\operatorname{diag}(d^{-2})+VV^\top$, $\mathcal{O}(rN)$ memory, zero eigendecomposition.
- `linear_exact.py`, `ma_fbm.py`, `simulate.py`, `train_graph.py`, `train_maps.py` — exact linear-Gaussian references, MA-fBm analytical covariance law, Davies–Harte path simulation, and CUDA-graph training.
- `code/tests/test_cosvi.py` — 28 unit tests verifying `fast_hfc`, $C_R\succ 0$, Simpson-exact grid laws, and all variational families.

## Experiment codes and manuscript mapping

| Code | Description | Manuscript / SM location |
|---|---|---|
| G1 | Noise fidelity at equal dimension (exact) | Main Sec. 4.1, Fig. 3 |
| G2, G2b, V1 | Exact linear-Gaussian inference & resolution limit | Main Sec. 4.2, Fig. 4 |
| G0 (a–c, r), N3, X6, X7 | Published bridge benchmark (authors' JAX code), read-out control, CosVI | Main Sec. 4.3, Fig. 5, Table 3, SM Sec. S.2 |
| G0-H, N1c | Joint Hurst-index learning in MA-fBm | Main Sec. 4.4, Fig. 1b |
| N1, N1b, X1 | 1-D double well (6 representations, 3 & 5 seeds, budget ablation) | Main Sec. 5.2, Fig. 6, Table 4, Table S.2 |
| N2, X10 | Bound vs. evidence & coupled importance-proposal check | Main Sec. 5.2, 5.6 |
| X8 (a–d), X11a | Coordinates vs. solver & 4-arm Haar-random rotation causal control | Main Sec. 5.3, Fig. 7, Table S.3, Table S.4 |
| X4, N6, X11b | Latent-path calibration, ablations, structured $\mathcal{O}(N)$ families & VRAM scaling | Main Sec. 5.4, 5.7, Fig. 8, Fig. 11c, Table 5, Tables S.1, S.7 |
| X2, X11c (Pt 1) | 2-D damped Duffing oscillator, PL and grid evaluators, 2-D BlockJoint | Main Sec. 5.5, Fig. 9, Table S.5 |
| N4, N4c, X3, X5, X9, X11c (Pt 2) | Real series (Nile minima, NH temperature), window audit, Nile structured $q$ | Main Sec. 5.6, Fig. 10, Table 6, Tables S.4, S.7 |
| N5, N7, X11, X11d | Step cost, fast precomputation, Simpson-exact check, clean standard-scale long horizons | Main Sec. 3.1, 3.5, 5.7, Fig. 11, Tables S.6, S.8 |

## Files intentionally excluded

1. **Competitor source code (Daems et al., ICLR 2024):** Carries no open-source licence and is not redistributed (neither `mafbm_torch.py` nor `code/competitor/`). Five drivers (`check_n4c_units.py`, `n1c_competitor_system.py`, `n4c_competitor_nile.py`, `n5_scaling.py`, `n7_horizon_exact.py`) require the port if re-run; all their recorded JSON outputs are included under `results/`.
2. **Third-party CRAN CSVs (`nile_min.csv`, `nhemi_temp.csv`):** Public in CRAN package `longmemo` 1.1-4 (`GPL-2 | GPL-3`); `code/experiments/n4_prepare_data.py` documents parsing from CRAN, while our generated metadata summary `results/n4/data/provenance.json` is included so all macro, table, and figure scripts run out of the box.

## Requirements

Python 3.12 with NumPy, SciPy, PyTorch (CUDA for GPU training scripts), Matplotlib, pytest, and mpmath (required for the high-precision unit tests).

### Refitting archived comparisons

The recorded competitor fits used a local PyTorch port with SHA-256 4ae56f48bdf05da02bc02b76beaeb9b1759a7cb1d9f6d6a41634add39d71c984. An authorised copy belongs at the relative path code/fnvi/mafbm_torch.py. The upstream implementation uses JAX (https://github.com/VideoNeuralSDE/MAFBM/tree/c4690b6b8de046ba71487c28d849eb58cfa58307). All numeric reporting and figure regeneration run from the distributed results without the port. To refit the real-series models, extract CRAN longmemo 1.1-4 under external/data_raw/longmemo/longmemo (the inner directory contains data/; retain longmemo_1.1-4.tar.gz in its parent directory) and run python code/experiments/n4_prepare_data.py. Verify the input tarball against results/n4/data/provenance.json. The public repository distributes the code and recorded results; the modular article sources accompany the journal software archive. Figure scripts require a LaTeX installation with the packages specified in manuscript/scripts/style.py.
