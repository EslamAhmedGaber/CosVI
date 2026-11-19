# CosVI: code and results

Supplementary archive to the manuscript "Exact whitened coordinates for fractional noise in neural stochastic differential equations". Built 2026-11-19.

## Layout (mirrors the working tree, so the scripts run unchanged from this folder)
- `code/fnvi/` - the CosVI implementation: cosine and two-scale coordinates (`coords.py`, `noise_maps.py`,
  `vendor/hfc.py`), exact linear-Gaussian references (`linear_exact.py`), the MA-fBm law re-implemented from the
  published formulas for exact error computations (`ma_fbm.py`), training on a graphics processor (`train_graph.py`,
  `train_maps.py`).
- `code/experiments/` - one driver per experiment and one summariser per reading rule.
- `code/queue_gpu.jsonl`, `code/queue_cpu.jsonl` - the command of every run, in the order run.
- `manuscript/scripts/` - `make_macros.py` (every number in the text), `make_tables.py`, `fig*.py`, `build.py`.
- `results/` - 960 files: per-run records (`*.json`), summaries, trained models (`*.pt`), the synthetic data sets
  (`*.npz`) and the run status files.
- `MANIFEST.sha256` - checksum of every file.

## Not included
- The competitor's code (Daems et al., ICLR 2024) carries no licence. It is not redistributed, and neither is our
  PyTorch port of it (`code/fnvi/mafbm_torch.py`) nor our drivers of their code (`code/competitor/`). These included
  drivers import the port and run only once it is supplied: `code/experiments/check_n4c_units.py`, `code/experiments/n1c_competitor_system.py`, `code/experiments/n4c_competitor_nile.py`, `code/experiments/n5_scaling.py`, `code/experiments/n7_horizon_exact.py`. Their recorded outputs are under `results/`.
- The two real series (Nile minima, Northern-Hemisphere temperature anomalies) are public in the CRAN package
  longmemo; `code/experiments/n4_prepare_data.py` documents how they were read.
- `results/cache/` (linear maps, regenerated on first use) and machine-specific logs.

## Requirements
Python 3.12 with NumPy, SciPy, PyTorch (CUDA for the training runs) and Matplotlib. Times in the records were measured
on one consumer graphics processor and one processor core and are specific to that machine.
