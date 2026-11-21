# X10 — proposal sensitivity of the importance scores (post hoc, CPU)

Protocols fixed before the runs: X10 on 18 Nov 2026 at 19:28, X10(c) at 19:46.

- `x10a_twoscale_on-{twoscale,grid,markov}_seed{1,2}_H{H}.json` — part (a): the frozen CosVI-trained N2 models, evidence of
  three discrete models under the joint Gauss-Newton proposal (`code/experiments/x10_proposal_sensitivity.py --part a`).
- `x10b_{rep}_on-grid_seed{s}_H{H}.json` — part (b): saved double-well models under the grid evaluator (`--part b`).
- `x10c_{nile,temp}_{rep}_seed{s}.json` — part (c): real-series models under the smooth-stage evaluator
  (`code/experiments/x10c_real_series.py`).
- `x10_summary.json` — `code/experiments/summarize_x10.py` (reading rules as fixed in the protocol).
- `n2_joint_candidates.json` — joint-minus-product candidate ELBO contrasts and joint readouts for the six N2 cases
  (computed 18 Nov 2026 with a separate verification script, not included); read by the number generator.
- `*_smoke.json`, `*.log` — smoke test and worker logs (not used by the summariser).
