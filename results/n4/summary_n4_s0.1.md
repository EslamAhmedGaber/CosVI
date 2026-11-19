N4 v2, s_obs = 0.1. Held-out = exact-law IWAE per test window, two-scale refit (mean ± sd over seeds).
| representation | seeds | profile Ĥ (training bound) | held-out at profile group | held-out per group H=0.5…0.9 (mean over seeds) | best group (held-out) | diverged groups | grid-refit failures |
|---|---|---|---|---|---|---|---|
| CosVI-64 [cosine64] | 3 | 0.778 ± 0.025 | -50.776 ± 0.425 | -53.94 / -53.13 / -51.53 / -50.36 / -49.83 | H=0.9: -49.829 ± 0.815 | 0/15 | 13/15 |
| exact grid [grid] | 3 | 0.618 ± 0.044 | -53.899 ± 2.904 | -62.01 / -53.33 / -50.29 / -349.00 / -96.93 | H=0.7: -50.289 ± 2.184 | 6/15 | 2/15 |
| MA-fBm law, published nodes [markov] | 3 | 0.715 ± 0.001 | -51.469 ± 0.067 | -58.67 / -64.52 / -51.47 / -56.32 / -126.60 | H=0.7: -51.469 ± 0.067 | 0/15 | 12/15 |
| MA-fBm law, tuned nodes [markov_tuned] | 3 | 0.611 ± 0.042 | -51.325 ± 2.926 | -61.71 / -51.99 / -49.48 / -66.16 / -203.08 | H=0.7: -49.480 ± 2.308 | 7/15 | 3/15 |
| CosVI [twoscale] | 3 | 0.794 ± 0.035 | -47.644 ± 0.327 | -49.21 / -49.11 / -48.33 / -47.64 / -47.69 | H=0.8: -47.644 ± 0.327 | 0/15 | 10/15 |
