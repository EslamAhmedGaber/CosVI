N4 v2, s_obs = 0.2. Held-out = exact-law IWAE per test window, two-scale refit (mean ± sd over seeds).
| representation | seeds | profile Ĥ (training bound) | held-out at profile group | held-out per group H=0.5…0.9 (mean over seeds) | best group (held-out) | diverged groups | grid-refit failures |
|---|---|---|---|---|---|---|---|
| CosVI-64 [cosine64] | 1 | 0.826 ± nan | -52.358 ± nan | -54.65 / -54.08 / -53.34 / -52.36 / -51.94 | H=0.9: -51.935 ± nan | 0/5 | 3/5 |
| exact grid [grid] | 1 | 0.753 ± nan | -47.413 ± nan | -62.84 / -58.42 / -47.86 / -47.41 / -54.63 | H=0.8: -47.413 ± nan | 1/5 | 2/5 |
| CosVI [twoscale] | 1 | 0.843 ± nan | -49.792 ± nan | -50.67 / -50.67 / -50.15 / -49.79 / -49.78 | H=0.9: -49.781 ± nan | 0/5 | 3/5 |

Competitor full system (N4c):
| mode | seed | Ĥ | held-out (two-scale refit) | held-out (grid refit) | σ (years, standard scale) |
|---|---|---|---|---|---|
| MA-fBm (H learned jointly) | 0 | 0.578 | -246.949 | -247.115 | 0.134 |
| MA-fBm (H learned jointly) | 1 | 0.576 | -249.515 | -250.179 | 0.133 |
| MA-fBm (H learned jointly) | 2 | 0.580 | -250.960 | -251.640 | 0.133 |
| MA-fBm (H fixed at the truth) | 0 | 0.840 | -360.502 | -379.778 | 0.106 |
| MA-fBm (H fixed at the truth) | 1 | 0.840 | -357.675 | -386.733 | 0.107 |
| MA-fBm (H fixed at the truth) | 2 | 0.840 | -366.973 | -379.151 | 0.103 |
