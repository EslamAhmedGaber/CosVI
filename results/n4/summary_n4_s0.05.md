N4 v2, s_obs = 0.05. Held-out = exact-law IWAE per test window, two-scale refit (mean ± sd over seeds).
| representation | seeds | profile Ĥ (training bound) | held-out at profile group | held-out per group H=0.5…0.9 (mean over seeds) | best group (held-out) | diverged groups | grid-refit failures |
|---|---|---|---|---|---|---|---|
| CosVI-64 [cosine64] | 1 | 0.741 ± nan | -49.450 ± nan | -52.43 / -51.61 / -49.45 / -48.94 / -47.93 | H=0.9: -47.929 ± nan | 0/5 | 5/5 |
| exact grid [grid] | 1 | 0.609 ± nan | -58.029 ± nan | -63.83 / -58.03 / -54.79 / -155.30 / -169.76 | H=0.7: -54.794 ± nan | 2/5 | 1/5 |
| CosVI [twoscale] | 1 | 0.695 ± nan | -47.626 ± nan | -48.58 / -48.60 / -47.63 / -47.89 / -48.05 | H=0.7: -47.626 ± nan | 0/5 | 4/5 |

Competitor full system (N4c):
| mode | seed | Ĥ | held-out (two-scale refit) | held-out (grid refit) | σ (years, standard scale) |
|---|---|---|---|---|---|
| MA-fBm (H learned jointly) | 0 | 0.578 | -246.949 | -247.115 | 0.134 |
| MA-fBm (H learned jointly) | 1 | 0.576 | -249.515 | -250.179 | 0.133 |
| MA-fBm (H learned jointly) | 2 | 0.580 | -250.960 | -251.640 | 0.133 |
| MA-fBm (H fixed at the truth) | 0 | 0.840 | -360.502 | -379.778 | 0.106 |
| MA-fBm (H fixed at the truth) | 1 | 0.840 | -357.675 | -386.733 | 0.107 |
| MA-fBm (H fixed at the truth) | 2 | 0.840 | -366.973 | -379.151 | 0.103 |
