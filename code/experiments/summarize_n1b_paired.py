r"""N1b, paired comparison by data seed (read-only; writes nothing).

The N1b runs of every representation share the data and the initial weights of each data seed, so differences between
representations are best read seed by seed. For each H_true this prints, for the main 5000-step runs at the oracle H
(the last group): mean and sample sd over seeds of the held-out exact-law log-likelihood (grid yardstick, the primary
one, and the two-scale yardstick where present), drift RMSE and profile H; then paired differences between
representations (mean, sd and the per-seed values); then the drift RMSE of CosVI against the best MA-fBm variant of
the same seed. Tagged runs (for example the 10 000-step ablation) are skipped.

Run from work\code\experiments:  python summarize_n1b_paired.py   (set PYTHONIOENCODING=utf-8 on Windows)
Written 15 Nov 2026 for the three-seed reading in 04_THREAT_AND_PILOT.md §4b.
"""
import glob
import json
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from fnvi import labels as L  # noqa: E402

R = os.path.join(HERE, '..', '..', 'results', 'n1b')
KEYS = ['twoscale', 'cosine', 'grid', 'markov', 'markov_tuned', 'markov_best']
PAIRS = [('twoscale', 'markov'), ('twoscale', 'grid'), ('cosine', 'markov'), ('cosine', 'twoscale'),
         ('twoscale', 'markov_best'), ('twoscale', 'markov_tuned')]
MA_KEYS = ('markov', 'markov_tuned', 'markov_best')
pat = re.compile(r'^n1b_(?P<rep>[a-z_]+?)_seed(?P<seed>\d+)_H(?P<H>\d\.\d+)\.json$')


def load():
    rows = {}
    for f in glob.glob(os.path.join(R, 'n1b_*_seed*_H*.json')):
        m = pat.match(os.path.basename(f))
        if not m:
            continue
        with open(f) as fh:
            d = json.load(fh)
        g = len(d['groups']) - 1
        ll2 = d.get('test_loglik_exact_law_twoscale_coords_mean')
        rows[(m['rep'], float(m['H']), int(m['seed']))] = {
            'grid': d['test_loglik_exact_grid_law_mean'][g], 'two': ll2[g] if ll2 else None,
            'drift': d['drift_rmse_occupation'][g], 'H': d['H_profile_own_model']}
    return rows


def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return '   -        '
    return f'{st.mean(xs):+6.2f}' + (f' ± {st.stdev(xs):4.2f}' if len(xs) > 1 else '       ') + f' (n={len(xs)})'


def main():
    rows = load()
    for H in sorted({k[1] for k in rows}):
        seeds = sorted({k[2] for k in rows if k[1] == H})
        print(f'\n=== H_true = {H}  (data seeds {seeds})')
        for r in KEYS:
            v = [rows[(r, H, s)] for s in seeds if (r, H, s) in rows]
            if v:
                print(f'  {L.rep_name(r):30s} loglik grid {ms([x["grid"] for x in v])}  two-scale '
                      f'{ms([x["two"] for x in v])}  drift {st.mean([x["drift"] for x in v]):.3f}  '
                      f'H {st.mean([x["H"] for x in v]):.2f} [{", ".join(f"{x["H"]:.2f}" for x in v)}]')
        print('  paired differences (mean ± sd, n) [per seed], grid yardstick | two-scale yardstick:')
        for a, b in PAIRS:
            dg, dt = [], []
            for s in seeds:
                if (a, H, s) in rows and (b, H, s) in rows:
                    dg.append(rows[(a, H, s)]['grid'] - rows[(b, H, s)]['grid'])
                    if rows[(a, H, s)]['two'] is not None and rows[(b, H, s)]['two'] is not None:
                        dt.append(rows[(a, H, s)]['two'] - rows[(b, H, s)]['two'])
            if dg:
                print(f'    {L.rep_name(a)} minus {L.rep_name(b)}: {ms(dg)} '
                      f'[{", ".join(f"{x:+.2f}" for x in dg)}] | {ms(dt)}')
        print(f'  drift RMSE, {L.METHOD} against the best {L.MA} variant of the same seed:')
        for s in seeds:
            ma = [rows[(r, H, s)]['drift'] for r in MA_KEYS if (r, H, s) in rows]
            if ('twoscale', H, s) in rows and ma:
                ours = rows[('twoscale', H, s)]['drift']
                print(f'    seed {s}: {ours:.3f} vs {min(ma):.3f} ({100 * (1 - ours / min(ma)):+.0f} %)')


if __name__ == '__main__':
    main()
