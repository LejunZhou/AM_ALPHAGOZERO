"""Cross-run comparison table for probe_sibling_ranking.py CSVs.

Usage: PYTHONPATH=src python src/scripts/aggregate_sibling_ranking.py <label=path.csv> ...
Prints one row per run with the decision-quality numbers that matter for MCTS.
"""
import sys
import numpy as np
import pandas as pd


def summarize(d):
    bad = d[d['isopt_pol'] == 0]
    return dict(
        states=len(d),
        children=int(d['m'].sum()),
        P_opt_pol=d['isopt_pol'].mean(),
        P_opt_ro=d['isopt_ro'].mean(),
        P_opt_vh=d['isopt_vh'].mean(),
        regret_pol=d['ropt_pol'].mean(),
        regret_ro=d['ropt_ro'].mean(),
        regret_vh=d['ropt_vh'].mean(),
        fix_ro=bad['isopt_ro'].mean() if len(bad) else np.nan,
        fix_vh=bad['isopt_vh'].mean() if len(bad) else np.nan,
        vh_worse_pol=(d['ropt_vh'] > d['ropt_pol'] + 1e-9).mean(),
        vh_better_pol=(d['ropt_vh'] < d['ropt_pol'] - 1e-9).mean(),
        P_opt_top3_vh=d['isopt_top3_vh'].mean() if 'isopt_top3_vh' in d else np.nan,
        P_opt_top3_ro=d['isopt_top3_ro'].mean() if 'isopt_top3_ro' in d else np.nan,
        slope_vh=d['slope_vh'].median() if 'slope_vh' in d else np.nan,
        slope_ro=d['slope_ro'].median() if 'slope_ro' in d else np.nan,
        margin_med=d['margin_opt'].median(),
        err_vh_med=d['err_std_vh'].median(),
        err_ro_med=d['err_std_ro'].median(),
    )


rows = []
for arg in sys.argv[1:]:
    label, path = arg.split('=', 1)
    d = pd.read_csv(path)
    s = summarize(d)
    s['run'] = label
    rows.append(s)
df = pd.DataFrame(rows).set_index('run')
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 40)
print(df.T.to_string(float_format=lambda x: f'{x:.4f}'))
