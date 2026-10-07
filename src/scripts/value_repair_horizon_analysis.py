"""Post-hoc horizon analysis of a value-repair results bundle.

Reads ``sibling_child_scores.npz`` (every child of every probed parent, all
horizons) and asks, per remaining-horizon bucket, how well each scorer agrees
with the frozen greedy rollout -- the quantity a learned evaluator is meant to
replace -- and, where an exact oracle exists, with the optimum.

Per parent: within-parent Pearson correlation of scorer vs rollout total score,
whether the scorer's pick equals the rollout's pick, regret of the scorer's pick
measured by the rollout score, and regret measured by the exact oracle when all
children are certified. Head seeds are averaged within a variant family.

Usage:
    PYTHONPATH=src .venv/bin/python src/scripts/value_repair_horizon_analysis.py \
        <results_dir> --out <csv>
"""
import argparse
import collections
import csv
import json
from pathlib import Path

import numpy as np


def bucket_of(remaining):
    if remaining <= 5:
        return '01-05'
    if remaining <= 10:
        return '06-10'
    if remaining <= 20:
        return '11-20'
    if remaining <= 30:
        return '21-30'
    if remaining <= 40:
        return '31-40'
    return '41+'


def family(name):
    stem = name[len('score_'):]
    base, _, tail = stem.rpartition('_s')
    return base if base and tail.isdigit() else stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results_dir')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = Path(args.results_dir)
    n_cities = json.loads((root / 'config.json').read_text())['graph_size']
    z = np.load(root / 'sibling_child_scores.npz')
    key = z['instance'] * 1_000_000 + z['source'] * 10_000 + z['step']
    order = np.argsort(key, kind='stable')
    col = {k: z[k][order] for k in z.files}
    key = key[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    ends = np.r_[starts[1:], len(key)]
    scorers = [k for k in z.files if k.startswith('score_') and k != 'score_rollout']
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        src = 'greedy' if int(col['source'][s]) == 0 else 'sample_tau3'
        b = bucket_of(n_cities - int(col['step'][s]))
        ro = col['score_rollout'][s:e].astype(np.float64)
        best_ro = ro.min()
        oracle = col['oracle'][s:e]
        exact = not np.isnan(oracle).any()
        acc[(src, b, 'rollout')]['signal'].append(ro.std())
        if exact:  # exact regret of the rollout's own pick (a decision, not a completion, error)
            acc[(src, b, 'rollout')]['regret_exact'].append(oracle[int(np.argmin(ro))] - oracle.min())
        for name in scorers:
            sc = col[name][s:e].astype(np.float64)
            if name == 'score_prior':
                sc = -col['prior'][s:e].astype(np.float64)
            fam = family(name)
            d = acc[(src, b, fam)]
            pick = int(np.argmin(sc))
            d['regret_rollout'].append(ro[pick] - best_ro)
            d['agree'].append(float(ro[pick] <= best_ro + 1e-9))
            if sc.std() > 0 and ro.std() > 0:
                d['corr'].append(float(np.corrcoef(sc, ro)[0, 1]))
            if exact:
                d['regret_exact'].append(oracle[pick] - oracle.min())
    rows = []
    for (src, b, fam), d in sorted(acc.items()):
        mean = lambda k: float(np.mean(d[k])) if d[k] else float('nan')
        sig = acc[(src, b, 'rollout')]
        rows.append(dict(source=src, bucket=b, scorer=fam,
                         parents=len(sig['signal']), exact_parents=len(sig['regret_exact']),
                         rollout_signal_std=float(np.mean(sig['signal'])),
                         corr_with_rollout=mean('corr'), agree_with_rollout=mean('agree'),
                         regret_vs_rollout=mean('regret_rollout'),
                         regret_vs_exact=mean('regret_exact')))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows)} rows -> {out}')
    print(f"{'source':12s} {'bucket':6s} {'scorer':16s} {'par':>5s} {'exact':>5s} {'signal':>7s} "
          f"{'corr':>6s} {'agree':>6s} {'regret/ro':>10s} {'regret/opt':>10s}")
    for r in rows:
        print(f"{r['source']:12s} {r['bucket']:6s} {r['scorer']:16s} {r['parents']:5d} "
              f"{r['exact_parents']:5d} {r['rollout_signal_std']:7.3f} {r['corr_with_rollout']:6.3f} "
              f"{r['agree_with_rollout']:6.3f} {r['regret_vs_rollout']:10.4f} {r['regret_vs_exact']:10.4f}")


if __name__ == '__main__':
    main()
