"""Stage 3 aggregator: roll up per-instance CSVs into a comparison table.

Reads Stage 2 MCTS K-curve CSVs (which predate Phase A instrumentation) and
Stage 3 sampling CSVs, joins them with the decode_step probe cache for the
forward-pass-budget x-axis, and writes a single tidy CSV per graph size:

    method, leaf_eval, K, decode_steps_mean, mean_cost, std_cost,
    mean_gap_to_opt_pct, gap_reduction_vs_greedy_pct,
    win_rate_vs_greedy, n_instances, source_csv

The plot script consumes this comparison CSV directly.

Usage:
  PYTHONPATH=src python -m scripts.aggregate_stage3 \
      --graph_size 20 \
      --optimum 3.8279 \
      --probe_cache outputs/stage3/decode_step_cache.json \
      --output_csv outputs/stage3/comparison_tsp20.csv
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics
import sys
from typing import Optional


# Mapping from (method, leaf_eval) → (glob pattern, regex pattern) for source CSVs.
# Glob pattern uses '*' for K; regex pattern has one capture group on K (integer).
SOURCES_TSP20 = [
    # Stage 3 cpp re-runs (preferred — already instrumented at val_size=1000).
    ('mcts', 'rollout',
     'outputs/stage3/tsp20_K*_rollout.csv',
     r'tsp20_K(\d+)_rollout\.csv'),
    ('mcts', 'value_head',
     'outputs/stage3/tsp20_K*_value_head.csv',
     r'tsp20_K(\d+)_value_head\.csv'),
    # Stage 2 reuse (only kicks in for K not present above).
    ('mcts', 'rollout',
     'outputs/stage2/tsp20_K*_rollout_canonical.csv',
     r'tsp20_K(\d+)_rollout_canonical\.csv'),
    ('mcts', 'value_head',
     'outputs/stage2/tsp20_K*_canonical_v2.csv',
     r'tsp20_K(\d+)_canonical_v2\.csv'),
    ('sampling', '-',
     'outputs/stage3/tsp20_sampling_K*.csv',
     r'tsp20_sampling_K(\d+)\.csv'),
]
SOURCES_TSP50 = [
    # Stage 3 Phase C.3/C.4 outputs (already instrumented; preferred when present)
    ('mcts', 'rollout',
     'outputs/stage3/tsp50_K*_rollout.csv',
     r'tsp50_K(\d+)_rollout\.csv'),
    ('mcts', 'value_head',
     'outputs/stage3/tsp50_K*_value_head.csv',
     r'tsp50_K(\d+)_value_head\.csv'),
    # Stage 2 reuse (ordered so _clean wins over _canonical for K=100 rollout).
    ('mcts', 'rollout',
     'outputs/stage2/tsp50_K*_rollout_clean.csv',
     r'tsp50_K(\d+)_rollout_clean\.csv'),
    ('mcts', 'rollout',
     'outputs/stage2/tsp50_K*_rollout_canonical.csv',
     r'tsp50_K(\d+)_rollout_canonical\.csv'),
    ('mcts', 'value_head',
     'outputs/stage2/tsp50_K*_canonical.csv',
     r'tsp50_K(\d+)_canonical\.csv'),
    ('sampling', '-',
     'outputs/stage3/tsp50_sampling_K*.csv',
     r'tsp50_sampling_K(\d+)\.csv'),
]
SOURCES_TSP100 = [
    ('mcts', 'rollout',
     'outputs/stage3/tsp100_K*_rollout_canonical.csv',
     r'tsp100_K(\d+)_rollout_canonical\.csv'),
    ('sampling', '-',
     'outputs/stage3/tsp100_sampling_K*.csv',
     r'tsp100_sampling_K(\d+)\.csv'),
]
SOURCES_BY_GRAPH = {20: SOURCES_TSP20, 50: SOURCES_TSP50, 100: SOURCES_TSP100}


def _read_csv(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _per_instance_costs(rows: list[dict]) -> tuple[list[float], list[float]]:
    """Return (greedy_costs, search_costs)."""
    g = [float(r['greedy_cost']) for r in rows]
    if rows and 'mcts_cost' in rows[0]:
        s = [float(r['mcts_cost']) for r in rows]
    elif rows and 'sample_cost' in rows[0]:
        s = [float(r['sample_cost']) for r in rows]
    else:
        raise ValueError(f"row schema has no mcts_cost or sample_cost: {rows[0] if rows else 'empty'}")
    return g, s


def _decode_steps_from_csv(rows: list[dict]) -> Optional[float]:
    if not rows or 'decode_steps' not in rows[0]:
        return None
    counts = [int(r['decode_steps']) for r in rows]
    return statistics.mean(counts)


def _wall_clock_ms_from_csv(rows: list[dict]) -> Optional[tuple[float, float]]:
    """Return (median, mean) wall-clock ms per instance, or None if unavailable.

    Median is preferred for the headline plot because the laptop GPU
    occasionally stalls for one instance (suspend / driver / context-switch),
    producing multi-second outliers that swamp the arithmetic mean.
    """
    if not rows or 'wall_clock_ms' not in rows[0]:
        return None
    times = [float(r['wall_clock_ms']) for r in rows]
    return statistics.median(times), statistics.mean(times)


def _decode_steps_from_cache(cache: dict, graph_size: int, leaf_eval: str, K: int,
                              method: str) -> Optional[float]:
    """Look up a probed decode_steps mean from the cache JSON."""
    if method == 'sampling':
        # Sampling decode_steps is analytic: width × graph_size.
        return float(K * graph_size)
    if method == 'greedy':
        return float(graph_size)
    if method == 'mcts':
        bucket = cache.get(f'tsp{graph_size}', {})
        key = f'{leaf_eval}_K{K}'
        if key in bucket:
            return float(bucket[key]['decode_steps_mean'])
    return None


def _aggregate_one(method: str, leaf_eval: str, K: int, csv_path: str,
                    cache: dict, graph_size: int, optimum: Optional[float]) -> dict:
    rows = _read_csv(csv_path)
    n = len(rows)
    g_costs, s_costs = _per_instance_costs(rows)
    g_mean = statistics.mean(g_costs)
    s_mean = statistics.mean(s_costs)
    s_std = statistics.stdev(s_costs) if n > 1 else 0.0

    # Forward-pass budget.
    decode_steps = _decode_steps_from_csv(rows)
    if decode_steps is None:
        decode_steps = _decode_steps_from_cache(cache, graph_size, leaf_eval, K, method)
    if decode_steps is None:
        print(f"WARN: {csv_path}: no decode_steps in CSV or cache; "
              f"emit row with decode_steps_mean=NaN", file=sys.stderr)
        decode_steps = float('nan')

    # Wall-clock per instance (only present in cpp re-runs and Phase-A-instrumented sampling).
    wc_pair = _wall_clock_ms_from_csv(rows)
    if wc_pair is None:
        wall_clock_ms_median = float('nan')
        wall_clock_ms_mean = float('nan')
    else:
        wall_clock_ms_median, wall_clock_ms_mean = wc_pair

    # Quality metrics.
    if optimum is not None and optimum > 0:
        gap_to_opt_pct = (s_mean - optimum) / optimum * 100.0
        greedy_gap_pct = (g_mean - optimum) / optimum * 100.0
        if greedy_gap_pct != 0:
            gap_reduction_pct = (greedy_gap_pct - gap_to_opt_pct) / greedy_gap_pct * 100.0
        else:
            gap_reduction_pct = float('nan')
    else:
        gap_to_opt_pct = float('nan')
        gap_reduction_pct = float('nan')

    wins = sum(1 for gi, si in zip(g_costs, s_costs) if si < gi - 1e-9)
    ties = sum(1 for gi, si in zip(g_costs, s_costs) if abs(si - gi) <= 1e-9)
    win_rate = wins / n
    tie_rate = ties / n

    return {
        'method': method,
        'leaf_eval': leaf_eval,
        'K': K,
        'decode_steps_mean': decode_steps,
        'wall_clock_ms_median': wall_clock_ms_median,
        'wall_clock_ms_mean': wall_clock_ms_mean,
        'mean_cost': s_mean,
        'std_cost': s_std,
        'mean_gap_to_opt_pct': gap_to_opt_pct,
        'gap_reduction_vs_greedy_pct': gap_reduction_pct,
        'win_rate_vs_greedy': win_rate,
        'tie_rate_vs_greedy': tie_rate,
        'n_instances': n,
        'source_csv': csv_path.replace('\\', '/'),
        'greedy_mean_in_csv': g_mean,
    }


def _greedy_row(graph_size: int, optimum: Optional[float], n: int, g_mean: float) -> dict:
    """Synthesize the greedy row from any source CSV's greedy_cost column."""
    if optimum is not None and optimum > 0:
        gap_to_opt_pct = (g_mean - optimum) / optimum * 100.0
    else:
        gap_to_opt_pct = float('nan')
    return {
        'method': 'greedy',
        'leaf_eval': '-',
        'K': 1,
        'decode_steps_mean': float(graph_size),
        'wall_clock_ms_median': float('nan'),
        'wall_clock_ms_mean': float('nan'),
        'mean_cost': g_mean,
        'std_cost': float('nan'),
        'mean_gap_to_opt_pct': gap_to_opt_pct,
        'gap_reduction_vs_greedy_pct': 0.0,
        'win_rate_vs_greedy': 0.0,
        'tie_rate_vs_greedy': 1.0,
        'n_instances': n,
        'source_csv': '(synthesized from greedy_cost column)',
        'greedy_mean_in_csv': g_mean,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate Stage 2 + Stage 3 CSVs into a comparison table")
    parser.add_argument('--graph_size', type=int, required=True, choices=[20, 50, 100])
    parser.add_argument('--optimum', type=float, default=None,
                        help='Optimum reference for gap-to-opt %% (TSP-20 default 3.8279 from Stage 0 Gurobi)')
    parser.add_argument('--optima_csv', type=str, default=None,
                        help='Optional CSV with per-instance optima (e.g. from run_optima.py). '
                             'Mean of optimum_cost column is used as --optimum if --optimum is unset.')
    parser.add_argument('--probe_cache', type=str,
                        default='outputs/stage3/decode_step_cache.json',
                        help='JSON cache produced by probe_decode_steps.py')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='If set, write comparison CSV here; otherwise print to stdout')
    parser.add_argument('--repo_root', type=str, default='.',
                        help='Repository root for resolving CSV glob patterns')
    args = parser.parse_args()

    # Default optima (Stage 0 Gurobi for TSP-20; others to be filled by C.1 / D.2)
    if args.optimum is None and args.optima_csv:
        with open(args.optima_csv) as f:
            opt_rows = list(csv.DictReader(f))
        opts = [float(r['optimum_cost']) for r in opt_rows]
        args.optimum = statistics.mean(opts)
        print(f"Loaded {len(opts)} optima from {args.optima_csv}: mean={args.optimum:.4f}")
    if args.optimum is None and args.graph_size == 20:
        args.optimum = 3.8279

    sources = SOURCES_BY_GRAPH[args.graph_size]

    # Load probe cache.
    if os.path.exists(args.probe_cache):
        with open(args.probe_cache) as f:
            cache = json.load(f)
    else:
        cache = {}
        if any(method == 'mcts' for method, _, _, _ in sources):
            print(f"WARN: probe cache {args.probe_cache} missing; "
                  f"MCTS decode_steps will be NaN", file=sys.stderr)

    # Discover all matching CSVs and aggregate.
    rows_out = []
    seen_keys: set[tuple] = set()
    greedy_n = 0
    greedy_mean: Optional[float] = None

    for method, leaf_eval, glob_pat, regex_pat in sources:
        full_glob = os.path.join(args.repo_root, glob_pat)
        matches = sorted(glob.glob(full_glob))
        for path in matches:
            rel_path = os.path.relpath(path, args.repo_root).replace('\\', '/')
            m = re.search(regex_pat, rel_path)
            if not m:
                continue
            K = int(m.group(1))
            key = (method, leaf_eval, K)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            row = _aggregate_one(method, leaf_eval, K, path, cache,
                                  args.graph_size, args.optimum)
            rows_out.append(row)
            # Track greedy mean from any source.
            if greedy_mean is None:
                greedy_mean = row['greedy_mean_in_csv']
                greedy_n = row['n_instances']

    # Add a synthetic greedy row at K=1, decode_steps=N.
    if greedy_mean is not None:
        rows_out.insert(0, _greedy_row(args.graph_size, args.optimum, greedy_n, greedy_mean))

    # Sort: greedy first, then sampling by K, then MCTS rollout by K, then MCTS value_head by K.
    method_order = {'greedy': 0, 'sampling': 1, 'mcts': 2}
    leaf_order = {'-': 0, 'rollout': 0, 'value_head': 1}
    rows_out.sort(key=lambda r: (method_order[r['method']], leaf_order[r['leaf_eval']], r['K']))

    # Print summary.
    print(f"\n=== Stage 3 aggregate for TSP-{args.graph_size} ===")
    if args.optimum is not None:
        print(f"Optimum reference: {args.optimum:.4f}")
    if greedy_mean is None:
        print("Greedy baseline:   (no source CSVs found)")
        if args.output_csv:
            print("Nothing to aggregate. Exiting.", file=sys.stderr)
        return 1
    if args.optimum is not None:
        gap_g = (greedy_mean - args.optimum) / args.optimum * 100.0
        print(f"Greedy baseline:   {greedy_mean:.4f} (gap to opt: {gap_g:.3f}%)")
    else:
        print(f"Greedy baseline:   {greedy_mean:.4f}")
    header = ['method', 'leaf_eval', 'K', 'decode_steps_mean', 'wall_ms_med',
              'mean_cost', 'gap_to_opt_pct', 'gap_red_pct', 'win_rate', 'n']
    fmt = "{:8s} {:11s} {:>5} {:>12.1f} {:>12.1f} {:>9.4f} {:>10.3f} {:>10.2f} {:>9.3f} {:>5}"
    print()
    print('  '.join(f'{h:>10s}' for h in header))
    for r in rows_out:
        print(fmt.format(
            r['method'], r['leaf_eval'], r['K'],
            r['decode_steps_mean'], r['wall_clock_ms_median'],
            r['mean_cost'],
            r['mean_gap_to_opt_pct'], r['gap_reduction_vs_greedy_pct'],
            r['win_rate_vs_greedy'], r['n_instances'],
        ))

    # Write CSV.
    if args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)) or '.', exist_ok=True)
        with open(args.output_csv, 'w', newline='') as f:
            w = csv.writer(f)
            cols = ['method', 'leaf_eval', 'K', 'decode_steps_mean',
                    'wall_clock_ms_median', 'wall_clock_ms_mean',
                    'mean_cost', 'std_cost',
                    'mean_gap_to_opt_pct', 'gap_reduction_vs_greedy_pct',
                    'win_rate_vs_greedy', 'tie_rate_vs_greedy',
                    'n_instances', 'source_csv']
            w.writerow(cols)
            for r in rows_out:
                w.writerow([
                    r['method'], r['leaf_eval'], r['K'],
                    f"{r['decode_steps_mean']:.2f}",
                    f"{r['wall_clock_ms_median']:.3f}",
                    f"{r['wall_clock_ms_mean']:.3f}",
                    f"{r['mean_cost']:.6f}",
                    f"{r['std_cost']:.6f}",
                    f"{r['mean_gap_to_opt_pct']:.4f}",
                    f"{r['gap_reduction_vs_greedy_pct']:.3f}",
                    f"{r['win_rate_vs_greedy']:.4f}",
                    f"{r['tie_rate_vs_greedy']:.4f}",
                    r['n_instances'],
                    r['source_csv'],
                ])
        print(f"\nWrote {args.output_csv}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
