"""Stage 3 headline plot: optimality gap vs. forward-pass budget.

Reads `outputs/stage3/comparison_tsp{N}.csv` produced by aggregate_stage3.py
and renders one figure per graph size. The figure is the proposal's
"key metric" (proposal.md:114): three curves on a log-scaled
decode_steps x-axis — sampling-K, MCTS rollout-K, MCTS value_head-K — with
greedy and (if known) sampling-1280 marked as reference points.

Usage:
  PYTHONPATH=src python -m scripts.plot_stage3 \
      --graph_size 20 \
      --comparison_csv outputs/stage3/comparison_tsp20.csv \
      --output_png outputs/stage3/figures/budget_curve_tsp20.png
"""
import argparse
import csv
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SERIES_STYLE = {
    ('sampling', '-'): {
        'label': 'Sampling-K',
        'color': '#666666',
        'marker': 's',
        'linestyle': '--',
    },
    ('mcts', 'rollout'): {
        'label': 'MCTS (rollout leaf)',
        'color': '#d62728',
        'marker': 'o',
        'linestyle': '-',
    },
    ('mcts', 'value_head'): {
        'label': 'MCTS (value_head leaf)',
        'color': '#1f77b4',
        'marker': '^',
        'linestyle': '-',
    },
}


def _load(path: str, require_x: str = 'decode_steps') -> list[dict]:
    """Load comparison CSV. Rows missing the requested x-axis are dropped.

    require_x is one of 'decode_steps' or 'wall_clock_ms' — only rows with a
    finite value for that field are kept (greedy synthesises decode_steps but
    not wall_clock_ms, so it is dropped from the wall-clock view).

    For wall_clock_ms the **median** column is used (the laptop GPU
    occasionally stalls one instance for seconds, polluting the mean).
    """
    with open(path) as f:
        rows = list(csv.DictReader(f))
    parsed = []
    for r in rows:
        try:
            ds = float(r['decode_steps_mean'])
        except (ValueError, KeyError):
            ds = float('nan')
        # Prefer median for wall-clock; fall back to mean for older comparison CSVs.
        wc_raw = r.get('wall_clock_ms_median') or r.get('wall_clock_ms_mean') or 'nan'
        try:
            wc = float(wc_raw)
        except ValueError:
            wc = float('nan')
        if require_x == 'decode_steps' and (ds != ds):  # NaN check
            continue
        if require_x == 'wall_clock_ms' and (wc != wc):
            continue
        parsed.append({
            'method': r['method'],
            'leaf_eval': r['leaf_eval'],
            'K': int(r['K']),
            'decode_steps': ds,
            'wall_clock_ms': wc,
            'mean_cost': float(r['mean_cost']),
            'gap_to_opt_pct': float(r['mean_gap_to_opt_pct']),
            'gap_red_pct': float(r['gap_reduction_vs_greedy_pct']),
            'win_rate': float(r['win_rate_vs_greedy']),
            'n': int(r['n_instances']),
        })
    return parsed


def _series(rows: list[dict], method: str, leaf: str, x_key: str) -> list[dict]:
    s = [r for r in rows if r['method'] == method and r['leaf_eval'] == leaf]
    s.sort(key=lambda r: r[x_key])
    return s


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 3 budget-curve plot")
    parser.add_argument('--graph_size', type=int, required=True)
    parser.add_argument('--comparison_csv', type=str, required=True)
    parser.add_argument('--output_png', type=str, required=True)
    parser.add_argument('--y_metric', choices=['gap_to_opt_pct', 'gap_red_pct'],
                        default='gap_to_opt_pct',
                        help="Y-axis: 'gap_to_opt_pct' (lower better) or "
                             "'gap_red_pct' (higher better)")
    parser.add_argument('--x_metric', choices=['decode_steps', 'wall_clock_ms'],
                        default='decode_steps',
                        help="X-axis: 'decode_steps' (forward passes per instance) "
                             "or 'wall_clock_ms' (mean wall-clock ms per instance, "
                             "RTX 4060 Laptop GPU). Sampling and MCTS-cpp are timed "
                             "on the same hardware.")
    parser.add_argument('--title', type=str, default=None)
    args = parser.parse_args()

    rows = _load(args.comparison_csv, require_x=args.x_metric)
    if not rows:
        print(f"No data in {args.comparison_csv}", file=sys.stderr)
        return 1

    # Greedy reference (synthesized row, K=1, method=greedy).
    greedy_row = next((r for r in rows if r['method'] == 'greedy'), None)

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=140)

    for (method, leaf), style in SERIES_STYLE.items():
        s = _series(rows, method, leaf, args.x_metric)
        if not s:
            continue
        xs = [r[args.x_metric] for r in s]
        ys = [r[args.y_metric] for r in s]
        ax.plot(xs, ys, **style, markersize=6, linewidth=1.5)
        # K labels next to each point.
        for r in s:
            ax.annotate(
                f"K={r['K']}",
                xy=(r[args.x_metric], r[args.y_metric]),
                xytext=(4, 4),
                textcoords='offset points',
                fontsize=7,
                color=style['color'],
                alpha=0.8,
            )

    # Greedy reference (horizontal line at greedy's y).
    if greedy_row is not None:
        gy = greedy_row[args.y_metric]
        ax.axhline(gy, color='black', linestyle=':', linewidth=1.0, alpha=0.6)
        ax.annotate(
            f"greedy ({gy:.3f}%)",
            xy=(rows[-1][args.x_metric], gy),
            xytext=(0, 4),
            textcoords='offset points',
            fontsize=8,
            color='black',
            alpha=0.7,
            ha='right',
        )

    ax.set_xscale('log')
    if args.x_metric == 'decode_steps':
        ax.set_xlabel('Forward passes per instance (decode_steps)')
    else:
        ax.set_xlabel('Wall-clock per instance (ms, RTX 4060 Laptop GPU)')
    ylabel = ('Optimality gap (% above optimum) — lower is better'
              if args.y_metric == 'gap_to_opt_pct'
              else 'Gap reduction vs greedy (%)')
    ax.set_ylabel(ylabel)
    axis_label = ('forward-pass budget' if args.x_metric == 'decode_steps'
                  else 'wall-clock budget')
    title = args.title or (
        f"TSP-{args.graph_size}: search-efficiency curve "
        f"(MCTS vs. sampling, 1000 instances seed=1234) — {axis_label}"
    )
    ax.set_title(title)
    ax.grid(True, which='major', alpha=0.3)
    ax.grid(True, which='minor', alpha=0.15)
    ax.legend(loc='best', framealpha=0.9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_png)) or '.', exist_ok=True)
    fig.savefig(args.output_png)
    print(f"Wrote {args.output_png}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
