"""Stage 4 headline plot — sample-efficiency on TSP-20.

Reads:
  - Stage 4: `iterations.csv` from a coach run dir (per-iteration row).
  - Stage 1: `epochs.csv` from `outputs/tsp_20/stage1_tsp20_canonical_*/`.

Produces:
  - PNG at `--out` (default `outputs/stage4/figures/sample_efficiency_tsp20.png`).
  - x-axis: total_instances (log scale).
  - y-axis: val_avg_cost.
  - Two curves: Stage 1 REINFORCE (epoch checkpoints) + Stage 4 (per-iteration).
  - Reference horizontal lines:
      * Gurobi optimum            = 3.8279
      * Stage 1 final canonical   = 3.83943
      * Stage 3 K=400 rollout MCTS = 3.8312

Stage-1 cumulative-instances axis: each Stage 1 epoch processes
`epoch_size` instances (default 1_280_000). We read `epoch_size` and
`batch_size` from the run's `args.json` if present; otherwise we fall back to
defaults that match the canonical run and warn.

Self-test mode `--smoke`:
  Fabricates a synthetic 5-row iterations.csv + 5-row epochs.csv inside a
  temp dir, calls the plotter, and verifies the PNG file is written and
  non-empty. Intended for CI / dev-box smoke without a real Stage 4 run.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile


# Reference annotations (from plan F.5).
_GUROBI_OPT = 3.8279
_STAGE1_FINAL_CANONICAL = 3.83943
_STAGE3_K400_ROLLOUT = 3.8312

# Stage-1 canonical defaults (used as fallback if args.json is absent).
_STAGE1_DEFAULT_EPOCH_SIZE = 1_280_000


def _read_csv_rows(path):
    """Read a CSV file as a list of dicts (header-keyed)."""
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)


def _stage1_total_instances(rows, epoch_size):
    """Map each Stage-1 epoch row to (cumulative_instances, val_avg_cost).

    Cumulative instances after epoch e = (e + 1) * epoch_size, since epoch e
    processes one full epoch_size-many TSP instances. (Epoch numbering in
    `epochs.csv` is 0-based by Stage 1 convention.)
    """
    out = []
    for row in rows:
        if row.get('val_avg_cost', '') in ('', None):
            continue
        try:
            e = int(row['epoch'])
            v = float(row['val_avg_cost'])
        except (TypeError, ValueError):
            continue
        out.append(((e + 1) * int(epoch_size), v))
    return out


def _stage4_total_instances(rows):
    """Map each Stage-4 iteration row to (total_instances, val_avg_cost)."""
    out = []
    for row in rows:
        if row.get('val_avg_cost', '') in ('', None):
            continue
        try:
            ti = int(row['total_instances'])
            v = float(row['val_avg_cost'])
        except (TypeError, ValueError):
            continue
        out.append((ti, v))
    return out


def _read_stage1_epoch_size(stage1_dir):
    """Read `epoch_size` from `args.json`, else fall back to the canonical default."""
    args_path = os.path.join(stage1_dir, 'args.json')
    if os.path.exists(args_path):
        try:
            with open(args_path) as f:
                args = json.load(f)
            return int(args.get('epoch_size', _STAGE1_DEFAULT_EPOCH_SIZE))
        except Exception as e:
            print(f'Warning: failed to read {args_path}: {e}; using default '
                  f'epoch_size={_STAGE1_DEFAULT_EPOCH_SIZE}')
    return _STAGE1_DEFAULT_EPOCH_SIZE


def _plot(stage1_xy, stage4_xy, out_path):
    """Render the two curves + reference lines."""
    # Local matplotlib import keeps `import plot_stage4` cheap when only
    # _read_csv_rows is needed (e.g. by tests).
    import matplotlib
    matplotlib.use('Agg')  # headless
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    if stage1_xy:
        xs1, ys1 = zip(*stage1_xy)
        ax.plot(xs1, ys1, marker='o', linestyle='-', linewidth=1.2,
                markersize=3, label='Stage 1 REINFORCE')
    if stage4_xy:
        xs4, ys4 = zip(*stage4_xy)
        ax.plot(xs4, ys4, marker='s', linestyle='-', linewidth=1.2,
                markersize=4, label='Stage 4 AGZ-style MCTS')

    # Reference horizontal lines (annotate at the rightmost x on the data).
    ax.axhline(_GUROBI_OPT, color='black', linestyle=':', linewidth=1,
               label=f'Gurobi optimum ({_GUROBI_OPT:.4f})')
    ax.axhline(_STAGE1_FINAL_CANONICAL, color='gray', linestyle='--', linewidth=1,
               label=f'Stage 1 final ({_STAGE1_FINAL_CANONICAL:.5f})')
    ax.axhline(_STAGE3_K400_ROLLOUT, color='tab:green', linestyle='--', linewidth=1,
               label=f'Stage 3 K=400 rollout MCTS ({_STAGE3_K400_ROLLOUT:.4f})')

    ax.set_xscale('log')
    ax.set_xlabel('Total instances seen (log scale)')
    ax.set_ylabel('Validation avg tour cost (TSP-20)')
    ax.set_title('Sample efficiency: Stage 1 REINFORCE vs. Stage 4 AGZ-style MCTS')
    ax.grid(True, which='both', linestyle=':', alpha=0.4)
    ax.legend(loc='best', fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_plot(stage1_dir, stage4_dir, out_path):
    """Driver: read CSVs, project x-axis, render PNG."""
    stage1_csv = os.path.join(stage1_dir, 'epochs.csv')
    stage4_csv = os.path.join(stage4_dir, 'iterations.csv')

    if not os.path.exists(stage1_csv):
        raise FileNotFoundError(f'Stage 1 epochs.csv not found at {stage1_csv}')
    if not os.path.exists(stage4_csv):
        raise FileNotFoundError(f'Stage 4 iterations.csv not found at {stage4_csv}')

    epoch_size = _read_stage1_epoch_size(stage1_dir)
    stage1_xy = _stage1_total_instances(_read_csv_rows(stage1_csv), epoch_size)
    stage4_xy = _stage4_total_instances(_read_csv_rows(stage4_csv))

    _plot(stage1_xy, stage4_xy, out_path)
    return out_path


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------


def _smoke():
    """Fabricate tiny CSVs + run make_plot; assert PNG written and non-empty."""
    print('plot_stage4 smoke: synthesizing tiny CSVs and rendering a PNG ...')

    with tempfile.TemporaryDirectory() as tmp:
        stage1_dir = os.path.join(tmp, 'stage1')
        stage4_dir = os.path.join(tmp, 'stage4')
        os.makedirs(stage1_dir)
        os.makedirs(stage4_dir)

        # Synthetic Stage 1 epochs.csv — header matches MetricsLogger schema.
        with open(os.path.join(stage1_dir, 'epochs.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                'epoch', 'val_avg_cost', 'epoch_duration', 'lr', 'baseline_updated',
                'val_value_r2_overall', 'val_value_r2_early', 'val_value_r2_mid',
                'val_value_r2_late', 'val_value_loss', 'val_value_residual_mean',
                'val_value_mean', 'val_target_mean',
            ])
            for e, v in enumerate([3.96, 3.91, 3.87, 3.85, 3.84]):
                w.writerow([e, v, 100.0, 1e-4, '',
                            '', '', '', '', '', '', '', ''])
        # Synthetic args.json so the script reads epoch_size and doesn't fall
        # back to the canonical default.
        with open(os.path.join(stage1_dir, 'args.json'), 'w') as f:
            json.dump({'epoch_size': 12800, 'batch_size': 64}, f)

        # Synthetic Stage 4 iterations.csv — header matches log_iteration schema.
        with open(os.path.join(stage4_dir, 'iterations.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                'iter', 'total_instances', 'val_avg_cost',
                'policy_loss_mean', 'value_loss_mean', 'mean_entropy_pi',
                'gated', 'accepted', 'mcts_wall_s', 'train_wall_s', 'buffer_size',
            ])
            for i, (ti, v) in enumerate([
                (1000, 3.85), (2000, 3.84), (3000, 3.835),
                (4000, 3.83), (5000, 3.829),
            ]):
                w.writerow([i, ti, v, 1.0, 0.5, 1.5, 0, '', 1.0, 0.1, ti])

        out_path = os.path.join(tmp, 'figures', 'smoke.png')
        make_plot(stage1_dir, stage4_dir, out_path)

        assert os.path.exists(out_path), f'PNG not written at {out_path}'
        size = os.path.getsize(out_path)
        assert size > 0, f'PNG file is empty: {out_path}'
        print(f'  smoke PNG written: {out_path} ({size} bytes)')
    print('plot_stage4 smoke OK')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Stage 4 sample-efficiency plot')
    p.add_argument('--stage1_dir', type=str, default=None,
                   help='Directory containing Stage 1 epochs.csv (and args.json).')
    p.add_argument('--stage4_dir', type=str, default=None,
                   help='Directory containing Stage 4 iterations.csv.')
    p.add_argument('--out', type=str,
                   default='outputs/stage4/figures/sample_efficiency_tsp20.png',
                   help='Output PNG path.')
    p.add_argument('--smoke', action='store_true',
                   help='Run the synthetic-data self-test (no real run dirs needed).')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.smoke:
        _smoke()
        return 0
    if not args.stage1_dir or not args.stage4_dir:
        print('Error: --stage1_dir and --stage4_dir are required when not running --smoke',
              file=sys.stderr)
        return 2
    out = make_plot(args.stage1_dir, args.stage4_dir, args.out)
    print(f'Wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
