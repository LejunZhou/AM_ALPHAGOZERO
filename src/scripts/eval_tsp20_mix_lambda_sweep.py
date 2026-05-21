"""Stage 5 §H Phase A — TSP-20 inference-time λ-sweep for mixed leaf eval.

For each λ in the grid, runs MCTS with `leaf_eval='mix', mix_lambda=λ`
on a single fixed Stage 4 checkpoint over a held-out TSP-20 val set
(default val_size=10000, val_seed=42 — the canonical training-time val).

Emits one CSV row per λ with mean cost, std, wall, fwd_count_{decode,value,
rollout}, and the per-instance numpy array saved separately for paired
statistics downstream.

Reference anchors (from §C.3 / §D.5 on F.6.1.6):
  λ=0 ≈ 3.834 (K=40 rollout)   λ=1 ≈ 3.868 (K=40 vh)

The script is Colab-T4-friendly: K=40 on 10000 instances takes ~10-15 min
per λ, so the full 5-λ grid is ~1.5-2 h.

Usage:
  PYTHONPATH=src python -m scripts.eval_tsp20_mix_lambda_sweep \\
      --ckpt outputs/tsp_20/f616_400iter_step_decay_.../iter-361_accepted.pt \\
      --K 40 --val_size 10000 --val_seed 42 \\
      --lambdas 0.0,0.25,0.5,0.75,1.0 \\
      --out_csv _progress/eval_logs/tsp20_mix_lambda_sweep_K40.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from am_baseline.problem.tsp import TSP
from am_baseline.search.mcts import MCTSConfig
from am_baseline.utils.misc import load_args, torch_load_cpu

from scripts.val_stage4_mcts import load_model as load_s4_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True,
                   help='Stage 4 checkpoint with a trained value head '
                        '(e.g. F.6.1.6 best_model).')
    p.add_argument('--ckpt_key', type=str, default='best_model',
                   choices=['best_model', 'model'],
                   help='State-dict key inside the .pt to load.')
    p.add_argument('--K', type=int, default=40,
                   help='MCTS simulations per root.')
    p.add_argument('--val_size', type=int, default=10000)
    p.add_argument('--val_seed', type=int, default=42,
                   help='Val set RNG seed. Default 42 matches Stage 4 training-time val.')
    p.add_argument('--lambdas', type=str, default='0.0,0.25,0.5,0.75,1.0',
                   help='Comma-separated list of mix_lambda values to sweep.')
    p.add_argument('--c_puct', type=float, default=0.05)
    p.add_argument('--device', type=str, default='cuda',
                   choices=['cuda', 'cpu'])
    p.add_argument('--mcts_batch_size', type=int, default=1000,
                   help='Cross-instance batch (CppBatchMCTSSolver chunk). '
                        '1000 fits A10/T4; lower if OOM.')
    p.add_argument('--mcts_seed', type=int, default=20260430,
                   help='MCTS RNG seed (Dirichlet, τ-sampling). Eval default is '
                        'ε=0 τ=0 so this is rarely consulted.')
    p.add_argument('--out_csv', type=str, default=None,
                   help='Where to write the result CSV. Defaults to a path '
                        'inside the checkpoint directory.')
    return p.parse_args()


def main() -> None:
    opts = parse_args()

    lambdas = [float(s.strip()) for s in opts.lambdas.split(',') if s.strip()]
    for lam in lambdas:
        if not (0.0 <= lam <= 1.0):
            raise ValueError(f'mix_lambda={lam} outside [0, 1]')

    device = torch.device(
        opts.device if (opts.device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    )
    print(f'device={device}  ckpt={opts.ckpt}')

    # Read sibling args.json for architecture; required by load_s4_model.
    ckpt_dir = os.path.dirname(os.path.abspath(opts.ckpt))
    train_args_path = os.path.join(ckpt_dir, 'args.json')
    train_args = load_args(train_args_path) if os.path.exists(train_args_path) else None
    if train_args is None:
        print(f'[warn] no args.json next to {opts.ckpt}; using AM defaults.')

    # Build val set with the canonical train-time seed.
    torch.manual_seed(opts.val_seed)
    np.random.seed(opts.val_seed)
    dataset = TSP.make_dataset(size=20, num_samples=opts.val_size)
    coords = torch.stack([x for x in dataset]).to(device)
    print(f'val: {opts.val_size} instances at seed={opts.val_seed}, K={opts.K}')

    # Load checkpoint (value-head required for mix mode).
    ckpt = torch_load_cpu(opts.ckpt)
    model = load_s4_model(ckpt, opts.ckpt_key, train_args=train_args, device=device)
    if model.value_head is None:
        raise RuntimeError(
            "Checkpoint has no value_head; mix mode requires one. "
            "Pass a Stage 4 checkpoint trained with value_enabled=True."
        )

    out_csv = opts.out_csv
    if out_csv is None:
        out_csv = os.path.join(ckpt_dir, f'mix_lambda_sweep_K{opts.K}.csv')
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or '.', exist_ok=True)

    out_npz = out_csv.replace('.csv', '.npz')
    per_lambda_costs: dict[str, np.ndarray] = {}

    rows = []
    for lam in lambdas:
        cfg = MCTSConfig(
            n_simulations=opts.K,
            simulation_batch_size=1,
            c_puct=opts.c_puct,
            temperature=0.0,
            temperature_schedule=None,
            dirichlet_alpha=10.0 / 20,
            dirichlet_epsilon=0.0,
            leaf_eval='mix',
            mix_lambda=lam,
            value_norm='bl',
            value_target_norm=str((train_args or {}).get('value_target_norm', 'none')),
            fpu_mode='running_q',
            fpu_fallback=-1.0,
            root_select='visits',
            tree_reuse=True,
            return_root_visits=False,
            seed=opts.mcts_seed,
        )

        # Use the cross-instance C++ batched solver — same path that drives
        # production self-play / val.
        from am_baseline.search.mcts_cpp.solver import CppBatchMCTSSolver
        solver = CppBatchMCTSSolver(
            model, cfg, device=device, mcts_batch_size=opts.mcts_batch_size
        )

        t0 = time.time()
        costs, _tours = solver.solve_batch(coords)
        wall = time.time() - t0

        costs_np = costs.detach().cpu().numpy().astype(np.float64)
        mean = float(costs_np.mean())
        std = float(costs_np.std(ddof=1))
        se = std / float(np.sqrt(len(costs_np)))
        decode = int(getattr(solver, 'fwd_count_decode', 0))
        value = int(getattr(solver, 'fwd_count_value', 0))
        rollout = int(getattr(solver, 'fwd_count_rollout', 0))
        cache_hits = int(getattr(solver, 'eval_cache_hits', 0))
        cache_misses = int(getattr(solver, 'eval_cache_misses', 0))

        print(f'  λ={lam:.2f}  mean={mean:.5f}  SE={se:.5f}  wall={wall:.1f}s  '
              f'decode={decode}  value={value}  rollout={rollout}')

        rows.append({
            'mix_lambda': lam,
            'K': opts.K,
            'val_size': opts.val_size,
            'val_seed': opts.val_seed,
            'mean_cost': mean,
            'std_cost': std,
            'se_cost': se,
            'wall_s': wall,
            'fwd_decode': decode,
            'fwd_value': value,
            'fwd_rollout': rollout,
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'ckpt': os.path.abspath(opts.ckpt),
        })
        per_lambda_costs[f'lambda_{lam:.4f}'] = costs_np

    fieldnames = list(rows[0].keys())
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f'\n[done] wrote {out_csv}')

    np.savez(out_npz, **per_lambda_costs)
    print(f'[done] wrote per-instance arrays to {out_npz}')


if __name__ == '__main__':
    main()
