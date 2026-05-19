"""TSP-50 batch eval: Gurobi vs AM greedy vs AM sampling vs MCTS K=25 (rollout).

Mirrors the per-instance logic in `notebooks/tsp50_demo.ipynb` but scales to 1000
instances and reports a single comparison table with optimality gap vs Gurobi.

Default checkpoint: Stage 1 AM baseline (no value head). MCTS uses leaf_eval=
rollout with value_norm=bl, so the value head is not required.

Usage:
    python src/scripts/eval_tsp50_1000_K25.py \
        --am_ckpt outputs/tsp_50/stage1_tsp50_am_baseline_20260424T032356/epoch-99.pt \
        --num_test 1000 --seed 1234 --K 25 --sample_width 1280

The script prints a comparison table:
    Method                          Avg Cost     Gap%      Time
    -----------------------------------------------------------
    Gurobi (optimal)                  ...        0.000%    ...
    AM greedy                         ...        ...%      ...
    AM sample(x1280)                  ...        ...%      ...
    MCTS K=25 rollout                 ...        ...%      ...
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch
from tqdm import tqdm

from am_baseline.problem.tsp import TSP
from am_baseline.search.mcts import MCTSConfig
from am_baseline.search.mcts_cpp.solver import CppBatchMCTSSolver
from scripts.eval_baselines import solve_gurobi
from scripts.val_stage4_mcts import (
    load_am_model,
    greedy_eval,
    sample_eval,
)


def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--am_ckpt', type=str, required=True,
                   help='Path to AM checkpoint (Stage 1 canonical or reference release).')
    p.add_argument('--graph_size', type=int, default=50)
    p.add_argument('--num_test', type=int, default=1000)
    p.add_argument('--seed', type=int, default=1234,
                   help='Seed for instance generation. Matches notebook default (1234).')

    # MCTS knobs — Stage 2 canonical (rollout + value_norm=bl + running_q FPU).
    p.add_argument('--K', type=int, default=25, help='MCTS simulations per tour-step.')
    p.add_argument('--c_puct', type=float, default=0.05)
    p.add_argument('--mcts_batch_size', type=int, default=64,
                   help='Cross-instance MCTS batch size (C++ scheduler).')

    # AM sampling knobs (paper-style best-of-K test-time search).
    p.add_argument('--sample_width', type=int, default=1280)
    p.add_argument('--sample_batch_rep', type=int, default=128)
    p.add_argument('--sample_outer_batch', type=int, default=64)
    p.add_argument('--no_sample', action='store_true',
                   help='Skip AM sampling (faster sanity runs).')

    # Skips for partial runs.
    p.add_argument('--no_gurobi', action='store_true',
                   help='Skip Gurobi (use AM greedy as the reference instead).')

    # Runtime.
    p.add_argument('--no_cuda', action='store_true')
    p.add_argument('--batch_size', type=int, default=2048,
                   help='Greedy-eval batch size.')

    # Output.
    p.add_argument('--out_csv', type=str, default=None,
                   help='Optional CSV path for per-instance costs (Gurobi/greedy/sample/MCTS).')
    return p.parse_args()


def main():
    opts = parse()
    device = torch.device('cuda' if torch.cuda.is_available() and not opts.no_cuda else 'cpu')
    print(f'device     = {device}')
    print(f'am_ckpt    = {opts.am_ckpt}')
    print(f'graph_size = {opts.graph_size}, num_test = {opts.num_test}, seed = {opts.seed}')
    print(f'MCTS K     = {opts.K}, leaf_eval = rollout, c_puct = {opts.c_puct}')

    # -------------------------------------------------------------------------
    # 1) Build the pinned val set (matches notebook seed convention).
    # -------------------------------------------------------------------------
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    instances = TSP.make_dataset(size=opts.graph_size, num_samples=opts.num_test)
    coords_cpu = torch.stack([x for x in instances])  # (N_test, n_nodes, 2) on cpu

    # -------------------------------------------------------------------------
    # 2) Gurobi exact (sequential, single-threaded per instance — matches notebook).
    # -------------------------------------------------------------------------
    results = {}  # name -> (cost_array, elapsed_s)
    if not opts.no_gurobi:
        print('\n[*] Gurobi (exact)...')
        t0 = time.time()
        gur_costs = np.empty(opts.num_test, dtype=np.float64)
        coords_np = coords_cpu.numpy()
        for i in tqdm(range(opts.num_test), desc='Gurobi'):
            c, _ = solve_gurobi(coords_np[i].tolist())
            gur_costs[i] = c
        elapsed = time.time() - t0
        results['Gurobi (optimal)'] = (gur_costs, elapsed)
        print(f'  Gurobi: mean = {gur_costs.mean():.5f}   SE = {gur_costs.std() / np.sqrt(len(gur_costs)):.5f}'
              f'   wall = {elapsed:.1f}s')

    # -------------------------------------------------------------------------
    # 3) Load AM checkpoint (Stage 1 baseline — no value head).
    # -------------------------------------------------------------------------
    print(f'\n[*] Loading AM checkpoint...')
    am_model = load_am_model(opts.am_ckpt, device)

    # -------------------------------------------------------------------------
    # 4) AM greedy (batched).
    # -------------------------------------------------------------------------
    print('\n[*] AM greedy...')
    t0 = time.time()
    am_greedy = greedy_eval(am_model, coords_cpu, device, batch_size=opts.batch_size)
    elapsed = time.time() - t0
    results['AM greedy'] = (am_greedy, elapsed)
    print(f'  AM greedy: mean = {am_greedy.mean():.5f}   SE = {am_greedy.std() / np.sqrt(len(am_greedy)):.5f}'
          f'   wall = {elapsed:.1f}s')

    # -------------------------------------------------------------------------
    # 5) AM sampling K=sample_width (paper-style best-of-K).
    # -------------------------------------------------------------------------
    am_sample = None
    if not opts.no_sample:
        print(f'\n[*] AM sampling x{opts.sample_width}...')
        torch.manual_seed(opts.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(opts.seed)
        t0 = time.time()
        am_sample = sample_eval(
            am_model, coords_cpu, device,
            width=opts.sample_width,
            batch_rep=opts.sample_batch_rep,
            outer_batch=opts.sample_outer_batch,
        )
        elapsed = time.time() - t0
        results[f'AM sample(x{opts.sample_width})'] = (am_sample, elapsed)
        print(f'  AM sample: mean = {am_sample.mean():.5f}   SE = {am_sample.std() / np.sqrt(len(am_sample)):.5f}'
              f'   wall = {elapsed:.1f}s ({elapsed / opts.num_test * 1000:.1f} ms/instance)')

    # -------------------------------------------------------------------------
    # 6) MCTS K=25 rollout (canonical Stage 2 config: c_puct=0.05, leaf=rollout,
    #    value_norm=bl, fpu=running_q, tree_reuse=True, eps=0). CppBatchMCTSSolver
    #    computes per-instance bl_val internally via greedy decode (matches the
    #    notebook's bl_val=cost_greedy convention).
    # -------------------------------------------------------------------------
    cfg = MCTSConfig(
        n_simulations=opts.K,
        c_puct=opts.c_puct,
        leaf_eval='rollout',
        value_norm='bl',
        fpu_mode='running_q',
        fpu_fallback=-1.0,
        root_select='visits',
        temperature=0.0,
        temperature_schedule=None,
        dirichlet_alpha=10.0 / opts.graph_size,
        dirichlet_epsilon=0.0,
        tree_reuse=True,
        return_root_visits=False,
        seed=opts.seed,
    )
    print(f'\n[*] MCTS K={opts.K} rollout (C++ batched, mcts_batch_size={opts.mcts_batch_size})...')
    solver = CppBatchMCTSSolver(am_model, cfg, device=device, mcts_batch_size=opts.mcts_batch_size)
    t0 = time.time()
    mcts_costs_t, _ = solver.solve_batch(coords_cpu.to(device))
    elapsed = time.time() - t0
    mcts_costs = mcts_costs_t.cpu().numpy()
    label = f'MCTS K={opts.K} rollout'
    results[label] = (mcts_costs, elapsed)
    print(f'  MCTS: mean = {mcts_costs.mean():.5f}   SE = {mcts_costs.std() / np.sqrt(len(mcts_costs)):.5f}'
          f'   wall = {elapsed:.1f}s ({elapsed / opts.num_test * 1000:.1f} ms/instance)')
    print(f'  fwd-pass mix: decode={solver.fwd_count_decode}  rollout={solver.fwd_count_rollout}'
          f'   batch_eval_calls={solver.batch_eval_calls}  rows={solver.batch_eval_rows}')

    # -------------------------------------------------------------------------
    # 7) Final comparison table — gap vs Gurobi (or AM greedy if --no_gurobi).
    # -------------------------------------------------------------------------
    if not opts.no_gurobi:
        ref_costs = results['Gurobi (optimal)'][0]
        ref_name = 'Gurobi'
    else:
        ref_costs = results['AM greedy'][0]
        ref_name = 'AM greedy'

    print('\n' + '=' * 78)
    print(f'TSP-{opts.graph_size}  ({opts.num_test} instances, seed={opts.seed})')
    print(f'Optimality gap vs {ref_name}:')
    print('=' * 78)
    print(f'{"Method":<32} {"Avg Cost":>10} {"SE":>9} {"Gap%":>9} {"Time (s)":>10}')
    print('-' * 78)
    for name, (costs, elapsed) in results.items():
        avg = costs.mean()
        se = costs.std() / np.sqrt(len(costs))
        gap = (avg - ref_costs.mean()) / ref_costs.mean() * 100.0
        print(f'{name:<32} {avg:>10.5f} {se:>9.5f} {gap:>+8.3f}% {elapsed:>10.1f}')
    print('=' * 78)

    # Per-instance paired diff vs reference (informative for MCTS vs greedy).
    if not opts.no_gurobi:
        print(f'\nPaired diffs vs {ref_name} (mean cost above optimum, per instance):')
        for name, (costs, _) in results.items():
            if name == 'Gurobi (optimal)':
                continue
            d = costs - ref_costs
            print(f'  {name:<32}  Δ={d.mean():+.5f}  SE={d.std() / np.sqrt(len(d)):.5f}'
                  f'  n_optimal={(d <= 1e-5).sum()}/{len(d)} ({(d <= 1e-5).mean() * 100:.1f}%)')

    # Optional CSV dump for later analysis.
    if opts.out_csv:
        import csv
        header = ['instance']
        cols = []
        for name, (costs, _) in results.items():
            header.append(name)
            cols.append(costs)
        with open(opts.out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            for i in range(opts.num_test):
                w.writerow([i] + [float(c[i]) for c in cols])
        print(f'\nWrote per-instance costs to {opts.out_csv}')


if __name__ == '__main__':
    main()
