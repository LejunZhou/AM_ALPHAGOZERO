"""TSP-20 full inference comparison: classical solvers + S1 AM + S4 K=10 ckpt.

Generates the same 1000 instances used by val_stage4_mcts.py (seed=20260430),
runs each method, captures per-instance costs, and prints a unified table with
{val_avg_cost, total wall, n_optimal/1000, gap-to-optimum %}.
"""
import os
import sys
import time
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from am_baseline.problem.tsp import TSP
from am_baseline.utils.misc import torch_load_cpu

# Reuse helpers from sibling scripts.
from eval_baselines import (
    solve_gurobi,
    solve_lkh_elkai,
    nearest_neighbour_batch,
    run_insertion,
)
from val_stage4_mcts import (
    load_am_model,
    load_model as load_s4_model,
    greedy_eval,
    sample_eval,
    mcts_eval,
    _build_mcts_config,
)


class _Opts:
    """Minimal stand-in for argparse.Namespace expected by _build_mcts_config."""
    def __init__(self, K):
        self.K = K
        self.leaf_eval = 'rollout'
        self.eps = 0.0
        self.alpha_factor = 10.0
        self.temperature_schedule = 'const'
        self.c_puct = 0.05
        self.seed = SEED
        self.match_train = False


SEED = 20260430
NUM = 1000
GRAPH = 20
S4_CKPT = 'outputs/tsp_20/tsp20_k10_lv0_step50_100iter_20260513T230051_20260513T230102/iter-99.pt'
AM_CKPT = 'outputs/tsp_20/stage1_tsp20_canonical_20260423T103541/epoch-99.pt'
AM_TSP50_CKPT = 'ref/attention-learn-to-route-master/pretrained/tsp_50/epoch-99.pt'


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  seed={SEED}  num={NUM}  graph={GRAPH}')

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dataset = TSP.make_dataset(size=GRAPH, num_samples=NUM)
    coords = torch.stack([x for x in dataset])
    data_np = [inst.numpy() for inst in dataset]

    results = {}  # name -> {'costs': np.ndarray, 'wall': float}

    # --- Gurobi (optimal reference) ---
    print('\n[*] Gurobi...')
    t0 = time.time()
    costs = []
    for inst in tqdm(data_np, desc='Gurobi'):
        c, _ = solve_gurobi(inst.tolist())
        costs.append(c)
    results['Gurobi (optimal)'] = {'costs': np.array(costs), 'wall': time.time() - t0}

    # --- LKH (elkai) ---
    print('\n[*] LKH (elkai)...')
    t0 = time.time()
    costs = []
    for inst in tqdm(data_np, desc='LKH'):
        c, _ = solve_lkh_elkai(inst)
        costs.append(c)
    results['LKH (elkai)'] = {'costs': np.array(costs), 'wall': time.time() - t0}

    # --- AM S1 canonical (trained on TSP-20): greedy + sampling x1280 ---
    print('\n[*] S1 canonical (bs=512, trained TSP-20)...')
    am_model = load_am_model(AM_CKPT, device)
    t0 = time.time()
    am_greedy = greedy_eval(am_model, coords, device, batch_size=2048)
    results['S1 greedy (TSP-20)'] = {'costs': am_greedy, 'wall': time.time() - t0}

    t0 = time.time()
    am_sample = sample_eval(am_model, coords, device, width=1280)
    results['S1 sample x1280 (TSP-20)'] = {'costs': am_sample, 'wall': time.time() - t0}

    # --- AM-paper released TSP-50 ckpt: OOD on TSP-20 (greedy + sampling x1280) ---
    print('\n[*] AM-paper released ckpt trained on TSP-50, applied to TSP-20...')
    am50_model = load_am_model(AM_TSP50_CKPT, device)
    t0 = time.time()
    am50_greedy = greedy_eval(am50_model, coords, device, batch_size=2048)
    results['AM-paper tsp50->20 greedy'] = {'costs': am50_greedy, 'wall': time.time() - t0}

    t0 = time.time()
    am50_sample = sample_eval(am50_model, coords, device, width=1280)
    results['AM-paper tsp50->20 sample x1280'] = {'costs': am50_sample, 'wall': time.time() - t0}

    # --- S4 K=10 ckpt: greedy + MCTS K=10/25/40 ---
    print('\n[*] S4 K=10 ckpt (best_model / theta-star)...')
    s4_ckpt = torch_load_cpu(S4_CKPT)
    # train_args=None -> use defaults; matches the K=10 step50 recipe (AM-arch defaults).
    s4_model = load_s4_model(s4_ckpt, 'best_model', train_args=None, device=device)

    t0 = time.time()
    s4_greedy = greedy_eval(s4_model, coords, device, batch_size=2048)
    results['S4 greedy'] = {'costs': s4_greedy, 'wall': time.time() - t0}

    t0 = time.time()
    s4_sample = sample_eval(s4_model, coords, device, width=1280)
    results['S4 sample x1280'] = {'costs': s4_sample, 'wall': time.time() - t0}

    for K in (10, 25, 40):
        cfg, _ = _build_mcts_config(_Opts(K), graph_size=GRAPH, train_args=None)
        t0 = time.time()
        s4_mcts = mcts_eval(s4_model, coords, cfg, device, mcts_batch_size=1000)
        results[f'S4 MCTS K={K}'] = {'costs': s4_mcts, 'wall': time.time() - t0}

    # --- Classical heuristics ---
    print('\n[*] Nearest Neighbour...')
    t0 = time.time()
    nn = nearest_neighbour_batch(coords, device).cpu().numpy()
    results['Nearest Neighbour'] = {'costs': nn, 'wall': time.time() - t0}

    for label, key in [('Farthest Insertion', 'farthest'),
                       ('Random Insertion', 'random'),
                       ('Nearest Insertion', 'nearest')]:
        print(f'\n[*] {label}...')
        t0 = time.time()
        costs = []
        for inst in tqdm(data_np, desc=label):
            c, _ = run_insertion(inst, key)
            costs.append(c)
        results[label] = {'costs': np.array(costs), 'wall': time.time() - t0}

    # --- Compose final table ---
    opt = results['Gurobi (optimal)']['costs']
    tol = 1e-4  # absolute tolerance for "matches optimum"

    rows = []
    for name, r in results.items():
        c = r['costs']
        mean = float(c.mean())
        n_opt = int((np.abs(c - opt) <= tol).sum())
        gap = 100.0 * (mean / opt.mean() - 1.0)
        rows.append((name, mean, r['wall'], n_opt, gap))
    rows.sort(key=lambda x: x[1])

    print('\n' + '=' * 90)
    print(f"TSP-{GRAPH} comparison ({NUM} instances, seed={SEED})")
    print('=' * 90)
    print(f"{'Method':<22} {'Avg cost':>10} {'Total wall (s)':>16} {'n_opt/'+str(NUM):>12} {'Gap (%)':>10}")
    print('-' * 90)
    for name, mean, wall, n_opt, gap in rows:
        print(f'{name:<22} {mean:>10.5f} {wall:>16.2f} {n_opt:>9d}/{NUM} {gap:>10.4f}')
    print('=' * 90)

    # Save raw per-instance arrays for downstream analysis.
    out_dir = os.path.dirname(S4_CKPT)
    np.savez(
        os.path.join(out_dir, f'eval_full_comparison_seed{SEED}.npz'),
        **{name: r['costs'] for name, r in results.items()},
    )
    print(f"\nSaved per-instance costs to {out_dir}/eval_full_comparison_seed{SEED}.npz")


if __name__ == '__main__':
    main()
