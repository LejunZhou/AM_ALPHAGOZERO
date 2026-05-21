"""TSP-50 full inference comparison: classical solvers + S1 AM + AM-paper + S4 K=25 ckpt.

Generates 1000 instances at seed=20260430 (same convention as the TSP-20
comparison), runs each method, captures per-instance costs, prints unified
table with {val_avg_cost, total wall, n_optimal/1000, gap-to-optimum %}.

MCTS only at K=25 per user request.
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


SEED = 20260430
NUM = 1000
GRAPH = 50
S4_CKPT = 'outputs/tsp_50/tsp50_k25_lv0_step50_100iter_20260513T233139_20260513T233149/iter-94_accepted.pt'
S1_CKPT = 'outputs/tsp_50/stage1_tsp50_am_baseline_20260424T032356/epoch-99.pt'
AM_PAPER_CKPT = 'ref/attention-learn-to-route-master/pretrained/tsp_50/epoch-99.pt'
MCTS_K = 25


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


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  seed={SEED}  num={NUM}  graph={GRAPH}  MCTS_K={MCTS_K}')

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dataset = TSP.make_dataset(size=GRAPH, num_samples=NUM)
    coords = torch.stack([x for x in dataset])
    data_np = [inst.numpy() for inst in dataset]

    results = {}

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

    # --- AM S1 canonical (our reproduction, in-dist TSP-50): greedy + sampling ---
    print('\n[*] S1 canonical (bs=512, trained TSP-50)...')
    s1_model = load_am_model(S1_CKPT, device)
    t0 = time.time()
    s1_greedy = greedy_eval(s1_model, coords, device, batch_size=2048)
    results['S1 greedy (TSP-50)'] = {'costs': s1_greedy, 'wall': time.time() - t0}

    t0 = time.time()
    s1_sample = sample_eval(s1_model, coords, device, width=1280)
    results['S1 sample x1280 (TSP-50)'] = {'costs': s1_sample, 'wall': time.time() - t0}

    # --- AM-paper released TSP-50 ckpt (in-dist): greedy + sampling ---
    print('\n[*] AM-paper released TSP-50 ckpt...')
    am_paper = load_am_model(AM_PAPER_CKPT, device)
    t0 = time.time()
    am_paper_greedy = greedy_eval(am_paper, coords, device, batch_size=2048)
    results['AM-paper TSP-50 greedy'] = {'costs': am_paper_greedy, 'wall': time.time() - t0}

    t0 = time.time()
    am_paper_sample = sample_eval(am_paper, coords, device, width=1280)
    results['AM-paper TSP-50 sample x1280'] = {'costs': am_paper_sample, 'wall': time.time() - t0}

    # --- S4 K=25 step50 ckpt (best = iter-94_accepted): greedy + sampling + MCTS K=25 ---
    print('\n[*] S4 K=25 step50 ckpt (iter-94_accepted, best)...')
    s4_ckpt = torch_load_cpu(S4_CKPT)
    s4_model = load_s4_model(s4_ckpt, 'best_model', train_args=None, device=device)

    t0 = time.time()
    s4_greedy = greedy_eval(s4_model, coords, device, batch_size=2048)
    results['S4 greedy'] = {'costs': s4_greedy, 'wall': time.time() - t0}

    t0 = time.time()
    s4_sample = sample_eval(s4_model, coords, device, width=1280)
    results['S4 sample x1280'] = {'costs': s4_sample, 'wall': time.time() - t0}

    cfg, _ = _build_mcts_config(_Opts(MCTS_K), graph_size=GRAPH, train_args=None)
    t0 = time.time()
    s4_mcts = mcts_eval(s4_model, coords, cfg, device, mcts_batch_size=1000)
    results[f'S4 MCTS K={MCTS_K}'] = {'costs': s4_mcts, 'wall': time.time() - t0}

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
    tol = 1e-4

    rows = []
    for name, r in results.items():
        c = r['costs']
        mean = float(c.mean())
        n_opt = int((np.abs(c - opt) <= tol).sum())
        gap = 100.0 * (mean / opt.mean() - 1.0)
        rows.append((name, mean, r['wall'], n_opt, gap))
    rows.sort(key=lambda x: x[1])

    print('\n' + '=' * 100)
    print(f"TSP-{GRAPH} comparison ({NUM} instances, seed={SEED})")
    print('=' * 100)
    print(f"{'Method':<32} {'Avg cost':>10} {'Total wall (s)':>16} {'n_opt/'+str(NUM):>12} {'Gap (%)':>10}")
    print('-' * 100)
    for name, mean, wall, n_opt, gap in rows:
        print(f'{name:<32} {mean:>10.5f} {wall:>16.2f} {n_opt:>9d}/{NUM} {gap:>10.4f}')
    print('=' * 100)

    out_dir = os.path.dirname(S4_CKPT)
    np.savez(
        os.path.join(out_dir, f'eval_full_comparison_seed{SEED}.npz'),
        **{name: r['costs'] for name, r in results.items()},
    )
    print(f"\nSaved per-instance costs to {out_dir}/eval_full_comparison_seed{SEED}.npz")


if __name__ == '__main__':
    main()
