"""One-off: rerun MCTS K=10 and K=40, pair vs saved Gurobi to compute n_optimal."""
import os
import sys
import time
import csv

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

import numpy as np
import torch

from am_baseline.problem.tsp import TSP
from am_baseline.search.mcts import MCTSConfig
from am_baseline.search.mcts_cpp.solver import CppBatchMCTSSolver
from scripts.val_stage4_mcts import load_am_model


SEED = 1234
N_TEST = 1000
GRAPH_SIZE = 50
CKPT = 'outputs/tsp_50/stage1_tsp50_am_baseline_20260424T032356/epoch-99.pt'
CSV = '_progress/eval_logs/tsp50_1000_K25_seed1234.csv'

# Load saved Gurobi per-instance costs.
gur = np.empty(N_TEST)
with open(CSV) as f:
    r = csv.DictReader(f)
    for row in r:
        i = int(row['instance'])
        gur[i] = float(row['Gurobi (optimal)'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Regenerate same val set.
torch.manual_seed(SEED)
np.random.seed(SEED)
coords = torch.stack([x for x in TSP.make_dataset(size=GRAPH_SIZE, num_samples=N_TEST)])

model = load_am_model(CKPT, device)

results = {}
for K in (10, 40):
    cfg = MCTSConfig(
        n_simulations=K,
        c_puct=0.05,
        leaf_eval='rollout',
        value_norm='bl',
        fpu_mode='running_q',
        fpu_fallback=-1.0,
        root_select='visits',
        temperature=0.0,
        temperature_schedule=None,
        dirichlet_alpha=10.0 / GRAPH_SIZE,
        dirichlet_epsilon=0.0,
        tree_reuse=True,
        return_root_visits=False,
        seed=SEED,
    )
    solver = CppBatchMCTSSolver(model, cfg, device=device, mcts_batch_size=1000)
    t0 = time.time()
    costs_t, _ = solver.solve_batch(coords.to(device))
    elapsed = time.time() - t0
    costs = costs_t.cpu().numpy()
    results[K] = (costs, elapsed)
    print(f'MCTS K={K}: mean={costs.mean():.5f}  wall={elapsed:.1f}s')

# n_optimal vs Gurobi.
TOL = 1e-5
print('\nn_optimal vs Gurobi (cost - gurobi <= 1e-5):')
print(f'  {"Method":<20} {"Δ mean":>10} {"Δ SE":>10} {"n_optimal":>14}')
for K, (c, _) in results.items():
    d = c - gur
    n_opt = (d <= TOL).sum()
    print(f'  MCTS K={K:<13}  {d.mean():+10.5f} {d.std()/np.sqrt(len(d)):10.5f}'
          f'   {n_opt:>5}/{N_TEST} ({n_opt/N_TEST*100:.1f}%)')

# Dump CSV for archive.
out_csv = '_progress/eval_logs/tsp50_1000_K10_K40_seed1234.csv'
with open(out_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['instance', 'Gurobi (optimal)', 'MCTS K=10 rollout', 'MCTS K=40 rollout'])
    for i in range(N_TEST):
        w.writerow([i, float(gur[i]), float(results[10][0][i]), float(results[40][0][i])])
print(f'\nWrote {out_csv}')
