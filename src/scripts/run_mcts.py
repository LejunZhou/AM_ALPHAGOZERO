"""Stage 2 CLI: run MCTS on a TSP dataset with a trained AM + value-head checkpoint.

Examples:
  # Headline — TSP-20 Stage 1 canonical, K=200
  PYTHONPATH=src python -m scripts.run_mcts \
      --model outputs/tsp_20/stage1_tsp20_canonical_<ts>/epoch-99.pt \
      --graph_size 20 --val_size 1000 --seed 1234 \
      --n_simulations 200 --c_puct 1.0 --temperature 0.0 \
      --leaf_eval value_head --output_csv outputs/stage2/tsp20_K200.csv

  # Leaf-eval ablation
  PYTHONPATH=src python -m scripts.run_mcts \
      --model ... --n_simulations 200 --leaf_eval rollout
"""
import argparse
import csv
import os
import sys
import time

import torch
from tqdm import tqdm

from am_baseline.problem.tsp import TSP
from am_baseline.search import MCTSConfig, MCTSSolver
from am_baseline.utils.misc import load_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2: MCTS on TSP with trained AM + value head")
    parser.add_argument('--model', required=True,
                        help='Path to model checkpoint file (.pt) or directory')
    parser.add_argument('--graph_size', type=int, default=None,
                        help='Override graph size (default: read from checkpoint args.json)')
    parser.add_argument('--val_size', type=int, default=1000,
                        help='Number of test instances to solve')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Optional path to a .pkl dataset; if omitted, generates fresh')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--n_simulations', type=int, default=200)
    parser.add_argument('--c_puct', type=float, default=0.05,
                        help='PUCT exploration constant. TSP minimization needs a small value '
                             '(~0.05) because Q differences are small on near-optimal policies; '
                             'AlphaGo default 1.0 makes U dominate and MCTS collapses to greedy.')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Root action selection temperature (0 = argmax)')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.3)
    parser.add_argument('--dirichlet_epsilon', type=float, default=0.0,
                        help='Dirichlet noise weight at root (0 = off)')
    parser.add_argument('--leaf_eval', choices=['value_head', 'rollout'], default='rollout',
                        help="Default 'rollout' per Stage 2 leaf-eval ablation (uniformly +12-22pp "
                             "gap reduction over 'value_head' at every matched K). Use 'value_head' "
                             "for diagnostics or as required by Stage 4 training-loop semantics.")
    parser.add_argument('--value_norm', choices=['bl', 'sqrt_n'], default='bl',
                        help="Normalizer for cost-to-go. 'bl' = per-instance greedy cost (matches "
                             "Stage 1 training convention). 'sqrt_n' is only valid with "
                             "leaf_eval='rollout' (raises ValueError otherwise — the value head "
                             "was trained in bl-normalized units).")
    parser.add_argument('--fpu_mode', choices=['fallback', 'running_q', 'node_value'],
                        default='running_q',
                        help="How to init Q for unvisited actions. 'fallback' = constant "
                             "`fpu_fallback` everywhere (useful for sweeping). 'running_q' = "
                             "sum(W)/sum(N) at the parent (AlphaZero standard). 'node_value' = "
                             "-(c_path_norm + v_estimate) — total-from-root scale matching backed-up Q.")
    parser.add_argument('--fpu_fallback', type=float, default=-1.0,
                        help='Q_init used at brand-new nodes (N=0) regardless of fpu_mode, and '
                             'everywhere when fpu_mode=fallback. Default -1.0 matches typical '
                             'completed-tour Q on TSP (normalized cost ~ 1).')
    parser.add_argument('--root_select', choices=['visits', 'q'], default='visits',
                        help="Final action at root: 'visits' (AlphaGo default) or 'q' "
                             "(diagnostic — argmax Q among visited actions).")
    tree_reuse_group = parser.add_mutually_exclusive_group()
    tree_reuse_group.add_argument('--tree_reuse', dest='tree_reuse', action='store_true',
                                  help='Retain the subtree below the chosen action as the next root '
                                       '(default — Stage 2 promoted to canonical: 47/100 wins, +0.149%% '
                                       'quality, 17%% wall-clock saved on TSP-20).')
    tree_reuse_group.add_argument('--no_tree_reuse', dest='tree_reuse', action='store_false',
                                  help='Discard the tree between tour-steps (diagnostic only).')
    parser.set_defaults(tree_reuse=True)
    parser.add_argument('--output_csv', type=str, default=None,
                        help='If set, write per-instance CSV with (idx, greedy_cost, mcts_cost, gap)')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--no_progress_bar', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"device={device}")

    # --- Load model ---
    model, model_args = load_model(args.model)
    model.to(device).eval()
    graph_size = args.graph_size if args.graph_size is not None else model_args['graph_size']
    print(f"Loaded AttentionModel for TSP-{graph_size} from {args.model}")
    if model.value_head is None and args.leaf_eval == 'value_head':
        print(f"ERROR: checkpoint has no value head but --leaf_eval=value_head. "
              f"Retrain with value_enabled=True, or pass --leaf_eval rollout.",
              file=sys.stderr)
        return 2

    # --- Dataset ---
    if args.dataset:
        dataset = TSP.make_dataset(filename=args.dataset, num_samples=args.val_size)
    else:
        torch.manual_seed(args.seed)
        dataset = TSP.make_dataset(size=graph_size, num_samples=args.val_size)
    # Stack into (B, N, 2) for batched greedy pass + MCTS iteration.
    inputs = torch.stack([dataset[i] for i in range(len(dataset))], dim=0)
    B = inputs.size(0)
    print(f"Dataset: {B} instances of TSP-{graph_size}, seed={args.seed}")

    # --- Greedy baseline on the same instances for comparison ---
    model.set_decode_type('greedy')
    with torch.no_grad():
        greedy_cost, _ = model(inputs.to(device))
    greedy_cost = greedy_cost.detach().cpu()
    print(f"Model greedy: mean={greedy_cost.mean().item():.4f}  "
          f"std={greedy_cost.std().item():.4f}  "
          f"min={greedy_cost.min().item():.4f}  max={greedy_cost.max().item():.4f}")

    # --- MCTS run ---
    cfg = MCTSConfig(
        n_simulations=args.n_simulations,
        c_puct=args.c_puct,
        temperature=args.temperature,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_epsilon=args.dirichlet_epsilon,
        leaf_eval=args.leaf_eval,
        value_norm=args.value_norm,
        fpu_mode=args.fpu_mode,
        fpu_fallback=args.fpu_fallback,
        root_select=args.root_select,
        tree_reuse=args.tree_reuse,
        seed=args.seed,
    )
    solver = MCTSSolver(model, cfg, device=device)
    print(f"MCTSConfig: {cfg}")

    # Per-instance loop (sequential). We drive it from here so we can show a
    # progress bar and not hide it inside solve_batch. bl_val is still computed
    # in one batched greedy pass for efficiency.
    with torch.no_grad():
        bl_vals = solver._compute_bl_val_batch(inputs.to(device))

    costs = torch.empty(B)
    tours = torch.empty(B, graph_size, dtype=torch.long)
    t0 = time.time()
    iterator = range(B)
    if not args.no_progress_bar:
        iterator = tqdm(iterator, desc="MCTS")
    for i in iterator:
        c_i, t_i = solver.solve_instance(
            inputs[i:i+1].to(device),
            bl_val=float(bl_vals[i].item()),
        )
        costs[i] = c_i.detach().cpu()
        tours[i] = t_i.detach().cpu()
    elapsed = time.time() - t0

    # --- Report ---
    gap_vs_greedy = (costs - greedy_cost)                  # negative = improvement
    gap_pct = gap_vs_greedy / greedy_cost * 100.0
    print()
    print(f"MCTS(K={args.n_simulations}, leaf={args.leaf_eval}) results on {B} instances:")
    print(f"  mean cost  : {costs.mean().item():.4f}")
    print(f"  std        : {costs.std().item():.4f}")
    print(f"  min        : {costs.min().item():.4f}")
    print(f"  max        : {costs.max().item():.4f}")
    print(f"  median     : {costs.median().item():.4f}")
    print(f"  vs greedy  : delta mean = {gap_vs_greedy.mean().item():+.4f} "
          f"({gap_pct.mean().item():+.3f}%)")
    print(f"  win rate   : MCTS better on {(gap_vs_greedy < 0).sum().item()}/{B} instances; "
          f"tied on {(gap_vs_greedy == 0).sum().item()}")
    print(f"  wall-clock : {elapsed:.1f}s  ({elapsed/B*1000:.1f} ms/inst)")

    # --- Correctness check: every tour is a valid permutation ---
    expected = torch.arange(graph_size, dtype=torch.long)
    for i in range(B):
        sorted_tour, _ = tours[i].sort()
        assert torch.equal(sorted_tour, expected), (
            f"instance {i}: tour is not a permutation of [0,{graph_size}): {tours[i].tolist()}"
        )
    print(f"[OK] all {B} tours are valid permutations of [0,{graph_size})")

    # --- CSV dump ---
    if args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)) or ".", exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "greedy_cost", "mcts_cost", "delta", "gap_pct"])
            for i in range(B):
                w.writerow([
                    i,
                    f"{greedy_cost[i].item():.6f}",
                    f"{costs[i].item():.6f}",
                    f"{gap_vs_greedy[i].item():+.6f}",
                    f"{gap_pct[i].item():+.4f}",
                ])
        print(f"Wrote {args.output_csv}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
