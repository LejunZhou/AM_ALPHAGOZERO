"""Stage 3 CLI: sampling-K test-time search on a trained AM checkpoint.

Companion to `run_mcts.py`. Computes the AM-paper-style sampling baseline at
configurable width K, with a per-instance CSV that mirrors run_mcts.py's
schema so the Stage 3 aggregator can ingest both runners uniformly.

Examples:
  # Sampling-1280 on TSP-20 Stage 1 canonical
  PYTHONPATH=src python -m scripts.run_sampling \
      --model outputs/tsp_20/stage1_tsp20_canonical_<ts>/epoch-99.pt \
      --graph_size 20 --val_size 1000 --seed 1234 \
      --width 1280 --output_csv outputs/stage3/tsp20_sampling_K1280.csv

  # Sampling-2560 on TSP-50
  PYTHONPATH=src python -m scripts.run_sampling \
      --model outputs/tsp_50/stage1_tsp50_with_value_<ts>/epoch-99.pt \
      --graph_size 50 --width 2560 --output_csv outputs/stage3/tsp50_sampling_K2560.csv

Forward-pass accounting:
  decode_steps per instance = (batch_rep * iter_rep) * graph_size
  with (batch_rep, iter_rep) chosen so their product equals --width when --width
  factors cleanly under the --max_batch_rep cap; otherwise rounded up to the
  smallest multiple. The actual sample count is reported (and used for the
  decode_steps column), not the user-requested width.
"""
import argparse
import csv
import os
import sys
import time

import torch
from tqdm import tqdm

from am_baseline.problem.tsp import TSP
from am_baseline.utils.misc import load_model


def _factor_width(width: int, max_batch_rep: int) -> tuple[int, int]:
    """Return (batch_rep, iter_rep) such that batch_rep * iter_rep == width
    and batch_rep <= max_batch_rep, picking the largest such batch_rep that
    divides `width`. If no divisor <= max_batch_rep exists, falls back to
    ceiling division (actual_samples > width).
    """
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    if width <= max_batch_rep:
        return width, 1
    # Pick the largest divisor of `width` that's <= max_batch_rep.
    for d in range(max_batch_rep, 0, -1):
        if width % d == 0:
            return d, width // d
    # Fallback: ceiling division. Actual samples > requested width.
    batch_rep = max_batch_rep
    iter_rep = (width + batch_rep - 1) // batch_rep
    return batch_rep, iter_rep


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 3: sampling-K test-time search on trained AM")
    parser.add_argument('--model', required=True,
                        help='Path to model checkpoint file (.pt) or directory')
    parser.add_argument('--graph_size', type=int, default=None,
                        help='Override graph size (default: read from checkpoint args.json)')
    parser.add_argument('--val_size', type=int, default=1000,
                        help='Number of test instances to solve')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Optional path to a .pkl dataset; if omitted, generates fresh')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--width', type=int, default=1280,
                        help='Sampling width K (samples drawn per instance, best returned)')
    parser.add_argument('--max_batch_rep', type=int, default=128,
                        help='Cap on inner batch_rep replication factor (GPU-memory limit). '
                             'Effective per-batch GPU work = eval_batch_size * batch_rep.')
    parser.add_argument('--eval_batch_size', type=int, default=None,
                        help='Test instances processed per chunk. Default chooses '
                             'eval_batch_size = max(1, max_total_batch // batch_rep) '
                             'where max_total_batch is set by --max_total_batch.')
    parser.add_argument('--max_total_batch', type=int, default=2048,
                        help='Soft cap on (eval_batch_size * batch_rep). Default 2048 fits '
                             'an 8GB consumer GPU at TSP-100; raise for larger GPUs.')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='If set, write per-instance CSV: idx,greedy_cost,sample_cost,'
                             'delta,gap_pct,decode_steps')
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

    # --- Dataset (matching run_mcts.py seeding for cross-runner determinism) ---
    if args.dataset:
        dataset = TSP.make_dataset(filename=args.dataset, num_samples=args.val_size)
    else:
        torch.manual_seed(args.seed)
        dataset = TSP.make_dataset(size=graph_size, num_samples=args.val_size)
    inputs = torch.stack([dataset[i] for i in range(len(dataset))], dim=0)
    B = inputs.size(0)
    print(f"Dataset: {B} instances of TSP-{graph_size}, seed={args.seed}")

    # --- Greedy baseline (matches run_mcts.py) ---
    model.set_decode_type('greedy')
    with torch.no_grad():
        greedy_cost, _ = model(inputs.to(device))
    greedy_cost = greedy_cost.detach().cpu()
    print(f"Model greedy: mean={greedy_cost.mean().item():.4f}  "
          f"std={greedy_cost.std().item():.4f}  "
          f"min={greedy_cost.min().item():.4f}  max={greedy_cost.max().item():.4f}")

    # --- Sampling factoring ---
    batch_rep, iter_rep = _factor_width(args.width, args.max_batch_rep)
    actual_samples = batch_rep * iter_rep
    decode_steps_per_inst = actual_samples * graph_size
    if args.eval_batch_size is None:
        eval_batch_size = max(1, args.max_total_batch // batch_rep)
    else:
        eval_batch_size = args.eval_batch_size
    print(f"Sampling K={args.width} via batch_rep={batch_rep}, iter_rep={iter_rep} "
          f"({actual_samples} actual samples per instance; eval_batch_size={eval_batch_size})")
    if actual_samples != args.width:
        print(f"  note: requested width={args.width} did not factor cleanly; "
              f"actual samples = {actual_samples} (> requested)")

    # --- Sampling loop ---
    # Re-seed torch RNG so multinomial draws are reproducible at fixed --seed.
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    model.set_decode_type('sampling')
    sample_costs = torch.empty(B)
    sample_pis = []  # collect best-of-K tours per instance (variable padding ok)
    t0 = time.time()
    iterator = range(0, B, eval_batch_size)
    if not args.no_progress_bar:
        iterator = tqdm(iterator, desc=f"Sampling K={args.width}")
    with torch.no_grad():
        for start in iterator:
            end = min(start + eval_batch_size, B)
            batch = inputs[start:end].to(device)
            best_pi, best_cost = model.sample_many(
                batch, batch_rep=batch_rep, iter_rep=iter_rep
            )
            sample_costs[start:end] = best_cost.detach().cpu()
            sample_pis.append(best_pi.detach().cpu())
    elapsed = time.time() - t0

    # --- Report ---
    delta = sample_costs - greedy_cost
    gap_pct = delta / greedy_cost * 100.0
    print()
    print(f"Sampling(K={args.width}) results on {B} instances:")
    print(f"  mean cost  : {sample_costs.mean().item():.4f}")
    print(f"  std        : {sample_costs.std().item():.4f}")
    print(f"  min        : {sample_costs.min().item():.4f}")
    print(f"  max        : {sample_costs.max().item():.4f}")
    print(f"  median     : {sample_costs.median().item():.4f}")
    print(f"  vs greedy  : delta mean = {delta.mean().item():+.4f} "
          f"({gap_pct.mean().item():+.3f}%)")
    print(f"  win rate   : sampling better on {(delta < 0).sum().item()}/{B} instances; "
          f"tied on {(delta == 0).sum().item()}")
    print(f"  wall-clock : {elapsed:.1f}s  ({elapsed/B*1000:.1f} ms/inst)")
    print(f"  fwd passes : decode_steps per instance = {decode_steps_per_inst}  "
          f"(K={actual_samples} samples * N={graph_size} steps)")

    # --- Correctness check: best-of-K tours are valid permutations ---
    expected = torch.arange(graph_size, dtype=torch.long)
    pis_concat = torch.cat(sample_pis, dim=0)
    # sample_many can return varying-length sequences across iter_rep batches;
    # for TSP they all equal graph_size. Trim/pad if needed.
    if pis_concat.size(1) < graph_size:
        raise RuntimeError(
            f"sampled tours have length {pis_concat.size(1)} < graph_size {graph_size}"
        )
    pis_concat = pis_concat[:, :graph_size]
    for i in range(B):
        sorted_tour, _ = pis_concat[i].sort()
        if not torch.equal(sorted_tour, expected):
            raise AssertionError(
                f"instance {i}: tour is not a permutation of [0,{graph_size}): "
                f"{pis_concat[i].tolist()}"
            )
    print(f"[OK] all {B} best-of-K tours are valid permutations of [0,{graph_size})")

    # --- CSV dump ---
    if args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)) or ".", exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "idx", "greedy_cost", "sample_cost", "delta", "gap_pct", "decode_steps",
            ])
            for i in range(B):
                w.writerow([
                    i,
                    f"{greedy_cost[i].item():.6f}",
                    f"{sample_costs[i].item():.6f}",
                    f"{delta[i].item():+.6f}",
                    f"{gap_pct[i].item():+.4f}",
                    decode_steps_per_inst,
                ])
        print(f"Wrote {args.output_csv}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
