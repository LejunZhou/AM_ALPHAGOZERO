"""Probe decode_step counts for Stage 2 MCTS configurations.

Stage 2's K-curve CSVs predate the Phase A instrumentation, so they lack the
`decode_steps` column needed for Stage 3's forward-pass-budget plot. This
script runs a small (`--val_size`, default 20) instrumented MCTS pass for each
requested config and writes the mean `decode_steps` (and rollout/value subset
counts) to a JSON cache. The aggregator reads from this cache.

Per-instance decode_step counts vary across instances but the mean over
20 instances is a tight estimate of the 1000-instance mean (typical std/mean
ratio ≈ 5 % from Phase A.4).

Usage:
  PYTHONPATH=src python -m scripts.probe_decode_steps \
      --model outputs/tsp_20/stage1_tsp20_canonical_<ts>/epoch-99.pt \
      --graph_size 20 \
      --rollout_K 20,50,100,200,400 \
      --value_head_K 50,100,200,400,800 \
      --cache_path outputs/stage3/decode_step_cache.json
"""
import argparse
import json
import os
import sys
import time

import torch

from am_baseline.problem.tsp import TSP
from am_baseline.search import MCTSConfig, MCTSSolver
from am_baseline.utils.misc import load_model


def _parse_klist(s: str) -> list[int]:
    if not s:
        return []
    return [int(x) for x in s.split(',') if x.strip()]


def _probe_one(model, inputs, bl_vals, K: int, leaf_eval: str, device, seed: int) -> dict:
    cfg = MCTSConfig(
        n_simulations=K,
        c_puct=0.05,
        leaf_eval=leaf_eval,
        seed=seed,
    )
    solver = MCTSSolver(model, cfg, device=device)
    decode_counts = []
    rollout_counts = []
    value_counts = []
    t0 = time.time()
    for i in range(inputs.size(0)):
        _ = solver.solve_instance(
            inputs[i:i+1].to(device),
            bl_val=float(bl_vals[i].item()),
        )
        decode_counts.append(solver.fwd_count_decode)
        rollout_counts.append(solver.fwd_count_rollout)
        value_counts.append(solver.fwd_count_value)
    elapsed = time.time() - t0
    n = len(decode_counts)
    return {
        'K': K,
        'leaf_eval': leaf_eval,
        'n_probe': n,
        'decode_steps_mean': sum(decode_counts) / n,
        'rollout_steps_mean': sum(rollout_counts) / n,
        'value_calls_mean': sum(value_counts) / n,
        'wall_clock_s': elapsed,
        'wall_clock_per_inst_ms': elapsed / n * 1000,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe decode_step counts for Stage 2 MCTS configs")
    parser.add_argument('--model', required=True)
    parser.add_argument('--graph_size', type=int, default=None)
    parser.add_argument('--val_size', type=int, default=20,
                        help='Probe size; smaller = faster, larger = tighter mean estimate')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--rollout_K', type=str, default='',
                        help='Comma-separated K values for rollout leaf-eval probes')
    parser.add_argument('--value_head_K', type=str, default='',
                        help='Comma-separated K values for value_head leaf-eval probes')
    parser.add_argument('--cache_path', type=str, required=True,
                        help='JSON cache file to update (created if missing)')
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"device={device}")

    rollout_Ks = _parse_klist(args.rollout_K)
    value_head_Ks = _parse_klist(args.value_head_K)
    if not rollout_Ks and not value_head_Ks:
        print("Nothing to probe (both --rollout_K and --value_head_K empty).", file=sys.stderr)
        return 1

    model, model_args = load_model(args.model)
    model.to(device).eval()
    graph_size = args.graph_size if args.graph_size is not None else model_args['graph_size']

    torch.manual_seed(args.seed)
    dataset = TSP.make_dataset(size=graph_size, num_samples=args.val_size)
    inputs = torch.stack([dataset[i] for i in range(len(dataset))], dim=0)
    print(f"Probe set: {inputs.size(0)} TSP-{graph_size} instances at seed={args.seed}")

    # bl_vals (single greedy pass; matches run_mcts.py).
    model.set_decode_type('greedy')
    with torch.no_grad():
        bl_vals_full, _ = model(inputs.to(device))
    bl_vals = bl_vals_full.detach().cpu()

    # Load existing cache.
    if os.path.exists(args.cache_path):
        with open(args.cache_path, 'r') as f:
            cache = json.load(f)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.cache_path)) or '.', exist_ok=True)
        cache = {}

    cache.setdefault(f'tsp{graph_size}', {})
    bucket = cache[f'tsp{graph_size}']

    for K in rollout_Ks:
        key = f'rollout_K{K}'
        print(f"\n[probe] {key}...")
        res = _probe_one(model, inputs, bl_vals, K=K, leaf_eval='rollout',
                          device=device, seed=args.seed)
        print(f"  decode_steps mean = {res['decode_steps_mean']:.1f}  "
              f"(rollout subset {res['rollout_steps_mean']:.1f}, "
              f"value_calls {res['value_calls_mean']:.1f})  "
              f"wall_clock {res['wall_clock_s']:.1f}s")
        bucket[key] = res

    for K in value_head_Ks:
        if model.value_head is None:
            print(f"[probe] skipping value_head_K{K} — checkpoint has no value head", file=sys.stderr)
            continue
        key = f'value_head_K{K}'
        print(f"\n[probe] {key}...")
        res = _probe_one(model, inputs, bl_vals, K=K, leaf_eval='value_head',
                          device=device, seed=args.seed)
        print(f"  decode_steps mean = {res['decode_steps_mean']:.1f}  "
              f"(rollout subset {res['rollout_steps_mean']:.1f}, "
              f"value_calls {res['value_calls_mean']:.1f})  "
              f"wall_clock {res['wall_clock_s']:.1f}s")
        bucket[key] = res

    with open(args.cache_path, 'w') as f:
        json.dump(cache, f, indent=2)
    print(f"\nWrote {args.cache_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
