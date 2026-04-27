"""Micro-benchmark: per-call cost of decoder.decode_step as a function of N.

Decouples MCTS overhead from the question "why is rollout barely slower than
value_head on TSP-20, and does that flip at large N?". Measures the pure GPU
cost of one decode_step call so we can fit a launch_overhead + alpha*N model.

Usage:
  python src/scripts/bench_decode_step.py \
    --model outputs/tsp_50/stage1_tsp50_with_value_20260424T032357/epoch-99.pt \
    --graph_sizes 20,50,100,200 --n_rollouts 200 --seed 1234
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import time

import torch

from am_baseline.problem.tsp import TSP
from am_baseline.utils.misc import load_model


def bench_one(model, N, n_rollouts, device):
    """Run n_rollouts greedy rollouts at graph size N. Return per-call mean (us)."""
    inputs = torch.rand(1, N, 2, device=device)
    embeddings = model.encode(inputs)
    fixed = model.precompute_decoder(embeddings)

    def one_rollout():
        state = TSP.make_state(inputs)
        while not state.all_finished():
            log_p, _ = model.decoder.decode_step(fixed, state, return_glimpse=False)
            a = int(log_p.view(-1).argmax().item())
            state = state.update(torch.tensor([a], dtype=torch.long, device=device))
        return state

    for _ in range(3):
        one_rollout()
    if device.type == 'cuda':
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_rollouts):
        one_rollout()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_us = (t1 - t0) * 1e6
    n_calls = n_rollouts * N
    return total_us / n_calls, total_us / n_rollouts


def linear_fit(xs, ys):
    """OLS for y = a + b*x. Returns (a, b)."""
    n = len(xs)
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a = (sy - b * sx) / n
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--graph_sizes', default='20,50,100,200',
                    help='Comma-separated list of N to benchmark')
    ap.add_argument('--n_rollouts', type=int, default=200)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--no_cuda', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Device: {device}")
    torch.manual_seed(args.seed)

    model, _ = load_model(args.model)
    model.to(device)
    model.eval()
    model.set_decode_type('greedy')

    Ns = [int(x) for x in args.graph_sizes.split(',')]
    rows = []
    print(f"\n{'N':>5} {'per_call_us':>14} {'per_rollout_ms':>16} {'n_calls':>10}")
    with torch.no_grad():
        for N in Ns:
            per_call, per_rollout = bench_one(model, N, args.n_rollouts, device)
            rows.append((N, per_call, per_rollout))
            print(f"{N:>5} {per_call:>14.2f} {per_rollout/1e3:>16.3f} {args.n_rollouts*N:>10}")

    if len(Ns) >= 2:
        a, b = linear_fit([r[0] for r in rows], [r[1] for r in rows])
        print(f"\nLinear fit:  per_call_us(N) = {a:.2f} + {b:.4f} * N")
        print(f"  launch_overhead floor (intercept): {a:.2f} us")
        print(f"  arithmetic per N (slope):          {b:.4f} us / city")

        print("\nPredicted rollout/value_head wall-clock ratio at avg leaf depth = N/2:")
        print(f"  ratio(N) = (N/2 + 1) * decode_step(N) / (1 * decode_step(N) + value_head_MLP)")
        print(f"  approx   = (N/2 + 1) / 1   when MLP ~ 0 cost")
        print(f"  but kernel-launch amortization shrinks the effective ratio by overhead/(overhead+arith).")
        print(f"\n{'N':>5} {'pred_decode_us':>16} {'pred_rollout_ratio':>20}")
        for N in [20, 50, 100, 200, 500]:
            pred_us = a + b * N
            n_calls = N / 2 + 1
            ratio = (n_calls * pred_us) / (1 * pred_us)
            print(f"{N:>5} {pred_us:>16.2f} {ratio:>20.2f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
