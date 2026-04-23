"""
Standalone diagnostic for the Stage 1 value head.

Loads a trained AM checkpoint, runs greedy decoding on a held-out TSP dataset
with compute_values=True, and reports:
  - R^2 overall
  - R^2 bucketed by tour position (early / mid / late)
  - Calibration table (binned mean prediction vs. binned mean target)

Usage:
  python src/scripts/eval_value.py --model outputs/tsp_20/stage1_run/epoch-99.pt \
    --val_size 10000 --seed 1234
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch
from torch.utils.data import DataLoader

from am_baseline.utils.misc import load_model
from am_baseline.utils.tensor_ops import move_to, value_targets_from_edges
from am_baseline.problem.tsp import TSP
from am_baseline.model.attention_model import set_decode_type
from am_baseline.training.trainer import _r2


def collect(model, dataset, device, value_target_norm, graph_size, batch_size):
    set_decode_type(model, "greedy")
    model.eval()
    all_v, all_t = [], []
    with torch.no_grad():
        for bat in DataLoader(dataset, batch_size=batch_size):
            x = move_to(bat, device)
            cost, _ll, pi, values = model(x, return_pi=True, compute_values=True)
            edge_costs = TSP.get_edge_costs(x, pi)
            rtg = value_targets_from_edges(edge_costs)
            if value_target_norm == 'bl':
                Z = cost.clamp(min=1e-6).unsqueeze(-1)
            else:
                Z = float(graph_size) ** 0.5
            targets = rtg / Z
            all_v.append(values.cpu())
            all_t.append(targets.cpu())
    return torch.cat(all_v, dim=0), torch.cat(all_t, dim=0)


def bucketed_r2(V, T):
    N = V.size(1)
    q = max(1, N // 4)
    early = slice(0, q)
    mid = slice(q, N - q) if N - q > q else slice(q, N)
    late = slice(N - q, N)
    return {
        'overall': _r2(V, T),
        'early': _r2(V[:, early], T[:, early]),
        'mid': _r2(V[:, mid], T[:, mid]),
        'late': _r2(V[:, late], T[:, late]),
    }


def calibration_table(V, T, n_bins=10):
    v_flat = V.flatten()
    t_flat = T.flatten()
    # Bin by prediction; compare mean target in each bin.
    lo, hi = v_flat.min().item(), v_flat.max().item()
    if hi - lo < 1e-9:
        return []
    edges = torch.linspace(lo, hi, n_bins + 1)
    rows = []
    for i in range(n_bins):
        mask = (v_flat >= edges[i]) & (v_flat <= edges[i + 1]) if i == n_bins - 1 \
               else (v_flat >= edges[i]) & (v_flat < edges[i + 1])
        count = int(mask.sum().item())
        if count == 0:
            continue
        rows.append({
            'bin_lo': edges[i].item(),
            'bin_hi': edges[i + 1].item(),
            'count': count,
            'mean_pred': v_flat[mask].mean().item(),
            'mean_target': t_flat[mask].mean().item(),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--val_size', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--value_target_norm', choices=['bl', 'sqrt_n'], default='bl')
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    model, model_args = load_model(args.model)
    model.to(device)
    if not getattr(model, 'value_enabled', False) or model.value_head is None:
        print("ERROR: this checkpoint has no value head (value_enabled=False).")
        sys.exit(1)
    graph_size = model_args['graph_size']
    print(f"Loaded model for TSP-{graph_size}")

    torch.manual_seed(args.seed)
    dataset = TSP.make_dataset(size=graph_size, num_samples=args.val_size)

    V, T = collect(model, dataset, device, args.value_target_norm, graph_size, args.batch_size)
    r2 = bucketed_r2(V, T)
    print("\nR^2:")
    for k, v in r2.items():
        print(f"  {k:8s} {v:+.4f}")

    print("\nValue stats:")
    print(f"  pred   mean={V.mean().item():.4f}  std={V.std().item():.4f}")
    print(f"  target mean={T.mean().item():.4f}  std={T.std().item():.4f}")
    print(f"  residual mean={(T - V).mean().item():+.4f}  std={(T - V).std().item():.4f}")

    rows = calibration_table(V, T, n_bins=10)
    if rows:
        print("\nCalibration (binned by prediction):")
        print(f"  {'bin_lo':>8} {'bin_hi':>8} {'count':>8} {'mean_pred':>10} {'mean_target':>12}")
        for r in rows:
            print(f"  {r['bin_lo']:8.4f} {r['bin_hi']:8.4f} {r['count']:8d} "
                  f"{r['mean_pred']:10.4f} {r['mean_target']:12.4f}")


if __name__ == '__main__':
    main()
