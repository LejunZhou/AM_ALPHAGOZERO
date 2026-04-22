"""
Evaluation script for trained AM models.

Usage:
  python src/scripts/evaluate.py --model outputs/tsp_20/.../epoch-99.pt --decode greedy
  python src/scripts/evaluate.py --model outputs/tsp_20/.../epoch-99.pt --decode sample --width 1280
  python src/scripts/evaluate.py --model ref/attention-learn-to-route-master/pretrained/tsp_20 --decode greedy
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from am_baseline.utils.misc import load_model
from am_baseline.utils.tensor_ops import move_to
from am_baseline.problem.tsp import TSP


def evaluate(model, dataset, decode_type='greedy', width=1, device='cpu', batch_size=256):
    model.eval()
    model.to(device)

    results = []
    start = time.time()

    if decode_type == 'greedy':
        model.set_decode_type('greedy')
        for batch in tqdm(DataLoader(dataset, batch_size=batch_size)):
            with torch.no_grad():
                cost, _ = model(move_to(batch, device))
            results.append(cost.cpu())

    elif decode_type == 'sample':
        model.set_decode_type('sampling')
        for batch in tqdm(DataLoader(dataset, batch_size=batch_size)):
            with torch.no_grad():
                batch = move_to(batch, device)
                # Split width into manageable chunks
                batch_rep = min(width, 128)
                iter_rep = (width + batch_rep - 1) // batch_rep
                _, best_cost = model.sample_many(batch, batch_rep=batch_rep, iter_rep=iter_rep)
            results.append(best_cost.cpu())
    else:
        raise ValueError(f"Unknown decode type: {decode_type}")

    elapsed = time.time() - start
    costs = torch.cat(results)
    return costs, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to model checkpoint or directory')
    parser.add_argument('--decode', default='greedy', choices=['greedy', 'sample'])
    parser.add_argument('--width', type=int, default=1280, help='Sampling width (for sample decode)')
    parser.add_argument('--dataset', default=None, help='Path to test dataset (.pkl)')
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--no_cuda', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    model, model_args = load_model(args.model)
    model.to(device)
    graph_size = model_args['graph_size']
    print(f"Loaded model for TSP-{graph_size}")

    # Dataset
    if args.dataset:
        dataset = TSP.make_dataset(filename=args.dataset, num_samples=args.num_samples)
    else:
        torch.manual_seed(args.seed)
        dataset = TSP.make_dataset(size=graph_size, num_samples=args.num_samples)

    # Evaluate
    print(f"\nEvaluating {args.decode} (width={args.width if args.decode == 'sample' else 1})...")
    costs, elapsed = evaluate(model, dataset, args.decode, args.width, device, args.batch_size)

    print(f"\nResults for TSP-{graph_size} ({len(costs)} instances):")
    print(f"  Mean cost: {costs.mean().item():.4f}")
    print(f"  Std:       {costs.std().item():.4f}")
    print(f"  Min:       {costs.min().item():.4f}")
    print(f"  Max:       {costs.max().item():.4f}")
    print(f"  Time:      {elapsed:.1f}s ({elapsed/len(costs)*1000:.1f}ms/instance)")


if __name__ == '__main__':
    main()
