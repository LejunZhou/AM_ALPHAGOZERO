"""Stage 3 CLI: compute per-instance optima for a TSP dataset.

Reuses the solver implementations from `src/scripts/eval_baselines.py` but
writes a per-instance CSV with the same `idx,...,cost` schema that the Stage 3
aggregator expects, so the comparison plot can use true per-instance gap.

Usage:
  PYTHONPATH=src python -m scripts.run_optima \
      --solver gurobi --graph_size 50 --val_size 1000 --seed 1234 \
      --output_csv outputs/baselines/tsp50_gurobi_seed1234.csv

  PYTHONPATH=src python -m scripts.run_optima \
      --solver lkh_elkai --graph_size 100 --val_size 1000 --seed 1234 \
      --output_csv outputs/baselines/tsp100_lkh_seed1234.csv
"""
import argparse
import csv
import os
import sys
import time

import torch
from tqdm import tqdm

from am_baseline.problem.tsp import TSP


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute per-instance optima for a TSP dataset")
    parser.add_argument('--solver', required=True,
                        choices=['gurobi', 'lkh_elkai', 'lkh3'],
                        help="Solver: 'gurobi' (exact MIP), 'lkh_elkai' (LKH via elkai), "
                             "'lkh3' (LKH-3 binary via TSPLIB files)")
    parser.add_argument('--graph_size', type=int, required=True)
    parser.add_argument('--val_size', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--time_limit', type=float, default=None,
                        help='Optional per-instance time limit in seconds. Only honored '
                             'by Gurobi (uses TimeLimit param). Other solvers ignore.')
    parser.add_argument('--no_progress_bar', action='store_true')
    args = parser.parse_args()

    # Defer scipy + solver imports until after arg parse.
    from scripts import eval_baselines as eb

    torch.manual_seed(args.seed)
    dataset = TSP.make_dataset(size=args.graph_size, num_samples=args.val_size)
    data_np = [inst.numpy().tolist() for inst in dataset]
    n_inst = len(data_np)
    print(f"Dataset: {n_inst} instances of TSP-{args.graph_size}, seed={args.seed}")

    if args.solver == 'gurobi':
        from gurobipy import GRB

        def solve(loc):
            # Wrap eval_baselines.solve_gurobi to honor TimeLimit if set.
            # The base function builds + optimizes a fresh model each call;
            # easiest path: monkey-patch a TimeLimit via env after Model() is
            # created — but eb.solve_gurobi doesn't expose that hook. So if a
            # time_limit is requested, fall back to a thin reimplementation.
            if args.time_limit is None:
                return eb.solve_gurobi(loc)
            # Replicate eb.solve_gurobi but set TimeLimit.
            import math, itertools
            from gurobipy import Model, quicksum, tuplelist
            n = len(loc)
            def subtourelim(model, where):
                if where == GRB.Callback.MIPSOL:
                    vals = model.cbGetSolution(model._vars)
                    selected = tuplelist((i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5)
                    tour = subtour(selected)
                    if len(tour) < n:
                        model.cbLazy(
                            quicksum(model._vars[i, j] for i, j in itertools.combinations(tour, 2))
                            <= len(tour) - 1
                        )
            def subtour(edges):
                unvisited = list(range(n))
                cycle = range(n + 1)
                while unvisited:
                    thiscycle = []
                    neighbors = unvisited
                    while neighbors:
                        current = neighbors[0]
                        thiscycle.append(current)
                        unvisited.remove(current)
                        neighbors = [j for i, j in edges.select(current, '*') if j in unvisited]
                    if len(cycle) > len(thiscycle):
                        cycle = thiscycle
                return cycle
            dist = {(i, j): math.sqrt(sum((loc[i][k] - loc[j][k]) ** 2 for k in range(2)))
                    for i in range(n) for j in range(i)}
            m = Model()
            m.Params.outputFlag = False
            m.Params.threads = 1
            m.Params.TimeLimit = args.time_limit
            evars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')
            for i, j in list(evars.keys()):
                evars[j, i] = evars[i, j]
            m.addConstrs(evars.sum(i, '*') == 2 for i in range(n))
            m._vars = evars
            m.Params.lazyConstraints = 1
            m.optimize(subtourelim)
            vals = m.getAttr('x', evars)
            selected = tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
            tour = subtour(selected)
            return m.objVal, tour
    elif args.solver == 'lkh_elkai':
        solve = eb.solve_lkh_elkai
    elif args.solver == 'lkh3':
        solve = eb.solve_lkh_binary

    iterator = data_np
    if not args.no_progress_bar:
        iterator = tqdm(data_np, desc=f"{args.solver}")

    costs = []
    times = []
    t0 = time.time()
    for loc in iterator:
        t1 = time.time()
        c, _ = solve(loc)
        costs.append(float(c))
        times.append(time.time() - t1)
    elapsed = time.time() - t0

    # Summary.
    mean_cost = sum(costs) / n_inst
    print()
    print(f"{args.solver} on {n_inst} TSP-{args.graph_size} instances:")
    print(f"  mean cost  : {mean_cost:.6f}")
    print(f"  min cost   : {min(costs):.6f}")
    print(f"  max cost   : {max(costs):.6f}")
    print(f"  wall-clock : {elapsed:.1f}s  ({elapsed/n_inst*1000:.1f} ms/inst)")

    # CSV.
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)) or '.', exist_ok=True)
    with open(args.output_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['idx', 'optimum_cost', 'solve_time_s', 'solver'])
        for i, (c, t) in enumerate(zip(costs, times)):
            w.writerow([i, f"{c:.6f}", f"{t:.4f}", args.solver])
    print(f"Wrote {args.output_csv}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
