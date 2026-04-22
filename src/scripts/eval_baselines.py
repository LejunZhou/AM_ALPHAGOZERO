"""
Evaluate baseline TSP solvers and compare with AM model.

Usage:
  python src/scripts/eval_baselines.py --graph_size 20 --num_samples 10000
  python src/scripts/eval_baselines.py --graph_size 20 --num_samples 10000 --model_path outputs/tsp_20/.../epoch-99.pt
  python src/scripts/eval_baselines.py --graph_size 20 --num_samples 1000 --gurobi  # exact solver (slower)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import math
import argparse
import itertools
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.spatial import distance_matrix
from tqdm import tqdm

from am_baseline.config import Config
from am_baseline.model.attention_model import AttentionModel, set_decode_type
from am_baseline.problem.tsp import TSP
from am_baseline.utils.misc import torch_load_cpu


# ---------------------------------------------------------------------------
# Classical heuristics (ported from ref/attention-learn-to-route-master)
# ---------------------------------------------------------------------------

def calc_tsp_length(loc, tour):
    """Calculate TSP tour length given locations and tour order."""
    loc = np.array(loc)
    sorted_locs = loc[np.concatenate((tour, [tour[0]]))]
    return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()


def _calc_insert_cost(D, prv, nxt, ins):
    return D[prv, ins] + D[ins, nxt] - D[prv, nxt]


def run_insertion(loc, method):
    """Insertion heuristic: random, nearest, or farthest."""
    n = len(loc)
    D = distance_matrix(loc, loc)
    mask = np.zeros(n, dtype=bool)
    tour = []
    for i in range(n):
        feas = mask == 0
        feas_ind = np.flatnonzero(feas)
        if method == 'random':
            a = i
        elif method == 'nearest':
            if i == 0:
                a = 0
            else:
                a = feas_ind[D[np.ix_(feas, ~feas)].min(1).argmin()]
        elif method == 'farthest':
            if i == 0:
                a = D.max(1).argmax()
            else:
                a = feas_ind[D[np.ix_(feas, ~feas)].min(1).argmax()]
        else:
            raise ValueError(f"Unknown insertion method: {method}")
        mask[a] = True
        if len(tour) == 0:
            tour = [a]
        else:
            ind_insert = np.argmin(_calc_insert_cost(D, tour, np.roll(tour, -1), a))
            tour.insert(ind_insert + 1, a)
    cost = D[tour, np.roll(tour, -1)].sum()
    return cost, tour


def nearest_neighbour_batch(dataset, device):
    """Batched nearest-neighbour using PyTorch (from reference code)."""
    dist = (dataset[:, :, None, :] - dataset[:, None, :, :]).norm(dim=-1)
    batch_size, graph_size, _ = dataset.size()
    total_dist = dataset.new(batch_size).zero_()
    current = dataset.new(batch_size).long().zero_()  # start from node 0
    dist_to_start = dist[:, :, 0].clone()
    tour = [current.clone()]
    for i in range(graph_size - 1):
        dist.scatter_(2, current.view(-1, 1, 1).expand(batch_size, graph_size, 1), float('inf'))
        nn_dist = torch.gather(dist, 1, current.view(-1, 1, 1).expand(batch_size, 1, graph_size)).squeeze(1)
        min_nn_dist, current = nn_dist.min(1)
        total_dist += min_nn_dist
        tour.append(current.clone())
    total_dist += torch.gather(dist_to_start, 1, current.view(-1, 1)).squeeze(1)
    return total_dist


# ---------------------------------------------------------------------------
# LKH-3 via TSPLIB files + subprocess (following ref/problems/tsp/tsp_baseline.py)
# ---------------------------------------------------------------------------

LKH_EXECUTABLE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'solvers', 'LKH-3.0.4', 'LKH')
)


def write_tsplib(filename, loc, name="problem"):
    """Write TSP instance in TSPLIB format (coords scaled to int, same as reference)."""
    with open(filename, 'w') as f:
        f.write("\n".join([
            "{} : {}".format(k, v)
            for k, v in (
                ("NAME", name),
                ("TYPE", "TSP"),
                ("DIMENSION", len(loc)),
                ("EDGE_WEIGHT_TYPE", "EUC_2D"),
            )
        ]))
        f.write("\n")
        f.write("NODE_COORD_SECTION\n")
        f.write("\n".join([
            "{}\t{}\t{}".format(i + 1, int(x * 10000000 + 0.5), int(y * 10000000 + 0.5))
            for i, (x, y) in enumerate(loc)
        ]))
        f.write("\n")
        f.write("EOF\n")


def write_lkh_par(filename, parameters):
    """Write LKH parameter file."""
    default_parameters = {
        "MAX_TRIALS": 10000,
        "RUNS": 10,
        "TRACE_LEVEL": 0,
        "SEED": 1234,
    }
    with open(filename, 'w') as f:
        for k, v in {**default_parameters, **parameters}.items():
            if v is None:
                f.write("{}\n".format(k))
            else:
                f.write("{} = {}\n".format(k, v))


def read_tsplib_tour(filename):
    """Read tour from TSPLIB tour file."""
    with open(filename, 'r') as f:
        tour = []
        dimension = 0
        started = False
        for line in f:
            if started:
                loc = int(line)
                if loc == -1:
                    break
                tour.append(loc)
            if line.startswith("DIMENSION"):
                dimension = int(line.split(" ")[-1])
            if line.startswith("TOUR_SECTION"):
                started = True
    assert len(tour) == dimension
    return (np.array(tour) - 1).tolist()  # TSPLIB is 1-indexed


def solve_lkh_binary(loc, executable=None, runs=1):
    """Solve TSP using LKH-3 binary via TSPLIB files."""
    import tempfile
    from subprocess import check_call

    if executable is None:
        executable = LKH_EXECUTABLE
    # Add .exe on Windows if needed
    if sys.platform == 'win32' and not executable.endswith('.exe'):
        executable = executable + '.exe'

    with tempfile.TemporaryDirectory() as tmpdir:
        problem_file = os.path.join(tmpdir, "problem.tsp")
        tour_file = os.path.join(tmpdir, "output.tour")
        param_file = os.path.join(tmpdir, "params.par")

        write_tsplib(problem_file, loc)
        write_lkh_par(param_file, {
            "PROBLEM_FILE": problem_file,
            "OUTPUT_TOUR_FILE": tour_file,
            "RUNS": runs,
            "SEED": 1234,
        })
        check_call([executable, param_file],
                    stdout=open(os.devnull, 'w'), stderr=open(os.devnull, 'w'))
        tour = read_tsplib_tour(tour_file)

    cost = calc_tsp_length(loc, tour)
    return cost, tour


def solve_lkh_elkai(loc):
    """Solve TSP using LKH (via elkai wrapper). Fast, near-optimal."""
    import elkai
    n = len(loc)
    D = distance_matrix(loc, loc)
    D_int = (D * 10000000 + 0.5).astype(int).tolist()
    tour = elkai.solve_int_matrix(D_int)
    if len(tour) == n + 1:
        tour = tour[:-1]
    cost = calc_tsp_length(loc, tour)
    return cost, tour


# ---------------------------------------------------------------------------
# Gurobi exact solver (ported from ref/problems/tsp/tsp_gurobi.py)
# ---------------------------------------------------------------------------

def solve_gurobi(loc):
    """Solve TSP to optimality using Gurobi MIP with lazy subtour elimination."""
    from gurobipy import Model, GRB, quicksum, tuplelist

    n = len(loc)

    def subtourelim(model, where):
        if where == GRB.Callback.MIPSOL:
            vals = model.cbGetSolution(model._vars)
            selected = tuplelist(
                (i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5
            )
            tour = subtour(selected)
            if len(tour) < n:
                model.cbLazy(
                    quicksum(model._vars[i, j]
                             for i, j in itertools.combinations(tour, 2))
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
                neighbors = [j for i, j in edges.select(current, '*')
                             if j in unvisited]
            if len(cycle) > len(thiscycle):
                cycle = thiscycle
        return cycle

    dist = {
        (i, j): math.sqrt(sum((loc[i][k] - loc[j][k]) ** 2 for k in range(2)))
        for i in range(n) for j in range(i)
    }

    m = Model()
    m.Params.outputFlag = False
    m.Params.threads = 1

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


# ---------------------------------------------------------------------------
# AM model evaluation
# ---------------------------------------------------------------------------

def eval_am_model(model_path, dataset, graph_size, device):
    """Load and evaluate an AM model checkpoint."""
    opts = Config(graph_size=graph_size)
    opts.device = device
    model = AttentionModel(opts).to(device)

    load_data = torch_load_cpu(model_path)
    model_state = load_data.get('model', load_data)

    # Handle reference model key remapping
    from am_baseline.utils.misc import _remap_ref_keys
    current_keys = set(model.state_dict().keys())
    if not all(k in current_keys for k in model_state.keys()):
        model_state = _remap_ref_keys(model_state)

    model.load_state_dict({**model.state_dict(), **model_state})

    set_decode_type(model, "greedy")
    model.eval()

    costs = []
    dataloader = DataLoader(dataset, batch_size=1024)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="AM greedy"):
            cost, _ = model(batch.to(device))
            costs.append(cost.cpu())
    return torch.cat(costs, 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate TSP baselines")
    parser.add_argument('--graph_size', type=int, default=20)
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--model_path', type=str, default=None,
                        help="Path to AM checkpoint (optional)")
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--lkh', action='store_true', help="Run LKH solver (via elkai, fast)")
    parser.add_argument('--lkh3', action='store_true', help="Run LKH-3 binary solver (via TSPLIB files)")
    parser.add_argument('--gurobi', action='store_true', help="Run Gurobi exact solver")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    # Generate test dataset
    print(f"Generating {args.num_samples} TSP-{args.graph_size} instances (seed={args.seed})...")
    dataset = TSP.make_dataset(size=args.graph_size, num_samples=args.num_samples)
    data_np = [inst.numpy() for inst in dataset]

    results = {}

    # --- Gurobi exact solver (optional, slow) ---
    if args.gurobi:
        print("\n[*] Gurobi (exact)...")
        t0 = time.time()
        costs = []
        for inst in tqdm(data_np, desc="Gurobi"):
            c, _ = solve_gurobi(inst.tolist())
            costs.append(c)
        costs = np.array(costs)
        results['Gurobi (optimal)'] = (costs.mean(), costs.std(), time.time() - t0)

    # --- LKH via elkai (optional, fast) ---
    if args.lkh:
        print("\n[*] LKH (elkai)...")
        t0 = time.time()
        costs = []
        for inst in tqdm(data_np, desc="LKH (elkai)"):
            c, _ = solve_lkh_elkai(inst)
            costs.append(c)
        costs = np.array(costs)
        results['LKH (elkai)'] = (costs.mean(), costs.std(), time.time() - t0)

    # --- LKH-3 binary via TSPLIB files (optional, reference-faithful) ---
    if args.lkh3:
        print("\n[*] LKH-3 (binary)...")
        t0 = time.time()
        costs = []
        for inst in tqdm(data_np, desc="LKH-3 (binary)"):
            c, _ = solve_lkh_binary(inst.tolist())
            costs.append(c)
        costs = np.array(costs)
        results['LKH-3 (binary)'] = (costs.mean(), costs.std(), time.time() - t0)

    # --- AM model (if checkpoint provided) ---
    if args.model_path:
        print(f"\n[*] AM Model ({args.model_path})...")
        t0 = time.time()
        am_costs = eval_am_model(args.model_path, dataset, args.graph_size, device)
        am_time = time.time() - t0
        am_np = am_costs.numpy()
        results['AM (greedy)'] = (am_np.mean(), am_np.std(), am_time)

    # --- Nearest Neighbour (batched, fast) ---
    print("\n[*] Nearest Neighbour...")
    t0 = time.time()
    dataloader = DataLoader(dataset, batch_size=1024)
    nn_costs = []
    for batch in dataloader:
        nn_costs.append(nearest_neighbour_batch(batch, device))
    nn_costs = torch.cat(nn_costs, 0).numpy()
    results['Nearest Neighbour'] = (nn_costs.mean(), nn_costs.std(), time.time() - t0)

    # --- Insertion heuristics ---
    for method_name, method_key in [('Farthest Insertion', 'farthest'),
                                     ('Random Insertion', 'random'),
                                     ('Nearest Insertion', 'nearest')]:
        print(f"\n[*] {method_name}...")
        t0 = time.time()
        costs = []
        for inst in tqdm(data_np, desc=method_name):
            c, _ = run_insertion(inst, method_key)
            costs.append(c)
        costs = np.array(costs)
        results[method_name] = (costs.mean(), costs.std(), time.time() - t0)

    # --- Print results table ---
    print("\n" + "=" * 70)
    print(f"TSP-{args.graph_size} Baseline Comparison ({args.num_samples} instances)")
    print("=" * 70)
    print(f"{'Method':<25} {'Avg Cost':>10} {'Std':>10} {'Time (s)':>10}")
    print("-" * 70)

    # Sort by avg cost
    for name, (avg, std, elapsed) in sorted(results.items(), key=lambda x: x[1][0]):
        print(f"{name:<25} {avg:>10.4f} {std:>10.4f} {elapsed:>10.2f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
