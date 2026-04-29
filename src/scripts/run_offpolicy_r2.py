"""Stage 3 Phase E.1 — off-policy R² probe.

Runs MCTS rollout on a TSP dataset with a value-head logging hook that records
(v_predicted, z_realized) at every leaf where rollout fires. Computes R² on
the off-policy distribution (states that MCTS actually visits during search)
and compares to Stage 1's in-distribution R² (training-time canonical: 0.9965).

Output:
  outputs/stage3/value_head_offpolicy_r2_tsp20.csv
  outputs/stage3/value_head_offpolicy_r2_tsp20_summary.txt

Backend is C++ (uses CppMCTSSolver with `enable_r2_log=True`).

Example:
  PYTHONPATH=src python -m scripts.run_offpolicy_r2 \
      --model outputs/tsp_20/stage1_tsp20_canonical_<ts>/epoch-99.pt \
      --graph_size 20 --val_size 1000 --seed 1234 --n_simulations 400
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

from am_baseline.problem.tsp import TSP
from am_baseline.search import CppMCTSSolver, HAVE_CPP_MCTS, MCTSConfig
from am_baseline.utils.misc import load_model


def r2_score(v: np.ndarray, z: np.ndarray) -> float:
    """Coefficient of determination R² = 1 - SS_res / SS_tot. NaN-safe for SS_tot=0."""
    if v.size == 0:
        return float("nan")
    ss_res = float(((z - v) ** 2).sum())
    ss_tot = float(((z - z.mean()) ** 2).sum())
    if ss_tot <= 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage 3 E.1: off-policy R² probe via cpp MCTS."
    )
    parser.add_argument('--model', required=True,
                        help='Path to model checkpoint (.pt) — must have a value head')
    parser.add_argument('--graph_size', type=int, default=None,
                        help='Override graph size (default: read from checkpoint args.json)')
    parser.add_argument('--val_size', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--n_simulations', type=int, default=400)
    parser.add_argument('--c_puct', type=float, default=0.05)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--value_norm', choices=['bl', 'sqrt_n'], default='bl')
    parser.add_argument('--fpu_mode', choices=['fallback', 'running_q', 'node_value'],
                        default='running_q')
    parser.add_argument('--fpu_fallback', type=float, default=-1.0)
    parser.add_argument('--root_select', choices=['visits', 'q'], default='visits')
    tree_reuse_group = parser.add_mutually_exclusive_group()
    tree_reuse_group.add_argument('--tree_reuse', dest='tree_reuse', action='store_true')
    tree_reuse_group.add_argument('--no_tree_reuse', dest='tree_reuse', action='store_false')
    parser.set_defaults(tree_reuse=True)
    parser.add_argument('--output_csv', type=str,
                        default='outputs/stage3/value_head_offpolicy_r2_tsp20.csv')
    parser.add_argument('--output_summary', type=str,
                        default='outputs/stage3/value_head_offpolicy_r2_tsp20_summary.txt')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--no_progress_bar', action='store_true')
    parser.add_argument('--stage1_in_dist_r2', type=float, default=0.9965,
                        help="Reference Stage 1 in-distribution R² for the comparison summary.")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"device={device}")

    if not HAVE_CPP_MCTS:
        print("ERROR: cpp MCTS extension not built. Run `pip install -e .` "
              "in the AM_AlphaGoZero env.", file=sys.stderr)
        return 2

    model, model_args = load_model(args.model)
    model.to(device).eval()
    if model.value_head is None:
        print("ERROR: checkpoint has no value head — off-policy R² probe requires "
              "a Stage 1 value-head'd checkpoint.", file=sys.stderr)
        return 2

    graph_size = args.graph_size if args.graph_size is not None else model_args['graph_size']
    print(f"Loaded AttentionModel for TSP-{graph_size} from {args.model}")

    torch.manual_seed(args.seed)
    dataset = TSP.make_dataset(size=graph_size, num_samples=args.val_size)
    inputs = torch.stack([dataset[i] for i in range(len(dataset))], dim=0)
    B = inputs.size(0)
    print(f"Dataset: {B} instances of TSP-{graph_size}, seed={args.seed}")

    cfg = MCTSConfig(
        n_simulations=args.n_simulations,
        c_puct=args.c_puct,
        temperature=args.temperature,
        leaf_eval='rollout',
        value_norm=args.value_norm,
        fpu_mode=args.fpu_mode,
        fpu_fallback=args.fpu_fallback,
        root_select=args.root_select,
        tree_reuse=args.tree_reuse,
        seed=args.seed,
        # Sequential MCTS — required for FIFO pairing in the r2_log hook.
        simulation_batch_size=1,
    )
    solver = CppMCTSSolver(model, cfg, device=device)
    print(f"MCTSConfig: {cfg}")

    with torch.no_grad():
        bl_vals = solver._compute_bl_val_batch(inputs.to(device))

    all_records: list[dict] = []
    costs = torch.empty(B)
    decode_steps = torch.empty(B, dtype=torch.long)
    rollout_steps = torch.empty(B, dtype=torch.long)
    leaves_per_inst = torch.empty(B, dtype=torch.long)

    t0 = time.time()
    iterator = range(B)
    if not args.no_progress_bar:
        iterator = tqdm(iterator, desc="MCTS R² probe")
    for i in iterator:
        c_i, _ = solver.solve_instance(
            inputs[i:i+1].to(device),
            bl_val=float(bl_vals[i].item()),
            enable_r2_log=True,
        )
        costs[i] = c_i.detach().cpu()
        decode_steps[i] = solver.fwd_count_decode
        rollout_steps[i] = solver.fwd_count_rollout
        leaves_per_inst[i] = len(solver.r2_records)
        for k, rec in enumerate(solver.r2_records):
            all_records.append({
                "instance_idx": i,
                "leaf_idx": k,
                "step": int(rec["step"]),
                "v_predicted": float(rec["v_predicted"]),
                "z_realized": float(rec["z_realized"]),
            })
    elapsed = time.time() - t0

    print()
    print(f"MCTS rollout K={args.n_simulations} on {B} instances of TSP-{graph_size}:")
    print(f"  mean cost      : {costs.mean().item():.4f}")
    print(f"  wall-clock     : {elapsed:.1f}s  ({elapsed / B * 1000:.1f} ms/inst)")
    print(f"  decode_steps   : mean={decode_steps.float().mean().item():.1f} per inst")
    print(f"  rollout_steps  : mean={rollout_steps.float().mean().item():.1f} per inst")
    print(f"  R² leaf rows   : {len(all_records)} total "
          f"(mean {leaves_per_inst.float().mean().item():.1f} per inst)")

    if not all_records:
        print("ERROR: no R² records collected — log_offpolicy hook never fired.",
              file=sys.stderr)
        return 3

    # --- Compute R² ---
    v = np.fromiter((r["v_predicted"] for r in all_records), dtype=np.float64,
                    count=len(all_records))
    z = np.fromiter((r["z_realized"] for r in all_records), dtype=np.float64,
                    count=len(all_records))
    steps = np.fromiter((r["step"] for r in all_records), dtype=np.int64,
                        count=len(all_records))

    r2_overall = r2_score(v, z)

    # Bucket by tour-step matching Stage 1's training-time bucket convention
    # (early = [0, N/4), mid = [N/4, N - N/4), late = [N - N/4, N)).
    n = graph_size
    early_mask = steps < n // 4
    late_mask = steps >= n - n // 4
    mid_mask = ~(early_mask | late_mask)
    r2_early = r2_score(v[early_mask], z[early_mask])
    r2_mid = r2_score(v[mid_mask], z[mid_mask])
    r2_late = r2_score(v[late_mask], z[late_mask])

    resid = z - v
    print()
    print("Off-policy R² results:")
    print(f"  R² overall                     : {r2_overall:.4f}  ({len(all_records)} pts)")
    print(f"  R² early  (step <  {n // 4:>2})         : {r2_early:.4f}  "
          f"({int(early_mask.sum())} pts)")
    print(f"  R² mid    ({n // 4:>2} <= step < {n - n // 4:>2})  : {r2_mid:.4f}  "
          f"({int(mid_mask.sum())} pts)")
    print(f"  R² late   (step >= {n - n // 4:>2})        : {r2_late:.4f}  "
          f"({int(late_mask.sum())} pts)")
    print(f"  Mean residual (z - v)          : {resid.mean():+.6f}")
    print(f"  Std residual                   : {resid.std():.6f}")
    print(f"  Mean v_predicted               : {v.mean():.6f}")
    print(f"  Mean z_realized                : {z.mean():.6f}")
    print()
    print(f"Stage 1 in-distribution R² (reference): {args.stage1_in_dist_r2:.4f}")
    print(f"Off-policy delta vs in-dist            : {r2_overall - args.stage1_in_dist_r2:+.4f}")

    # --- Write CSV ---
    if args.output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)) or ".", exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["instance_idx", "leaf_idx", "step", "v_predicted", "z_realized"])
            for r in all_records:
                w.writerow([
                    r["instance_idx"], r["leaf_idx"], r["step"],
                    f"{r['v_predicted']:.6f}", f"{r['z_realized']:.6f}",
                ])
        print(f"Wrote {args.output_csv}")

    # --- Summary file ---
    if args.output_summary:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_summary)) or ".", exist_ok=True)
        with open(args.output_summary, "w") as f:
            f.write("Off-policy R² probe — Stage 3 Phase E.1\n")
            f.write(f"Model     : {args.model}\n")
            f.write(f"Graph size: TSP-{graph_size}\n")
            f.write(f"val_size  : {B}\n")
            f.write(f"seed      : {args.seed}\n")
            f.write(f"K         : {args.n_simulations}\n")
            f.write(f"leaf_eval : rollout\n")
            f.write(f"value_norm: {args.value_norm}\n")
            f.write(f"tree_reuse: {args.tree_reuse}\n")
            f.write(f"Wall-clock: {elapsed:.1f} s ({elapsed / B * 1000:.1f} ms/inst)\n\n")
            f.write(f"Total leaves logged: {len(all_records)} "
                    f"(mean {leaves_per_inst.float().mean().item():.1f} per inst)\n\n")
            f.write(f"R² overall                       : {r2_overall:.4f}  "
                    f"({len(all_records)} pts)\n")
            f.write(f"R² early  (step <  {n // 4:>2})           : {r2_early:.4f}  "
                    f"({int(early_mask.sum())} pts)\n")
            f.write(f"R² mid    ({n // 4:>2} <= step < {n - n // 4:>2})    : {r2_mid:.4f}  "
                    f"({int(mid_mask.sum())} pts)\n")
            f.write(f"R² late   (step >= {n - n // 4:>2})          : {r2_late:.4f}  "
                    f"({int(late_mask.sum())} pts)\n\n")
            f.write(f"Mean residual (z - v): {resid.mean():+.6f}\n")
            f.write(f"Std residual         : {resid.std():.6f}\n")
            f.write(f"Mean v_predicted     : {v.mean():.6f}\n")
            f.write(f"Mean z_realized      : {z.mean():.6f}\n\n")
            f.write(f"Stage 1 in-distribution R² (reference): {args.stage1_in_dist_r2:.4f}\n")
            f.write(f"Off-policy delta vs in-distribution    : "
                    f"{r2_overall - args.stage1_in_dist_r2:+.4f}\n")
        print(f"Wrote {args.output_summary}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
