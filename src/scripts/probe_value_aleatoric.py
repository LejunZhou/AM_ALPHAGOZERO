"""Aleatoric residual decomposition for the Stage 4 value head.

This is a standalone diagnostic. It loads a Stage 4 checkpoint and replay
buffer, samples fixed partial states, repeatedly completes each state with
fresh MCTS, and decomposes value-head error into downstream target variance
versus systematic bias:

    E[(v(s) - z)^2 | s] = Var[z | s] + (v(s) - E[z | s])^2

The script intentionally does not modify training code or checkpoints.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch

from am_baseline.model.attention_model import AttentionModel
from am_baseline.search.mcts import MCTSConfig, MCTSSolver
from am_baseline.search.tree import MCTSNode
from am_baseline.training.coach import MCTSReplayBuffer, reconstruct_state
from am_baseline.utils.misc import torch_load_cpu


def parse_args():
    p = argparse.ArgumentParser(
        description="Decompose Stage 4 value-head residuals into aleatoric variance and bias."
    )
    p.add_argument("--ckpt", required=True, help="Stage 4 iter checkpoint path.")
    p.add_argument("--buffer", required=True, help="Stage 4 replay buffer.pt path.")
    p.add_argument("--which", choices=["best", "working"], default="best")
    p.add_argument("--num_states", type=int, default=100)
    p.add_argument("--rollouts_per_state", type=int, default=20)
    p.add_argument("--K", type=int, default=None, help="MCTS simulations per partial root.")
    p.add_argument("--leaf_eval", choices=["value_head", "rollout"], default=None)
    p.add_argument(
        "--temperature_schedule",
        choices=["const", "step10", "step30", "step50"],
        default=None,
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Root action sampling temperature. Defaults to 1.0 to match self-play.",
    )
    p.add_argument("--dirichlet_epsilon", type=float, default=None)
    p.add_argument("--dirichlet_alpha_factor", type=float, default=None)
    p.add_argument("--value_target_norm", choices=["bl", "sqrt_n", "none"], default=None)
    p.add_argument("--out_csv", default=None)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--graph_size", type=int, default=None)
    p.add_argument("--buffer_capacity", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=512, help="Reserved for future batched caching.")
    p.add_argument("--no_cuda", action="store_true")
    p.add_argument(
        "--steps",
        type=str,
        default=None,
        help="Optional comma-separated tour steps to sample, e.g. 0,5,10,19.",
    )
    return p.parse_args()


def _read_json_next_to(path: str) -> Optional[dict]:
    args_path = os.path.join(os.path.dirname(os.path.abspath(path)), "args.json")
    if not os.path.exists(args_path):
        return None
    with open(args_path, "r") as f:
        return json.load(f)


def _read_buffer_meta(path: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "graph_size": int(payload["graph_size"]),
        "capacity_instances": int(payload["capacity_instances"]),
        "n_filled_instances": int(payload.get("n_filled_instances", 0)),
        "n_filled_tuples": int(payload.get("n_filled_tuples", 0)),
    }


def _build_model_cfg(train_args: Optional[dict], graph_size: int, value_target_norm: str):
    class Cfg:
        embedding_dim = 128
        hidden_dim = 128
        n_encode_layers = 3
        n_heads = 8
        tanh_clipping = 10.0
        normalization = "batch"
        feed_forward_hidden = 512
        value_enabled = True
        value_hidden_dim = 128
        value_target_norm = "bl"
        graph_size = 20

    if train_args is not None:
        for k in (
            "embedding_dim",
            "hidden_dim",
            "n_encode_layers",
            "n_heads",
            "tanh_clipping",
            "normalization",
            "feed_forward_hidden",
            "value_enabled",
            "value_hidden_dim",
            "value_target_norm",
            "graph_size",
        ):
            if k in train_args:
                setattr(Cfg, k, train_args[k])
    # CLI-resolved value_target_norm is authoritative for interpreting the head.
    Cfg.value_target_norm = value_target_norm
    Cfg.graph_size = graph_size
    return Cfg()


def _load_stage4_model(
    ckpt_path: str,
    which: str,
    train_args: Optional[dict],
    graph_size: int,
    value_target_norm: str,
    device: torch.device,
):
    ckpt = torch_load_cpu(ckpt_path)
    key = "best_model" if which == "best" else "model"
    if key not in ckpt:
        raise KeyError(f"checkpoint has no {key!r}; available keys: {list(ckpt.keys())}")
    cfg = _build_model_cfg(train_args, graph_size, value_target_norm)
    model = AttentionModel(cfg)
    model.load_state_dict(ckpt[key])
    model.to(device)
    model.eval()
    return model


def _resolve_config(args, train_args: Optional[dict], graph_size: int) -> Dict:
    def pick(name, default):
        val = getattr(args, name)
        if val is not None:
            return val
        if train_args is not None and name in train_args:
            return train_args[name]
        return default

    if args.K is not None:
        k = int(args.K)
    elif train_args is not None and "n_simulations_train" in train_args:
        k = int(train_args["n_simulations_train"])
    else:
        k = 50

    return {
        "K": k,
        "leaf_eval": str(pick("leaf_eval", "value_head")),
        "temperature_schedule": str(pick("temperature_schedule", "step30")),
        "temperature": float(pick("temperature", 1.0)),
        "dirichlet_epsilon": float(pick("dirichlet_epsilon", 0.25)),
        "dirichlet_alpha_factor": float(pick("dirichlet_alpha_factor", 10.0)),
        "value_target_norm": str(pick("value_target_norm", "bl")),
        "n_simulations_per_step": (
            train_args.get("n_simulations_per_step")
            if train_args is not None and args.K is None
            else None
        ),
        "graph_size": int(graph_size),
    }


def _load_buffer(path: str, graph_size: int, capacity: int) -> MCTSReplayBuffer:
    buf = MCTSReplayBuffer(graph_size=graph_size, capacity_instances=capacity, device="cpu")
    buf.load(path)
    return buf


def _target_from_remaining(remaining_real: float, bl_val: float, graph_size: int, norm: str) -> float:
    if norm == "bl":
        return remaining_real / max(float(bl_val), 1e-6)
    if norm == "sqrt_n":
        return remaining_real / math.sqrt(float(graph_size))
    if norm == "none":
        return remaining_real
    raise ValueError(f"unknown value_target_norm: {norm!r}")


def _buffer_target(cost_to_go: float, bl_val: float, graph_size: int, norm: str) -> float:
    return _target_from_remaining(cost_to_go, bl_val, graph_size, norm)


def _sample_slots(
    buf: MCTSReplayBuffer,
    num_states: int,
    rng: np.random.Generator,
    steps_filter: Optional[List[int]] = None,
) -> List[int]:
    steps = [
        int(t)
        for t, arr in enumerate(buf._step_index)
        if arr.shape[0] > 0 and (steps_filter is None or t in steps_filter)
    ]
    if not steps:
        raise RuntimeError("no non-empty step buckets available for sampling")

    slots: List[int] = []
    for i in range(num_states):
        step = steps[i % len(steps)]
        candidates = buf._step_index[step]
        slots.append(int(candidates[int(rng.integers(0, candidates.shape[0]))]))
    rng.shuffle(slots)
    return slots


def _batch_from_slot(buf: MCTSReplayBuffer, slot: int) -> Dict:
    step = int(slot % buf.N)
    idx_t = torch.tensor([slot], dtype=torch.long, device=buf.device)
    inst = buf.inst_idx[idx_t].to(torch.long)
    return {
        "state_i": step,
        "coords": buf.coords[inst].clone(),
        "visited": buf.visited[idx_t].clone(),
        "first_a": buf.first_a[idx_t].to(torch.long).clone(),
        "prev_a": buf.prev_a[idx_t].to(torch.long).clone(),
        "lengths": buf.lengths[idx_t].clone(),
        "pi": buf.pi[idx_t].clone(),
        "z": (buf.cost_to_go[idx_t] / buf.bl_val[inst].clamp(min=1e-6)).clone(),
        "bl_val": buf.bl_val[inst].clone(),
        "tour_cost": buf.tour_cost[inst].clone(),
        "cost_to_go": buf.cost_to_go[idx_t].clone(),
        "slot": slot,
        "inst": int(inst.view(-1)[0].item()),
    }


@dataclass
class StateEval:
    state: object
    fixed: object
    v_pred: float
    buffer_z: float
    bl_val: float
    start_length: float
    coords: torch.Tensor


@torch.no_grad()
def _prepare_state_eval(
    model: AttentionModel,
    batch: Dict,
    device: torch.device,
    value_target_norm: str,
    graph_size: int,
) -> StateEval:
    coords = batch["coords"].to(device=device, dtype=torch.float32)
    encoded = model.encode(coords)
    fixed = model.precompute_decoder(encoded)
    state = reconstruct_state(batch, device=device)
    _log_p, _mask, glimpse = model.decode_step(fixed, state, return_glimpse=True)
    if model.value_head is None:
        raise RuntimeError("value head is disabled; cannot run value residual probe")
    v_pred = float(model.value_head(glimpse).view(-1)[0].item())
    bl_val = float(batch["bl_val"].view(-1)[0].item())
    cost_to_go = float(batch["cost_to_go"].view(-1)[0].item())
    buffer_z = _buffer_target(cost_to_go, bl_val, graph_size, value_target_norm)
    return StateEval(
        state=state,
        fixed=fixed,
        v_pred=v_pred,
        buffer_z=buffer_z,
        bl_val=bl_val,
        start_length=float(state.lengths.view(-1)[0].item()),
        coords=coords,
    )


def _k_for_step(cfg: MCTSConfig, step: int) -> int:
    if cfg.n_simulations_per_step is not None and len(cfg.n_simulations_per_step) > 0:
        if step >= len(cfg.n_simulations_per_step):
            raise IndexError(
                f"n_simulations_per_step has length {len(cfg.n_simulations_per_step)} "
                f"but reached step {step}"
            )
        return int(cfg.n_simulations_per_step[step])
    return int(cfg.n_simulations)


@torch.no_grad()
def _complete_from_partial_state(
    solver: MCTSSolver,
    base_state,
    fixed,
    bl_val: float,
) -> float:
    """Run a fresh MCTS completion from a fixed partial StateTSP.

    Returns the real remaining cost from base_state to terminal.
    """
    state = base_state
    root = None
    start_length = float(base_state.lengths.view(-1)[0].item())
    while not state.all_finished():
        if root is None or not solver.cfg.tree_reuse:
            root = MCTSNode(state=state, parent=None, action_into_me=None)
        else:
            root.parent = None
            root.action_into_me = None

        if not root.is_expanded() and not root.is_terminal():
            solver._populate_priors(root, fixed, bl_val)
        if solver.cfg.dirichlet_epsilon > 0 and not root.is_terminal():
            solver._apply_dirichlet(root)

        step = int(state.i.view(-1)[0].item())
        for _ in range(_k_for_step(solver.cfg, step)):
            solver._simulate(root, fixed, bl_val)

        action = solver._pick_root_action(root)
        state = state.update(torch.tensor([action], dtype=torch.long, device=solver.device))

        if solver.cfg.tree_reuse and action in root.children:
            root = root.children[action]
        else:
            root = None

    total_real = float(state.get_final_cost().view(-1)[0].item())
    remaining = total_real - start_length
    if remaining < -1e-6:
        raise RuntimeError(f"negative remaining cost {remaining}; state accounting is broken")
    return max(0.0, remaining)


def _make_solver(model: AttentionModel, cfg_dict: Dict, device: torch.device, seed: int) -> MCTSSolver:
    graph_size = int(cfg_dict["graph_size"])
    n_simulations_per_step = cfg_dict.get("n_simulations_per_step")
    if n_simulations_per_step is not None:
        n_simulations_per_step = tuple(int(k) for k in n_simulations_per_step)
    cfg = MCTSConfig(
        n_simulations=int(cfg_dict["K"]),
        n_simulations_per_step=n_simulations_per_step,
        c_puct=0.05,
        temperature=float(cfg_dict["temperature"]),
        temperature_schedule=cfg_dict["temperature_schedule"],
        dirichlet_alpha=float(cfg_dict["dirichlet_alpha_factor"]) / graph_size,
        dirichlet_epsilon=float(cfg_dict["dirichlet_epsilon"]),
        leaf_eval=cfg_dict["leaf_eval"],
        value_norm="bl",
        value_target_norm=cfg_dict["value_target_norm"],
        fpu_mode="running_q",
        fpu_fallback=-1.0,
        root_select="visits",
        tree_reuse=True,
        return_root_visits=False,
        seed=seed,
    )
    return MCTSSolver(model, cfg, device=device)


def _write_csv(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = [
        "state_id",
        "slot",
        "inst",
        "step",
        "bl_val",
        "start_length",
        "v_pred",
        "buffer_z",
        "mean_z",
        "var_z",
        "bias2",
        "mse_decomp",
        "empirical_mse",
        "aleatoric_fraction",
        "invariant_abs_err",
        "min_z",
        "max_z",
        "n_rollouts",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _print_summary(rows: List[Dict]) -> None:
    var_sum = sum(r["var_z"] for r in rows)
    bias_sum = sum(r["bias2"] for r in rows)
    mse_sum = var_sum + bias_sum
    frac = var_sum / mse_sum if mse_sum > 0 else float("nan")
    mean_emp = float(np.mean([r["empirical_mse"] for r in rows]))
    mean_decomp = float(np.mean([r["mse_decomp"] for r in rows]))
    mean_buffer_mse = float(np.mean([(r["v_pred"] - r["buffer_z"]) ** 2 for r in rows]))
    mean_fresh_shift2 = float(np.mean([(r["buffer_z"] - r["mean_z"]) ** 2 for r in rows]))
    mean_fresh_shift = float(np.mean([r["buffer_z"] - r["mean_z"] for r in rows]))
    max_err = max(r["invariant_abs_err"] for r in rows)
    print("\nOverall")
    print(f"  states              : {len(rows)}")
    print(f"  mean empirical MSE  : {mean_emp:.8f}")
    print(f"  mean decomp MSE     : {mean_decomp:.8f}")
    print(f"  mean buffer MSE     : {mean_buffer_mse:.8f}  (v_pred vs stored buffer_z)")
    print(f"  mean target shift^2 : {mean_fresh_shift2:.8f}  (stored buffer_z vs fresh mean_z)")
    print(f"  mean target shift   : {mean_fresh_shift:+.8f}  (stored buffer_z - fresh mean_z)")
    print(f"  weighted var_z      : {var_sum / len(rows):.8f}")
    print(f"  weighted bias2      : {bias_sum / len(rows):.8f}")
    print(f"  aleatoric fraction  : {frac:.4f}")
    print(f"  max invariant error : {max_err:.3e}")

    print("\nPer-step")
    by_step: Dict[int, List[Dict]] = {}
    for row in rows:
        by_step.setdefault(int(row["step"]), []).append(row)
    print("  step  n  var_z      bias2      frac")
    for step in sorted(by_step):
        group = by_step[step]
        vs = sum(r["var_z"] for r in group)
        bs = sum(r["bias2"] for r in group)
        denom = vs + bs
        gfrac = vs / denom if denom > 0 else float("nan")
        print(f"  {step:>4} {len(group):>2} {vs/len(group):.8f} {bs/len(group):.8f} {gfrac:.4f}")


def main():
    args = parse_args()
    if args.num_states <= 0:
        raise ValueError("--num_states must be positive")
    if args.rollouts_per_state <= 0:
        raise ValueError("--rollouts_per_state must be positive")

    train_args = _read_json_next_to(args.ckpt)
    buffer_meta = _read_buffer_meta(args.buffer)
    graph_size = int(
        args.graph_size
        or (train_args.get("graph_size") if train_args is not None and "graph_size" in train_args else 0)
        or buffer_meta["graph_size"]
    )
    capacity = int(args.buffer_capacity or buffer_meta["capacity_instances"])
    cfg_dict = _resolve_config(args, train_args, graph_size)
    value_target_norm = cfg_dict["value_target_norm"]

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    np_rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print("Loading")
    print(f"  ckpt        : {args.ckpt}")
    print(f"  buffer      : {args.buffer}")
    print(f"  args.json   : {'yes' if train_args is not None else 'no'}")
    print(f"  device      : {device}")
    print(f"  graph_size  : {graph_size}")
    print(f"  capacity    : {capacity}")
    print(f"  buffer filled instances/tuples: {buffer_meta['n_filled_instances']} / {buffer_meta['n_filled_tuples']}")
    print("MCTS config")
    for k in (
        "K",
        "leaf_eval",
        "temperature",
        "temperature_schedule",
        "dirichlet_epsilon",
        "dirichlet_alpha_factor",
        "value_target_norm",
    ):
        print(f"  {k:22s}: {cfg_dict[k]}")

    model = _load_stage4_model(
        args.ckpt, args.which, train_args, graph_size, value_target_norm, device
    )
    buf = _load_buffer(args.buffer, graph_size, capacity)

    steps_filter = None
    if args.steps:
        steps_filter = [int(x.strip()) for x in args.steps.split(",") if x.strip()]
    slots = _sample_slots(buf, args.num_states, np_rng, steps_filter=steps_filter)
    solver = _make_solver(model, cfg_dict, device, seed=args.seed + 17)

    rows: List[Dict] = []
    t0 = time.time()
    for state_id, slot in enumerate(slots):
        batch = _batch_from_slot(buf, slot)
        state_eval = _prepare_state_eval(
            model, batch, device, value_target_norm, graph_size
        )
        z_vals = []
        for _ in range(args.rollouts_per_state):
            remaining = _complete_from_partial_state(
                solver, state_eval.state, state_eval.fixed, state_eval.bl_val
            )
            z_vals.append(
                _target_from_remaining(
                    remaining, state_eval.bl_val, graph_size, value_target_norm
                )
            )
        z = np.asarray(z_vals, dtype=np.float64)
        mean_z = float(z.mean())
        var_z = float(z.var(ddof=0))
        bias2 = float((state_eval.v_pred - mean_z) ** 2)
        mse_decomp = var_z + bias2
        empirical_mse = float(((state_eval.v_pred - z) ** 2).mean())
        invariant_abs_err = abs(empirical_mse - mse_decomp)
        rows.append(
            {
                "state_id": state_id,
                "slot": int(slot),
                "inst": int(batch["inst"]),
                "step": int(batch["state_i"]),
                "bl_val": state_eval.bl_val,
                "start_length": state_eval.start_length,
                "v_pred": state_eval.v_pred,
                "buffer_z": state_eval.buffer_z,
                "mean_z": mean_z,
                "var_z": var_z,
                "bias2": bias2,
                "mse_decomp": mse_decomp,
                "empirical_mse": empirical_mse,
                "aleatoric_fraction": var_z / mse_decomp if mse_decomp > 0 else float("nan"),
                "invariant_abs_err": invariant_abs_err,
                "min_z": float(z.min()),
                "max_z": float(z.max()),
                "n_rollouts": int(args.rollouts_per_state),
            }
        )
        if (state_id + 1) % max(1, min(10, args.num_states)) == 0:
            elapsed = time.time() - t0
            print(f"  processed {state_id + 1}/{args.num_states} states ({elapsed:.1f}s)")

    _print_summary(rows)
    out_csv = args.out_csv
    if out_csv is None:
        stem = f"aleatoric_{args.which}_n{args.num_states}_m{args.rollouts_per_state}.csv"
        out_csv = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), stem)
    _write_csv(out_csv, rows)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
