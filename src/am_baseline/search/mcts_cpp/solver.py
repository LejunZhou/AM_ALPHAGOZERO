"""Python wrapper for the optional pybind11 MCTS backend."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict
from typing import Optional, Tuple

import numpy as np
import torch

from am_baseline.model.attention_model import AttentionModel
from am_baseline.problem.state import StateTSP
from am_baseline.search.mcts import MCTSConfig, MCTSSolver

try:
    from am_baseline.search.mcts_cpp import _mcts_cpp
    HAVE_CPP_MCTS = True
    _IMPORT_ERROR: Optional[BaseException] = None
except ImportError as exc:  # pragma: no cover - exercised only before extension build.
    _mcts_cpp = None
    HAVE_CPP_MCTS = False
    _IMPORT_ERROR = exc


class CppMCTSSolver:
    """Drop-in solver wrapper whose tree walk runs in C++.

    The AttentionModel forward boundary intentionally remains in Python/PyTorch:
    C++ owns node/state/PUCT bookkeeping, calls a small Python evaluator for
    `decode_step` and optional `value_head`, then continues the search.
    """

    def __init__(
        self,
        model: AttentionModel,
        cfg: MCTSConfig,
        device: Optional[torch.device] = None,
    ):
        if _mcts_cpp is None:
            raise ImportError(
                "The C++ MCTS extension is not built. Run `pip install -e .` "
                "inside the AM_AlphaGoZero environment, then retry `--backend cpp`."
            ) from _IMPORT_ERROR

        MCTSSolver._validate_config(cfg, model)
        self.model = model
        self.cfg = cfg
        self.device = device if device is not None else next(model.parameters()).device
        self.model.eval()

        self.fwd_count_decode = 0
        self.fwd_count_value = 0
        self.fwd_count_rollout = 0
        self.eval_cache_hits = 0
        self.eval_cache_misses = 0
        self.batch_eval_calls = 0
        self.batch_eval_rows = 0
        self.pending_batch_calls = 0
        self.pending_batch_rows = 0
        self.pending_collection_attempts = 0
        self.pending_collection_successes = 0
        self.virtual_collision_count = 0
        self.max_virtual_visits_remaining = 0
        # Stage 3 Phase E.1 — off-policy R² probe records. Populated only when
        # `enable_r2_log=True` is passed to `solve_instance`; one row per
        # rollout-leaf, in the order C++ visits them:
        #   {"step": tour-step index of the leaf,
        #    "v_predicted": value_head(glimpse) at the leaf state,
        #    "z_realized": (greedy rollout remaining real cost) / bl_val}
        self.r2_records: list = []

    @torch.no_grad()
    def solve_batch(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = inputs.to(self.device)
        bsz, n_nodes, _ = inputs.shape
        bl_vals = self._compute_bl_val_batch(inputs)

        costs = torch.empty(bsz, device=self.device)
        tours = torch.empty(bsz, n_nodes, dtype=torch.long, device=self.device)
        for i in range(bsz):
            cost_i, tour_i = self.solve_instance(
                inputs[i : i + 1],
                bl_val=float(bl_vals[i].item()),
            )
            costs[i] = cost_i
            tours[i] = tour_i
        return costs, tours

    def _compute_bl_val_batch(self, inputs: torch.Tensor) -> torch.Tensor:
        bsz, n_nodes, _ = inputs.shape
        if self.cfg.value_norm == "sqrt_n":
            return torch.full((bsz,), math.sqrt(n_nodes), device=self.device)
        prev = self.model.decoder.decode_type
        self.model.set_decode_type("greedy")
        cost, _ = self.model(inputs)
        if prev is not None:
            self.model.set_decode_type(prev)
        return cost.detach()

    @torch.no_grad()
    def solve_instance(
        self,
        input_1: torch.Tensor,
        bl_val: Optional[float] = None,
        enable_r2_log: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert input_1.dim() == 3 and input_1.size(0) == 1, "solve_instance expects (1, N, 2)"

        input_1 = input_1.to(self.device)
        if bl_val is None:
            bl_val = float(self._compute_bl_val_batch(input_1).item())

        if enable_r2_log:
            if self.cfg.leaf_eval != "rollout":
                raise ValueError(
                    "enable_r2_log=True requires cfg.leaf_eval='rollout' (the probe pairs "
                    "value-head predictions with rollout-realized remaining costs at every leaf)."
                )
            if self.model.value_head is None:
                raise ValueError(
                    "enable_r2_log=True requires a model with a value head (Stage 1 ckpt)."
                )
            if self.cfg.simulation_batch_size != 1:
                raise ValueError(
                    "enable_r2_log=True requires simulation_batch_size=1 (FIFO pairing between "
                    "leaf priors-evaluator and rollout-evaluator is only well-defined in the "
                    "sequential C++ path)."
                )

        self.fwd_count_decode = 0
        self.fwd_count_value = 0
        self.fwd_count_rollout = 0
        self.eval_cache_hits = 0
        self.eval_cache_misses = 0
        self.batch_eval_calls = 0
        self.batch_eval_rows = 0
        self.pending_batch_calls = 0
        self.pending_batch_rows = 0
        self.pending_collection_attempts = 0
        self.pending_collection_successes = 0
        self.virtual_collision_count = 0
        self.max_virtual_visits_remaining = 0
        self.r2_records = []
        r2_records_out = self.r2_records if enable_r2_log else None

        embeddings = self.model.encode(input_1)
        fixed = self.model.precompute_decoder(embeddings)
        dist = (input_1[:, :, None, :] - input_1[:, None, :, :]).norm(p=2, dim=-1)
        coords = input_1.detach().cpu().double().numpy()[0]

        evaluator, rollout_evaluator, eval_stats = self._make_evaluators(
            input_1, dist, fixed, coords,
            bl_val=float(bl_val),
            r2_records=r2_records_out,
        )
        result = _mcts_cpp.solve_instance(
            coords,
            evaluator,
            self._cfg_dict(),
            float(bl_val),
            rollout_evaluator if self.cfg.leaf_eval == "rollout" else None,
        )

        self.fwd_count_decode = int(result["decode_steps"])
        self.fwd_count_rollout = int(result["rollout_steps"])
        self.fwd_count_value = int(result["value_calls"])
        self.eval_cache_hits = int(eval_stats["hits"])
        self.eval_cache_misses = int(eval_stats["misses"])
        self.batch_eval_calls = int(eval_stats["batch_calls"])
        self.batch_eval_rows = int(eval_stats["batch_rows"])
        self.pending_batch_calls = int(result["pending_batch_calls"])
        self.pending_batch_rows = int(result["pending_batch_rows"])
        self.pending_collection_attempts = int(result["pending_collection_attempts"])
        self.pending_collection_successes = int(result["pending_collection_successes"])
        self.virtual_collision_count = int(result["virtual_collision_count"])
        self.max_virtual_visits_remaining = int(result["max_virtual_visits_remaining"])

        cost = torch.tensor(float(result["cost"]), device=self.device, dtype=input_1.dtype)
        tour = torch.tensor(list(result["tour"]), device=self.device, dtype=torch.long)
        return cost, tour

    def _cfg_dict(self) -> dict:
        cfg = asdict(self.cfg)
        if cfg.get("seed") is None:
            cfg["seed"] = 0
        return cfg

    def _make_evaluators(
        self,
        input_1: torch.Tensor,
        dist: torch.Tensor,
        fixed,
        coords,
        bl_val: float = 1.0,
        r2_records: Optional[list] = None,
    ):
        device = self.device
        dtype = input_1.dtype
        n_nodes = int(input_1.size(1))
        eval_cache = {}
        eval_stats = {"hits": 0, "misses": 0, "batch_calls": 0, "batch_rows": 0}
        # Stage 3 Phase E.1 — off-policy R² probe FIFO. When `r2_records` is
        # provided (sequential leaf_eval='rollout' path only), `evaluator`
        # forces a value-head computation at every leaf and pushes
        # `(step, v_predicted)` here. `rollout_evaluator` pops one entry per
        # leaf and emits `{step, v_predicted, z_realized}` to `r2_records`.
        # The pairing is FIFO 1:1 because in simulation_batch_size=1 the C++
        # side calls evaluator(leaf) immediately followed by
        # rollout_evaluator(leaf) for each leaf expansion.
        pending_v_preds: list = []
        log_offpolicy = (r2_records is not None) and (self.cfg.leaf_eval == "rollout")

        # Pairwise distance matrix (float64 from `coords`, which is itself
        # float64-from-coords). Computed once per instance and shared with
        # `rollout_evaluator` for edge-cost lookups.
        dist_table = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

        def cache_key(snapshot: dict, need_value: bool):
            visited_key = tuple(bool(v) for v in snapshot["visited"])
            return (
                bool(need_value),
                int(snapshot["step"]),
                int(snapshot["first"]),
                int(snapshot["prev"]),
                visited_key,
            )

        def eval_many(snapshots, need_value: bool):
            # Decoder outputs depend on first/current city and the visited mask.
            # They do not depend on accumulated path length, so it is safe to
            # cache network evaluations for transposed TSP decoder states while
            # leaving MCTS backup costs untouched. The unique root state carries
            # `first=-1` (C++ TspState placeholder before any action is taken);
            # benign because the root is unique per instance and is overridden
            # to `first=0` inside `_state_batch_from_snapshots` for StateTSP.
            snapshots = list(snapshots)
            results = [None] * len(snapshots)
            misses_by_step = defaultdict(list)

            for row, snapshot in enumerate(snapshots):
                key = cache_key(snapshot, need_value)
                cached = eval_cache.get(key)
                if cached is not None:
                    eval_stats["hits"] += 1
                    results[row] = cached
                else:
                    eval_stats["misses"] += 1
                    misses_by_step[int(snapshot["step"])].append((row, snapshot, key))

            for _step, rows in misses_by_step.items():
                row_snapshots = [snapshot for _row, snapshot, _key in rows]
                state = self._state_batch_from_snapshots(
                    row_snapshots, input_1, dist, dtype, device
                )
                fixed_b = self._expand_fixed(fixed, len(row_snapshots))
                eval_stats["batch_calls"] += 1
                eval_stats["batch_rows"] += len(row_snapshots)

                if need_value:
                    log_p, mask, glimpse = self.model.decoder.decode_step(
                        fixed_b, state, return_glimpse=True
                    )
                    values = self.model.value_head(glimpse).view(-1)
                else:
                    log_p, mask = self.model.decoder.decode_step(
                        fixed_b, state, return_glimpse=False
                    )
                    values = None

                probs_b = log_p.exp().detach().cpu().double()
                mask_b = mask.detach().cpu().bool()
                values_b = values.detach().cpu().double() if values is not None else None

                for local_row, (global_row, _snapshot, key) in enumerate(rows):
                    probs = probs_b[local_row].view(-1).tolist()
                    mask_list = mask_b[local_row].view(-1).tolist()
                    value = float(values_b[local_row].item()) if values_b is not None else 0.0
                    result = (probs, mask_list, value)
                    eval_cache[key] = result
                    results[global_row] = result

            return results

        def eval_one(snapshot: dict, need_value: bool):
            return eval_many([snapshot], need_value)[0]

        def evaluator(snapshot_or_snapshots, need_value: bool):
            is_list = isinstance(snapshot_or_snapshots, list)
            snapshots = snapshot_or_snapshots if is_list else [snapshot_or_snapshots]

            # In off-policy R² probe mode (leaf_eval='rollout' + enable_r2_log),
            # force a value-head computation at every leaf so we can log the
            # value-head's prediction alongside the rollout-realized cost.
            actual_need_value = need_value or log_offpolicy
            results = eval_many(snapshots, actual_need_value)

            if log_offpolicy:
                for snap, (_probs, _mask, value) in zip(snapshots, results):
                    pending_v_preds.append((int(snap["step"]), float(value)))
                if not need_value:
                    # Strip the value back to 0.0 to preserve the cpp-side
                    # contract that leaf_eval='rollout' callers see no
                    # value-head signal in the returned tuple.
                    results = [(probs, mask, 0.0) for probs, mask, _value in results]

            return results if is_list else results[0]

        def rollout_many(snapshots):
            rollouts = []
            for snapshot in snapshots:
                rollouts.append({
                    "start_length": float(snapshot["length"]),
                    "step": int(snapshot["step"]),
                    "first": int(snapshot["first"]),
                    "prev": int(snapshot["prev"]),
                    "length": float(snapshot["length"]),
                    "visited": [bool(v) for v in snapshot["visited"]],
                    "done": False,
                    "remaining_cost": None,
                })

            decode_steps = 0
            while True:
                active = [
                    (idx, ro)
                    for idx, ro in enumerate(rollouts)
                    if not ro["done"] and ro["step"] < n_nodes
                ]
                if not active:
                    break

                by_step = defaultdict(list)
                for idx, ro in active:
                    by_step[int(ro["step"])].append((idx, ro))

                for _step, group in by_step.items():
                    cur_snapshots = []
                    for _idx, ro in group:
                        cur_snapshots.append({
                            "step": ro["step"],
                            "first": ro["first"],
                            "prev": ro["prev"],
                            "length": ro["length"],
                            "visited": ro["visited"],
                        })
                    evals = eval_many(cur_snapshots, False)
                    decode_steps += len(group)

                    for (_idx, ro), (probs, mask, _value) in zip(group, evals):
                        best_action = -1
                        best_prob = -math.inf
                        for action, (prob, is_masked) in enumerate(zip(probs, mask)):
                            if is_masked:
                                continue
                            if prob > best_prob:
                                best_prob = prob
                                best_action = action
                        if best_action < 0:
                            raise RuntimeError("batched rollout evaluator found no legal action")

                        if ro["step"] > 0:
                            ro["length"] += dist_table[ro["prev"]][best_action]
                        else:
                            ro["first"] = best_action
                        ro["prev"] = best_action
                        ro["visited"][best_action] = True
                        ro["step"] += 1

                        if ro["step"] >= n_nodes:
                            final_cost = ro["length"]
                            if n_nodes > 1:
                                final_cost += dist_table[ro["prev"]][ro["first"]]
                            ro["remaining_cost"] = final_cost - ro["start_length"]
                            ro["done"] = True

            return (
                [float(ro["remaining_cost"]) for ro in rollouts],
                decode_steps,
                decode_steps,
            )

        def _log_r2_pair(leaf_snapshot, remaining_cost):
            """Pop the matching pending v_predicted (FIFO) and emit one R² row."""
            if not log_offpolicy or not pending_v_preds:
                return
            step, v_pred = pending_v_preds.pop(0)
            r2_records.append({
                "step": step,
                "v_predicted": v_pred,
                "z_realized": float(remaining_cost) / float(bl_val),
            })

        def rollout_evaluator(snapshot_or_snapshots):
            if isinstance(snapshot_or_snapshots, list):
                snapshots = snapshot_or_snapshots
                remaining_costs, decode_total, rollout_total = rollout_many(snapshots)
                if log_offpolicy:
                    for snap, rcost in zip(snapshots, remaining_costs):
                        _log_r2_pair(snap, rcost)
                return (remaining_costs, decode_total, rollout_total)

            snapshot = snapshot_or_snapshots
            start_length = float(snapshot["length"])
            step = int(snapshot["step"])
            first = int(snapshot["first"])
            prev = int(snapshot["prev"])
            length = start_length
            visited = [bool(v) for v in snapshot["visited"]]
            decode_steps = 0

            while step < n_nodes:
                cur_snapshot = {
                    "step": step,
                    "first": first,
                    "prev": prev,
                    "length": length,
                    "visited": visited,
                }
                probs, mask, _ = eval_one(cur_snapshot, False)
                decode_steps += 1

                best_action = -1
                best_prob = -math.inf
                for action, (prob, is_masked) in enumerate(zip(probs, mask)):
                    if is_masked:
                        continue
                    if prob > best_prob:
                        best_prob = prob
                        best_action = action
                if best_action < 0:
                    raise RuntimeError("rollout evaluator found no legal action")

                if step > 0:
                    length += dist_table[prev][best_action]
                else:
                    first = best_action
                prev = best_action
                visited[best_action] = True
                step += 1

            final_cost = length
            if n_nodes > 1:
                final_cost += dist_table[prev][first]
            remaining_cost = final_cost - start_length
            _log_r2_pair(snapshot, remaining_cost)
            return (remaining_cost, decode_steps, decode_steps)

        return evaluator, rollout_evaluator, eval_stats

    @staticmethod
    def _state_from_snapshot(
        snapshot: dict,
        input_1: torch.Tensor,
        dist: torch.Tensor,
        ids: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> StateTSP:
        step = int(snapshot["step"])
        first = int(snapshot["first"])
        prev = int(snapshot["prev"])
        length = float(snapshot["length"])

        if step == 0:
            first = 0
            prev = 0
            cur_coord = None
        else:
            cur_coord = input_1[:, prev : prev + 1, :]

        first_a = torch.tensor([[first]], dtype=torch.long, device=device)
        prev_a = torch.tensor([[prev]], dtype=torch.long, device=device)
        visited = torch.tensor(snapshot["visited"], dtype=torch.bool, device=device).view(1, 1, -1)
        lengths = torch.tensor([[length]], dtype=dtype, device=device)
        i = torch.tensor([step], dtype=torch.long, device=device)

        return StateTSP(
            loc=input_1,
            dist=dist,
            ids=ids,
            first_a=first_a,
            prev_a=prev_a,
            visited_=visited,
            lengths=lengths,
            cur_coord=cur_coord,
            i=i,
        )

    @staticmethod
    def _state_batch_from_snapshots(
        snapshots,
        input_1: torch.Tensor,
        dist: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> StateTSP:
        if not snapshots:
            raise ValueError("_state_batch_from_snapshots requires at least one snapshot")

        bsz = len(snapshots)
        step = int(snapshots[0]["step"])
        n_nodes = int(input_1.size(1))

        first_vals = []
        prev_vals = []
        lengths = []
        visited_rows = []
        for snapshot in snapshots:
            if int(snapshot["step"]) != step:
                raise ValueError("batched decoder snapshots must share the same step")

            first = int(snapshot["first"])
            prev = int(snapshot["prev"])
            if step == 0:
                first = 0
                prev = 0
            first_vals.append(first)
            prev_vals.append(prev)
            lengths.append(float(snapshot["length"]))
            visited = [bool(v) for v in snapshot["visited"]]
            if len(visited) != n_nodes:
                raise ValueError("snapshot visited mask has wrong graph size")
            visited_rows.append(visited)

        loc = input_1.expand(bsz, -1, -1)
        dist_b = dist.expand(bsz, -1, -1)
        ids = torch.arange(bsz, dtype=torch.long, device=device)[:, None]
        first_a = torch.tensor(first_vals, dtype=torch.long, device=device).view(bsz, 1)
        prev_a = torch.tensor(prev_vals, dtype=torch.long, device=device).view(bsz, 1)
        visited_t = torch.tensor(visited_rows, dtype=torch.bool, device=device).view(bsz, 1, n_nodes)
        lengths_t = torch.tensor(lengths, dtype=dtype, device=device).view(bsz, 1)
        cur_coord = None if step == 0 else loc[ids, prev_a]
        i = torch.tensor([step], dtype=torch.long, device=device)

        return StateTSP(
            loc=loc,
            dist=dist_b,
            ids=ids,
            first_a=first_a,
            prev_a=prev_a,
            visited_=visited_t,
            lengths=lengths_t,
            cur_coord=cur_coord,
            i=i,
        )

    @staticmethod
    def _expand_fixed(fixed, batch_size: int):
        if batch_size == 1:
            return fixed
        return fixed._replace(
            node_embeddings=fixed.node_embeddings.expand(batch_size, -1, -1),
            context_node_projected=fixed.context_node_projected.expand(batch_size, -1, -1),
            glimpse_key=fixed.glimpse_key.expand(-1, batch_size, -1, -1, -1),
            glimpse_val=fixed.glimpse_val.expand(-1, batch_size, -1, -1, -1),
            logit_key=fixed.logit_key.expand(batch_size, -1, -1, -1),
        )
