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
        # Stage 4 Phase A — per-tour-step raw root visit counts. Populated by
        # `solve_instance` only when `cfg.return_root_visits=True`. One dict
        # per tour-step (length N after a full solve); each dict maps
        # action -> visit count for actions touched in backup at that step.
        self.root_visit_dists: list = []

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
        self.root_visit_dists = []
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
        # Stage 4 Phase A — Plumb per-step root visit dumps back to Python.
        # The C++ side emits the field as `list[list[(int, int)]]` only when
        # cfg.return_root_visits is True; we convert each step's pairs into
        # a {action: count} dict here so callers see the same shape as the
        # pure-Python `MCTSSolver.root_visit_dists`.
        if self.cfg.return_root_visits and "root_visit_dists" in result:
            self.root_visit_dists = [
                {int(a): int(c) for a, c in step} for step in result["root_visit_dists"]
            ]
        else:
            self.root_visit_dists = []

        cost = torch.tensor(float(result["cost"]), device=self.device, dtype=input_1.dtype)
        tour = torch.tensor(list(result["tour"]), device=self.device, dtype=torch.long)
        return cost, tour

    # Mapping for the temperature_schedule field. Mirrors the C++ encoding
    # in mcts.hpp (Config::temperature_schedule). None and 'const' both map
    # to 0 (constant τ = cfg.temperature) so default behavior is unchanged.
    _SCHEDULE_TO_INT = {
        None: 0,
        "const": 0,
        "step30": 1,
        "step50": 2,
        "step10": 3,
    }

    def _cfg_dict(self) -> dict:
        cfg = asdict(self.cfg)
        if cfg.get("seed") is None:
            cfg["seed"] = 0
        # Translate the Python-side string schedule into the int the C++
        # Config struct expects. Validation already ran in MCTSSolver._validate_config.
        sched = cfg.get("temperature_schedule")
        if sched not in self._SCHEDULE_TO_INT:
            raise ValueError(
                f"unsupported temperature_schedule for C++ backend: {sched!r}"
            )
        cfg["temperature_schedule"] = self._SCHEDULE_TO_INT[sched]
        # Per-tour-step K override: marshal to a list of ints (or empty list
        # if None / empty). The C++ side uses an empty vector as the
        # "no override" sentinel (falls back to scalar n_simulations).
        per_step = cfg.get("n_simulations_per_step")
        if per_step is None:
            cfg["n_simulations_per_step"] = []
        else:
            cfg["n_simulations_per_step"] = [int(k) for k in per_step]
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


class CppBatchMCTSSolver(CppMCTSSolver):
    """Cross-instance batched C++ MCTS scheduler.

    C++ still owns each tree's state, PUCT selection, and backup. Python keeps a
    pool of independent trees active and batches their NN evaluator requests.
    Each tree has at most one pending request, so per-tree search order matches
    the sequential C++ backend.
    """

    def __init__(
        self,
        model: AttentionModel,
        cfg: MCTSConfig,
        device: Optional[torch.device] = None,
        mcts_batch_size: int = 32,
    ):
        super().__init__(model, cfg, device)
        if cfg.simulation_batch_size != 1:
            raise ValueError(
                "CppBatchMCTSSolver preserves one pending simulation per tree; "
                "use simulation_batch_size=1."
            )
        if mcts_batch_size < 1:
            raise ValueError("mcts_batch_size must be >= 1")
        self.mcts_batch_size = int(mcts_batch_size)

        self.fwd_count_decode_per_instance = []
        self.fwd_count_rollout_per_instance = []
        self.fwd_count_value_per_instance = []
        self.eval_cache_hits_per_instance = []
        self.eval_cache_misses_per_instance = []
        # Stage 4 Phase A — per-instance per-step root visit dumps. Each entry
        # is a list (length N) of dicts {action: visit_count}, mirroring the
        # `MCTSSolver.root_visit_dists` shape but indexed by instance.
        # Populated by `solve_batch` / `solve_instance` only when
        # `cfg.return_root_visits=True`.
        self.root_visit_dists_per_instance: list = []

    @torch.no_grad()
    def solve_instance(
        self,
        input_1: torch.Tensor,
        bl_val: Optional[float] = None,
        enable_r2_log: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if enable_r2_log:
            raise ValueError("enable_r2_log is only supported by the sequential C++ backend")
        bl_vals = None
        if bl_val is not None:
            bl_vals = torch.tensor([float(bl_val)], device=self.device)
        costs, tours = self.solve_batch(input_1, bl_vals=bl_vals)
        # Stage 4 Phase A — for a single-instance solve through the batched
        # backend, expose the per-step visit dists via the same attribute name
        # as the sequential solver (`root_visit_dists`).
        if self.cfg.return_root_visits and self.root_visit_dists_per_instance:
            self.root_visit_dists = self.root_visit_dists_per_instance[0]
        else:
            self.root_visit_dists = []
        return costs[0], tours[0]

    @torch.no_grad()
    def solve_batch(
        self,
        inputs: torch.Tensor,
        bl_vals: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = inputs.to(self.device)
        bsz, n_nodes, _ = inputs.shape
        if bl_vals is None:
            bl_vals = self._compute_bl_val_batch(inputs)
        else:
            bl_vals = bl_vals.to(self.device)

        costs = torch.empty(bsz, device=self.device)
        tours = torch.empty(bsz, n_nodes, dtype=torch.long, device=self.device)

        self.fwd_count_decode_per_instance = []
        self.fwd_count_rollout_per_instance = []
        self.fwd_count_value_per_instance = []
        self.eval_cache_hits_per_instance = []
        self.eval_cache_misses_per_instance = []
        self.batch_eval_calls = 0
        self.batch_eval_rows = 0
        self.root_visit_dists_per_instance = []

        for start in range(0, bsz, self.mcts_batch_size):
            end = min(start + self.mcts_batch_size, bsz)
            chunk_costs, chunk_tours, chunk_stats = self._solve_chunk(
                inputs[start:end],
                bl_vals[start:end],
            )
            costs[start:end] = chunk_costs
            tours[start:end] = chunk_tours
            self.fwd_count_decode_per_instance.extend(chunk_stats["decode_steps"])
            self.fwd_count_rollout_per_instance.extend(chunk_stats["rollout_steps"])
            self.fwd_count_value_per_instance.extend(chunk_stats["value_calls"])
            self.eval_cache_hits_per_instance.extend(chunk_stats["cache_hits"])
            self.eval_cache_misses_per_instance.extend(chunk_stats["cache_misses"])
            self.batch_eval_calls += chunk_stats["batch_eval_calls"]
            self.batch_eval_rows += chunk_stats["batch_eval_rows"]
            if "root_visit_dists" in chunk_stats:
                self.root_visit_dists_per_instance.extend(chunk_stats["root_visit_dists"])

        self.fwd_count_decode = int(sum(self.fwd_count_decode_per_instance))
        self.fwd_count_rollout = int(sum(self.fwd_count_rollout_per_instance))
        self.fwd_count_value = int(sum(self.fwd_count_value_per_instance))
        self.eval_cache_hits = int(sum(self.eval_cache_hits_per_instance))
        self.eval_cache_misses = int(sum(self.eval_cache_misses_per_instance))
        return costs, tours

    def _solve_chunk(self, input_b: torch.Tensor, bl_vals_b: torch.Tensor):
        bsz, n_nodes, _ = input_b.shape
        dtype = input_b.dtype
        device = self.device

        embeddings = self.model.encode(input_b)
        fixed = self.model.precompute_decoder(embeddings)
        dist_b = (input_b[:, :, None, :] - input_b[:, None, :, :]).norm(p=2, dim=-1)
        coords = input_b.detach().cpu().double().numpy()
        dist_table = np.linalg.norm(coords[:, :, None, :] - coords[:, None, :, :], axis=-1)

        engine = _mcts_cpp.BatchSearch(
            coords,
            self._cfg_dict(),
            [float(v.item()) for v in bl_vals_b],
        )

        eval_cache = {}
        cache_hits = [0 for _ in range(bsz)]
        cache_misses = [0 for _ in range(bsz)]
        batch_eval_calls = 0
        batch_eval_rows = 0

        def cache_key(item, need_value: bool):
            # Hot path — called once per row in eval_many. Updated 2026-05-10
            # (Fix #5) to use the same compact 2-tuple `(packed_header_int,
            # visited_bytes)` shape as `eval_many_arrays`, so the cache is
            # shared and Python dict hashing is faster on a 2-tuple of
            # (int, bytes) than on the previous 6-tuple of mixed types.
            #
            # Header layout (29 bits used out of int64):
            #   bits 19+ : slot
            #   bit  18  : need_value
            #   bits 12-17 : step
            #   bits  6-11 : first  (zeroed when step == 0)
            #   bits  0-5  : prev   (zeroed when step == 0)
            snap = item["snapshot"]
            slot = int(item["slot"])
            step = int(snap["step"])
            if step == 0:
                first = 0
                prev = 0
            else:
                first = int(snap["first"])
                prev = int(snap["prev"])
            header = (
                (slot << 19)
                | ((1 if need_value else 0) << 18)
                | (step << 12)
                | (first << 6)
                | prev
            )
            return (header, bytes(snap["visited"]))

        def state_batch_from_items(items):
            if not items:
                raise ValueError("state_batch_from_items requires at least one item")
            step = int(items[0]["snapshot"]["step"])
            slots = []
            first_vals = []
            prev_vals = []
            lengths = []
            visited_rows = []
            for item in items:
                snap = item["snapshot"]
                if int(snap["step"]) != step:
                    raise ValueError("batched cross-instance decoder requests must share step")
                slot = int(item["slot"])
                first = int(snap["first"])
                prev = int(snap["prev"])
                if step == 0:
                    first = 0
                    prev = 0
                visited = [bool(v) for v in snap["visited"]]
                if len(visited) != n_nodes:
                    raise ValueError("snapshot visited mask has wrong graph size")
                slots.append(slot)
                first_vals.append(first)
                prev_vals.append(prev)
                lengths.append(float(snap["length"]))
                visited_rows.append(visited)

            slot_t = torch.tensor(slots, dtype=torch.long, device=device)
            loc = input_b.index_select(0, slot_t)
            dist = dist_b.index_select(0, slot_t)
            row_ids = torch.arange(len(items), dtype=torch.long, device=device)[:, None]
            first_a = torch.tensor(first_vals, dtype=torch.long, device=device).view(len(items), 1)
            prev_a = torch.tensor(prev_vals, dtype=torch.long, device=device).view(len(items), 1)
            visited_t = torch.tensor(
                visited_rows, dtype=torch.bool, device=device
            ).view(len(items), 1, n_nodes)
            lengths_t = torch.tensor(lengths, dtype=dtype, device=device).view(len(items), 1)
            cur_coord = None if step == 0 else loc[row_ids, prev_a]
            i = torch.tensor([step], dtype=torch.long, device=device)
            state = StateTSP(
                loc=loc,
                dist=dist,
                ids=row_ids,
                first_a=first_a,
                prev_a=prev_a,
                visited_=visited_t,
                lengths=lengths_t,
                cur_coord=cur_coord,
                i=i,
            )
            return state, slot_t

        def eval_many_arrays(
            slot_a_in,
            step_a_in,
            first_a_in,
            prev_a_in,
            length_a_in,
            visited_a_in,
            need_value: bool = False,
        ):
            """Numpy-direct leaf evaluator (Fix #3, 2026-05-10).

            Mirrors `eval_many` but takes numpy arrays directly:
              - slot_a_in (B,) int64
              - step_a_in (B,) int64
              - first_a_in (B,) int64
              - prev_a_in (B,) int64
              - length_a_in (B,) float64
              - visited_a_in (B, N) bool

            Returns (probs_out (B,N) float64, mask_out (B,N) bool, values_out
            (B,) float64 or None). Used by `rollout_many` so the rollout path
            avoids two layers of overhead the dict-based `eval_many` carries:

              1. The list-of-dicts construction at the rollout call site
                 (line 944 listcomp, ~16s tottime in TSP-50 K=50 M=200 profile).
              2. `state_batch_from_items` rebuilds: `[bool(v) for v in
                 snap["visited"]]` per row, then 5× `torch.tensor(...)` on
                 list-of-lists (~64s cumtime in same profile).
              3. The post-call rebuild `np.asarray([e[0] for e in evals])`
                 inside `rollout_many` (~27s tottime).

            All replaced by per-row numpy slicing + `torch.from_numpy` (one
            allocation per per-step group, no list-of-lists materialization).

            Cache: shares `eval_cache` with `eval_many`. Both paths build the
            same compact 2-tuple key `(packed_header_int, visited_bytes)`
            (Fix #5 2026-05-10), so hits cross-pollinate. Cache values are
            stored as `(probs_arr, mask_arr, value)` numpy arrays (Fix #4).
            """
            nonlocal batch_eval_calls, batch_eval_rows
            B = int(slot_a_in.shape[0])
            probs_out = np.empty((B, n_nodes), dtype=np.float64)
            mask_out = np.empty((B, n_nodes), dtype=bool)
            values_out = np.empty(B, dtype=np.float64) if need_value else None

            if B == 0:
                return probs_out, mask_out, values_out

            # Fix #5 (2026-05-10): bulk-vectorize the cache-key construction.
            # Previous code did per-row int casts on 4 numpy scalars + 6-tuple
            # build = ~6µs/row × 58.7M rows = ~350s on Modal M=1000. Replace
            # with one vectorized numpy bit-pack + `.tolist()` so per-row work
            # collapses to list-index + 2-tuple build = ~1-2µs/row.
            #
            # Header layout (29 bits used out of int64):
            #   bits 19+ : slot (supports M up to 1024)
            #   bit  18  : need_value
            #   bits 12-17 : step (supports N up to 64)
            #   bits  6-11 : first (zeroed when step==0)
            #   bits  0-5  : prev  (zeroed when step==0)
            need_value_int = 1 if need_value else 0
            slot_a_64 = slot_a_in.astype(np.int64, copy=False)
            step_a_64 = step_a_in.astype(np.int64, copy=False)
            first_a_64 = first_a_in.astype(np.int64, copy=False)
            prev_a_64 = prev_a_in.astype(np.int64, copy=False)
            step_eq_zero = step_a_64 == 0
            first_eff = np.where(step_eq_zero, np.int64(0), first_a_64)
            prev_eff = np.where(step_eq_zero, np.int64(0), prev_a_64)
            header_a = (
                (slot_a_64 << 19)
                | (np.int64(need_value_int) << 18)
                | (step_a_64 << 12)
                | (first_eff << 6)
                | prev_eff
            )
            # Bulk numpy→Python via tolist() — one C call instead of B per-row
            # int casts.
            header_list = header_a.tolist()
            slot_list = slot_a_64.tolist()
            step_list = step_a_64.tolist()

            # Track A (2026-05-11): single-batch miss processing, no step
            # grouping. Rows are at heterogeneous steps; state.i is a per-row
            # (B_miss,) tensor and Decoder._get_step_context dispatches per-row
            # via torch.where between the W_placeholder context (step==0) and
            # the gathered first/current-node context (step>=1). This collapses
            # what was previously up to N=50 small NN calls per outer iter into
            # one large call, saving kernel-launch overhead.
            misses = []
            cache_get = eval_cache.get
            for row in range(B):
                slot_i = slot_list[row]
                key = (header_list[row], visited_a_in[row].tobytes())
                cached = cache_get(key)
                if cached is not None:
                    cache_hits[slot_i] += 1
                    probs_out[row] = cached[0]
                    mask_out[row] = cached[1]
                    if values_out is not None:
                        values_out[row] = cached[2]
                else:
                    cache_misses[slot_i] += 1
                    misses.append((row, key))

            if misses:
                miss_idx = np.fromiter(
                    (r for r, _k in misses), dtype=np.int64, count=len(misses)
                )
                B_g = int(miss_idx.shape[0])

                slot_slice = slot_a_in[miss_idx]
                slot_t = torch.from_numpy(np.ascontiguousarray(slot_slice)).to(device)
                loc = input_b.index_select(0, slot_t)
                dist = dist_b.index_select(0, slot_t)
                row_ids = torch.arange(B_g, dtype=torch.long, device=device)[:, None]

                # Per-row step indices. For step==0 rows the decoder branches
                # to the W_placeholder context, so first/prev for those rows
                # don't affect the NN output — but they must still be valid
                # indices in [0, N-1] for the unused gather. Zero them out
                # per-row to match the canonical cache-key shape.
                step_slice = step_a_in[miss_idx].astype(np.int64, copy=False)
                first_slice = first_a_in[miss_idx].astype(np.int64, copy=False).copy()
                prev_slice = prev_a_in[miss_idx].astype(np.int64, copy=False).copy()
                zero_step = step_slice == 0
                first_slice[zero_step] = 0
                prev_slice[zero_step] = 0

                first_t = torch.from_numpy(first_slice).to(device).view(B_g, 1)
                prev_t = torch.from_numpy(prev_slice).to(device).view(B_g, 1)
                cur_coord = loc[row_ids, prev_t]  # valid for all rows; unused
                                                  # by _get_step_context but
                                                  # needed for StateTSP shape.

                # Match `lengths_t` dtype to `input_b` (float32 or float64).
                length_slice = length_a_in[miss_idx]
                if dtype == torch.float32:
                    length_slice_np = length_slice.astype(np.float32, copy=False)
                else:
                    length_slice_np = np.ascontiguousarray(length_slice)
                lengths_t = torch.from_numpy(length_slice_np).to(device).view(B_g, 1)

                visited_t = torch.from_numpy(
                    np.ascontiguousarray(visited_a_in[miss_idx])
                ).to(device).view(B_g, 1, n_nodes)
                i_tensor = torch.from_numpy(step_slice).to(device)  # (B_g,)

                state = StateTSP(
                    loc=loc,
                    dist=dist,
                    ids=row_ids,
                    first_a=first_t,
                    prev_a=prev_t,
                    visited_=visited_t,
                    lengths=lengths_t,
                    cur_coord=cur_coord,
                    i=i_tensor,
                )
                fixed_b = fixed[slot_t]
                batch_eval_calls += 1
                batch_eval_rows += B_g

                if need_value:
                    log_p, mask, glimpse = self.model.decoder.decode_step(
                        fixed_b, state, return_glimpse=True
                    )
                    values_t = self.model.value_head(glimpse).view(-1)
                    values_np = values_t.detach().cpu().double().numpy()
                else:
                    log_p, mask = self.model.decoder.decode_step(
                        fixed_b, state, return_glimpse=False
                    )
                    values_np = None

                probs_np = log_p.exp().detach().cpu().double().numpy().reshape(B_g, n_nodes)
                mask_np = mask.detach().cpu().bool().numpy().reshape(B_g, n_nodes)

                for local_row, (global_row, key) in enumerate(misses):
                    probs_row = probs_np[local_row]
                    mask_row = mask_np[local_row]
                    value = float(values_np[local_row]) if values_np is not None else 0.0
                    if key is not None:
                        eval_cache[key] = (probs_row.copy(), mask_row.copy(), value)
                    probs_out[global_row] = probs_row
                    mask_out[global_row] = mask_row
                    if values_out is not None:
                        values_out[global_row] = value

            return probs_out, mask_out, values_out

        def eval_many(items, need_value: bool, use_cache: bool = True):
            """Batched cross-instance leaf evaluator.

            `use_cache=True` (default) builds a per-row cache key, checks
            `eval_cache`, batches misses by step, and stores results back. This
            is the right behavior for selection/expansion leaf evals where the
            same (slot, step, prefix) recurs across MCTS simulations and the
            cache hit rate is high (>99% in production).

            `use_cache=False` skips all cache_key construction and lookup. Use
            this for rollout inner-loop calls — every rollout state is unique
            (no repeats across simulations or instances), so the cache always
            misses, making `cache_key` (which iterates the `visited` mask
            element-by-element) ~20% of total wall on TSP-50 K=50.
            """
            nonlocal batch_eval_calls, batch_eval_rows
            items = list(items)
            results = [None] * len(items)
            misses_by_step = defaultdict(list)

            if use_cache:
                for row, item in enumerate(items):
                    slot = int(item["slot"])
                    key = cache_key(item, need_value)
                    cached = eval_cache.get(key)
                    if cached is not None:
                        cache_hits[slot] += 1
                        results[row] = cached
                    else:
                        cache_misses[slot] += 1
                        misses_by_step[int(item["snapshot"]["step"])].append((row, item, key))
            else:
                # Skip cache_key + lookup entirely. Every row is a "miss"; key=None.
                for row, item in enumerate(items):
                    slot = int(item["slot"])
                    cache_misses[slot] += 1
                    misses_by_step[int(item["snapshot"]["step"])].append((row, item, None))

            for _step, rows in misses_by_step.items():
                row_items = [item for _row, item, _key in rows]
                state, slot_t = state_batch_from_items(row_items)
                fixed_b = fixed[slot_t]
                batch_eval_calls += 1
                batch_eval_rows += len(row_items)

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

                # Fix #4 (2026-05-10): store numpy arrays in cache instead of
                # Python lists. The C++ `apply_results` reads cache values via
                # `sequence_to_doubles` which works on any PySequence (including
                # numpy ndarrays), so the rollout path's `eval_many_arrays` and
                # this selection path can share a cache of numpy values. Saves
                # ~96s tolist() tottime on the M=1000 probe.
                probs_np_b = log_p.exp().detach().cpu().double().numpy().reshape(
                    len(rows), n_nodes
                )
                mask_np_b = mask.detach().cpu().bool().numpy().reshape(
                    len(rows), n_nodes
                )
                values_np_b = (
                    values.detach().cpu().double().numpy()
                    if values is not None
                    else None
                )

                for local_row, (global_row, _item, key) in enumerate(rows):
                    probs_row = probs_np_b[local_row].copy()
                    mask_row = mask_np_b[local_row].copy()
                    value = float(values_np_b[local_row]) if values_np_b is not None else 0.0
                    result = (probs_row, mask_row, value)
                    if use_cache and key is not None:
                        eval_cache[key] = result
                    results[global_row] = result

            return results

        def rollout_many(requests):
            # Vectorized rollout (Fix #2, 2026-05-10).
            #
            # Prior implementation kept rollout state as a list of dicts and
            # ran a per-rollout Python loop both for masked-argmax action
            # selection (line 937: `for action, (prob, is_masked) in enumerate
            # (zip(probs, mask)) ...`) and for state updates. cProfile on
            # TSP-50 K=50 M=200 measured rollout_many at 89s tottime out of
            # 361s total (25%) — dominated by these inner Python loops.
            #
            # This version packs all rollout state into numpy arrays and uses
            # torch.argmax + numpy fancy indexing to do per-step action
            # selection and state updates in batched form. Behavior preserved
            # exactly: torch.argmax breaks ties by returning the first (= lowest)
            # index, matching the original `prob > best_prob` strict-greater
            # tie-break.
            B = len(requests)
            if B == 0:
                return []

            slot_a = np.empty(B, dtype=np.int64)
            step_a = np.empty(B, dtype=np.int64)
            first_a = np.empty(B, dtype=np.int64)
            prev_a = np.empty(B, dtype=np.int64)
            length_a = np.empty(B, dtype=np.float64)
            start_length_a = np.empty(B, dtype=np.float64)
            visited_a = np.zeros((B, n_nodes), dtype=bool)
            done_a = np.zeros(B, dtype=bool)
            remaining_cost_a = np.zeros(B, dtype=np.float64)

            for i, request in enumerate(requests):
                snap = request["snapshot"]
                slot_a[i] = int(request["slot"])
                step_a[i] = int(snap["step"])
                first_a[i] = int(snap["first"])
                prev_a[i] = int(snap["prev"])
                length_a[i] = float(snap["length"])
                start_length_a[i] = float(snap["length"])
                visited_a[i] = np.asarray(snap["visited"], dtype=bool)

            while True:
                active = (~done_a) & (step_a < n_nodes)
                if not active.any():
                    break
                active_idx = np.nonzero(active)[0]

                # Track A (2026-05-11): all active rollouts share one NN call
                # per outer iter, regardless of step. Decoder accepts a per-row
                # state.i and branches per-row between W_placeholder (step==0)
                # and gathered context (step>=1). Eliminates the prior
                # `for _step in np.unique(active_steps):` loop that previously
                # split into ~50 small NN calls per outer iter.
                probs_arr, mask_arr, _ = eval_many_arrays(
                    slot_a[active_idx],
                    step_a[active_idx],
                    first_a[active_idx],
                    prev_a[active_idx],
                    length_a[active_idx],
                    visited_a[active_idx],
                    need_value=False,
                )

                # Sanity: at least one legal action per rollout. Pure numpy.
                if not (~mask_arr).any(axis=-1).all():
                    raise RuntimeError(
                        "cross-instance rollout found no legal action"
                    )

                masked_probs = np.where(mask_arr, -np.inf, probs_arr)
                best_actions = masked_probs.argmax(axis=-1).astype(np.int64)

                # Vectorized state updates.
                cur_steps = step_a[active_idx]                  # (B_a,)
                cur_prev = prev_a[active_idx]                    # (B_a,)
                cur_slot = slot_a[active_idx]                    # (B_a,)
                non_zero = cur_steps > 0                         # (B_a,) bool

                # Edge cost: dist_table[slot, prev, best] for step>0, else 0.
                edge_costs = np.where(
                    non_zero,
                    dist_table[cur_slot, cur_prev, best_actions],
                    0.0,
                )
                length_a[active_idx] += edge_costs

                # For step==0 rollouts, set first = best_action.
                first_a[active_idx] = np.where(
                    non_zero, first_a[active_idx], best_actions
                )
                prev_a[active_idx] = best_actions
                visited_a[active_idx, best_actions] = True
                step_a[active_idx] += 1

                # Terminal handling (closing edge + remaining_cost).
                new_steps = step_a[active_idx]
                terminal = new_steps >= n_nodes
                if terminal.any():
                    term_idx = active_idx[terminal]
                    if n_nodes > 1:
                        closing = dist_table[
                            slot_a[term_idx],
                            prev_a[term_idx],
                            first_a[term_idx],
                        ]
                    else:
                        closing = np.zeros(len(term_idx), dtype=np.float64)
                    final_cost = length_a[term_idx] + closing
                    remaining_cost_a[term_idx] = (
                        final_cost - start_length_a[term_idx]
                    )
                    done_a[term_idx] = True

            return remaining_cost_a.tolist()

        def evaluate_requests(requests):
            requests = list(requests)
            results = [None] * len(requests)

            by_need = defaultdict(list)
            for row, request in enumerate(requests):
                by_need[bool(request["need_value"])].append((row, request))
            for need_value, rows in by_need.items():
                evals = eval_many([request for _row, request in rows], need_value)
                for (row, _request), eval_result in zip(rows, evals):
                    results[row] = eval_result

            rollout_rows = [
                (row, request)
                for row, request in enumerate(requests)
                if bool(request["need_rollout"])
            ]
            rollout_remaining = {}
            if rollout_rows:
                remaining = rollout_many([request for _row, request in rollout_rows])
                for (row, _request), value in zip(rollout_rows, remaining):
                    rollout_remaining[row] = value

            out = []
            for row, request in enumerate(requests):
                probs, mask, value = results[row]
                result = {
                    "slot": int(request["slot"]),
                    "probs": probs,
                    "mask": mask,
                    "value": float(value),
                }
                if row in rollout_remaining:
                    result["rollout_remaining"] = float(rollout_remaining[row])
                out.append(result)
            return out

        while not engine.is_done():
            requests = list(engine.collect_requests())
            if not requests:
                if engine.is_done():
                    break
                raise RuntimeError("BatchSearch made no progress but is not done")
            engine.apply_results(evaluate_requests(requests))

        raw = engine.results()
        costs = torch.tensor(list(raw["costs"]), dtype=dtype, device=device)
        tours = torch.tensor(
            [list(row) for row in raw["tours"]],
            dtype=torch.long,
            device=device,
        )
        stats = {
            "decode_steps": [int(v) for v in raw["decode_steps"]],
            "rollout_steps": [int(v) for v in raw["rollout_steps"]],
            "value_calls": [int(v) for v in raw["value_calls"]],
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "batch_eval_calls": batch_eval_calls,
            "batch_eval_rows": batch_eval_rows,
        }
        # Stage 4 Phase A — convert raw `list[list[(int, int)]]` per instance
        # into `list[list[dict]]` to match the sequential solver's shape.
        if self.cfg.return_root_visits and "root_visit_dists_per_instance" in raw:
            stats["root_visit_dists"] = [
                [{int(a): int(c) for a, c in step} for step in inst]
                for inst in raw["root_visit_dists_per_instance"]
            ]
        return costs, tours, stats
