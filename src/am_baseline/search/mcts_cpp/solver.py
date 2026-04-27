"""Python wrapper for the optional pybind11 MCTS backend."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Optional, Tuple

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert input_1.dim() == 3 and input_1.size(0) == 1, "solve_instance expects (1, N, 2)"

        input_1 = input_1.to(self.device)
        if bl_val is None:
            bl_val = float(self._compute_bl_val_batch(input_1).item())

        self.fwd_count_decode = 0
        self.fwd_count_value = 0
        self.fwd_count_rollout = 0
        self.eval_cache_hits = 0
        self.eval_cache_misses = 0

        embeddings = self.model.encode(input_1)
        fixed = self.model.precompute_decoder(embeddings)
        dist = (input_1[:, :, None, :] - input_1[:, None, :, :]).norm(p=2, dim=-1)
        coords = input_1.detach().cpu().double().numpy()[0]

        evaluator, rollout_evaluator, eval_stats = self._make_evaluators(
            input_1, dist, fixed, coords
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
    ):
        device = self.device
        dtype = input_1.dtype
        n_nodes = int(input_1.size(1))
        ids = torch.zeros(1, 1, dtype=torch.long, device=device)
        eval_cache = {}
        eval_stats = {"hits": 0, "misses": 0}

        dist_table = [
            [
                math.sqrt(
                    (float(coords[i, 0]) - float(coords[j, 0])) ** 2
                    + (float(coords[i, 1]) - float(coords[j, 1])) ** 2
                )
                for j in range(n_nodes)
            ]
            for i in range(n_nodes)
        ]

        def eval_one(snapshot: dict, need_value: bool):
            # Decoder outputs depend on first/current city and the visited mask.
            # They do not depend on accumulated path length, so it is safe to
            # cache network evaluations for transposed TSP decoder states while
            # leaving MCTS backup costs untouched.
            visited_key = tuple(bool(v) for v in snapshot["visited"])
            key = (
                bool(need_value),
                int(snapshot["step"]),
                int(snapshot["first"]),
                int(snapshot["prev"]),
                visited_key,
            )
            cached = eval_cache.get(key)
            if cached is not None:
                eval_stats["hits"] += 1
                return cached

            eval_stats["misses"] += 1
            state = self._state_from_snapshot(snapshot, input_1, dist, ids, dtype, device)
            if need_value:
                log_p, mask, glimpse = self.model.decoder.decode_step(
                    fixed, state, return_glimpse=True
                )
                value = float(self.model.value_head(glimpse).view(-1)[0].item())
            else:
                log_p, mask = self.model.decoder.decode_step(
                    fixed, state, return_glimpse=False
                )
                value = 0.0
            probs = log_p.exp().view(-1).detach().cpu().double().tolist()
            mask_list = mask.view(-1).detach().cpu().bool().tolist()
            result = (probs, mask_list, value)
            eval_cache[key] = result
            return result

        def evaluator(snapshot: dict, need_value: bool):
            return eval_one(snapshot, need_value)

        def rollout_evaluator(snapshot: dict):
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
