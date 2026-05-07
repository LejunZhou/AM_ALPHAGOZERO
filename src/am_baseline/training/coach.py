"""Stage 4 coach module — replay buffer and state reconstruction utilities.

Phase B of `_plans/stage4_plan.md` lives here. The full coach orchestrator
(`MCTSCoach.learn`) is reserved for Phase D. This module provides:

  - `MCTSReplayBuffer`: flat dict-of-pre-allocated-tensors replay buffer with
    ring-buffer eviction and stratified-by-step sampling.
  - `reconstruct_state`: rebuild a `StateTSP` named-tuple from a buffer batch
    so the AM decoder can score it inside `train_step_alphazero`.
  - Phase C additions: `make_self_play_config` and `generate_self_play_batch`
    drive `CppBatchMCTSSolver` with `return_root_visits=True` and pack the
    results into the buffer's `push_instance` schema.

Design notes (see plan §Phase B and spec §3.5/§3.6):
  - Storage is CPU-resident (~520 MB at default capacity); training-time
    transfer to the model device happens inside the train step.
  - `_step_index[t]` caches the slot indices currently filled at step `t`.
    It is rebuilt deterministically on `load()` from the locked invariant
    `tuple_slot = inst_idx * N + step`.
  - Stratified sampling fixes the mixed-step decoder bug — AM's decoder
    consumes a scalar `state.i` per call, so a minibatch must share one step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from am_baseline.problem.state import StateTSP


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


class MCTSReplayBuffer:
    """AlphaGo-Zero-style replay buffer for TSP-N MCTS self-play.

    Storage layout (per-instance and per-step tensors), with default sizes:

        coords      (capacity_instances, N, 2)  float32   ~ 32 MB
        bl_val      (capacity_instances,)        float32  ~  0.8 MB
        tour_cost   (capacity_instances,)        float32  ~  0.8 MB

        pi          (capacity_tuples, N)         float32  ~ 320 MB
        visited     (capacity_tuples, N)         bool     ~  80 MB
        first_a     (capacity_tuples,)           int16    ~  8 MB
        prev_a      (capacity_tuples,)           int16    ~  8 MB
        lengths     (capacity_tuples,)           float32  ~ 16 MB
        cost_to_go  (capacity_tuples,)           float32  ~ 16 MB
        inst_idx    (capacity_tuples,)           int32    ~ 16 MB

    where `capacity_tuples = capacity_instances * N`.

    Eviction is ring-buffer (instance write head modulo capacity); dropping an
    instance drops all its `N` tuple slots atomically because the layout
    `tuple_slot = inst_idx * N + step` is fixed and bijective.
    """

    def __init__(
        self,
        graph_size: int,
        capacity_instances: int = 200_000,
        device: str = "cpu",
    ) -> None:
        if capacity_instances <= 0:
            raise ValueError("capacity_instances must be positive")
        if graph_size <= 0:
            raise ValueError("graph_size must be positive")

        self.N = int(graph_size)
        self.capacity_instances = int(capacity_instances)
        self.capacity_tuples = self.capacity_instances * self.N
        # Buffer is CPU-resident — 520 MB at default capacity would OOM many GPUs.
        self.device = torch.device(device)

        # Per-instance tensors
        self.coords = torch.zeros(
            (self.capacity_instances, self.N, 2), dtype=torch.float32, device=self.device
        )
        self.bl_val = torch.zeros(
            (self.capacity_instances,), dtype=torch.float32, device=self.device
        )
        self.tour_cost = torch.zeros(
            (self.capacity_instances,), dtype=torch.float32, device=self.device
        )

        # Per-step tensors
        self.pi = torch.zeros(
            (self.capacity_tuples, self.N), dtype=torch.float32, device=self.device
        )
        self.visited = torch.zeros(
            (self.capacity_tuples, self.N), dtype=torch.bool, device=self.device
        )
        self.first_a = torch.zeros(
            (self.capacity_tuples,), dtype=torch.int16, device=self.device
        )
        self.prev_a = torch.zeros(
            (self.capacity_tuples,), dtype=torch.int16, device=self.device
        )
        self.lengths = torch.zeros(
            (self.capacity_tuples,), dtype=torch.float32, device=self.device
        )
        self.cost_to_go = torch.zeros(
            (self.capacity_tuples,), dtype=torch.float32, device=self.device
        )
        self.inst_idx = torch.zeros(
            (self.capacity_tuples,), dtype=torch.int32, device=self.device
        )

        # Bookkeeping
        self.write_head_inst = 0          # next instance slot to (over)write
        self.write_head_tuple = 0         # mirrors inst write head: head_inst * N
        self.n_filled_instances = 0       # for diagnostics; bounded by capacity
        self.n_filled_tuples = 0          # for diagnostics; bounded by capacity_tuples

        # Step index — `_step_index[t]` lists tuple-slot indices currently filled
        # with `step == t`. Updated atomically on push/eviction.
        self._step_index: List[np.ndarray] = [
            np.zeros(0, dtype=np.int64) for _ in range(self.N)
        ]
        # _step_index_set caches set-form for O(1) eviction membership;
        # rebuilt on load via _rebuild_step_index().
        self._step_index_sets: List[set] = [set() for _ in range(self.N)]

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def push_instance(
        self,
        coords: torch.Tensor,
        bl_val: float,
        tour_cost: float,
        per_step: List[Dict],
    ) -> int:
        """Append one self-play instance + its `N` per-step records.

        Args:
            coords: (N, 2) float32 — graph coordinates.
            bl_val: scalar float32 — frozen baseline normalizer (typically
                `cost(greedy_rollout(theta_star, coords))`).
            tour_cost: scalar float32 — full tour cost from MCTS.
            per_step: list of length N. Each dict carries:
                {
                    'visited': (N,) bool,
                    'first':   int (-1 if step == 0, else first city),
                    'prev':    int (-1 if step == 0, else previous city),
                    'lengths': float (state.lengths at step t),
                    'pi':      (N,) float32, raw tau=1 normalized visits,
                    'cost_to_go': float (= tour_cost - lengths_t),
                }

        Returns:
            instance slot index (post-modulo) where this record landed.
        """
        if len(per_step) != self.N:
            raise ValueError(
                f"per_step must have length N={self.N}, got {len(per_step)}"
            )
        coords_t = coords if isinstance(coords, torch.Tensor) else torch.as_tensor(coords)
        if coords_t.shape != (self.N, 2):
            raise ValueError(
                f"coords must be shape (N={self.N}, 2), got {tuple(coords_t.shape)}"
            )

        slot_inst = self.write_head_inst % self.capacity_instances

        # Drop everything currently living in this slot before overwriting.
        # On a fresh buffer (n_filled_instances < capacity), this is a no-op
        # because slot_inst hasn't been written yet.
        if self.n_filled_instances >= self.capacity_instances:
            # Evict every per-step tuple owned by this instance slot.
            for t in range(self.N):
                old_tuple_slot = slot_inst * self.N + t
                self._step_index_sets[t].discard(old_tuple_slot)

        # Write per-instance row.
        self.coords[slot_inst] = coords_t.to(dtype=torch.float32, device=self.device)
        self.bl_val[slot_inst] = float(bl_val)
        self.tour_cost[slot_inst] = float(tour_cost)

        # Write per-step rows — fixed layout: tuple_slot = slot_inst * N + step.
        for t in range(self.N):
            rec = per_step[t]
            tuple_slot = slot_inst * self.N + t

            visited_arr = rec["visited"]
            visited_t = (
                visited_arr if isinstance(visited_arr, torch.Tensor)
                else torch.as_tensor(visited_arr, dtype=torch.bool)
            )
            if visited_t.shape != (self.N,):
                raise ValueError(
                    f"per_step[{t}]['visited'] must be (N,), got {tuple(visited_t.shape)}"
                )

            pi_arr = rec["pi"]
            pi_t = (
                pi_arr if isinstance(pi_arr, torch.Tensor)
                else torch.as_tensor(pi_arr, dtype=torch.float32)
            )
            if pi_t.shape != (self.N,):
                raise ValueError(
                    f"per_step[{t}]['pi'] must be (N,), got {tuple(pi_t.shape)}"
                )

            self.visited[tuple_slot] = visited_t.to(dtype=torch.bool, device=self.device)
            self.pi[tuple_slot] = pi_t.to(dtype=torch.float32, device=self.device)
            self.first_a[tuple_slot] = int(rec.get("first", -1))
            self.prev_a[tuple_slot] = int(rec.get("prev", -1))
            self.lengths[tuple_slot] = float(rec["lengths"])
            self.cost_to_go[tuple_slot] = float(rec["cost_to_go"])
            self.inst_idx[tuple_slot] = slot_inst

            # Re-add to step index (eviction above already removed if needed).
            self._step_index_sets[t].add(tuple_slot)

        # Refresh the np.array view of the step index (kept in sync with the set).
        for t in range(self.N):
            self._step_index[t] = np.fromiter(
                self._step_index_sets[t], dtype=np.int64, count=len(self._step_index_sets[t])
            )

        # Advance heads / counters.
        self.write_head_inst = (self.write_head_inst + 1) % (self.capacity_instances * 2)
        # Note on write_head_inst overflow: we keep it modulo (2 * capacity) so
        # `n_filled_instances >= capacity` detection above is robust without
        # ever overflowing int64 in long runs. Slot resolution always uses
        # `% capacity_instances`.
        self.n_filled_instances = min(self.n_filled_instances + 1, self.capacity_instances)
        self.n_filled_tuples = min(self.n_filled_tuples + self.N, self.capacity_tuples)
        self.write_head_tuple = (slot_inst * self.N + self.N) % self.capacity_tuples

        return slot_inst

    def __len__(self) -> int:
        return self.n_filled_tuples

    @property
    def n_instances(self) -> int:
        return self.n_filled_instances

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_step(self, step: int, batch_size: int) -> Dict:
        """Deterministic-step sampling — returns `batch_size` records from
        records currently filled at step == `step`.

        All rows in the returned dict share the same step value; `state_i` is
        a Python `int`, matching AM decoder's scalar `state.i` requirement.
        """
        if step < 0 or step >= self.N:
            raise ValueError(f"step {step} out of range [0, {self.N})")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        step_indices = self._step_index[step]
        n_at_step = step_indices.shape[0]
        if n_at_step == 0:
            raise RuntimeError(
                f"sample_step({step}) called on empty step bucket — buffer "
                f"contains no records at step {step}"
            )

        if n_at_step < batch_size:
            # Fresh buffer fallback — sample with replacement so callers can
            # always get a full minibatch even before the buffer fills out.
            idx_within = np.random.randint(0, n_at_step, batch_size)
        else:
            idx_within = np.random.choice(n_at_step, batch_size, replace=False)
        idx = step_indices[idx_within]
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=self.device)

        inst = self.inst_idx[idx_t].to(torch.long)

        # Per-step targets (read from the per-step tensor slabs).
        pi_b = self.pi[idx_t]                       # (B, N)
        visited_b = self.visited[idx_t]             # (B, N)
        first_b = self.first_a[idx_t].to(torch.long)
        prev_b = self.prev_a[idx_t].to(torch.long)
        lengths_b = self.lengths[idx_t]
        cost_to_go_b = self.cost_to_go[idx_t]

        # Per-instance fan-out (fancy index by inst_idx).
        coords_b = self.coords[inst]                # (B, N, 2)
        bl_val_b = self.bl_val[inst]
        tour_cost_b = self.tour_cost[inst]

        # Frozen-at-generation invariant — bl_val and cost_to_go are both
        # written at push time, never refreshed. z is therefore stationary
        # for any given record across all train steps that draw it.
        z_b = cost_to_go_b / bl_val_b.clamp(min=1e-6)

        return {
            "state_i": int(step),                   # SCALAR (decoder requires)
            "coords": coords_b,                     # (B, N, 2)
            "visited": visited_b,                   # (B, N) bool
            "first_a": first_b,                     # (B,)
            "prev_a": prev_b,                       # (B,)
            "lengths": lengths_b,                   # (B,)
            "pi": pi_b,                             # (B, N)
            "z": z_b,                               # (B,)
            "bl_val": bl_val_b,                     # (B,)
            "tour_cost": tour_cost_b,               # (B,)
        }

    def sample(self, batch_size: int) -> Dict:
        """Training-loop sampling — picks one step uniformly per minibatch.

        `state_i` in the returned dict is a Python int (the chosen step) —
        every row shares it, satisfying the AM decoder's scalar `state.i`
        requirement.
        """
        # Restrict to non-empty step buckets so a fresh buffer (which only
        # has step 0 filled after the first push) doesn't crash.
        non_empty = [t for t in range(self.N) if self._step_index[t].shape[0] > 0]
        if not non_empty:
            raise RuntimeError("sample() called on empty buffer")
        step = int(np.random.choice(non_empty))
        return self.sample_step(step, batch_size)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist tensor slabs and write heads. `_step_index` is intentionally
        NOT saved — it is rebuilt on `load()` from the locked invariant
        `tuple_slot = inst_idx * N + step`. See plan lines 155-172."""
        payload = {
            "graph_size": self.N,
            "capacity_instances": self.capacity_instances,
            # Tensor slabs
            "coords": self.coords,
            "bl_val": self.bl_val,
            "tour_cost": self.tour_cost,
            "pi": self.pi,
            "visited": self.visited,
            "first_a": self.first_a,
            "prev_a": self.prev_a,
            "lengths": self.lengths,
            "cost_to_go": self.cost_to_go,
            "inst_idx": self.inst_idx,
            # Bookkeeping
            "write_head_inst": self.write_head_inst,
            "write_head_tuple": self.write_head_tuple,
            "n_filled_instances": self.n_filled_instances,
            "n_filled_tuples": self.n_filled_tuples,
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        """Restore from disk. Validates schema and rebuilds `_step_index`."""
        d = torch.load(path, map_location=self.device, weights_only=False)
        if int(d["graph_size"]) != self.N:
            raise ValueError(
                f"saved graph_size {d['graph_size']} != buffer N={self.N}"
            )
        if int(d["capacity_instances"]) != self.capacity_instances:
            raise ValueError(
                f"saved capacity_instances {d['capacity_instances']} != "
                f"buffer capacity {self.capacity_instances}"
            )

        # Restore slabs in-place (same shapes).
        self.coords.copy_(d["coords"])
        self.bl_val.copy_(d["bl_val"])
        self.tour_cost.copy_(d["tour_cost"])
        self.pi.copy_(d["pi"])
        self.visited.copy_(d["visited"])
        self.first_a.copy_(d["first_a"])
        self.prev_a.copy_(d["prev_a"])
        self.lengths.copy_(d["lengths"])
        self.cost_to_go.copy_(d["cost_to_go"])
        self.inst_idx.copy_(d["inst_idx"])

        self.write_head_inst = int(d["write_head_inst"])
        self.write_head_tuple = int(d["write_head_tuple"])
        self.n_filled_instances = int(d["n_filled_instances"])
        self.n_filled_tuples = int(d["n_filled_tuples"])

        self._rebuild_step_index()

    def _rebuild_step_index(self) -> None:
        """Rebuild `_step_index` and `_step_index_sets` from the dense slot
        layout. Exploits the locked invariant `tuple_slot = inst_idx * N + step`.

        Implementation: a slot is currently filled iff its owning instance
        slot is filled. The owning instance is `tuple_slot // N`. We mark
        the first `n_filled_instances` instance slots in write order as
        filled (ring-buffer head minus n_filled, modulo capacity). For
        simplicity, we approximate by treating *every* instance slot up to
        `min(n_filled_instances, capacity)` after the head as filled, which
        for any buffer that has wrapped covers all `capacity_instances` slots.
        """
        cap_inst = self.capacity_instances
        filled_inst = np.zeros(cap_inst, dtype=bool)
        if self.n_filled_instances >= cap_inst:
            filled_inst[:] = True
        else:
            # Pre-wrap: instance slots [0, n_filled_instances) are valid.
            # (write_head advances 0,1,2,... before any wrap, so the slots
            # written so far are exactly [0, n_filled_instances).)
            filled_inst[: self.n_filled_instances] = True

        slots = np.arange(self.capacity_tuples, dtype=np.int64)
        owning_inst = slots // self.N
        step_per_slot = slots % self.N
        slot_filled = filled_inst[owning_inst]

        self._step_index = []
        self._step_index_sets = []
        for t in range(self.N):
            mask = (step_per_slot == t) & slot_filled
            arr = slots[mask].astype(np.int64)
            self._step_index.append(arr)
            self._step_index_sets.append(set(int(x) for x in arr))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def step_counts(self) -> List[int]:
        """Return a length-N list with the number of records filled at each step.

        Useful for verifying the dense layout invariant in tests / smoke runs.
        """
        return [int(arr.shape[0]) for arr in self._step_index]


# ---------------------------------------------------------------------------
# State-tensor reconstruction
# ---------------------------------------------------------------------------


def reconstruct_state(
    batch: Dict,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> StateTSP:
    """Rebuild a `StateTSP` from a stratified buffer batch.

    Mirrors `mcts_cpp/solver.py:_state_from_snapshot` — same fields, same
    dtypes. The decoder consumes `state.i` as a SCALAR step value, so this
    helper requires `batch['state_i']` to be a Python int (which is exactly
    what `MCTSReplayBuffer.sample_step` returns).

    The returned tensors live on `device` so the caller can plug them into
    `model.decode_step` directly. `loc` is set to `coords.to(device)`; the
    encoded embeddings (which are what the decoder actually consumes via
    `precompute_decoder`) must be computed by the caller — typically inside
    the train step so gradients flow through `model.encode`.
    """
    state_i = int(batch["state_i"])
    coords = batch["coords"].to(device=device, dtype=dtype)        # (B, N, 2)
    visited = batch["visited"].to(device=device, dtype=torch.bool)  # (B, N)
    first_a = batch["first_a"].to(device=device, dtype=torch.long)  # (B,)
    prev_a = batch["prev_a"].to(device=device, dtype=torch.long)    # (B,)
    lengths = batch["lengths"].to(device=device, dtype=dtype)       # (B,)

    bsz, n_nodes, _ = coords.shape

    # Distance matrix (B, N, N). Same construction as StateTSP.initialize.
    dist = (coords[:, :, None, :] - coords[:, None, :, :]).norm(p=2, dim=-1)

    ids = torch.arange(bsz, dtype=torch.long, device=device)[:, None]

    # For step == 0, AM decoder branches on `state.i.item() == 0` and uses
    # the placeholder embedding; first_a / prev_a / cur_coord are unused on
    # that branch. Mirror solver.py:466-471 — clamp first/prev to 0 and
    # set cur_coord = None for safety.
    if state_i == 0:
        first_a_in = torch.zeros(bsz, 1, dtype=torch.long, device=device)
        prev_a_in = torch.zeros(bsz, 1, dtype=torch.long, device=device)
        cur_coord = None
    else:
        first_a_in = first_a.view(bsz, 1)
        prev_a_in = prev_a.view(bsz, 1)
        # cur_coord = coords[b, prev_a[b]] for each row in the batch.
        cur_coord = coords.gather(
            1, prev_a_in.unsqueeze(-1).expand(bsz, 1, 2)
        )

    visited_t = visited.view(bsz, 1, n_nodes)
    lengths_t = lengths.view(bsz, 1)
    i_t = torch.tensor([state_i], dtype=torch.long, device=device)

    return StateTSP(
        loc=coords,
        dist=dist,
        ids=ids,
        first_a=first_a_in,
        prev_a=prev_a_in,
        visited_=visited_t,
        lengths=lengths_t,
        cur_coord=cur_coord,
        i=i_t,
    )


# ---------------------------------------------------------------------------
# Phase C — self-play config + batched self-play generator
# ---------------------------------------------------------------------------


@dataclass
class InstanceRecord:
    """Phase C self-play record. Mirrors the per-instance arguments of
    `MCTSReplayBuffer.push_instance`. Kept as a tiny carrier dataclass so the
    coach orchestrator (Phase D) can iterate `records` and unpack into
    `buf.push_instance(coords, bl_val, tour_cost, per_step)`.
    """
    coords: torch.Tensor          # (N, 2) float32, CPU
    bl_val: float                 # scalar, frozen at generation
    tour_cost: float              # scalar, MCTS-played tour cost
    per_step: List[Dict]          # length N; per-step dict matches push_instance


def make_self_play_config(
    graph_size: int,
    n_simulations: int,
    leaf_eval: str = 'value_head',
    dirichlet_epsilon: float = 0.25,
    dirichlet_alpha_factor: float = 10.0,
    temperature_schedule: str = 'step30',
    value_target_norm: str = 'bl',
    n_simulations_per_step=None,
):
    """AlphaGo-Zero-style self-play preset.

    Returns an `MCTSConfig` configured per Stage 4 plan §Phase C.1. The
    AGZ-canonical defaults (value_head, ε=0.25, step30) work from a
    random-init policy but **destroy MCTS quality on a converged warm-start**
    (see _progress/stage4_progress.md F.3 attempt 1). For warm-started
    Stage 4, override via Coach's CLI (`--leaf_eval rollout`,
    `--dirichlet_epsilon 0.05`).

    Args:
        graph_size: TSP-N size.
        n_simulations: K — MCTS simulations per root.
        leaf_eval: 'value_head' (AGZ canonical) or 'rollout'. The probe in
            `src/scripts/probe_mcts_quality.py` shows rollout consistently
            beats value_head on the Stage 1 checkpoint.
        dirichlet_epsilon: root-noise mixing weight. AGZ canonical 0.25
            assumes a near-uniform prior; ε ≤ 0.05 is recommended for
            warm-started training (probe-validated).
        dirichlet_alpha_factor: scale for α = factor / N (AGZ default 10/N).
        temperature_schedule: 'const' | 'step30' | 'step50'. Affects only
            action-selection σ_t at the root; π_t (the training target) is
            always raw τ=1 (decoupled per spec §4.2 choice (B)).
    """
    # Local import to avoid circular import at module load.
    from am_baseline.search.mcts import MCTSConfig
    cfg = MCTSConfig(
        n_simulations=n_simulations,
        n_simulations_per_step=(
            tuple(int(k) for k in n_simulations_per_step)
            if n_simulations_per_step else None
        ),
        leaf_eval=leaf_eval,
        value_norm='bl',
        value_target_norm=value_target_norm,
        c_puct=0.05,
        temperature=1.0,
        temperature_schedule=temperature_schedule,
        dirichlet_alpha=dirichlet_alpha_factor / graph_size,
        dirichlet_epsilon=dirichlet_epsilon,
        fpu_mode='running_q',
        fpu_fallback=-1.0,
        root_select='visits',
        tree_reuse=True,
        return_root_visits=True,
    )
    return cfg


def _compute_edge_costs(coords: np.ndarray, tour: np.ndarray) -> np.ndarray:
    """Per-edge costs along a played tour, matching `TSP.get_edge_costs`.

    Args:
        coords: (N, 2) float32 — graph coordinates.
        tour:   (N,) int — visit order.

    Returns:
        (N,) float32 with edge_costs[k] = ||coords[tour[k+1]] - coords[tour[k]]||
        for k in [0, N-2], and edge_costs[N-1] = ||coords[tour[0]] - coords[tour[N-1]]||
        (the closing edge).

    Sanity: edge_costs.sum() == realized tour cost.
    """
    N = tour.shape[0]
    if N == 0:
        return np.zeros(0, dtype=np.float32)
    ordered = coords[tour]                                  # (N, 2)
    fwd = np.linalg.norm(ordered[1:] - ordered[:-1], axis=-1)            # (N-1,)
    closing = np.linalg.norm(ordered[0] - ordered[-1])                   # scalar
    return np.concatenate([fwd, np.array([closing])]).astype(np.float32)


def _mask_from_tour(prefix_tour: np.ndarray, N: int) -> np.ndarray:
    """Boolean (N,) mask: True for cities visited so far (i.e. in `prefix_tour`)."""
    mask = np.zeros(N, dtype=bool)
    if prefix_tour.size > 0:
        mask[prefix_tour] = True
    return mask


def _normalize_visit_dict(d: Dict[int, int], N: int) -> np.ndarray:
    """Scatter sparse visit dict into dense (N,) τ=1 distribution.

    Sparse format: `d` only contains keys for actions that were touched in
    backup at this tour-step (Phase A note in `_progress/stage4_progress.md`).
    Visited cities are never keys.

    Args:
        d: dict mapping action -> visit count (non-negative ints).
        N: graph size (output length).

    Returns:
        (N,) float32 with sum == 1 (or 0 if `d` is empty — a degenerate case
        that should not occur with K > 0; we guard with an assertion).
    """
    out = np.zeros(N, dtype=np.float32)
    total = 0
    for a, c in d.items():
        out[int(a)] = float(c)
        total += int(c)
    if total <= 0:
        # Degenerate: no visits at this step. Cannot happen with K > 0 unless
        # the step is forced (one legal action) and tree_reuse fast-paths it.
        # Surface the bug clearly rather than emit a uniform fallback that
        # would silently corrupt the training target.
        raise AssertionError(
            f"_normalize_visit_dict: empty visit dict (total visits == 0). "
            f"Either K == 0 or an upstream bug; refusing to fabricate a target."
        )
    out /= float(total)
    return out


def generate_self_play_batch(
    best_model,
    M: int,
    graph_size: int,
    cfg,
    device: torch.device,
    mcts_batch_size: int = 64,
) -> List[InstanceRecord]:
    """Generate `M` TSP-`graph_size` self-play instances under MCTS guided by
    `best_model` (θ★) using config `cfg`.

    For each instance we:
      1. Sample fresh coordinates via `TSP.make_dataset`.
      2. Compute `bl_val = greedy_cost(theta_star, coords)` once and freeze it.
      3. Run `CppBatchMCTSSolver.solve_batch` to produce a tour and per-step
         root visit dicts (sparse `dict[int,int]` per tour-step per instance).
      4. Pack into `InstanceRecord` matching `MCTSReplayBuffer.push_instance`'s
         per-step schema:
             { 'visited', 'first', 'prev', 'lengths', 'pi', 'cost_to_go' }

    The cost-to-go targets are derived from the played tour's edge costs via
    `value_targets_from_edges` — same convention the value head was trained
    against in Stage 1 (closing edge included exactly once; entry [t] is the
    cost of every edge still to traverse from state s_t).

    Args:
        best_model: AttentionModel — θ★ that runs MCTS and provides bl_val.
        M: number of instances to generate.
        graph_size: TSP graph size N.
        cfg: MCTSConfig (typically from `make_self_play_config`).
        device: torch device for solver / model.
        mcts_batch_size: cross-instance NN-eval batch size for
            `CppBatchMCTSSolver`. Default 64 is fine on CPU.

    Returns:
        list of `InstanceRecord` with length M, ready for
        `MCTSReplayBuffer.push_instance(rec.coords, rec.bl_val, rec.tour_cost,
        rec.per_step)`.
    """
    # Local imports defer heavy / circular deps to call time.
    from am_baseline.problem.tsp import TSP
    from am_baseline.search.mcts_cpp.solver import CppBatchMCTSSolver
    from am_baseline.utils.tensor_ops import value_targets_from_edges

    if M <= 0:
        raise ValueError(f"M must be positive, got {M}")
    if graph_size <= 0:
        raise ValueError(f"graph_size must be positive, got {graph_size}")
    if not cfg.return_root_visits:
        raise ValueError(
            "generate_self_play_batch requires cfg.return_root_visits=True so "
            "the generator can read root visit dicts; got False."
        )

    # 1. Sample instances.
    dataset = TSP.make_dataset(size=graph_size, num_samples=M)
    coords = torch.stack([dataset[i] for i in range(M)]).to(device)   # (M, N, 2)

    # 2. Frozen baseline normalizer = greedy cost under θ★. One forward pass.
    best_model.eval()
    prev_decode_type = best_model.decoder.decode_type
    with torch.no_grad():
        best_model.set_decode_type('greedy')
        bl_costs, _ = best_model(coords)
        bl_val_t = bl_costs.detach().cpu()                            # (M,) float32
    if prev_decode_type is not None:
        best_model.set_decode_type(prev_decode_type)

    # 3. Run cross-instance batched MCTS. Pass our frozen bl_val so the solver
    #    does not silently recompute one — keeps generator targets consistent.
    solver = CppBatchMCTSSolver(
        best_model, cfg, device=device, mcts_batch_size=mcts_batch_size
    )
    tour_costs, tours = solver.solve_batch(
        coords, bl_vals=bl_val_t.to(device)
    )
    visits_per_inst = solver.root_visit_dists_per_instance            # list[list[dict]]
    if len(visits_per_inst) != M:
        raise RuntimeError(
            f"CppBatchMCTSSolver returned {len(visits_per_inst)} visit-dist "
            f"rows for {M} instances — return_root_visits not propagating."
        )

    # 4. Pack records. cost-to-go is derived per-instance from realized edges.
    coords_cpu = coords.detach().cpu()
    tours_cpu = tours.detach().cpu()
    tour_costs_cpu = tour_costs.detach().cpu()

    records: List[InstanceRecord] = []
    for i in range(M):
        coords_i = coords_cpu[i].numpy()                              # (N, 2)
        tour_i = tours_cpu[i].numpy().astype(np.int64)                # (N,)
        edge_costs_i = _compute_edge_costs(coords_i, tour_i)          # (N,)

        # cost_to_go[t] = total cost still to traverse from s_t (V_CURRENT
        # convention; see utils.tensor_ops.value_targets_from_edges).
        ec_t = torch.from_numpy(edge_costs_i).unsqueeze(0)            # (1, N)
        ctg = value_targets_from_edges(ec_t).squeeze(0).numpy()       # (N,)

        per_step: List[Dict] = []
        for t in range(graph_size):
            visited_mask = _mask_from_tour(tour_i[:t], graph_size)
            pi_t = _normalize_visit_dict(visits_per_inst[i][t], graph_size)
            # state.lengths convention from Stage 1: 0 at t=0 and t=1, else
            # cumulative cost of edges traversed BEFORE arriving at s_t.
            if t < 2:
                length_t = 0.0
            else:
                length_t = float(edge_costs_i[: t - 1].sum())

            per_step.append({
                'visited': visited_mask,
                'first': int(tour_i[0]) if t > 0 else -1,
                'prev':  int(tour_i[t - 1]) if t > 0 else -1,
                'lengths': length_t,
                'pi': pi_t,
                'cost_to_go': float(ctg[t]),
            })

        records.append(InstanceRecord(
            coords=coords_cpu[i],
            bl_val=float(bl_val_t[i].item()),
            tour_cost=float(tour_costs_cpu[i].item()),
            per_step=per_step,
        ))

    return records


# ---------------------------------------------------------------------------
# Phase D — MCTSCoach orchestrator
# ---------------------------------------------------------------------------


class MCTSCoach:
    """Phase D AlphaGo-Zero-style self-improvement orchestrator.

    Per-iteration loop (mirrors `_plans/stage4_plan.md` lines 322-378):
      1. **Generate** M self-play instances under θ★ (`best_model`) using
         `make_self_play_config` + `generate_self_play_batch`.
      2. **Push** each `InstanceRecord` into `MCTSReplayBuffer` via
         `push_instance(coords, bl_val, tour_cost, per_step)`.
      3. **Train** the working model for `train_steps_per_iter` minibatches
         drawn from the buffer's stratified-by-step sampler; each step calls
         `train_step_alphazero` and accumulates running means in
         `MetricsLogger.log_alphazero_step`.
      4. **Validate** the working model on `val_dataset` via Stage 1 `validate`.
      5. **Gate** every `opts.gate_every` iterations: a paired t-test
         (`RolloutBaseline.epoch_callback`) decides whether to promote the
         working model to the new θ★. **No rollback on reject** (scope
         decision 3 in the plan).
      6. **Checkpoint** every iteration so a long run can be resumed.

    Init-order trap (caught at review): `RolloutBaseline.__init__` builds and
    caches its validation dataset using `opts.val_size` *at construction time*.
    Subsequent `epoch_callback` calls do **not** re-read `opts.val_size`.
    Therefore `MCTSCoach.__init__` must be called *after* `opts.val_size` has
    been finalized (typically from CLI) — there is **no** Stage-4-specific
    `gate_val_size` flag.
    """

    def __init__(self, model, problem, opts, val_dataset, device=None):
        # Local imports keep heavy / circular deps deferred.
        import copy
        import os
        import time
        import torch as _torch
        from am_baseline.training.logging import MetricsLogger
        from am_baseline.training.trainer import rollout
        from am_baseline.baseline.baselines import RolloutBaseline

        self._copy = copy
        self._os = os
        self._time = time
        self._torch = _torch

        self.model = model
        self.best_model = copy.deepcopy(model)
        self.problem = problem
        self.opts = opts
        self.val_dataset = val_dataset
        self.device = (
            device
            if device is not None
            else getattr(opts, 'device', _torch.device('cpu'))
        )
        if not isinstance(self.device, _torch.device):
            self.device = _torch.device(self.device)

        # Replay buffer — capacity from opts.buffer_capacity (Phase F default
        # 50_000 per AGZ-proportional pilot; Phase B default 200_000 if unset).
        capacity = int(getattr(opts, 'buffer_capacity', 50_000))
        self.buffer = MCTSReplayBuffer(
            graph_size=int(opts.graph_size),
            capacity_instances=capacity,
            device='cpu',
        )

        # Optionally freeze the encoder (init_embed + embedder) so distillation
        # only updates decoder + value_head. Tests the "shared backbone is the
        # noise channel" hypothesis from F.3 v3-v5 plateau diagnosis: if
        # freezing the encoder lets the heads cleanly absorb MCTS targets,
        # the F.3 plateau was encoder-poisoning rather than capacity ceiling.
        if bool(getattr(opts, 'freeze_encoder', False)):
            for p in self.model.init_embed.parameters():
                p.requires_grad_(False)
            for p in self.model.embedder.parameters():
                p.requires_grad_(False)
            n_total = sum(p.numel() for p in self.model.parameters())
            n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f'[freeze_encoder] froze {n_total - n_train:,} encoder params; '
                  f'{n_train:,} trainable (decoder + value_head + ...)')

        # Optimizer over the WORKING copy (not best_model). Stage 1 default
        # picks Adam; SGD+momentum is the AGZ-canonical alternative under
        # plan G.8 ablation.
        self.optimizer = _torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(getattr(opts, 'lr_model', 1e-4)),
            weight_decay=float(self._weight_decay()),
        )

        # LR scheduler — multiplicative per-iteration decay.
        # lr(iter k) = lr_model * lr_decay**k. Default lr_decay=1.0 → no decay.
        # Stage 1 uses the same LambdaLR pattern (train.py:102).
        _lr_decay = float(getattr(opts, 'lr_decay', 1.0))
        self.lr_scheduler = _torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda iter_k: _lr_decay ** iter_k
        )

        # Gating — verbatim Stage 1 RolloutBaseline. Construct AFTER
        # `opts.val_size` is finalized (init-order trap). Note that
        # `RolloutBaseline.__init__` runs a greedy rollout over `val_size`
        # samples right here — for a small smoke this is cheap, for the full
        # pilot it is one-time at startup.
        self.gating_baseline = RolloutBaseline(
            self.best_model, problem, opts, rollout_fn=rollout, epoch=0
        )

        self.iter_idx = 0
        self.total_instances_seen = 0

        # Logger — under opts.save_dir (Stage 1 convention) so iterations.csv
        # ends up next to metrics.csv / epochs.csv. Honor opts.no_wandb (or
        # any of the Stage 1 W&B kill-switch flags) by passing
        # `wandb_project=None` to the logger.
        # `wandb_group=f"tsp_{graph_size}"` matches Stage 1's grouping
        # convention ([scripts/train.py:45]) so Stage 1 + Stage 4 runs land
        # in the same W&B group for side-by-side comparison.
        log_dir = getattr(opts, 'save_dir', getattr(opts, 'output_dir', '.'))
        wandb_project = self._wandb_project()
        graph_size_for_group = int(getattr(opts, 'graph_size', 20))
        self.logger = MetricsLogger(
            log_dir=log_dir,
            use_tensorboard=False,
            wandb_project=wandb_project,
            wandb_entity=getattr(opts, 'wandb_entity', None),
            wandb_group=getattr(opts, 'wandb_group', f"tsp_{graph_size_for_group}"),
            wandb_name=getattr(opts, 'run_name', None),
            wandb_mode=getattr(opts, 'wandb_mode', 'online'),
            track_gpu_memory=getattr(opts, 'use_cuda', False),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _weight_decay(self) -> float:
        # AGZ-canonical L2 reg = 1e-4. Allow opts override under either name.
        if hasattr(self.opts, 'weight_decay'):
            return float(self.opts.weight_decay)
        if hasattr(self.opts, 'l2_reg'):
            return float(self.opts.l2_reg)
        return 1e-4

    def _wandb_project(self):
        """Return the W&B project name, honoring the Stage 1 kill-switch
        `opts.no_wandb` (and the `wandb_mode='disabled'` form)."""
        if bool(getattr(self.opts, 'no_wandb', False)):
            return None
        mode = getattr(self.opts, 'wandb_mode', 'online')
        if mode == 'disabled':
            return None
        return getattr(self.opts, 'wandb_project', None)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _checkpoint_dir(self) -> str:
        """Directory where iter-{i}.pt and buffer.pt live."""
        return getattr(
            self.opts, 'save_dir',
            self._os.path.join(getattr(self.opts, 'output_dir', '.'), 'stage4'),
        )

    def save_checkpoint(self, tag: str) -> str:
        """Persist the coach state to `save_dir/iter-{tag}.pt` (+ buffer.pt).

        File contents (small):
            model:      state_dict
            best_model: state_dict
            optimizer:  state_dict
            iter_idx:   int
            total_instances_seen: int
            rng_state:  {torch, numpy, python}

        The replay buffer is written separately to `buffer.pt` (overwritten
        each iter — its slabs are large; checkpoint files stay small).
        Returns the path written for the iteration checkpoint.
        """
        import random as _random
        import numpy as _np

        ckpt_dir = self._checkpoint_dir()
        self._os.makedirs(ckpt_dir, exist_ok=True)

        rng_state = {
            'torch': self._torch.get_rng_state(),
            'numpy': _np.random.get_state(),
            'python': _random.getstate(),
        }
        if self._torch.cuda.is_available():
            try:
                rng_state['cuda'] = self._torch.cuda.get_rng_state_all()
            except Exception:
                pass

        payload = {
            'model': self.model.state_dict(),
            'best_model': self.best_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'iter_idx': int(self.iter_idx),
            'total_instances_seen': int(self.total_instances_seen),
            'rng_state': rng_state,
        }
        ckpt_path = self._os.path.join(ckpt_dir, f'iter-{tag}.pt')
        self._torch.save(payload, ckpt_path)

        # Buffer — large; overwrite the single rolling buffer.pt.
        buf_path = self._os.path.join(ckpt_dir, 'buffer.pt')
        try:
            self.buffer.save(buf_path)
        except Exception as e:
            print(f'Warning: buffer.save({buf_path}) failed: {e}')
        return ckpt_path

    # Private alias the plan pseudocode uses.
    def _save_checkpoint(self, tag: str) -> str:
        return self.save_checkpoint(tag)

    def load_checkpoint(self, path: str) -> None:
        """Restore coach state from `path` (an `iter-{i}.pt` file).

        Loads model + best_model + optimizer + iter_idx + total_instances_seen
        + rng_state. Then loads the sibling `buffer.pt` if present (warns and
        continues if missing — buffer is recoverable in O(1 iter)).
        """
        import random as _random
        import numpy as _np

        d = self._torch.load(path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(d['model'])
        self.best_model.load_state_dict(d['best_model'])
        self.optimizer.load_state_dict(d['optimizer'])
        if 'lr_scheduler' in d:
            self.lr_scheduler.load_state_dict(d['lr_scheduler'])
        # Stored `iter_idx` is the LAST COMPLETED iteration; advance one so
        # `learn(...)` resumes at the next integer.
        self.iter_idx = int(d['iter_idx']) + 1
        self.total_instances_seen = int(d['total_instances_seen'])

        rng = d.get('rng_state', {})
        if 'torch' in rng:
            try:
                self._torch.set_rng_state(rng['torch'])
            except Exception as e:
                print(f'Warning: torch RNG restore failed: {e}')
        if 'numpy' in rng:
            try:
                _np.random.set_state(rng['numpy'])
            except Exception as e:
                print(f'Warning: numpy RNG restore failed: {e}')
        if 'python' in rng:
            try:
                _random.setstate(rng['python'])
            except Exception as e:
                print(f'Warning: python RNG restore failed: {e}')
        if 'cuda' in rng and self._torch.cuda.is_available():
            try:
                self._torch.cuda.set_rng_state_all(rng['cuda'])
            except Exception as e:
                print(f'Warning: cuda RNG restore failed: {e}')

        # Buffer — best-effort.
        buf_path = self._os.path.join(self._os.path.dirname(path), 'buffer.pt')
        if self._os.path.exists(buf_path):
            try:
                self.buffer.load(buf_path)
            except Exception as e:
                print(
                    f'Warning: buffer.load({buf_path}) failed: {e}. '
                    f'Continuing — buffer will refill in one iteration.'
                )
        else:
            print(
                f'Warning: no buffer.pt next to {path}. Continuing — buffer '
                f'will refill in one iteration.'
            )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def learn(self, n_iterations: int) -> None:
        """Run `n_iterations` of generate → train → (gate) → checkpoint.

        Resume-aware: starts at the current `self.iter_idx` (which
        `load_checkpoint` restores to the last completed iteration + 1) and
        runs for `n_iterations` more iterations.
        """
        # Local imports avoid circular load (trainer imports from coach).
        from am_baseline.training.trainer import validate, train_step_alphazero

        opts = self.opts
        start = int(self.iter_idx)
        end = start + int(n_iterations)
        for self.iter_idx in range(start, end):
            t0 = self._time.time()

            cfg = make_self_play_config(
                int(opts.graph_size),
                int(getattr(opts, 'n_simulations_train', 50)),
                leaf_eval=str(getattr(opts, 'leaf_eval', 'value_head')),
                dirichlet_epsilon=float(getattr(opts, 'dirichlet_epsilon', 0.25)),
                dirichlet_alpha_factor=float(getattr(opts, 'dirichlet_alpha_factor', 10.0)),
                temperature_schedule=str(getattr(opts, 'temperature_schedule', 'step30')),
                value_target_norm=str(getattr(opts, 'value_target_norm', 'bl')),
                n_simulations_per_step=getattr(opts, 'n_simulations_per_step', None),
            )
            records = generate_self_play_batch(
                self.best_model,
                int(opts.M_instances),
                int(opts.graph_size),
                cfg,
                self.device,
                mcts_batch_size=int(getattr(opts, 'mcts_batch_size', 64)),
            )
            for r in records:
                self.buffer.push_instance(r.coords, r.bl_val, r.tour_cost, r.per_step)
            self.total_instances_seen += int(opts.M_instances)
            t1 = self._time.time()

            # Train. Stratified-by-step sampler ensures every minibatch has a
            # single scalar `state_i` (decoder requirement).
            train_steps = int(getattr(opts, 'train_steps_per_iter', 100))
            batch_size = int(getattr(opts, 'batch_size', 256))
            for step in range(train_steps):
                batch = self.buffer.sample(batch_size)
                metrics = train_step_alphazero(
                    self.model, self.optimizer, batch, opts
                )
                self.logger.log_alphazero_step(metrics, self.iter_idx, step)
            t2 = self._time.time()

            val_cost_t = validate(self.model, self.val_dataset, opts)
            val_cost = (
                float(val_cost_t.item()) if hasattr(val_cost_t, 'item') else float(val_cost_t)
            )

            gated = False
            accepted = None
            gate_every = int(getattr(opts, 'gate_every', 1))
            gate_mode = str(getattr(opts, 'gate_mode', 'ttest'))
            if gate_every > 0 and (self.iter_idx + 1) % gate_every == 0:
                gated = True
                if gate_mode == 'always':
                    # Phase G.5.c — skip the t-test, always accept.
                    accepted = True
                elif gate_mode == 'never':
                    accepted = False
                else:  # 'ttest' — Stage 1's paired-t at α=0.05 (current default).
                    accepted = self.gating_baseline.epoch_callback(
                        self.model, epoch=self.iter_idx
                    )
                if accepted:
                    self.best_model = self._copy.deepcopy(self.model)
                    self._save_checkpoint(tag=f'{self.iter_idx}_accepted')
                # NB: per scope decision 3, NO rollback on reject.

            # Capture lr USED this iter (before stepping the scheduler for next iter).
            current_lr = float(self.optimizer.param_groups[0]['lr'])

            self.logger.log_iteration(
                iter=self.iter_idx,
                total_instances=self.total_instances_seen,
                val_avg_cost=val_cost,
                gated=gated,
                accepted=accepted,
                mcts_wall_s=t1 - t0,
                train_wall_s=t2 - t1,
                buffer_size=int(len(self.buffer)),
                lr=current_lr,
            )
            self._save_checkpoint(tag=f'{self.iter_idx}')

            # Advance lr scheduler for next iter. lr(iter k+1) = lr_model * lr_decay**(k+1).
            self.lr_scheduler.step()

        # Position iter_idx at the next integer so a follow-up call to
        # `learn(...)` resumes after the last completed iteration.
        self.iter_idx = end

    def close(self) -> None:
        """Flush + close the underlying logger."""
        if self.logger is not None:
            self.logger.close()
