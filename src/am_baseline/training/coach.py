"""Stage 4 coach module — replay buffer and state reconstruction utilities.

Phase B of `_plans/stage4_plan.md` lives here. The full coach orchestrator
(`MCTSCoach.learn`) is reserved for Phase D. This module provides:

  - `MCTSReplayBuffer`: flat dict-of-pre-allocated-tensors replay buffer with
    ring-buffer eviction and stratified-by-step sampling.
  - `reconstruct_state`: rebuild a `StateTSP` named-tuple from a buffer batch
    so the AM decoder can score it inside `train_step_alphazero`.

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
