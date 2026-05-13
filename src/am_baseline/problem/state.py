import torch
from typing import NamedTuple


class StateTSP(NamedTuple):
    # Fixed input
    loc: torch.Tensor       # (batch, n, 2)
    dist: torch.Tensor      # (batch, n, n)

    # Beam search indexing
    ids: torch.Tensor       # (batch, 1)

    # State
    first_a: torch.Tensor   # (batch, 1)
    prev_a: torch.Tensor    # (batch, 1)
    visited_: torch.Tensor  # (batch, 1, n) bool mask
    lengths: torch.Tensor   # (batch, 1)
    cur_coord: torch.Tensor # (batch, 1, 2) or None
    # Step counter. Two shapes are supported (Track A 2026-05-11):
    #   - scalar (1,)  : all rows share the same step (non-MCTS callers,
    #                    training-time decode, selection-path eval_many).
    #   - per-row (B,) : rows are at heterogeneous steps (rollout_many fast
    #                    path; eval_many_arrays after merging step groups).
    # Per-row callers must use the Decoder fast paths in `_get_step_context`
    # which branch on `state.i.numel()`.
    i: torch.Tensor

    @property
    def visited(self):
        return self.visited_

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        # Slice `i` only when it is per-row (B,); leave scalar `i` alone so
        # the non-MCTS callers preserve their existing semantics.
        new_i = self.i
        if self.i.numel() > 1:
            new_i = self.i[key]
        return self._replace(
            ids=self.ids[key],
            first_a=self.first_a[key],
            prev_a=self.prev_a[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key],
            cur_coord=self.cur_coord[key] if self.cur_coord is not None else None,
            i=new_i,
        )

    @staticmethod
    def initialize(loc):
        batch_size, n_loc, _ = loc.size()
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)
        return StateTSP(
            loc=loc,
            dist=(loc[:, :, None, :] - loc[:, None, :, :]).norm(p=2, dim=-1),
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],
            first_a=prev_a,
            prev_a=prev_a,
            visited_=torch.zeros(batch_size, 1, n_loc, dtype=torch.bool, device=loc.device),
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            cur_coord=None,
            i=torch.zeros(1, dtype=torch.int64, device=loc.device),
        )

    def get_final_cost(self):
        assert self.all_finished()
        return self.lengths + (self.loc[self.ids, self.first_a, :] - self.cur_coord).norm(p=2, dim=-1)

    def update(self, selected):
        prev_a = selected[:, None]
        cur_coord = self.loc[self.ids, prev_a]

        lengths = self.lengths
        if self.cur_coord is not None:
            lengths = self.lengths + (cur_coord - self.cur_coord).norm(p=2, dim=-1)

        # Per-row-safe first_a selection: rows where i==0 take prev_a as the
        # tour starting node; rows where i>=1 keep the existing first_a.
        # `torch.where` broadcasts correctly whether self.i is scalar (1,) or
        # per-row (B,) — for scalar, the condition is a single bool that selects
        # the whole tensor (preserving the original `prev_a if i==0 else first_a`
        # semantics bit-for-bit).
        i_is_zero = (self.i.view(-1, 1) == 0)
        first_a = torch.where(i_is_zero, prev_a, self.first_a)

        visited_ = self.visited_.scatter(-1, prev_a[:, :, None], True)

        return self._replace(
            first_a=first_a, prev_a=prev_a, visited_=visited_,
            lengths=lengths, cur_coord=cur_coord, i=self.i + 1
        )

    def all_finished(self):
        return self.i.item() >= self.loc.size(-2)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        return self.visited_
