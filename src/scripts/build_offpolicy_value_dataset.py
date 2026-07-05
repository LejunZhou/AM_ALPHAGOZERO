"""Stage 5 §V0 — build an OFF-POLICY value dataset by rollout-labeling children.

Root cause from §H.7: the value head is calibrated on the self-play trajectory
distribution but ~0.6 raw optimistic on UNTAKEN sibling children — exactly the
states MCTS reads. Fix candidate: distill the greedy rollout into the head on
those states. This script builds the dataset:

For sampled buffer states s at step t, enumerate (a subset of) legal children
s' = s.update(a), and label each with its GREEDY-rollout cost-to-go under the
frozen policy:

    label(s') = rollout_greedy(s').final_cost - s'.lengths       (raw units)

This is the same function `leaf_eval='rollout'` computes inside MCTS
(mcts.py::_rollout_remaining_real), so the trained head is an amortized
rollout. Labels exclude the edge into the child (probe_action_ranking
convention: action value = edge_a + v(s'_a)).

The dataset stores only (slot, action, label) triples + the parent step; the
trainer reconstructs child states from the buffer, so the file stays small.

Instance split (matches probe_action_ranking / train_value_trunk_supervised):
with --holdout_k k, 'train' keeps inst % k != 0, 'eval' keeps inst % k == 0.
Train the head ONLY on the train split; gate on the eval split.

Repro (Colab T4, ~5-10 min at 20k states):
    PYTHONPATH=src python -m scripts.build_offpolicy_value_dataset \
        --ckpt outputs/.../iter-99.pt --buffer outputs/.../buffer.pt --which best \
        --num_states 20000 --holdout_k 5 --split train --seed 1234
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch

from am_baseline.training.coach import MCTSReplayBuffer, reconstruct_state
from scripts.probe_value_aleatoric import _load_stage4_model, _read_json_next_to
from scripts.train_value_trunk_supervised import _batch_from_slots


def parse_args():
    p = argparse.ArgumentParser(description="Rollout-label legal children of buffer states (off-policy value dataset).")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--buffer", required=True)
    p.add_argument("--which", choices=["best", "working"], default="best")
    p.add_argument("--num_states", type=int, default=20000,
                   help="Total parent states sampled, spread uniformly over steps.")
    p.add_argument("--children_per_state", type=int, default=0,
                   help="Random children per parent (0 = ALL legal children).")
    p.add_argument("--holdout_k", type=int, default=5)
    p.add_argument("--split", choices=["train", "eval", "all"], default="train")
    p.add_argument("--max_rows_per_chunk", type=int, default=4096,
                   help="Child rows per batched rollout chunk (memory bound).")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default=None,
                   help="Output .pt path (default: <ckpt_dir>/offpolicy_value_ds_<split>.pt)")
    p.add_argument("--no_cuda", action="store_true")
    return p.parse_args()


def _split_slots_by_step(buf, holdout_k: int, split: str):
    """Per-step slot arrays restricted to the instance split."""
    per_step = []
    for t in range(buf.N):
        slots = np.asarray(buf._step_index[t])
        if slots.shape[0] > 0 and holdout_k > 0 and split != "all":
            inst = buf.inst_idx[torch.as_tensor(slots, dtype=torch.long)].cpu().numpy()
            keep = (inst % holdout_k != 0) if split == "train" else (inst % holdout_k == 0)
            slots = slots[keep]
        per_step.append(slots)
    return per_step


@torch.no_grad()
def _label_chunk(model, buf, step: int, slots: np.ndarray, actions: torch.Tensor, device):
    """Greedy-rollout ctg labels for children (slots[i], actions[i]) at parent step `step`.

    All rows share the parent step, so `state.i` stays a scalar and every row
    finishes the rollout at the same loop iteration (mirrors the production
    scalar fast-path of decode_step).
    """
    batch = _batch_from_slots(buf, step, slots)
    coords = batch["coords"].to(device=device, dtype=torch.float32)
    parent = reconstruct_state(
        {
            "state_i": batch["state_i"],
            "coords": coords,
            "visited": batch["visited"].to(device=device),
            "first_a": batch["first_a"].to(device=device),
            "prev_a": batch["prev_a"].to(device=device),
            "lengths": batch["lengths"].to(device=device),
        },
        device=device,
    )
    fixed = model.precompute_decoder(model.encode(coords))
    child = parent.update(actions.to(device=device))
    start_len = child.lengths.view(-1).clone()

    cur = child
    n_fwd = 0
    while not cur.all_finished():
        log_p, _mask = model.decode_step(fixed, cur)
        a = log_p.reshape(log_p.size(0), -1).argmax(dim=-1)
        cur = cur.update(a)
        n_fwd += 1
    final = cur.get_final_cost().view(-1)
    return (final - start_len).cpu(), n_fwd * int(actions.numel())


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    train_args = _read_json_next_to(args.ckpt)
    graph_size = int((train_args or {}).get("graph_size", 20))
    capacity = int((train_args or {}).get("buffer_capacity", 5000))
    vtn = (train_args or {}).get("value_target_norm", "none")

    model = _load_stage4_model(args.ckpt, args.which, train_args, graph_size, vtn, device)
    model.eval()
    buf = MCTSReplayBuffer(graph_size=graph_size, capacity_instances=capacity, device="cpu")
    buf.load(args.buffer)

    per_step = _split_slots_by_step(buf, args.holdout_k, args.split)
    # Parent step N-1 excluded: its children are TERMINAL states, where MCTS
    # never queries the value head (exact cost is used) and where decode_step
    # has no glimpse (all-visited mask -> NaN log_p).
    nonempty = [t for t in range(buf.N - 1) if per_step[t].shape[0] > 0]
    quota = max(1, args.num_states // len(nonempty))

    print(f"device={device} graph_size={graph_size} split={args.split} holdout_k={args.holdout_k}")
    print(f"num_states={args.num_states} -> {quota}/step over {len(nonempty)} steps; "
          f"children_per_state={'ALL' if args.children_per_state <= 0 else args.children_per_state}")

    out_slots, out_actions, out_steps, out_labels = [], [], [], []
    total_fwd, t0 = 0, time.time()

    for t in nonempty:
        slots_t = per_step[t]
        n_take = min(quota, slots_t.shape[0])
        chosen = rng.choice(slots_t, size=n_take, replace=False)

        # Legal-children matrix: at step t exactly t nodes are visited, so every
        # parent has N - t legal children — a rectangular (n_take, N-t) layout.
        vis = buf.visited[torch.as_tensor(chosen, dtype=torch.long)]  # (n_take, N) bool
        legal = (~vis).nonzero(as_tuple=False)                        # (n_take*(N-t), 2) row-major
        n_children = graph_size - t
        acts = legal[:, 1].view(n_take, n_children)                   # (n_take, N-t)

        if 0 < args.children_per_state < n_children:
            sel = torch.from_numpy(
                np.stack([rng.permutation(n_children)[: args.children_per_state] for _ in range(n_take)])
            ).long()
            acts = acts.gather(1, sel)
            n_children = args.children_per_state

        pair_slots = np.repeat(chosen, n_children)
        pair_actions = acts.reshape(-1)

        # Chunked batched rollouts.
        for lo in range(0, pair_slots.shape[0], args.max_rows_per_chunk):
            hi = min(lo + args.max_rows_per_chunk, pair_slots.shape[0])
            labels, n_fwd = _label_chunk(
                model, buf, t, pair_slots[lo:hi], pair_actions[lo:hi], device
            )
            out_slots.append(torch.as_tensor(pair_slots[lo:hi], dtype=torch.long))
            out_actions.append(pair_actions[lo:hi].to(torch.int16).cpu())
            out_steps.append(torch.full((hi - lo,), t, dtype=torch.int16))
            out_labels.append(labels.to(torch.float32))
            total_fwd += n_fwd
        print(f"  step {t:2d}: {n_take} states x {n_children} children = {n_take * n_children} labels "
              f"({time.time() - t0:.1f}s, {total_fwd} rollout fwd)")

    ds = {
        "slot": torch.cat(out_slots),
        "action": torch.cat(out_actions),
        "step": torch.cat(out_steps),
        "label": torch.cat(out_labels),
        "meta": {
            "ckpt": os.path.abspath(args.ckpt),
            "buffer": os.path.abspath(args.buffer),
            "which": args.which,
            "num_states": args.num_states,
            "children_per_state": args.children_per_state,
            "holdout_k": args.holdout_k,
            "split": args.split,
            "seed": args.seed,
            "graph_size": graph_size,
            "label_def": "greedy rollout final_cost - child.lengths (raw, excludes edge into child)",
        },
    }
    n = ds["label"].numel()
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.ckpt)), f"offpolicy_value_ds_{args.split}.pt"
    )
    torch.save(ds, out)
    print(f"\nwrote {out}")
    print(f"pairs={n}  label mean={ds['label'].mean():.4f} std={ds['label'].std():.4f} "
          f"min={ds['label'].min():.4f} max={ds['label'].max():.4f}")
    print(f"wall={time.time() - t0:.1f}s  rollout_fwd={total_fwd}")


if __name__ == "__main__":
    main()
