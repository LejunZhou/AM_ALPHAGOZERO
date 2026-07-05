"""Stage 5 §H.7 Phase 0 / 0b — supervised separate-value-trunk regression.

Can a value head with its OWN representation fit raw cost-to-go with lower bias
than the shared-glimpse head (~0.082 raw on §H.4)?

Two modes:
  - shared trunk (default): trunk attends over the FROZEN policy encoder's node
    embeddings (Phase 0). Encoder detached; only the trunk trains.
  - --value_own_encoder (Phase 0b): the value path gets its OWN encoder over
    coords. Policy encoder is untouched, so the E[z|s] reference in the probe is
    unchanged. Trains value_embedder + value_init_embed + value_trunk.

Loads a checkpoint, freezes the policy, trains the value path via
MSE(v(s), raw cost_to_go), saves the augmented checkpoint. No self-play/MCTS/C++.

Repro (0b):
    PYTHONPATH=src python -m scripts.train_value_trunk_supervised \
        --ckpt outputs/.../iter-99.pt --buffer outputs/.../buffer.pt --which best \
        --value_own_encoder --epochs 40 --lr 1e-3 --lr_decay 0.2 --lr_decay_step 25 \
        --out_ckpt outputs/.../iter-99_vtrunk_ownenc.pt
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn.functional as F

from am_baseline.model.attention_model import AttentionModel
from am_baseline.training.coach import MCTSReplayBuffer, reconstruct_state
from am_baseline.utils.misc import torch_load_cpu

TRAINABLE_PREFIXES = ("value_trunk", "value_embedder", "value_init_embed")


def parse_args():
    p = argparse.ArgumentParser(description="Supervised separate-value-trunk regression (raw cost-to-go).")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--buffer", required=True)
    p.add_argument("--which", choices=["best", "working"], default="best")
    p.add_argument("--value_own_encoder", action="store_true",
                   help="Phase 0b: value path uses its own encoder (fully separate value net).")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps_per_epoch", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_decay", type=float, default=1.0, help="StepLR gamma (1.0 = off).")
    p.add_argument("--lr_decay_step", type=int, default=0, help="StepLR step size in epochs (0 = off).")
    p.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clip norm (0 = off).")
    p.add_argument("--holdout_k", type=int, default=0,
                   help="If >0, train ONLY on instances where inst_id %% k != 0 (held-out = inst %% k == 0).")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out_ckpt", default=None)
    p.add_argument("--no_cuda", action="store_true")
    return p.parse_args()


def _read_train_args(ckpt_path):
    ap = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "args.json")
    if os.path.exists(ap):
        with open(ap) as f:
            return json.load(f)
    return None


def _batch_from_slots(buf, step, idx):
    idx_t = torch.as_tensor(idx, dtype=torch.long, device=buf.device)
    inst = buf.inst_idx[idx_t].to(torch.long)
    return {
        "state_i": int(step),
        "coords": buf.coords[inst].clone(),
        "visited": buf.visited[idx_t].clone(),
        "first_a": buf.first_a[idx_t].to(torch.long).clone(),
        "prev_a": buf.prev_a[idx_t].to(torch.long).clone(),
        "lengths": buf.lengths[idx_t].clone(),
        "z": (buf.cost_to_go[idx_t] / buf.bl_val[inst].clamp(min=1e-6)).clone(),
        "bl_val": buf.bl_val[inst].clone(),
    }


def _make_split_sampler(buf, holdout_k, split):
    """Minibatch sampler restricted to an instance split
    (train: inst%k != 0, eval: inst%k == 0). Same one-step-per-batch stratification
    as MCTSReplayBuffer.sample."""
    per_step = []
    for t in range(buf.N):
        slots = np.asarray(buf._step_index[t])
        if slots.shape[0] > 0:
            inst = buf.inst_idx[torch.as_tensor(slots, dtype=torch.long)].cpu().numpy()
            keep = (inst % holdout_k != 0) if split == "train" else (inst % holdout_k == 0)
            slots = slots[keep]
        per_step.append(slots)
    nonempty = [t for t in range(buf.N) if per_step[t].shape[0] > 0]
    if not nonempty:
        raise RuntimeError(f"no records in split={split!r}")

    def sample(bs):
        step = int(np.random.choice(nonempty))
        cand = per_step[step]
        if cand.shape[0] < bs:
            idx = cand[np.random.randint(0, cand.shape[0], bs)]
        else:
            idx = cand[np.random.choice(cand.shape[0], bs, replace=False)]
        return _batch_from_slots(buf, step, idx)

    return sample


def _build_cfg(train_args, graph_size, value_own_encoder):
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
        value_target_norm = "none"
        graph_size = 20
        value_head_type = "separate_trunk"
        value_own_encoder = False

    if train_args:
        for k in ("embedding_dim", "hidden_dim", "n_encode_layers", "n_heads",
                  "tanh_clipping", "normalization", "feed_forward_hidden",
                  "value_hidden_dim", "graph_size"):
            if k in train_args:
                setattr(Cfg, k, train_args[k])
    Cfg.graph_size = graph_size
    Cfg.value_enabled = True
    Cfg.value_head_type = "separate_trunk"
    Cfg.value_own_encoder = bool(value_own_encoder)
    return Cfg()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_args = _read_train_args(args.ckpt)
    graph_size = int((train_args or {}).get("graph_size", 20))
    capacity = int((train_args or {}).get("buffer_capacity", 5000))

    cfg = _build_cfg(train_args, graph_size, args.value_own_encoder)
    model = AttentionModel(cfg).to(device)

    ck = torch_load_cpu(args.ckpt)
    key = "best_model" if args.which == "best" else "model"
    missing, unexpected = model.load_state_dict(ck[key], strict=False)
    fresh = [m for m in missing if m.startswith(TRAINABLE_PREFIXES)]
    other_missing = [m for m in missing if not m.startswith(TRAINABLE_PREFIXES)]
    print(f"loaded {key}: fresh value-path keys={len(fresh)} | other_missing={other_missing} | "
          f"unexpected={list(unexpected)[:4]}")

    # Freeze the policy (eval -> BN uses its calibrated running stats). The value
    # path is FRESH, so it must stay in TRAIN mode: otherwise its BatchNorm uses
    # uninitialised running stats (0/1) and never updates them, which diverges.
    model.eval()
    if model.value_embedder is not None:
        model.value_embedder.train()
    if model.value_trunk is not None:
        model.value_trunk.train()
    for name, prm in model.named_parameters():
        prm.requires_grad = name.startswith(TRAINABLE_PREFIXES)
    train_params = [prm for n, prm in model.named_parameters() if prm.requires_grad]
    n_train = sum(p.numel() for p in train_params)
    opt = torch.optim.Adam(train_params, lr=args.lr)
    sched = None
    if args.lr_decay != 1.0 and args.lr_decay_step > 0:
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_decay_step, gamma=args.lr_decay)

    buf = MCTSReplayBuffer(graph_size=graph_size, capacity_instances=capacity, device="cpu")
    buf.load(args.buffer)
    sampler = _make_split_sampler(buf, args.holdout_k, "train") if args.holdout_k > 0 else buf.sample

    own = model.value_embedder is not None
    print(f"device={device} graph_size={graph_size} own_encoder={own} train_params={n_train} "
          f"epochs={args.epochs} steps/epoch={args.steps_per_epoch} bs={args.batch_size} lr={args.lr} "
          f"holdout_k={args.holdout_k} (train split={'inst%%k!=0' if args.holdout_k>0 else 'ALL'})")

    for ep in range(args.epochs):
        t0 = time.time()
        losses = []
        for _ in range(args.steps_per_epoch):
            batch = sampler(args.batch_size)
            coords = batch["coords"].to(device=device, dtype=torch.float32)
            z = batch["z"].to(device=device, dtype=torch.float32).view(-1)
            bl_val = batch["bl_val"].to(device=device, dtype=torch.float32).view(-1)
            target = z * bl_val  # raw cost-to-go

            state = reconstruct_state(
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
            if own:
                fixed = None  # own value encoder ignores `fixed`; skip the policy encode
            else:
                with torch.no_grad():
                    fixed = model.precompute_decoder(model.encode(coords))
            v = model.value_from_state(fixed, state, detach_encoder=False)
            loss = F.mse_loss(v, target)
            opt.zero_grad()
            loss.backward()
            if args.max_grad_norm and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(train_params, args.max_grad_norm)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if sched is not None:
            sched.step()
        lr_now = opt.param_groups[0]["lr"]
        print(f"epoch {ep:2d}: mse={np.mean(losses):.6f}  rmse={np.sqrt(np.mean(losses)):.6f}  "
              f"lr={lr_now:.1e}  ({time.time() - t0:.1f}s)")

    out = args.out_ckpt or os.path.join(
        os.path.dirname(os.path.abspath(args.ckpt)),
        "iter-99_vtrunk_ownenc.pt" if own else "iter-99_vtrunk.pt",
    )
    save = dict(ck)
    sd = model.state_dict()
    save["best_model"] = sd
    save["model"] = sd
    torch.save(save, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
