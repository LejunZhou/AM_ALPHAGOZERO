"""Stage 5 §V0 — distill the greedy rollout into the value head, OFF-POLICY.

Trains the value head on (child-state -> greedy-rollout cost-to-go) pairs from
`build_offpolicy_value_dataset.py`, i.e. on the counterfactual sibling states
where §H.7.2 measured the ~0.6-raw optimistic bias. The policy (encoder +
decoder) is frozen; only the value path trains:

    glimpse_mlp     -> 'value_head' MLP params only (glimpse computed no-grad)
    separate_trunk  -> value_trunk (+ value_embedder/value_init_embed with
                       --value_own_encoder), as in train_value_trunk_supervised

Target is RAW cost-to-go (value_target_norm='none' convention). Success is NOT
judged by MSE here — the pre-registered gate is `probe_action_ranking` decision
regret / within-node Spearman on the held-out instance split (plan
stage5_offpolicy_value_plan.md §V0).

Repro (Colab T4, ~10 min):
    PYTHONPATH=src python -m scripts.train_value_head_offpolicy \
        --ckpt outputs/.../iter-99.pt --buffer outputs/.../buffer.pt --which best \
        --dataset outputs/.../offpolicy_value_ds_train.pt \
        --eval_dataset outputs/.../offpolicy_value_ds_eval.pt \
        --epochs 20 --lr 1e-3 --lr_decay 0.2 --lr_decay_step 10 \
        --out_ckpt outputs/.../iter-99_vh_offpolicy.pt
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn.functional as F

from am_baseline.training.coach import MCTSReplayBuffer, reconstruct_state
from am_baseline.utils.misc import torch_load_cpu
from scripts.probe_value_aleatoric import _load_stage4_model, _read_json_next_to
from scripts.train_value_trunk_supervised import _batch_from_slots

TRUNK_PREFIXES = ("value_trunk", "value_embedder", "value_init_embed")


def parse_args():
    p = argparse.ArgumentParser(description="Off-policy rollout distillation into the value head.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--buffer", required=True)
    p.add_argument("--dataset", required=True, help="train-split .pt from build_offpolicy_value_dataset")
    p.add_argument("--eval_dataset", default=None, help="optional eval-split .pt for held-out MSE")
    p.add_argument("--which", choices=["best", "working"], default="best")
    p.add_argument("--value_head_type", choices=["glimpse_mlp", "separate_trunk"], default="glimpse_mlp")
    p.add_argument("--value_own_encoder", action="store_true")
    p.add_argument("--reinit_head", action="store_true",
                   help="Re-initialize the value path instead of warm-starting from the checkpoint.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps_per_epoch", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_decay", type=float, default=0.2)
    p.add_argument("--lr_decay_step", type=int, default=10, help="StepLR step in epochs (0 = off).")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--eval_batches", type=int, default=20, help="held-out MSE minibatches per epoch")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out_ckpt", default=None)
    p.add_argument("--no_cuda", action="store_true")
    return p.parse_args()


class PairSampler:
    """Stratified minibatch sampler over (slot, action, label) pairs.

    One parent step per batch (uniform over nonempty steps) so `state.i` stays
    a scalar — same stratification as MCTSReplayBuffer.sample.
    """

    def __init__(self, ds, n_steps: int, rng: np.random.Generator):
        self.rng = rng
        step = ds["step"].numpy()
        self.by_step = {
            t: np.nonzero(step == t)[0] for t in range(n_steps) if (step == t).any()
        }
        self.steps = sorted(self.by_step.keys())
        self.slot = ds["slot"]
        self.action = ds["action"]
        self.label = ds["label"]

    def sample(self, bs: int):
        t = int(self.rng.choice(self.steps))
        cand = self.by_step[t]
        idx = cand[self.rng.integers(0, cand.shape[0], bs)] if cand.shape[0] < bs \
            else self.rng.choice(cand, size=bs, replace=False)
        idx_t = torch.as_tensor(idx, dtype=torch.long)
        return (
            t,
            self.slot[idx_t].numpy(),
            self.action[idx_t].to(torch.long),
            self.label[idx_t],
        )


def _child_values(model, buf, step, slots, actions, device, own_encoder: bool, head_type: str):
    """v(child) for pairs at one parent step. Grad flows only into the value path."""
    batch = _batch_from_slots(buf, step, slots)
    coords = batch["coords"].to(device=device, dtype=torch.float32)
    with torch.no_grad():
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
        child = parent.update(actions.to(device=device))

    if head_type == "separate_trunk":
        if own_encoder:
            return model.value_from_state(None, child, detach_encoder=True)
        with torch.no_grad():
            fixed = model.precompute_decoder(model.encode(coords))
        return model.value_from_state(fixed, child, detach_encoder=True)

    # glimpse_mlp: the glimpse comes from the frozen decoder — no grad needed
    # through it; only the value_head MLP trains.
    with torch.no_grad():
        fixed = model.precompute_decoder(model.encode(coords))
        _lp, _m, glimpse = model.decode_step(fixed, child, return_glimpse=True)
    return model.value_head(glimpse)


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    train_args = _read_json_next_to(args.ckpt)
    graph_size = int((train_args or {}).get("graph_size", 20))
    capacity = int((train_args or {}).get("buffer_capacity", 5000))
    vtn = (train_args or {}).get("value_target_norm", "none")

    model = _load_stage4_model(
        args.ckpt, args.which, train_args, graph_size, vtn, device,
        value_head_type=args.value_head_type, value_own_encoder=args.value_own_encoder,
    )
    trainable_prefixes = TRUNK_PREFIXES if args.value_head_type == "separate_trunk" else ("value_head",)
    if args.reinit_head:
        for name, mod in model.named_modules():
            if name.startswith(trainable_prefixes) and hasattr(mod, "reset_parameters"):
                mod.reset_parameters()

    # Freeze the policy; value path stays trainable. The glimpse head is a
    # BN-free MLP, so eval() everywhere is safe for it; a FRESH trunk must
    # stay in train mode for its BN running stats (H.7 convention).
    model.eval()
    if args.value_head_type == "separate_trunk":
        if model.value_embedder is not None:
            model.value_embedder.train()
        if model.value_trunk is not None:
            model.value_trunk.train()
    for name, prm in model.named_parameters():
        prm.requires_grad = name.startswith(trainable_prefixes)
    train_params = [prm for _n, prm in model.named_parameters() if prm.requires_grad]
    opt = torch.optim.Adam(train_params, lr=args.lr)
    sched = None
    if args.lr_decay != 1.0 and args.lr_decay_step > 0:
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_decay_step, gamma=args.lr_decay)

    buf = MCTSReplayBuffer(graph_size=graph_size, capacity_instances=capacity, device="cpu")
    buf.load(args.buffer)

    ds = torch.load(args.dataset, map_location="cpu", weights_only=False)
    sampler = PairSampler(ds, graph_size, rng)
    eval_sampler = None
    if args.eval_dataset:
        ds_eval = torch.load(args.eval_dataset, map_location="cpu", weights_only=False)
        eval_sampler = PairSampler(ds_eval, graph_size, np.random.default_rng(args.seed + 1))

    n_train = sum(p.numel() for p in train_params)
    print(f"device={device} head={args.value_head_type} own_enc={args.value_own_encoder} "
          f"reinit={args.reinit_head} train_params={n_train}")
    print(f"train pairs={ds['label'].numel()} (split={ds['meta']['split']}, holdout_k={ds['meta']['holdout_k']})"
          + (f"  eval pairs={ds_eval['label'].numel()}" if eval_sampler else ""))

    for ep in range(args.epochs):
        t0, losses = time.time(), []
        for _ in range(args.steps_per_epoch):
            step, slots, actions, labels = sampler.sample(args.batch_size)
            v = _child_values(model, buf, step, slots, actions, device,
                              args.value_own_encoder, args.value_head_type)
            loss = F.mse_loss(v.view(-1), labels.to(device=device, dtype=torch.float32).view(-1))
            opt.zero_grad()
            loss.backward()
            if args.max_grad_norm and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(train_params, args.max_grad_norm)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if sched is not None:
            sched.step()

        msg = (f"epoch {ep:2d}: mse={np.mean(losses):.6f} rmse={np.sqrt(np.mean(losses)):.4f} "
               f"lr={opt.param_groups[0]['lr']:.1e}")
        if eval_sampler is not None:
            with torch.no_grad():
                ev = []
                for _ in range(args.eval_batches):
                    step, slots, actions, labels = eval_sampler.sample(args.batch_size)
                    v = _child_values(model, buf, step, slots, actions, device,
                                      args.value_own_encoder, args.value_head_type)
                    ev.append(float(F.mse_loss(
                        v.view(-1), labels.to(device=device, dtype=torch.float32).view(-1)
                    ).cpu()))
            msg += f"  heldout_mse={np.mean(ev):.6f} heldout_rmse={np.sqrt(np.mean(ev)):.4f}"
        print(msg + f"  ({time.time() - t0:.1f}s)")

    tag = "_ownenc" if args.value_own_encoder else ""
    kind = "vtrunk_offpolicy" if args.value_head_type == "separate_trunk" else "vh_offpolicy"
    out = args.out_ckpt or os.path.join(
        os.path.dirname(os.path.abspath(args.ckpt)),
        os.path.splitext(os.path.basename(args.ckpt))[0] + f"_{kind}{tag}.pt",
    )
    ck = torch_load_cpu(args.ckpt)
    sd = model.state_dict()
    save = dict(ck)
    save["best_model"] = sd
    save["model"] = sd
    torch.save(save, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
