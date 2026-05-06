"""Diagnostic probe — compares gradient norms between Stage 1 (REINFORCE +
optional value head MSE) and Stage 4 (CE distillation + value head MSE)
training steps at F.6.0 model scale.

Motivation
----------
F.6.0 locks lr=1e-4 to match Stage 1 for sample-efficiency comparison. But
the two regimes use *qualitatively different* gradient computations:
  - Stage 1: ∇ log π · (L − b) — score function × centered advantage.
    High-variance, near-zero-mean per-parameter gradient signal.
  - Stage 4: ∇ CE(π_θ, π_target) — supervised soft-target classification.
    Lower-variance, larger-mean per-parameter gradient signal.

If Stage 4's per-step gradient is materially larger, lr=1e-4 may be
effectively too aggressive even though it's "fair" by the same-hyperparam
argument. This probe measures the gradient norms under both regimes on
matched architecture + matched random init, isolating the gradient-shape
difference from any other factor.

Usage
-----
    PYTHONPATH=src python -m scripts.probe_grad_norm

Output: per-regime grad-norm statistics (count, mean, p50, p95, max).
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from am_baseline.config import Config
from am_baseline.model.attention_model import AttentionModel
from am_baseline.training.coach import MCTSReplayBuffer
from am_baseline.training.trainer import train_step_alphazero


# ---------------------------------------------------------------------------
# Synthetic batch construction (mirrors smoke_alphazero._synth_instance_records)
# ---------------------------------------------------------------------------


def _synth_instance(N, rng):
    coords = rng.random((N, 2)).astype(np.float32)
    tour = rng.permutation(N).astype(np.int64)

    edge_costs = np.zeros(N, dtype=np.float32)
    for k in range(N):
        a = tour[k]
        b = tour[(k + 1) % N]
        edge_costs[k] = float(np.linalg.norm(coords[a] - coords[b]))
    tour_cost = float(edge_costs.sum())
    bl_val = tour_cost * float(rng.uniform(0.95, 1.10))

    per_step = []
    for t in range(N):
        visited_mask = np.zeros(N, dtype=bool)
        visited_mask[tour[:t]] = True
        if t < 2:
            lengths_t = 0.0
        else:
            lengths_t = float(edge_costs[: t - 1].sum())
        cost_to_go_t = tour_cost - lengths_t
        pi_t = np.zeros(N, dtype=np.float32)
        unvisited = np.where(~visited_mask)[0]
        if len(unvisited) == 1:
            pi_t[unvisited[0]] = 1.0
        else:
            # Use Dirichlet(α=1) — matches what MCTS would produce when visit
            # counts are spread fairly evenly across legal actions. This is
            # the *near-uniform* π_target regime that random-init MCTS
            # produces (which is the regime relevant to F.6.0's first iters).
            alpha = np.full(len(unvisited), 1.0)
            draw = rng.dirichlet(alpha).astype(np.float32)
            pi_t[unvisited] = draw
        per_step.append({
            "visited": visited_mask,
            "first": int(tour[0]) if t > 0 else -1,
            "prev": int(tour[t - 1]) if t > 0 else -1,
            "lengths": lengths_t,
            "pi": pi_t,
            "cost_to_go": cost_to_go_t,
        })
    return coords, float(bl_val), float(tour_cost), per_step


def _build_buffer(N, n_instances, seed):
    rng = np.random.default_rng(seed)
    buf = MCTSReplayBuffer(graph_size=N, capacity_instances=max(8, n_instances * 2))
    for _ in range(n_instances):
        coords, bl_val, tour_cost, per_step = _synth_instance(N, rng)
        buf.push_instance(coords, bl_val, tour_cost, per_step)
    return buf


# ---------------------------------------------------------------------------
# F.6.0-scale model construction
# ---------------------------------------------------------------------------


def _build_f60_model(N=20, seed=0):
    """Same architecture as F.6.0 / Stage 1 canonical.
    embedding_dim=128, n_encode_layers=3, n_heads=8, feed_forward_hidden=512,
    normalization=batch, value_enabled=True with hidden_dim=128.
    """
    torch.manual_seed(seed)
    cfg = Config(
        graph_size=N,
        embedding_dim=128,
        hidden_dim=128,
        n_encode_layers=3,
        n_heads=8,
        feed_forward_hidden=512,
        normalization='batch',
        value_enabled=True,
        value_hidden_dim=128,
    )
    return AttentionModel(cfg).cpu()


# ---------------------------------------------------------------------------
# Stage 1 train step (replicates trainer.train_batch's gradient computation)
# ---------------------------------------------------------------------------


def stage1_train_step(model, optimizer, x, opts):
    """One Stage 1 REINFORCE-with-rollout-baseline + value MSE step.

    Returns the unclipped grad-norm (matching trainer.py's logging convention).
    Uses the model's own greedy-rollout cost as a proxy for the rollout
    baseline's bl_val — this is *not* the Stage 1 production path (production
    uses RolloutBaseline.eval(...) which freezes θ★) but the *gradient shape*
    is the same: REINFORCE × centered advantage + value MSE.
    """
    model.train()
    model.set_decode_type("sampling")

    # Forward — sampled rollout for the policy gradient.
    cost, log_likelihood, pi, values = model(x, return_pi=True, compute_values=True)

    # Construct a baseline by running greedy decode on the same model with
    # gradients disabled. Centers the REINFORCE advantage so the gradient is
    # zero-mean in expectation (the canonical variance-reduction trick).
    model.set_decode_type("greedy")
    with torch.no_grad():
        bl_cost, _ = model(x)
    model.set_decode_type("sampling")

    bl_val = bl_cost.detach()
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()

    # Value loss — V_CURRENT cost-to-go normalized by bl_val (Stage 1 default).
    from am_baseline.utils.tensor_ops import value_targets_from_edges
    edge_costs = model.problem.get_edge_costs(x, pi)
    rtg = value_targets_from_edges(edge_costs).detach()
    Z = bl_val.clamp(min=1e-6).unsqueeze(-1)
    targets = rtg / Z
    value_loss = F.mse_loss(values, targets)

    loss = reinforce_loss + opts.lambda_v * value_loss

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=math.inf)
    return float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm)


# ---------------------------------------------------------------------------
# Probe runner
# ---------------------------------------------------------------------------


def _stats(values):
    arr = np.array(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def _print_stats(label, s):
    print(f"  {label}")
    print(f"    count={s['count']}  mean={s['mean']:.4f}")
    print(f"    p50={s['p50']:.4f}  p95={s['p95']:.4f}  p99={s['p99']:.4f}  max={s['max']:.4f}")


def main():
    N = 20
    BATCH = 64                  # stage 4 train batch size proxy (F.6.0 uses 512; 64 is faster on CPU)
    INSTANCES = 32              # stage 1 batch size proxy
    N_STEPS = 30                # gradient samples per regime
    LR = 1e-4

    print(f"=== probe_grad_norm.py ===")
    print(f"N={N}, BATCH={BATCH}, INSTANCES={INSTANCES}, N_STEPS={N_STEPS}, LR={LR}")
    print()

    # Common opts shim — both train steps read .lambda_v, .max_grad_norm, .device
    class _Opts:
        device = torch.device("cpu")
        lambda_v = 1.0
        max_grad_norm = 1.0
        graph_size = N
        value_target_norm = 'bl'

    opts = _Opts()

    # ---- Stage 4 (CE distillation + value MSE) -----------------------------
    print("[Stage 4] CE distillation + value MSE — F.6.0 regime")
    torch.manual_seed(0)
    np.random.seed(0)
    model4 = _build_f60_model(N=N, seed=0)
    optim4 = torch.optim.Adam(model4.parameters(), lr=LR, weight_decay=1e-4)
    buf = _build_buffer(N=N, n_instances=BATCH, seed=42)

    grad_norms_stage4 = []
    for step in range(N_STEPS):
        # Cycle through different decoder steps to mimic stratified sampling.
        t = step % (N - 1)
        batch = buf.sample_step(step=t, batch_size=BATCH)
        m = train_step_alphazero(model4, optim4, batch, opts)
        grad_norms_stage4.append(m["gradient_norm"])
        if step < 5 or step == N_STEPS - 1:
            print(f"    step={step:>3}  t={t:>2}  grad_norm={m['gradient_norm']:8.4f}  "
                  f"policy_loss={m['policy_loss']:7.4f}  value_loss={m['value_loss']:.4f}")

    s4 = _stats(grad_norms_stage4)
    _print_stats("Stage 4 grad_norm distribution:", s4)

    print()

    # ---- Stage 1 (REINFORCE + value MSE) -----------------------------------
    print("[Stage 1] REINFORCE-with-baseline + value MSE")
    torch.manual_seed(0)
    np.random.seed(0)
    model1 = _build_f60_model(N=N, seed=0)
    optim1 = torch.optim.Adam(model1.parameters(), lr=LR, weight_decay=1e-4)

    grad_norms_stage1 = []
    rng_g = torch.Generator(device="cpu").manual_seed(123)
    for step in range(N_STEPS):
        x = torch.rand((INSTANCES, N, 2), generator=rng_g, dtype=torch.float32)
        gn = stage1_train_step(model1, optim1, x, opts)
        grad_norms_stage1.append(gn)
        if step < 5 or step == N_STEPS - 1:
            print(f"    step={step:>3}  grad_norm={gn:8.4f}")

    s1 = _stats(grad_norms_stage1)
    _print_stats("Stage 1 grad_norm distribution:", s1)

    print()
    print("=== Comparison ===")
    print(f"Stage 4 / Stage 1 ratio (median):  {s4['p50'] / max(1e-6, s1['p50']):.2f}x")
    print(f"Stage 4 / Stage 1 ratio (p95):     {s4['p95'] / max(1e-6, s1['p95']):.2f}x")
    print(f"Stage 4 / Stage 1 ratio (mean):    {s4['mean'] / max(1e-6, s1['mean']):.2f}x")


if __name__ == "__main__":
    main()
