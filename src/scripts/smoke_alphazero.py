"""Stage 4 Phase B smoke harness — replay buffer + distillation training step.

Implements case **A1** from `_plans/stage4_plan.md` line 222:
  - Construct a 5-instance buffer with hand-crafted records (random pi_t,
    random z_t).
  - Run one `train_step_alphazero` call.
  - Verify: loss is finite, no NaN, gradients flow into encoder + decoder +
    value_head (non-zero parameter delta after `optimizer.step()`).

Also runs the buffer-only invariants from A1.5 / A1.6:
  - sample_step(t) returns rows all reporting state_i == t.
  - sample(B) returns a batch sharing one state_i.
  - pi has zero mass on visited cities, sums to 1, is non-negative.
  - save / load round-trip rebuilds `_step_index` to the same set per step.

The records here are SYNTHESIZED — Phase A is exposing real visit dists in
parallel and Phase C will produce the records for real. Replace this stub
with `generate_self_play_batch(...)` output as soon as Phase A.4 + C.2 land.

Run:
    PYTHONPATH=src python -m scripts.smoke_alphazero
or:
    python src/scripts/smoke_alphazero.py
"""
from __future__ import annotations

import copy
import math
import os
import sys
import tempfile

import numpy as np
import torch

# Make the script runnable as both a module (-m scripts.smoke_alphazero) and
# as a plain file (`python src/scripts/smoke_alphazero.py`).
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from am_baseline.config import Config
from am_baseline.model.attention_model import AttentionModel
from am_baseline.training.coach import (
    MCTSReplayBuffer,
    reconstruct_state,
    make_self_play_config,
    generate_self_play_batch,
)
from am_baseline.training.trainer import train_step_alphazero


# ---------------------------------------------------------------------------
# Synthesis helpers — STUB. Replace once Phase A.4 + C.2 land.
# ---------------------------------------------------------------------------


def _synth_instance_records(N: int, rng: np.random.Generator):
    """Hand-craft a plausible (coords, bl_val, tour_cost, per_step) record.

    The played 'tour' is a random permutation of [0, N). At each step t:
      - visited = first t cities in the tour (mask bool length N).
      - first / prev derive from the tour prefix.
      - lengths_t = cumulative euclidean cost of edges traversed so far.
      - pi_t is a Dirichlet draw renormalized over UNVISITED cities (zero on
        visited cities) — mimics the τ=1 normalized visit dist.
      - cost_to_go = tour_cost - lengths_t (matches V_CURRENT).
    """
    coords = rng.random((N, 2)).astype(np.float32)
    tour = rng.permutation(N).astype(np.int64)

    edge_costs = np.zeros(N, dtype=np.float32)
    for k in range(N):
        a = tour[k]
        b = tour[(k + 1) % N]
        edge_costs[k] = float(np.linalg.norm(coords[a] - coords[b]))
    tour_cost = float(edge_costs.sum())

    # Approximate `bl_val` as the same tour cost scaled — Phase C will use
    # cost(greedy_rollout(theta_star, x)). Magnitude is what matters for the
    # smoke test (z_t should land in roughly [0, 1.5] range).
    bl_val = tour_cost * float(rng.uniform(0.95, 1.10))

    per_step = []
    for t in range(N):
        visited_mask = np.zeros(N, dtype=bool)
        visited_mask[tour[:t]] = True

        # State.lengths at step t = cumulative cost of edges traversed BEFORE
        # arriving at s_t. AM's `state.update` adds the edge cost at step t
        # only if step > 0, so lengths_t = sum of edge_costs[0..max(0,t-1)-1].
        # In our edge_costs indexing, edge_costs[k] is the (k -> k+1) hop, so
        # lengths_t = sum(edge_costs[:max(0, t-1)]) for t == 0 -> 0,
        # t == 1 -> 0, t >= 2 -> sum(edge_costs[:t-1]).
        if t < 2:
            lengths_t = 0.0
        else:
            lengths_t = float(edge_costs[: t - 1].sum())

        # cost_to_go via the same convention as `value_targets_from_edges`:
        # for t in {0, 1} the entire tour is still ahead; for t >= 2,
        # cost_to_go_t = sum(edge_costs[t-1:]).
        cost_to_go_t = tour_cost - lengths_t

        # pi_t — Dirichlet over unvisited, zero on visited. Sums to 1 unless
        # there is exactly one unvisited (forced step t == N-1) in which case
        # it is one-hot.
        pi_t = np.zeros(N, dtype=np.float32)
        unvisited = np.where(~visited_mask)[0]
        if len(unvisited) == 1:
            pi_t[unvisited[0]] = 1.0
        else:
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _build_buffer(N: int, n_instances: int, seed: int = 1234):
    """Push `n_instances` synthesized records into a fresh small buffer."""
    rng = np.random.default_rng(seed)
    buf = MCTSReplayBuffer(graph_size=N, capacity_instances=max(8, n_instances * 2))
    for _ in range(n_instances):
        coords, bl_val, tour_cost, per_step = _synth_instance_records(N, rng)
        buf.push_instance(coords, bl_val, tour_cost, per_step)
    return buf


def test_buffer_invariants(N: int = 8) -> None:
    print(f"  [B0] buffer invariants on N={N}, 5 instances ...")
    buf = _build_buffer(N, n_instances=5, seed=42)

    # step_counts: every step should have exactly 5 records (5 instances).
    counts = buf.step_counts()
    assert counts == [5] * N, f"step_counts mismatch: {counts}"

    # sample_step(t) -> all rows share state_i == t and obey pi invariants.
    for t in range(N):
        b = buf.sample_step(t, batch_size=4)
        assert b["state_i"] == t, f"sample_step({t}).state_i = {b['state_i']}"
        # pi sums to 1 and is non-negative.
        sums = b["pi"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
            f"pi rows do not sum to 1 at step {t}: {sums}")
        assert (b["pi"] >= 0).all(), f"pi has negative entries at step {t}"
        # Visited mass is exactly zero.
        masked_mass = (b["pi"] * b["visited"].float()).sum(dim=-1)
        assert torch.all(masked_mass == 0), (
            f"pi has nonzero mass on visited cities at step {t}: {masked_mass}")

    # sample() — random-step path, all rows must share state_i.
    b = buf.sample(batch_size=4)
    assert isinstance(b["state_i"], int)
    print("  [B0] OK")


def test_save_load_roundtrip(N: int = 6) -> None:
    print(f"  [B1] save/load + _step_index rebuild on N={N}, 5 instances ...")
    buf = _build_buffer(N, n_instances=5, seed=7)
    snap_sets = [set(int(x) for x in arr) for arr in buf._step_index]

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "buf.pt")
        buf.save(path)
        buf2 = MCTSReplayBuffer(
            graph_size=N, capacity_instances=buf.capacity_instances
        )
        buf2.load(path)

    # Set equality per step (rebuild order may differ).
    for t in range(N):
        reloaded = set(int(x) for x in buf2._step_index[t])
        assert reloaded == snap_sets[t], (
            f"_step_index mismatch at step {t}: "
            f"original={snap_sets[t]} reloaded={reloaded}")

    # Sample on reloaded buffer; same invariants must hold.
    for t in range(N):
        b = buf2.sample_step(t, batch_size=3)
        assert b["state_i"] == t
        assert torch.allclose(b["pi"].sum(-1), torch.ones(b["pi"].shape[0]), atol=1e-5)
        assert (b["pi"] >= 0).all()
        masked = (b["pi"] * b["visited"].float()).sum(-1)
        assert torch.all(masked == 0)
    print("  [B1] OK")


def test_train_step_alphazero(N: int = 8) -> None:
    """Smoke A1 — the load-bearing test.

    Build a 5-instance buffer, run one train_step_alphazero, verify finite
    loss, no NaN, and that *every* parameter group (encoder, decoder,
    value_head) saw a non-zero update.
    """
    print(f"  [A1] train_step_alphazero on N={N}, 5 instances ...")
    torch.manual_seed(0)

    buf = _build_buffer(N, n_instances=5, seed=11)

    # AttentionModel needs a Config-shaped opts. Use small dims for CPU smoke.
    cfg = Config(
        graph_size=N,
        embedding_dim=32,
        n_encode_layers=2,
        n_heads=4,
        value_enabled=True,
        value_hidden_dim=32,
    )
    model = AttentionModel(cfg).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Snapshot parameters before the step (per-named-parameter copies).
    pre = {name: p.detach().clone() for name, p in model.named_parameters()}

    # Choose a non-final step so policy-loss has gradient signal (forced
    # last step has near-one-hot pi vs the decoder's already near-one-hot
    # log_p, so gradient is small; we want an unambiguous A1 signal).
    batch = buf.sample_step(step=2, batch_size=5)

    class _Opts:
        device = torch.device("cpu")
        lambda_v = 1.0
        max_grad_norm = 1.0

    metrics = train_step_alphazero(model, optimizer, batch, _Opts())

    # Finiteness checks.
    for k, v in metrics.items():
        assert math.isfinite(v), f"metrics[{k}] is not finite: {v}"
    print(f"     metrics = {metrics}")

    # Gradient flow checks — every parameter must have a (non-None) grad,
    # and we expect at least the encoder, decoder, and value_head to have
    # actually moved. We bucket by submodule prefix.
    moved = {"encoder": False, "decoder": False, "value_head": False, "init_embed": False}
    for name, p in model.named_parameters():
        post = p.detach()
        delta = (post - pre[name]).abs().max().item()
        if name.startswith("embedder"):
            bucket = "encoder"
        elif name.startswith("decoder"):
            bucket = "decoder"
        elif name.startswith("value_head"):
            bucket = "value_head"
        elif name.startswith("init_embed"):
            bucket = "init_embed"
        else:
            bucket = None
        if bucket is not None and delta > 0:
            moved[bucket] = True

    for bucket, did_move in moved.items():
        assert did_move, (
            f"no parameters moved in submodule '{bucket}' — gradient flow broken")

    # Policy-loss + value-loss both contributed.
    assert metrics["policy_loss"] > 0, "policy_loss should be strictly positive (CE)"
    assert metrics["value_loss"] > 0, "value_loss should be strictly positive (MSE)"
    assert metrics["total_loss"] >= metrics["policy_loss"] - 1e-6
    print("  [A1] OK")


def test_state_reconstruction_roundtrip(N: int = 6) -> None:
    """Sanity: reconstruct_state produces a StateTSP that decode_step accepts.

    Not part of the official A1 case, but cheap to check and catches a whole
    class of dtype/device/shape regressions in `reconstruct_state`.
    """
    print(f"  [A1b] reconstruct_state -> decode_step round-trip on N={N} ...")
    buf = _build_buffer(N, n_instances=3, seed=99)
    cfg = Config(
        graph_size=N, embedding_dim=32, n_encode_layers=2, n_heads=4,
        value_enabled=True, value_hidden_dim=32,
    )
    model = AttentionModel(cfg).cpu().eval()

    for t in [0, 1, N // 2, N - 1]:
        batch = buf.sample_step(step=t, batch_size=3)
        coords = batch["coords"]
        encoded = model.encode(coords)
        fixed = model.precompute_decoder(encoded)
        state = reconstruct_state(batch, device=torch.device("cpu"))
        log_p, mask, glimpse = model.decode_step(fixed, state, return_glimpse=True)
        assert log_p.shape == (3, 1, N)
        assert mask.shape == (3, 1, N)
        assert glimpse.shape == (3, cfg.embedding_dim)
        # Visited cities -> mask True -> log_p == -inf.
        masked_logp = log_p.squeeze(1)[mask.squeeze(1)]
        assert torch.all(torch.isinf(masked_logp) & (masked_logp < 0)), (
            f"mask alignment broke at step {t} — visited cities should have log_p == -inf")
    print("  [A1b] OK")


def test_self_play_generator(N: int = 20, M: int = 10, K: int = 20) -> None:
    """Smoke A2 — self-play generator end-to-end on TSP-N.

    Drives `generate_self_play_batch` with `temperature=0` so MCTS picks the
    argmax-N action at every root, then verifies for every per-step π_t:
        (i)   sums to 1 within float tolerance,
        (ii)  has zero mass on visited cities,
        (iii) argmax(π_t) equals the action MCTS chose at step t (= tour[t]
              when temperature=0 because root_select='visits' picks argmax N
              and π_t = N / Σ N).
    Also verifies that `value_norm='bl' + leaf_eval='value_head'` is accepted
    by `MCTSConfig._validate_config` (Stage 3 E.2 explicitly allows this combo).
    """
    print(f"  [A2] generate_self_play_batch on N={N}, M={M}, K={K} ...")
    torch.manual_seed(0)
    np.random.seed(0)

    cfg_model = Config(
        graph_size=N,
        embedding_dim=32,
        n_encode_layers=2,
        n_heads=4,
        value_enabled=True,
        value_hidden_dim=32,
    )
    model = AttentionModel(cfg_model).cpu().eval()

    # A2 explicitly checks the bl + value_head combo is accepted (no raise).
    cfg = make_self_play_config(graph_size=N, n_simulations=K)
    assert cfg.leaf_eval == 'value_head', "A2 expects AGZ-canonical value_head leaf"
    assert cfg.value_norm == 'bl', "A2 expects bl normalization"
    # Greedy at the root so π argmax aligns with the chosen tour action.
    cfg.temperature = 0.0
    # Disable Dirichlet root noise so greedy argmax is deterministic across
    # runs and π_t reflects pure MCTS visits.
    cfg.dirichlet_epsilon = 0.0

    records = generate_self_play_batch(
        best_model=model,
        M=M,
        graph_size=N,
        cfg=cfg,
        device=torch.device("cpu"),
        mcts_batch_size=8,
    )
    assert len(records) == M, f"expected {M} records, got {len(records)}"

    # Push into a buffer to confirm the records match `push_instance`'s schema.
    buf = MCTSReplayBuffer(graph_size=N, capacity_instances=M)
    for rec in records:
        assert rec.coords.shape == (N, 2)
        assert math.isfinite(rec.bl_val) and rec.bl_val > 0
        assert math.isfinite(rec.tour_cost) and rec.tour_cost > 0
        assert len(rec.per_step) == N
        buf.push_instance(rec.coords, rec.bl_val, rec.tour_cost, rec.per_step)
    assert buf.step_counts() == [M] * N, f"step_counts mismatch: {buf.step_counts()}"

    # Per-step π invariants — across all instances and all tour-steps.
    for inst_i, rec in enumerate(records):
        # Reconstruct chosen tour from per_step['prev'] / 'first' for cross-check.
        # `tour[t]` = the action MCTS chose at step t. We can recover it from
        # the difference between visited masks at consecutive steps:
        #   tour[t] = the unique city where visited[t+1] is True but visited[t] is False.
        for t in range(N):
            ps = rec.per_step[t]
            pi_t = ps['pi']
            visited = ps['visited']

            # (i) sums to 1.
            s = float(pi_t.sum())
            assert abs(s - 1.0) < 1e-5, (
                f"inst={inst_i} step={t}: pi sums to {s}, not 1")
            assert (pi_t >= 0).all(), (
                f"inst={inst_i} step={t}: pi has negative entries")

            # (ii) zero mass on visited.
            masked_mass = float((pi_t * visited.astype(np.float32)).sum())
            assert masked_mass == 0.0, (
                f"inst={inst_i} step={t}: pi has {masked_mass} mass on visited cities")

            # (iii) argmax(pi_t) == the action chosen at step t.
            #       At step t, the chosen action is the city that is unvisited
            #       at step t but visited at step t+1 (or, for t == N-1, it's
            #       the unique remaining unvisited city).
            if t < N - 1:
                next_visited = rec.per_step[t + 1]['visited']
                chosen_arr = np.where(next_visited & ~visited)[0]
                assert chosen_arr.size == 1, (
                    f"inst={inst_i} step={t}: cannot recover chosen action "
                    f"from visited deltas (got {chosen_arr})")
                chosen = int(chosen_arr[0])
            else:
                # Last step: only one unvisited city left.
                rem = np.where(~visited)[0]
                assert rem.size == 1, (
                    f"inst={inst_i} step={t} (last): expected exactly 1 "
                    f"unvisited, got {rem}")
                chosen = int(rem[0])

            argmax_a = int(np.argmax(pi_t))
            assert argmax_a == chosen, (
                f"inst={inst_i} step={t}: argmax(pi)={argmax_a} but MCTS "
                f"chose action {chosen} (temperature=0, so they should match)")

    # Sanity on cost_to_go: for t == 0, cost_to_go should equal tour_cost
    # within float tolerance (V_CURRENT at s_0 is the entire tour).
    for inst_i, rec in enumerate(records):
        ctg0 = rec.per_step[0]['cost_to_go']
        assert abs(ctg0 - rec.tour_cost) < 1e-3, (
            f"inst={inst_i}: cost_to_go[0]={ctg0} but tour_cost={rec.tour_cost}")

    print(f"     M={M} records produced; all π_t invariants hold")
    print("  [A2] OK")


def main() -> int:
    print("Stage 4 Phase B + C smoke harness")
    print("=" * 64)
    try:
        test_buffer_invariants(N=8)
        test_save_load_roundtrip(N=6)
        test_state_reconstruction_roundtrip(N=6)
        test_train_step_alphazero(N=8)
        test_self_play_generator(N=20, M=10, K=20)
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print("=" * 64)
    print("PASS — all Phase B + C smoke checks succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
