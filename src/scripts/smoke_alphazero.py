"""Stage 4 Phase B/C/D/F smoke harness — replay buffer, distillation, coach loop.

Cases (cumulative across phases):
  - **B0**: buffer invariants + sample_step(t) + sample(B) shape rules.
  - **B1**: save/load round-trip rebuilds `_step_index`.
  - **A1b**: `reconstruct_state` -> `decode_step` round-trip.
  - **A1**: one `train_step_alphazero` call with finite loss + grad flow.
  - **A2**: `generate_self_play_batch` end-to-end with τ=0 invariants.
  - **A3** (Phase F): π_t target-distribution entropy invariant under
            `temperature_schedule='step30'`. Validates spec §4.2 choice (B):
            σ_t may collapse to one-hot late-game, but the *training target*
            π_t is always raw τ=1 normalized visits. We verify per-step
            entropy(π_t) > 0 across the run except at the forced last step
            (N-1) where exactly one legal action remains.
  - **A5**: gating no-op when `gate_every > n_iterations` (Phase D).
  - **A6**: 3-iteration coach loop end-to-end with checkpoint round-trip
            (Phase D).

Notes on cases that were considered but deliberately skipped:
  - A4 (legality/support/finiteness on `pi_t`): redundant with Phase A's A13
    (which checks the raw `solver.root_visit_dists`) and the existing A2
    here (which checks the same invariants on post-`generate_self_play_batch`
    `pi_t`). Adding a separate A4 would duplicate A2's checks verbatim under
    the same code path.

Run:
    PYTHONPATH=src python -m scripts.smoke_alphazero
or:
    python src/scripts/smoke_alphazero.py
"""
from __future__ import annotations

import copy
import contextlib
import math
import os
import shutil
import sys
import uuid

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
from am_baseline.problem.tsp import TSP
from am_baseline.training.coach import (
    MCTSCoach,
    MCTSReplayBuffer,
    reconstruct_state,
    make_self_play_config,
    generate_self_play_batch,
)
from am_baseline.training.trainer import train_step_alphazero


@contextlib.contextmanager
def _workspace_tempdir():
    """Temporary directory under the repo scratch area.

    Windows sandbox runs can create ACL-restricted dirs via `tempfile` that
    PyTorch's C++ zip writer cannot open. A normal workspace dir avoids that
    without changing the smoke semantics.
    """
    root = os.path.abspath(os.path.join(os.getcwd(), ".tmp", "smoke_alphazero"))
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, uuid.uuid4().hex)
    os.makedirs(path, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


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

    with _workspace_tempdir() as tmp:
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


def test_train_step_alphazero_lambda_zero(N: int = 8) -> None:
    """Smoke A1c - lambda_v=0 is a true policy-only update.

    The value head still reports a finite MSE metric, but value gradients must
    be exactly absent from the combined update. This is the rollout-teacher
    ablation contract.
    """
    print(f"  [A1c] train_step_alphazero lambda_v=0 on N={N} ...")
    torch.manual_seed(0)

    buf = _build_buffer(N, n_instances=5, seed=17)
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
    pre = {name: p.detach().clone() for name, p in model.named_parameters()}
    batch = buf.sample_step(step=2, batch_size=5)

    class _Opts:
        device = torch.device("cpu")
        lambda_v = 0.0
        max_grad_norm = 1.0

    metrics = train_step_alphazero(model, optimizer, batch, _Opts())

    for k, v in metrics.items():
        assert math.isfinite(v), f"metrics[{k}] is not finite: {v}"
    assert abs(metrics["total_loss"] - metrics["policy_loss"]) < 1e-6, (
        f"lambda_v=0 should make total_loss == policy_loss, got {metrics}"
    )
    assert metrics["value_grad_norm"] == 0.0, metrics
    assert metrics["value_grad_norm_vh"] == 0.0, metrics
    assert metrics["value_grad_norm_shared"] == 0.0, metrics
    assert "mean_entropy_policy" in metrics, "policy entropy metric missing"

    moved = {"shared": False, "value_head": False}
    for name, p in model.named_parameters():
        delta = (p.detach() - pre[name]).abs().max().item()
        if name.startswith("value_head"):
            moved["value_head"] = moved["value_head"] or delta > 0
        else:
            moved["shared"] = moved["shared"] or delta > 0
    assert moved["shared"], "policy-only update did not move shared policy parameters"
    assert not moved["value_head"], "lambda_v=0 unexpectedly moved value_head parameters"

    print("  [A1c] OK")


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


def test_target_entropy_under_schedule(N: int = 20, M: int = 10, K: int = 20) -> None:
    """Smoke A3 — π_t entropy invariant under `temperature_schedule='step30'`.

    Spec §4.2 choice (B): action selection σ_t honors the per-step temperature
    schedule (collapses to one-hot at step ⌈0.3·N⌉ for `step30`); the *training
    target* π_t is always the raw τ=1 normalized visit distribution. So
    entropy(π_t) at late steps must NOT collapse to zero just because σ_t did.

    This test runs `generate_self_play_batch` with `temperature_schedule='step30'`
    on M=10 TSP-N instances and verifies:
      (a) π_t sums to 1 and is non-negative (re-checks A2 invariants under
          the schedule code path).
      (b) entropy(π_t) > 0 for every step t ∈ [0, N-1) — the only legitimate
          collapse is the forced last step (t=N-1) where exactly one legal
          action remains.
      (c) entropy(π_t) at the final step (t=N-1) is exactly 0 (one-hot,
          dictated by legality, not by temperature).
    """
    print(f"  [A3] π_t entropy under temperature_schedule='step30' on N={N}, M={M}, K={K} ...")
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

    cfg = make_self_play_config(graph_size=N, n_simulations=K)
    # `make_self_play_config` already sets temperature_schedule='step30';
    # assert that to lock the test against a future default change.
    assert cfg.temperature_schedule == 'step30', (
        f"A3 expects make_self_play_config to default to 'step30', got "
        f"{cfg.temperature_schedule!r}"
    )
    # Disable Dirichlet root noise for a deterministic check on the
    # raw-visit-derived target distribution.
    cfg.dirichlet_epsilon = 0.0

    records = generate_self_play_batch(
        best_model=model,
        M=M,
        graph_size=N,
        cfg=cfg,
        device=torch.device("cpu"),
        mcts_batch_size=8,
    )
    assert len(records) == M

    cutoff = (N + 9) // 10 * 3 // 10  # not used; inline math kept for clarity below
    # (We don't need the cutoff index here — A14 in smoke_mcts.py checks σ_t
    # behavior at the cutoff. A3 only checks π_t.)

    eps_floor = 1e-6
    for inst_i, rec in enumerate(records):
        for t in range(N):
            ps = rec.per_step[t]
            pi_t = ps['pi'].astype(np.float64)
            visited = ps['visited']

            # (a) sum / non-negativity (A2 invariants, re-checked under schedule).
            s = float(pi_t.sum())
            assert abs(s - 1.0) < 1e-5, (
                f"A3 inst={inst_i} step={t}: pi sums to {s}, not 1"
            )
            assert (pi_t >= 0).all(), (
                f"A3 inst={inst_i} step={t}: pi has negative entries"
            )

            # Entropy with 0·log(0) := 0 (numpy broadcasts inf*0 = nan; mask).
            mask_pos = pi_t > eps_floor
            entropy = -float((pi_t[mask_pos] * np.log(pi_t[mask_pos])).sum())

            n_unvisited = int((~visited).sum())
            if t == N - 1:
                # Forced step — exactly one unvisited city — π_t is one-hot
                # and entropy MUST be 0 (legality, not temperature).
                assert n_unvisited == 1, (
                    f"A3 inst={inst_i} step=N-1: expected 1 unvisited, got {n_unvisited}"
                )
                assert entropy < 1e-6, (
                    f"A3 inst={inst_i} step=N-1: forced step but entropy={entropy} ≠ 0"
                )
            else:
                # σ_t may have collapsed to one-hot at step ≥ ⌈0.3·N⌉, but π_t
                # uses raw τ=1 visits and must stay above 0 because there are
                # multiple unvisited cities and MCTS explored more than one of
                # them under c_puct + Dirichlet (ε=0 here, but PUCT still
                # spreads visits across unvisited).
                if n_unvisited > 1:
                    assert entropy > 0.0, (
                        f"A3 inst={inst_i} step={t}: π_t entropy collapsed to 0 with "
                        f"{n_unvisited} unvisited cities — schedule must NOT couple to π_t"
                    )

    print(f"     M={M} instances; π_t entropy stays > 0 except at forced step N-1")
    print("  [A3] OK")


def _make_tiny_coach_opts(N: int, M: int, K: int, train_steps: int,
                          gate_every: int, save_dir: str,
                          val_size: int = 32, batch_size: int = 16):
    """Build an opts namespace adequate for a tiny CPU smoke MCTSCoach run."""
    import argparse
    opts = argparse.Namespace()
    # Problem
    opts.graph_size = N
    # Training
    opts.lr_model = 1e-3
    opts.weight_decay = 1e-4
    opts.batch_size = batch_size
    opts.max_grad_norm = 1.0
    opts.lambda_v = 1.0
    opts.value_target_norm = 'bl'
    # Coach
    opts.M_instances = M
    opts.n_simulations_train = K
    opts.train_steps_per_iter = train_steps
    opts.gate_every = gate_every
    opts.buffer_capacity = max(64, M * 4)
    opts.mcts_batch_size = 8
    # Validation / gating
    opts.val_size = val_size
    opts.eval_batch_size = max(1, val_size)
    opts.no_progress_bar = True
    opts.bl_alpha = 0.05
    # Device
    opts.use_cuda = False
    opts.device = torch.device('cpu')
    # Logging
    opts.save_dir = save_dir
    opts.output_dir = save_dir
    opts.run_name = 'smoke_d'
    opts.no_wandb = True
    opts.wandb_project = None
    opts.wandb_entity = None
    opts.wandb_mode = 'disabled'
    return opts


def _build_smoke_model(N: int) -> "AttentionModel":
    cfg = Config(
        graph_size=N,
        embedding_dim=16,
        n_encode_layers=1,
        n_heads=2,
        value_enabled=True,
        value_hidden_dim=16,
    )
    return AttentionModel(cfg).cpu()


def test_coach_gate_noop(N: int = 8) -> None:
    """A5 — gating no-op when `gate_every > n_iterations`.

    Constructs a tiny `MCTSCoach`, runs `learn(3)` with `gate_every=10`, and
    asserts that `gating_baseline.epoch_callback` was never called. This
    directly exercises the iter-wise scheduling guard in `MCTSCoach.learn`.
    """
    print(f"  [A5] gating no-op (gate_every > n_iterations) ...")
    torch.manual_seed(0)
    np.random.seed(0)

    M, K, train_steps = 10, 20, 5
    n_iterations = 3
    gate_every = 10  # > n_iterations -> never fires

    with _workspace_tempdir() as tmp:
        opts = _make_tiny_coach_opts(
            N=N, M=M, K=K, train_steps=train_steps,
            gate_every=gate_every, save_dir=tmp,
        )
        model = _build_smoke_model(N)
        problem = TSP
        # Tiny val dataset for `validate(...)` (separate from gating's frozen
        # one which lives inside RolloutBaseline).
        val_dataset = TSP.make_dataset(size=N, num_samples=opts.val_size)

        coach = MCTSCoach(
            model=model, problem=problem, opts=opts,
            val_dataset=val_dataset, device=torch.device('cpu'),
        )

        # Patch epoch_callback with a counting stub. We do NOT delegate to
        # the real callback because A5 only asserts the call count == 0.
        call_count = {'n': 0}
        original_cb = coach.gating_baseline.epoch_callback
        def _counting_cb(model_arg, epoch=0):
            call_count['n'] += 1
            return original_cb(model_arg, epoch)
        coach.gating_baseline.epoch_callback = _counting_cb

        try:
            coach.learn(n_iterations=n_iterations)
        finally:
            coach.close()

        assert call_count['n'] == 0, (
            f"A5: gating_baseline.epoch_callback was called {call_count['n']} "
            f"times with gate_every=10 and n_iterations=3 (expected 0)"
        )

        # Verify the iterations.csv was written with 3 rows + header.
        iter_csv = os.path.join(opts.save_dir, 'iterations.csv')
        assert os.path.exists(iter_csv), f"iterations.csv not written at {iter_csv}"
        with open(iter_csv) as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        assert len(lines) == 1 + n_iterations, (
            f"A5: expected {1 + n_iterations} CSV lines, got {len(lines)}"
        )
        # Header has 'gated' and 'accepted' columns; data rows must show
        # gated=0 and accepted='' (the latter because no gating ran).
        header = lines[0].split(',')
        gi = header.index('gated')
        ai = header.index('accepted')
        for row in lines[1:]:
            cells = row.split(',')
            assert cells[gi] == '0', f"A5: expected gated=0, got {cells[gi]}"
            assert cells[ai] == '', (
                f"A5: expected accepted='' (un-gated), got '{cells[ai]}'"
            )

    print("  [A5] OK")


def test_coach_three_iters(N: int = 8) -> None:
    """A6 — 3 iterations end-to-end + checkpoint round-trip.

    Exercises MCTSCoach.learn with M=10, K=20, gate_every=2 over 3 iterations.
    Verifies:
      (i)   no NaN in any iteration row of `iterations.csv`,
      (ii)  val_avg_cost is finite,
      (iii) at least one gating decision (iter 1 — `(iter+1)%2 == 0`) ran,
      (iv)  load_checkpoint round-trip restores model + best_model + iter_idx
            to a bit-identical state (parameter equality, deterministic on
            CPU within float tolerance).
    """
    print(f"  [A6] 3-iteration coach loop on N={N}, M=10, K=20 ...")
    torch.manual_seed(0)
    np.random.seed(0)

    M, K, train_steps = 10, 20, 5
    n_iterations = 3
    gate_every = 2  # gates at iter_idx in {1} -> (1+1)%2 == 0 -> fires once

    with _workspace_tempdir() as tmp:
        opts = _make_tiny_coach_opts(
            N=N, M=M, K=K, train_steps=train_steps,
            gate_every=gate_every, save_dir=tmp,
        )
        model = _build_smoke_model(N)
        problem = TSP
        val_dataset = TSP.make_dataset(size=N, num_samples=opts.val_size)

        coach = MCTSCoach(
            model=model, problem=problem, opts=opts,
            val_dataset=val_dataset, device=torch.device('cpu'),
        )
        try:
            coach.learn(n_iterations=n_iterations)
        finally:
            coach.close()

        # (i) + (ii) — no NaN in iterations.csv, val cost finite.
        iter_csv = os.path.join(opts.save_dir, 'iterations.csv')
        with open(iter_csv) as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        header = lines[0].split(',')
        v_idx = header.index('val_avg_cost')
        gi = header.index('gated')
        ai = header.index('accepted')
        gated_seen = 0
        for row in lines[1:]:
            cells = row.split(',')
            for c in cells:
                # Reject literal NaN spellings only; '' is allowed (e.g.
                # accepted column on un-gated iters).
                assert c.lower() not in ('nan', '+nan', '-nan'), (
                    f"A6: NaN cell in iterations.csv row: {row}"
                )
            v = float(cells[v_idx])
            assert math.isfinite(v), f"A6: val_avg_cost not finite: {v}"
            if cells[gi] == '1':
                gated_seen += 1
                # accepted column must be 0 or 1 for gated rows.
                assert cells[ai] in ('0', '1'), (
                    f"A6: gated row with non-bool accepted: {cells[ai]}"
                )

        # (iii) — at least one gating decision was logged.
        assert gated_seen >= 1, (
            f"A6: expected ≥1 gated iteration with gate_every=2, n=3; got {gated_seen}"
        )

        # (iv) — load_checkpoint round-trip on iter-2.pt (the last iter).
        last_ckpt = os.path.join(opts.save_dir, 'iter-2.pt')
        assert os.path.exists(last_ckpt), f"A6: missing checkpoint {last_ckpt}"

        # Snapshot working-model params (we'll compare against the restored).
        pre_state = {k: v.detach().clone() for k, v in coach.model.state_dict().items()}
        pre_best_state = {k: v.detach().clone() for k, v in coach.best_model.state_dict().items()}
        pre_iter = coach.iter_idx
        pre_total = coach.total_instances_seen

        # Build a fresh coach (different RNG-driven init) and load.
        torch.manual_seed(99)
        fresh_model = _build_smoke_model(N)
        coach2 = MCTSCoach(
            model=fresh_model, problem=problem, opts=opts,
            val_dataset=val_dataset, device=torch.device('cpu'),
        )
        try:
            coach2.load_checkpoint(last_ckpt)

            # Parameter equality
            post_state = coach2.model.state_dict()
            post_best_state = coach2.best_model.state_dict()
            for k in pre_state:
                assert torch.equal(pre_state[k], post_state[k]), (
                    f"A6: model state_dict mismatch after restore at key '{k}'"
                )
            for k in pre_best_state:
                assert torch.equal(pre_best_state[k], post_best_state[k]), (
                    f"A6: best_model state_dict mismatch after restore at key '{k}'"
                )

            # iter_idx restored to "next iter to run" (saved+1).
            assert coach2.iter_idx == pre_iter, (
                f"A6: iter_idx mismatch after restore — pre={pre_iter} "
                f"post={coach2.iter_idx}"
            )
            assert coach2.total_instances_seen == pre_total, (
                f"A6: total_instances_seen mismatch — pre={pre_total} "
                f"post={coach2.total_instances_seen}"
            )

            # Continue training one more iteration to confirm the resumed
            # coach can execute (this is what F.1 will exercise).
            coach2.learn(n_iterations=1)
            assert coach2.iter_idx == pre_iter + 1, (
                f"A6: iter_idx after resumed learn(1) wrong: {coach2.iter_idx}"
            )
        finally:
            coach2.close()

    print("  [A6] OK")


def main() -> int:
    print("Stage 4 Phase B + C + D + F smoke harness")
    print("=" * 64)
    try:
        test_buffer_invariants(N=8)
        test_save_load_roundtrip(N=6)
        test_state_reconstruction_roundtrip(N=6)
        test_train_step_alphazero(N=8)
        test_train_step_alphazero_lambda_zero(N=8)
        test_self_play_generator(N=20, M=10, K=20)
        test_target_entropy_under_schedule(N=20, M=10, K=20)
        test_coach_gate_noop(N=8)
        test_coach_three_iters(N=8)
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print("=" * 64)
    print("PASS — all Phase B + C + D + F smoke checks succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
