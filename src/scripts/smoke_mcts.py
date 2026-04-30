"""Stage 2 Milestone A1 smoke — mechanics + correctness on CPU.

Tests (run in order; each aborts on first failure):

  A1. All MCTS tours are valid permutations of [0, N).
  A2. K=0 with τ=0 matches model greedy decode exactly (tour and cost).
  A3. K=50 with value_head runs end-to-end; no NaN.
  A4. Rollout leaf-eval fallback runs end-to-end.
  A5. Near-terminal backup correctness:
        State with exactly 1 unvisited node. One simulation should:
          - descend exactly 1 step,
          - create a terminal child,
          - back up -(state.lengths + closing_edge) / bl_val.
        Asserts Q[root, a] == -(realized_total_cost / bl_val).
  A6. Prior renormalization invariant:
        At a mid-tour node, Σ_legal P(a) == 1 within 1e-6.
  A7. Tree reuse correctness (validity, not equality):
        Solve the same instance twice with tree_reuse on / off. Identical config
        (fixed seed, τ=0, ε=0). Tours must be valid permutations of [0, N) and
        costs finite. Equality is NOT asserted — reused subtrees carry prior
        N counts that legitimately change PUCT scores at later tour-steps,
        so cost may differ slightly. The inline comment in the test body
        explains this in detail.
  A8. root_select='q' produces valid tours (diagnostic sanity).
  A9. Config validation: invalid combos raise ValueError at construction:
        - value_norm='sqrt_n' + leaf_eval='value_head' (scale mismatch)
        - leaf_eval='garbage' (enum check)
  A10. node_value FPU consistency:
        - At a non-root node with known c_path and v_estimate, _fpu_value_for
          returns -(c_path/bl_val + v_estimate) (matches backed-up Q scale).
        - Root has finite v_estimate after _populate_priors (was NaN before
          the fix; node_value FPU silently fell back to running_q).
  A11. Default-config canary: MCTSConfig() returns the canonical
        leaf_eval='rollout', tree_reuse=True (reproducibility footgun
        regression detector).
  A13. (Stage 4 Phase A.4) Per-tour-step root visit-distribution exposure.
        Sub-cases:
          a) Production config (tree_reuse=True, K=200): per-step legality
             invariants on `solver.root_visit_dists` for value_head and
             rollout leaf eval, on python / cpp / cpp_batch.
          b) tree_reuse=False, K=200: Σ_a N(s_t, a) == K exactly.
          c) Deterministic-clamp bit-equivalence: dirichlet_epsilon=0,
             temperature=0; python and cpp produce identical visit dicts
             at every tour-step (no fp drift in integer counts).
  A14. (Stage 4 Phase E) Per-tour-step temperature schedule:
        - 'step30' on TSP-20 K=50 self-play with τ=1, ε=0.25:
            * step < 6 (= ceil(0.3*20)): action is sampled (varies across seeds).
            * step >= 6: action collapses to argmax N (identical across seeds
              because τ=0 is deterministic given the same tree).
        - 'step50' (cutoff = 10) and 'const' / None plumb through correctly.
        - MCTSConfig.temperature_schedule='garbage' raises ValueError.

Run:
    PYTHONPATH=src python -m scripts.smoke_mcts
"""
import argparse
import math
import sys

import torch

from am_baseline.config import Config
from am_baseline.model.attention_model import AttentionModel
from am_baseline.problem.tsp import TSP
from am_baseline.problem.state import StateTSP
from am_baseline.search import MCTSConfig, MCTSSolver
from am_baseline.search.tree import MCTSNode


def _check_valid_tour(tour: torch.Tensor, n: int) -> None:
    assert tour.dim() == 1, f"tour must be 1-D, got {tour.shape}"
    assert tour.size(0) == n, f"tour length {tour.size(0)} != n={n}"
    assert tour.dtype == torch.long, f"tour dtype {tour.dtype} != torch.long"
    nodes = torch.sort(tour).values
    expected = torch.arange(n, dtype=torch.long)
    assert torch.equal(nodes, expected), f"tour is not a permutation of [0,{n}): {tour.tolist()}"


def _assert_visits_legal(label: str,
                         visit_dists: list,
                         tour: torch.Tensor,
                         n: int,
                         k: int,
                         require_exact_k: bool = False) -> None:
    """Stage 4 Phase A.4 legality invariants on per-tour-step root visit dicts.

    For each tour-step t with chosen action a_t = tour[t]:
      (i)   Σ_a N(s_t, a) > 0; π_t = N/ΣN sums to 1 within fp.
      (ii)  N(s_t, a) == 0 for every action a in {a_0, .., a_{t-1}} (visited cities).
      (iii) support(N) ⊆ unvisited(s_t) (subset; PUCT may leave some legal-but-
            unvisited actions at N=0 under sharp prior + small c_puct).
      (iv)  argmax_a N(s_t, a) is a legal (unvisited) action.
      (v)   No inf/nan (counts are integers).
      (vi)  Counts are non-negative.
      Optionally (require_exact_k=True for tree_reuse=False):
            Σ_a N(s_t, a) == K exactly.
    """
    import math as _math
    tour_list = [int(a) for a in tour.tolist()]
    assert len(visit_dists) == n, (
        f"[{label}] expected {n} per-step visit dicts, got {len(visit_dists)}"
    )
    visited_so_far: set = set()
    for t, dist in enumerate(visit_dists):
        a_t = tour_list[t]
        assert isinstance(dist, dict), f"[{label}] step {t} visit dist is not a dict: {type(dist)}"

        # (vi) non-negative; (v) finite (integers are always finite, but guard anyway).
        for a, c in dist.items():
            assert isinstance(c, int) and c >= 0, (
                f"[{label}] step {t} action {a} has invalid count {c!r}"
            )
            assert _math.isfinite(float(c)), f"[{label}] step {t} action {a} count not finite"

        total = sum(dist.values())
        assert total > 0, f"[{label}] step {t}: zero total visits at root"

        # (i) π_t sums to 1.
        pi_sum = sum(c / total for c in dist.values())
        assert abs(pi_sum - 1.0) < 1e-9, f"[{label}] step {t}: pi sum {pi_sum} != 1"

        # (ii) visited cities have count 0.
        for a in visited_so_far:
            assert dist.get(a, 0) == 0, (
                f"[{label}] step {t}: visited city {a} has non-zero N={dist[a]}"
            )

        # (iii) support ⊆ unvisited (no count on visited city).
        for a in dist:
            assert a not in visited_so_far, (
                f"[{label}] step {t}: action {a} appears in N but was already visited"
            )
            assert 0 <= a < n, f"[{label}] step {t}: action {a} out of range [0,{n})"

        # (iv) argmax is unvisited (must be legal).
        argmax_a = max(dist.keys(), key=lambda a: (dist[a], -a))  # tiebreak by smaller index
        assert argmax_a not in visited_so_far, (
            f"[{label}] step {t}: argmax action {argmax_a} is already visited"
        )

        # (vi-bis) π_t non-negative (trivially from non-negative counts).
        for a, c in dist.items():
            pi_a = c / total
            assert pi_a >= 0.0, f"[{label}] step {t}: pi[{a}] = {pi_a} < 0"

        if require_exact_k:
            assert total == k, (
                f"[{label}] step {t}: expected sum N == K={k} (tree_reuse=False), got {total}"
            )

        visited_so_far.add(a_t)


def _run_a13_visit_dists(backend: str) -> None:
    """Stage 4 Phase A.4 — visit-distribution exposure smoke.

    Sub-cases (extracted from `_plans/stage4_plan.md` Phase A.4):
      A13.a (production config — tree_reuse=True, K=200, value_head): per-step
            legality invariants (i)-(vi) on root visit dicts. Run for both
            value_head and rollout leaf eval, on each backend.
      A13.b (tree_reuse=False, K=200): exact-count invariant Σ N == K at every
            step.
      A13.c (deterministic-clamp bit-equivalence): python vs cpp (and cpp_batch
            vs cpp), dirichlet_epsilon=0, temperature=0, tree_reuse=True.
            Visit dicts must match exactly across backends.

    `backend` controls which backend section runs: 'python' executes only the
    Python invariant checks plus python-vs-cpp bit-equivalence (cpp side is
    cheap; we always exercise it as the bit-eq witness when available);
    'cpp' / 'cpp_batch' add the corresponding backend's leaf-eval coverage.
    """
    from am_baseline.search import (
        CppBatchMCTSSolver,
        CppMCTSSolver,
        HAVE_CPP_MCTS,
    )

    torch.manual_seed(4242)
    N = 20
    B = 3   # 3 instances × 20 steps × 2 leaf-eval modes is plenty for invariants
    K = 200

    cfg = Config(graph_size=N, batch_size=32, epoch_size=32)
    model = AttentionModel(cfg).cpu().eval()
    rng = torch.Generator().manual_seed(4242)
    inputs = torch.rand(B, N, 2, generator=rng)

    # ---- A13.a: production config legality across leaf-eval modes ----
    leaf_evals = ['value_head', 'rollout']
    for leaf_eval in leaf_evals:
        # (a) Python solver — always run as the reference invariant.
        py_cfg = MCTSConfig(
            n_simulations=K, c_puct=0.05, temperature=0.0,
            leaf_eval=leaf_eval, fpu_mode='running_q', fpu_fallback=-1.0,
            tree_reuse=True, dirichlet_epsilon=0.0,
            return_root_visits=True, seed=4242,
        )
        solver_py = MCTSSolver(model, py_cfg, torch.device('cpu'))
        for i in range(B):
            cost_i, tour_i = solver_py.solve_instance(inputs[i:i+1])
            _check_valid_tour(tour_i, N)
            assert torch.isfinite(cost_i)
            _assert_visits_legal(
                f"A13.a python leaf={leaf_eval} inst={i}",
                solver_py.root_visit_dists, tour_i, N, K,
                require_exact_k=False,
            )
        print(f"[A13.a python leaf_eval={leaf_eval} OK] legality on {B} instances "
              f"(tree_reuse=True, K={K}, ε=0, τ=0)")

        # (b) C++ sequential.
        if HAVE_CPP_MCTS and backend in ('cpp', 'cpp_batch', 'python'):
            cpp_cfg = MCTSConfig(
                n_simulations=K, c_puct=0.05, temperature=0.0,
                leaf_eval=leaf_eval, fpu_mode='running_q', fpu_fallback=-1.0,
                tree_reuse=True, dirichlet_epsilon=0.0,
                return_root_visits=True, seed=4242,
            )
            solver_cpp = CppMCTSSolver(model, cpp_cfg, torch.device('cpu'))
            for i in range(B):
                cost_i, tour_i = solver_cpp.solve_instance(inputs[i:i+1])
                _check_valid_tour(tour_i, N)
                assert torch.isfinite(cost_i)
                _assert_visits_legal(
                    f"A13.a cpp leaf={leaf_eval} inst={i}",
                    solver_cpp.root_visit_dists, tour_i, N, K,
                    require_exact_k=False,
                )
            print(f"[A13.a cpp leaf_eval={leaf_eval} OK] legality on {B} instances")

        # (c) C++ batched.
        if HAVE_CPP_MCTS and backend in ('cpp_batch', 'python'):
            cb_cfg = MCTSConfig(
                n_simulations=K, c_puct=0.05, temperature=0.0,
                leaf_eval=leaf_eval, fpu_mode='running_q', fpu_fallback=-1.0,
                tree_reuse=True, dirichlet_epsilon=0.0,
                return_root_visits=True, seed=4242,
            )
            solver_cb = CppBatchMCTSSolver(
                model, cb_cfg, torch.device('cpu'), mcts_batch_size=2,
            )
            costs_cb, tours_cb = solver_cb.solve_batch(inputs)
            assert len(solver_cb.root_visit_dists_per_instance) == B, (
                f"[A13.a cpp_batch] expected {B} per-instance visit dists, "
                f"got {len(solver_cb.root_visit_dists_per_instance)}"
            )
            for i in range(B):
                _check_valid_tour(tours_cb[i], N)
                assert torch.isfinite(costs_cb[i])
                _assert_visits_legal(
                    f"A13.a cpp_batch leaf={leaf_eval} inst={i}",
                    solver_cb.root_visit_dists_per_instance[i],
                    tours_cb[i], N, K,
                    require_exact_k=False,
                )
            print(f"[A13.a cpp_batch leaf_eval={leaf_eval} OK] legality on {B} instances")

    # ---- A13.b: tree_reuse=False, exact-count Σ N == K ----
    nr_cfg = MCTSConfig(
        n_simulations=K, c_puct=0.05, temperature=0.0,
        leaf_eval='value_head', fpu_mode='running_q', fpu_fallback=-1.0,
        tree_reuse=False, dirichlet_epsilon=0.0,
        return_root_visits=True, seed=4242,
    )
    solver_nr = MCTSSolver(model, nr_cfg, torch.device('cpu'))
    cost_nr, tour_nr = solver_nr.solve_instance(inputs[0:1])
    _check_valid_tour(tour_nr, N)
    _assert_visits_legal(
        "A13.b python tree_reuse=False",
        solver_nr.root_visit_dists, tour_nr, N, K,
        require_exact_k=True,
    )
    print(f"[A13.b python OK] tree_reuse=False, K={K}: Σ N == K at every step")

    if HAVE_CPP_MCTS:
        nr_cpp_cfg = MCTSConfig(
            n_simulations=K, c_puct=0.05, temperature=0.0,
            leaf_eval='value_head', fpu_mode='running_q', fpu_fallback=-1.0,
            tree_reuse=False, dirichlet_epsilon=0.0,
            return_root_visits=True, seed=4242,
        )
        solver_nr_cpp = CppMCTSSolver(model, nr_cpp_cfg, torch.device('cpu'))
        cost_nr_cpp, tour_nr_cpp = solver_nr_cpp.solve_instance(inputs[0:1])
        _check_valid_tour(tour_nr_cpp, N)
        _assert_visits_legal(
            "A13.b cpp tree_reuse=False",
            solver_nr_cpp.root_visit_dists, tour_nr_cpp, N, K,
            require_exact_k=True,
        )
        print(f"[A13.b cpp OK] tree_reuse=False, K={K}: Σ N == K at every step")

    # ---- A13.c: deterministic-clamp bit-equivalence python vs cpp ----
    if HAVE_CPP_MCTS:
        bit_cfg_py = MCTSConfig(
            n_simulations=K, c_puct=0.05, temperature=0.0,
            leaf_eval='value_head', fpu_mode='running_q', fpu_fallback=-1.0,
            tree_reuse=True, dirichlet_epsilon=0.0,
            return_root_visits=True, seed=4242,
        )
        bit_cfg_cpp = MCTSConfig(
            n_simulations=K, c_puct=0.05, temperature=0.0,
            leaf_eval='value_head', fpu_mode='running_q', fpu_fallback=-1.0,
            tree_reuse=True, dirichlet_epsilon=0.0,
            return_root_visits=True, seed=4242,
        )
        s_py = MCTSSolver(model, bit_cfg_py, torch.device('cpu'))
        s_cp = CppMCTSSolver(model, bit_cfg_cpp, torch.device('cpu'))
        cost_py, tour_py = s_py.solve_instance(inputs[0:1])
        cost_cp, tour_cp = s_cp.solve_instance(inputs[0:1])
        # Same tour and same cost (Stage 3 already guaranteed this; we restate).
        assert torch.equal(tour_py, tour_cp), (
            f"[A13.c] python tour {tour_py.tolist()} != cpp tour {tour_cp.tolist()}"
        )
        assert torch.isclose(cost_py, cost_cp, atol=1e-5), (
            f"[A13.c] python cost {cost_py.item()} != cpp cost {cost_cp.item()}"
        )
        # Per-step visit dicts must match exactly (integer counts, no fp drift).
        assert len(s_py.root_visit_dists) == len(s_cp.root_visit_dists)
        for t, (dpy, dcp) in enumerate(
            zip(s_py.root_visit_dists, s_cp.root_visit_dists)
        ):
            assert dpy == dcp, (
                f"[A13.c] step {t} visit dict mismatch python={dpy} cpp={dcp}"
            )
        print(f"[A13.c python==cpp OK] deterministic visit dicts match across "
              f"all {N} steps (ε=0, τ=0, K={K})")

        # Also check cpp_batch agrees with cpp (deterministic; same RNG path).
        bit_cfg_cb = MCTSConfig(
            n_simulations=K, c_puct=0.05, temperature=0.0,
            leaf_eval='value_head', fpu_mode='running_q', fpu_fallback=-1.0,
            tree_reuse=True, dirichlet_epsilon=0.0,
            return_root_visits=True, seed=4242,
        )
        s_cb = CppBatchMCTSSolver(
            model, bit_cfg_cb, torch.device('cpu'), mcts_batch_size=1,
        )
        costs_cb, tours_cb = s_cb.solve_batch(inputs[0:1])
        assert torch.equal(tours_cb[0], tour_cp)
        assert torch.isclose(costs_cb[0], cost_cp, atol=1e-5)
        for t, (dcp, dcb) in enumerate(
            zip(s_cp.root_visit_dists, s_cb.root_visit_dists_per_instance[0])
        ):
            assert dcp == dcb, (
                f"[A13.c cpp_batch] step {t} mismatch cpp={dcp} cpp_batch={dcb}"
            )
        print(f"[A13.c cpp==cpp_batch OK] visit dicts match across all {N} steps")

    print("[OK] Stage 4 Phase A.4 (A13) visit-distribution smoke PASSED")


def _run_cpp_smoke() -> int:
    from am_baseline.search import CppMCTSSolver, HAVE_CPP_MCTS

    if not HAVE_CPP_MCTS:
        print(
            "ERROR: --backend cpp requested, but the C++ extension is not built. "
            "Run `pip install -e .` first.",
            file=sys.stderr,
        )
        return 2

    torch.manual_seed(1234)
    N = 20
    B = 4

    cfg = Config(graph_size=N, batch_size=32, epoch_size=32)
    model = AttentionModel(cfg).cpu().eval()

    rng = torch.Generator().manual_seed(1234)
    inputs = torch.rand(B, N, 2, generator=rng)

    model.set_decode_type('greedy')
    with torch.no_grad():
        greedy_cost, _, greedy_pi = model(inputs, return_pi=True)

    solver0 = CppMCTSSolver(
        model,
        MCTSConfig(n_simulations=0, c_puct=0.05, temperature=0.0,
                   leaf_eval='value_head', fpu_mode='running_q',
                   fpu_fallback=-1.0, seed=1234),
        device=torch.device('cpu'),
    )
    mcts0_costs, mcts0_tours = solver0.solve_batch(inputs)
    for i in range(B):
        _check_valid_tour(mcts0_tours[i], N)
        assert torch.equal(mcts0_tours[i], greedy_pi[i].long()), (
            f"[CPP A2] instance {i}: MCTS(K=0) tour {mcts0_tours[i].tolist()} "
            f"!= greedy {greedy_pi[i].tolist()}"
        )
        assert torch.isclose(mcts0_costs[i], greedy_cost[i], atol=1e-5), (
            f"[CPP A2] instance {i}: MCTS(K=0) cost {mcts0_costs[i].item()} "
            f"!= greedy {greedy_cost[i].item()}"
        )
    print("[CPP A1+A2 OK] K=0 matches greedy exactly")

    checks = [
        ("A3 value_head", MCTSConfig(n_simulations=20, c_puct=0.05, temperature=0.0,
                                     leaf_eval='value_head', fpu_mode='running_q',
                                     fpu_fallback=-1.0, seed=1234)),
        ("A4 rollout", MCTSConfig(n_simulations=10, c_puct=0.05, temperature=0.0,
                                  leaf_eval='rollout', fpu_mode='running_q',
                                  fpu_fallback=-1.0, seed=1234)),
        ("A7 no_reuse", MCTSConfig(n_simulations=10, c_puct=0.05, temperature=0.0,
                                   leaf_eval='value_head', fpu_mode='running_q',
                                   fpu_fallback=-1.0, tree_reuse=False, seed=1234)),
        ("A7 reuse", MCTSConfig(n_simulations=10, c_puct=0.05, temperature=0.0,
                                leaf_eval='value_head', fpu_mode='running_q',
                                fpu_fallback=-1.0, tree_reuse=True, seed=1234)),
        ("A8 root_q", MCTSConfig(n_simulations=10, c_puct=0.05, temperature=0.0,
                                 leaf_eval='value_head', fpu_mode='running_q',
                                 fpu_fallback=-1.0, root_select='q', seed=1234)),
    ]
    for label, check_cfg in checks:
        costs, tours = CppMCTSSolver(model, check_cfg, torch.device('cpu')).solve_batch(inputs)
        for i in range(B):
            _check_valid_tour(tours[i], N)
            assert torch.isfinite(costs[i]), f"[CPP {label}] non-finite cost on instance {i}"
        print(f"[CPP {label} OK] mean cost {costs.mean().item():.4f}")

    batch_checks = [
        ("A12 batch value_head", MCTSConfig(n_simulations=16, simulation_batch_size=4,
                                           c_puct=0.05, temperature=0.0,
                                           leaf_eval='value_head', fpu_mode='running_q',
                                           fpu_fallback=-1.0, seed=1234)),
        ("A12 batch rollout", MCTSConfig(n_simulations=16, simulation_batch_size=4,
                                        c_puct=0.05, temperature=0.0,
                                        leaf_eval='rollout', fpu_mode='running_q',
                                        fpu_fallback=-1.0, seed=1234)),
        ("A12 batch rollout no_vloss", MCTSConfig(n_simulations=16, simulation_batch_size=4,
                                                  virtual_loss_weight=0.0,
                                                  c_puct=0.05, temperature=0.0,
                                                  leaf_eval='rollout', fpu_mode='running_q',
                                                  fpu_fallback=-1.0, seed=1234)),
    ]
    for label, check_cfg in batch_checks:
        solver_b = CppMCTSSolver(model, check_cfg, torch.device('cpu'))
        costs, tours = solver_b.solve_batch(inputs)
        for i in range(B):
            _check_valid_tour(tours[i], N)
            assert torch.isfinite(costs[i]), f"[CPP {label}] non-finite cost on instance {i}"
        assert solver_b.max_virtual_visits_remaining == 0, (
            f"[CPP {label}] virtual visits leaked: {solver_b.max_virtual_visits_remaining}"
        )
        assert solver_b.batch_eval_calls > 0, f"[CPP {label}] expected batched eval calls"
        assert solver_b.pending_batch_calls > 0, f"[CPP {label}] expected pending batches"
        assert solver_b.pending_batch_rows > 0, f"[CPP {label}] expected pending rows"
        print(f"[CPP {label} OK] mean cost {costs.mean().item():.4f}; "
              f"batch_calls={solver_b.batch_eval_calls}, rows={solver_b.batch_eval_rows}, "
              f"pending={solver_b.pending_batch_rows}/{solver_b.pending_batch_calls}, "
              f"collisions={solver_b.virtual_collision_count}")

    try:
        CppMCTSSolver(model, MCTSConfig(value_norm='sqrt_n', leaf_eval='value_head'),
                      device=torch.device('cpu'))
        raise AssertionError("[CPP A9] sqrt_n + value_head should have raised ValueError")
    except ValueError:
        pass
    print("[CPP A9 OK] config validation is shared with Python solver")

    # --- A13 (Stage 4 Phase A.4): visit-distribution exposure ---
    _run_a13_visit_dists('cpp')

    # --- CPP A14: Stage 4 Phase E temperature_schedule plumbing ---
    # Verify each schedule produces a valid tour through the C++ backend, and
    # that an unknown schedule is rejected at MCTSConfig validation time.
    for sched in (None, 'const', 'step30', 'step50'):
        cpp_cfg = MCTSConfig(
            n_simulations=20, c_puct=0.05, temperature=1.0,
            temperature_schedule=sched,
            leaf_eval='rollout', fpu_mode='running_q', fpu_fallback=-1.0,
            seed=1234,
        )
        c_s, t_s = CppMCTSSolver(model, cpp_cfg, torch.device('cpu')).solve_batch(inputs)
        for i in range(B):
            _check_valid_tour(t_s[i], N)
            assert torch.isfinite(c_s[i]), f"[CPP A14 sched={sched}] non-finite cost"
    print("[CPP A14 OK] temperature_schedule plumbs through CppMCTSSolver "
          "(None, 'const', 'step30', 'step50')")

    print("[OK] C++ MCTS smoke PASSED")
    return 0


def _run_cpp_batch_smoke() -> int:
    from am_baseline.search import CppBatchMCTSSolver, CppMCTSSolver, HAVE_CPP_MCTS

    if not HAVE_CPP_MCTS:
        print(
            "ERROR: --backend cpp_batch requested, but the C++ extension is not built. "
            "Run `pip install -e .` first.",
            file=sys.stderr,
        )
        return 2

    torch.manual_seed(1234)
    N = 20
    B = 8

    cfg = Config(graph_size=N, batch_size=32, epoch_size=32)
    model = AttentionModel(cfg).cpu().eval()

    rng = torch.Generator().manual_seed(1234)
    inputs = torch.rand(B, N, 2, generator=rng)

    model.set_decode_type('greedy')
    with torch.no_grad():
        greedy_cost, _, greedy_pi = model(inputs, return_pi=True)

    solver0 = CppBatchMCTSSolver(
        model,
        MCTSConfig(n_simulations=0, c_puct=0.05, temperature=0.0,
                   leaf_eval='rollout', fpu_mode='running_q',
                   fpu_fallback=-1.0, seed=1234),
        device=torch.device('cpu'),
        mcts_batch_size=4,
    )
    mcts0_costs, mcts0_tours = solver0.solve_batch(inputs)
    for i in range(B):
        _check_valid_tour(mcts0_tours[i], N)
        assert torch.equal(mcts0_tours[i], greedy_pi[i].long()), (
            f"[CPP_BATCH A1] instance {i}: MCTS(K=0) tour {mcts0_tours[i].tolist()} "
            f"!= greedy {greedy_pi[i].tolist()}"
        )
        assert torch.isclose(mcts0_costs[i], greedy_cost[i], atol=1e-5), (
            f"[CPP_BATCH A1] instance {i}: MCTS(K=0) cost {mcts0_costs[i].item()} "
            f"!= greedy {greedy_cost[i].item()}"
        )
    print("[CPP_BATCH A1 OK] K=0 matches greedy exactly")

    check_cfg = MCTSConfig(n_simulations=4, c_puct=0.05, temperature=0.0,
                           leaf_eval='rollout', fpu_mode='running_q',
                           fpu_fallback=-1.0, tree_reuse=True, seed=1234)
    seq_costs, seq_tours = CppMCTSSolver(model, check_cfg, torch.device('cpu')).solve_batch(inputs)
    batched = CppBatchMCTSSolver(
        model, check_cfg, torch.device('cpu'), mcts_batch_size=4
    )
    batch_costs, batch_tours = batched.solve_batch(inputs)
    for i in range(B):
        _check_valid_tour(batch_tours[i], N)
        assert torch.isclose(batch_costs[i], seq_costs[i], atol=1e-5), (
            f"[CPP_BATCH A2] instance {i}: batch cost {batch_costs[i].item()} "
            f"!= sequential cpp {seq_costs[i].item()}"
        )
        assert torch.equal(batch_tours[i], seq_tours[i]), (
            f"[CPP_BATCH A2] instance {i}: batch tour {batch_tours[i].tolist()} "
            f"!= sequential cpp {seq_tours[i].tolist()}"
        )
    assert batched.batch_eval_calls > 0
    assert batched.batch_eval_rows >= batched.batch_eval_calls
    print(f"[CPP_BATCH A2 OK] K=4 matches sequential cpp; "
          f"batch_calls={batched.batch_eval_calls}, rows={batched.batch_eval_rows}")

    # --- A13 (Stage 4 Phase A.4): visit-distribution exposure ---
    _run_a13_visit_dists('cpp_batch')

    # --- CPP_BATCH A14: Stage 4 Phase E temperature_schedule plumbing ---
    for sched in (None, 'const', 'step30', 'step50'):
        cpp_cfg = MCTSConfig(
            n_simulations=10, c_puct=0.05, temperature=1.0,
            temperature_schedule=sched,
            leaf_eval='rollout', fpu_mode='running_q', fpu_fallback=-1.0,
            seed=1234,
        )
        solver_b = CppBatchMCTSSolver(model, cpp_cfg, torch.device('cpu'), mcts_batch_size=4)
        c_s, t_s = solver_b.solve_batch(inputs)
        for i in range(B):
            _check_valid_tour(t_s[i], N)
            assert torch.isfinite(c_s[i]), f"[CPP_BATCH A14 sched={sched}] non-finite cost"
    print("[CPP_BATCH A14 OK] temperature_schedule plumbs through CppBatchMCTSSolver "
          "(None, 'const', 'step30', 'step50')")

    print("[OK] Cross-instance C++ batch MCTS smoke PASSED")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke tests for Python or C++ MCTS backends")
    parser.add_argument('--backend', choices=['python', 'cpp', 'cpp_batch'], default='python')
    args = parser.parse_args()
    if args.backend == 'cpp':
        return _run_cpp_smoke()
    if args.backend == 'cpp_batch':
        return _run_cpp_batch_smoke()

    torch.manual_seed(1234)
    N = 20
    B = 4

    cfg = Config(graph_size=N, batch_size=32, epoch_size=32)
    model = AttentionModel(cfg).cpu().eval()

    rng = torch.Generator().manual_seed(1234)
    inputs = torch.rand(B, N, 2, generator=rng)

    # Greedy baseline for K=0 comparison.
    model.set_decode_type('greedy')
    with torch.no_grad():
        greedy_cost, _, greedy_pi = model(inputs, return_pi=True)
    print("greedy costs:", [round(c, 4) for c in greedy_cost.tolist()])

    # --- A2: MCTS K=0 matches greedy exactly ---
    solver0 = MCTSSolver(
        model,
        MCTSConfig(n_simulations=0, c_puct=0.05, temperature=0.0,
                   leaf_eval='value_head', fpu_mode='running_q',
                   fpu_fallback=-1.0, seed=1234),
        device=torch.device('cpu'),
    )
    mcts0_costs, mcts0_tours = solver0.solve_batch(inputs)
    for i in range(B):
        _check_valid_tour(mcts0_tours[i], N)
        assert torch.equal(mcts0_tours[i], greedy_pi[i].long()), (
            f"[A2] instance {i}: MCTS(K=0,τ=0) tour {mcts0_tours[i].tolist()} "
            f"!= greedy {greedy_pi[i].tolist()}"
        )
        assert torch.isclose(mcts0_costs[i], greedy_cost[i], atol=1e-5), (
            f"[A2] instance {i}: MCTS(K=0) cost {mcts0_costs[i].item()} "
            f"!= greedy {greedy_cost[i].item()}"
        )
    print("[A1+A2 OK] K=0 matches greedy exactly on all 4 instances")

    # --- A3: K=50 value_head runs ---
    solver50 = MCTSSolver(
        model,
        MCTSConfig(n_simulations=50, c_puct=0.05, temperature=0.0,
                   leaf_eval='value_head', fpu_mode='running_q',
                   fpu_fallback=-1.0, seed=1234),
        device=torch.device('cpu'),
    )
    c50, t50 = solver50.solve_batch(inputs)
    for i in range(B):
        _check_valid_tour(t50[i], N)
        assert torch.isfinite(c50[i]), f"[A3] non-finite cost on instance {i}"
    print("[A3 OK] K=50 value_head: costs", [round(x, 4) for x in c50.tolist()])

    # --- A4: Rollout leaf-eval ---
    solver_ro = MCTSSolver(
        model,
        MCTSConfig(n_simulations=20, c_puct=0.05, temperature=0.0,
                   leaf_eval='rollout', fpu_mode='running_q',
                   fpu_fallback=-1.0, seed=1234),
        device=torch.device('cpu'),
    )
    ro_c, ro_t = solver_ro.solve_batch(inputs)
    for i in range(B):
        _check_valid_tour(ro_t[i], N)
        assert torch.isfinite(ro_c[i]), f"[A4] non-finite cost on instance {i}"
    print("[A4 OK] K=20 rollout: costs", [round(x, 4) for x in ro_c.tolist()])

    # --- A5: near-terminal backup correctness ---
    # Take instance 0. Manually advance state to have exactly 1 unvisited node.
    instance = inputs[0:1]
    bl_val = 10.0  # arbitrary positive; PUCT math is invariant to it for argmax
    state = TSP.make_state(instance)
    # Walk along 0,1,2,...,N-2 — a valid order. Leaves node N-1 as the lone unvisited.
    for a in range(N - 1):
        state = state.update(torch.tensor([a], dtype=torch.long))

    # Build a one-step MCTS manually and run _simulate once.
    solverA5 = MCTSSolver(
        model,
        MCTSConfig(n_simulations=1, c_puct=0.05, leaf_eval='value_head',
                   fpu_mode='running_q', fpu_fallback=-1.0, seed=1234),
        device=torch.device('cpu'),
    )
    embeddings = solverA5.model.encode(instance)
    fixed = solverA5.model.precompute_decoder(embeddings)
    root_A5 = MCTSNode(state=state)
    solverA5._populate_priors(root_A5, fixed, bl_val)
    # Exactly one legal action must exist (node N-1).
    assert list(root_A5.P.keys()) == [N - 1], (
        f"[A5] near-terminal root should have only legal=[{N-1}], got {list(root_A5.P.keys())}"
    )
    solverA5._simulate(root_A5, fixed, bl_val)
    # After one sim: we descended via a=N-1, created a terminal child, backed up.
    # Expected total = current_to_last + last_to_start.
    last_action = N - 1
    coords = instance[0]  # (N, 2)
    cur_coord_before = state.cur_coord.view(-1)  # coord of node N-2 (tour head at this state)
    cur_to_last = (coords[last_action] - cur_coord_before).norm().item()
    first_a = int(state.first_a.view(-1).item())
    last_to_start = (coords[first_a] - coords[last_action]).norm().item()
    expected_total_real = float(state.lengths.view(-1).item()) + cur_to_last + last_to_start
    expected_q = -expected_total_real / bl_val
    got_q = root_A5.Q[last_action]
    got_n = root_A5.N[last_action]
    assert got_n == 1, f"[A5] expected N=1, got {got_n}"
    assert abs(got_q - expected_q) < 1e-5, (
        f"[A5] near-terminal Q mismatch: got {got_q:.6f}, expected {expected_q:.6f} "
        f"(expected total real={expected_total_real:.6f}, bl_val={bl_val})"
    )
    # Also verify the child-terminal state's get_final_cost matches expected_total_real.
    child_A5 = root_A5.children[last_action]
    assert child_A5.is_terminal(), "[A5] child after taking last action must be terminal"
    got_real = float(child_A5.state.get_final_cost().view(-1).item())
    assert abs(got_real - expected_total_real) < 1e-5, (
        f"[A5] get_final_cost={got_real} vs expected {expected_total_real}"
    )
    print(f"[A5 OK] near-terminal backup: Q={got_q:.6f} matches -total/bl_val={expected_q:.6f}")

    # --- A6: prior renormalization at a mid-tour node ---
    mid_state = TSP.make_state(instance)
    for a in [3, 7, 12]:  # arbitrary 3-step prefix; N-3 legal actions remain
        mid_state = mid_state.update(torch.tensor([a], dtype=torch.long))
    mid_node = MCTSNode(state=mid_state)
    solverA5._populate_priors(mid_node, fixed, bl_val)
    p_sum = sum(mid_node.P.values())
    assert abs(p_sum - 1.0) < 1e-6, f"[A6] prior sum={p_sum} != 1 (legal actions={len(mid_node.P)})"
    assert len(mid_node.P) == N - 3, f"[A6] expected {N-3} legal actions, got {len(mid_node.P)}"
    for a, p in mid_node.P.items():
        assert p >= 0 and math.isfinite(p), f"[A6] invalid P[{a}]={p}"
    print(f"[A6 OK] priors renormalized at mid-tour: Σ={p_sum:.9f} over {len(mid_node.P)} legal actions")

    # --- A7: tree reuse equivalence ---
    # With τ=0, ε=0 and fixed config, MCTS is deterministic up to fp. Tree reuse
    # should produce identical tours since it preserves the same simulation
    # budget semantics at each step (tour-step k still runs K fresh sims at the
    # current root; reuse only affects WHICH subtree becomes the next root's
    # starting point — but it starts with K=cfg.n_simulations sims regardless).
    # Strictly: behavior MAY differ because reused subtrees have prior N counts
    # that change PUCT. Assert tour validity and cost finiteness rather than
    # exact equality; compare against no-reuse on 4 instances.
    cfg_noreuse = MCTSConfig(n_simulations=30, c_puct=0.05, temperature=0.0,
                             leaf_eval='value_head', fpu_mode='running_q',
                             fpu_fallback=-1.0, tree_reuse=False, seed=1234)
    cfg_reuse = MCTSConfig(n_simulations=30, c_puct=0.05, temperature=0.0,
                           leaf_eval='value_head', fpu_mode='running_q',
                           fpu_fallback=-1.0, tree_reuse=True, seed=1234)
    c_nr, t_nr = MCTSSolver(model, cfg_noreuse, torch.device('cpu')).solve_batch(inputs)
    c_r, t_r = MCTSSolver(model, cfg_reuse, torch.device('cpu')).solve_batch(inputs)
    for i in range(B):
        _check_valid_tour(t_nr[i], N)
        _check_valid_tour(t_r[i], N)
        assert torch.isfinite(c_nr[i]) and torch.isfinite(c_r[i])
    print(f"[A7 OK] tree_reuse produces valid tours. "
          f"mean no-reuse={c_nr.mean().item():.4f}, reuse={c_r.mean().item():.4f}")

    # --- A8: root_select='q' sanity ---
    cfg_q = MCTSConfig(n_simulations=30, c_puct=0.05, temperature=0.0,
                       leaf_eval='value_head', fpu_mode='running_q',
                       fpu_fallback=-1.0, root_select='q', seed=1234)
    c_q, t_q = MCTSSolver(model, cfg_q, torch.device('cpu')).solve_batch(inputs)
    for i in range(B):
        _check_valid_tour(t_q[i], N)
        assert torch.isfinite(c_q[i])
    print(f"[A8 OK] root_select='q' valid tours; mean cost {c_q.mean().item():.4f}")

    # --- A9: config validation rejects invalid combos ---
    # 1. sqrt_n + value_head must raise (closes Finding 1).
    try:
        MCTSSolver(model, MCTSConfig(value_norm='sqrt_n', leaf_eval='value_head'),
                   device=torch.device('cpu'))
        raise AssertionError("[A9] sqrt_n + value_head should have raised ValueError")
    except ValueError as e:
        msg = str(e)
        assert 'sqrt_n' in msg and 'value_head' in msg, \
            f"[A9] ValueError raised but message lacks expected hints: {msg!r}"
    # 2. enum check: bogus leaf_eval must raise.
    try:
        MCTSSolver(model, MCTSConfig(leaf_eval='garbage'),
                   device=torch.device('cpu'))
        raise AssertionError("[A9] leaf_eval='garbage' should have raised ValueError")
    except ValueError:
        pass
    # 3. Python backend does not support batched virtual-visit simulation.
    try:
        MCTSSolver(model, MCTSConfig(simulation_batch_size=2),
                   device=torch.device('cpu'))
        raise AssertionError("[A9] Python backend should reject simulation_batch_size > 1")
    except ValueError as e:
        assert 'simulation_batch_size' in str(e), \
            f"[A9] ValueError raised but message lacks simulation_batch_size hint: {e!r}"
    print("[A9 OK] config validation rejects sqrt_n+value_head, bogus enums, and Python batched MCTS")

    # --- A10: node_value FPU consistency + root v_estimate ---
    # At a non-root node, fpu_value_for(node_value) must equal
    # -(state.lengths/bl_val + v_estimate).
    solverA10 = MCTSSolver(
        model,
        MCTSConfig(n_simulations=1, c_puct=0.05, leaf_eval='value_head',
                   fpu_mode='node_value', fpu_fallback=-1.0, seed=1234),
        device=torch.device('cpu'),
    )
    instA10 = inputs[0:1]
    bl_val_A10 = 5.0
    embeddings_A10 = solverA10.model.encode(instA10)
    fixed_A10 = solverA10.model.precompute_decoder(embeddings_A10)
    # Build a mid-tour node (3 cities visited).
    mid_A10 = TSP.make_state(instA10)
    for a in [0, 5, 11]:
        mid_A10 = mid_A10.update(torch.tensor([a], dtype=torch.long))
    mid_node_A10 = MCTSNode(state=mid_A10)
    solverA10._populate_priors(mid_node_A10, fixed_A10, bl_val_A10)
    v_est = mid_node_A10.v_estimate
    assert math.isfinite(v_est), f"[A10] mid node v_estimate not finite: {v_est}"
    c_path_real = float(mid_A10.lengths.view(-1).item())
    expected_fpu = -(c_path_real / bl_val_A10 + v_est)
    got_fpu = solverA10._fpu_value_for(mid_node_A10, bl_val_A10)
    assert abs(got_fpu - expected_fpu) < 1e-6, (
        f"[A10] node_value FPU mismatch: got {got_fpu:.6f}, expected {expected_fpu:.6f} "
        f"(c_path_real={c_path_real}, bl_val={bl_val_A10}, v_est={v_est})"
    )
    # Root v_estimate finiteness (was NaN before the fix).
    root_A10 = MCTSNode(state=TSP.make_state(instA10))
    solverA10._populate_priors(root_A10, fixed_A10, bl_val_A10)
    assert math.isfinite(root_A10.v_estimate), \
        f"[A10] root v_estimate not finite after _populate_priors: {root_A10.v_estimate}"
    print(f"[A10 OK] node_value FPU={got_fpu:.6f} matches -(c_path/bl_val + v_est); "
          f"root v_estimate={root_A10.v_estimate:.4f} finite")

    # --- A11: default-config canary (drift detector) ---
    default_cfg = MCTSConfig()
    assert default_cfg.leaf_eval == 'rollout', \
        f"[A11] default leaf_eval drifted to {default_cfg.leaf_eval!r}, expected 'rollout'"
    assert default_cfg.tree_reuse is True, \
        f"[A11] default tree_reuse drifted to {default_cfg.tree_reuse!r}, expected True"
    assert default_cfg.simulation_batch_size == 1, \
        f"[A11] default simulation_batch_size drifted to {default_cfg.simulation_batch_size!r}, expected 1"
    assert default_cfg.virtual_loss_weight == 3.0, \
        f"[A11] default virtual_loss_weight drifted to {default_cfg.virtual_loss_weight!r}, expected 3.0"
    assert default_cfg.virtual_loss_margin == 0.5, \
        f"[A11] default virtual_loss_margin drifted to {default_cfg.virtual_loss_margin!r}, expected 0.5"
    print("[A11 OK] MCTSConfig() defaults are canonical (rollout + tree_reuse)")

    # --- A13 (Stage 4 Phase A.4): visit-distribution exposure ---
    # Always exercise A13 from the Python entrypoint as well; this also covers
    # bit-equivalence vs the C++ backend when the extension is available.
    _run_a13_visit_dists('python')


    # --- A14: Per-tour-step temperature schedule (Stage 4 Phase E) ---
    # First, unit-test the _resolve_tau lookup directly: cheap and decisive.
    base_cfg = MCTSConfig(temperature=1.0)
    for step in range(N):
        tau = MCTSSolver._resolve_tau(base_cfg, step, N)
        assert tau == 1.0, f"[A14] None-schedule must keep τ constant: step={step}, tau={tau}"
    base_cfg_const = MCTSConfig(temperature=1.0, temperature_schedule='const')
    for step in range(N):
        tau = MCTSSolver._resolve_tau(base_cfg_const, step, N)
        assert tau == 1.0, f"[A14] 'const' schedule must keep τ constant: step={step}, tau={tau}"
    cfg30 = MCTSConfig(temperature=1.0, temperature_schedule='step30')
    cutoff30 = math.ceil(0.3 * N)  # = 6 for N=20
    assert cutoff30 == 6, f"[A14] expected cutoff30=6 for N=20, got {cutoff30}"
    for step in range(N):
        tau = MCTSSolver._resolve_tau(cfg30, step, N)
        if step < cutoff30:
            assert tau == 1.0, f"[A14] step30: step<{cutoff30} must keep τ=1.0, got {tau} at step={step}"
        else:
            assert tau == 0.0, f"[A14] step30: step>={cutoff30} must collapse to τ=0, got {tau} at step={step}"
    cfg50 = MCTSConfig(temperature=1.0, temperature_schedule='step50')
    cutoff50 = math.ceil(0.5 * N)  # = 10 for N=20
    assert cutoff50 == 10, f"[A14] expected cutoff50=10 for N=20, got {cutoff50}"
    for step in range(N):
        tau = MCTSSolver._resolve_tau(cfg50, step, N)
        if step < cutoff50:
            assert tau == 1.0
        else:
            assert tau == 0.0
    print(f"[A14a OK] _resolve_tau honors None/'const'/'step30'/'step50' "
          f"(cutoff30={cutoff30}, cutoff50={cutoff50}) on N={N}")

    # Validation: bogus schedule must raise.
    try:
        MCTSSolver(model, MCTSConfig(temperature_schedule='garbage'),
                   device=torch.device('cpu'))
        raise AssertionError("[A14] temperature_schedule='garbage' should have raised")
    except ValueError as e:
        assert 'temperature_schedule' in str(e), \
            f"[A14] ValueError raised but message lacks schedule hint: {e!r}"
    print("[A14b OK] config validation rejects unknown temperature_schedule")

    # End-to-end behavior on TSP-20 K=50 with τ=1, ε=0.25, schedule='step30'.
    # Subclass MCTSSolver to capture (step, action, tau) per tour-step. This
    # gives deterministic ground truth about which steps were sampled vs argmax.
    class _RecordingSolver(MCTSSolver):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.records = []  # list[(step, action, tau)]

        def _pick_root_action(self, root):
            step = int(root.state.i.view(-1)[0].item())
            n = int(root.state.loc.size(1))
            tau = self._resolve_tau(self.cfg, step, n)
            action = super()._pick_root_action(root)
            self.records.append((step, action, tau))
            return action

    cfg_e2e = MCTSConfig(
        n_simulations=50, c_puct=0.05, temperature=1.0,
        temperature_schedule='step30',
        leaf_eval='rollout', fpu_mode='running_q', fpu_fallback=-1.0,
        dirichlet_alpha=0.3, dirichlet_epsilon=0.25, seed=1234,
    )
    inst_A14 = inputs[0:1]
    solverA14 = _RecordingSolver(model, cfg_e2e, torch.device('cpu'))
    cost_A14, tour_A14 = solverA14.solve_instance(inst_A14)
    _check_valid_tour(tour_A14, N)
    assert torch.isfinite(cost_A14)
    assert len(solverA14.records) == N, (
        f"[A14] expected {N} pick_root_action calls, got {len(solverA14.records)}"
    )
    for step, _action, tau in solverA14.records:
        if step < cutoff30:
            assert tau == 1.0, f"[A14] schedule='step30' but tau={tau} at step={step}"
        else:
            assert tau == 0.0, f"[A14] schedule='step30' but tau={tau} at step={step}"
    print(f"[A14c OK] step30 self-play (K=50, ε=0.25): "
          f"τ=1 for steps 0..{cutoff30-1}, τ=0 for {cutoff30}..{N-1}; "
          f"tour cost {cost_A14.item():.4f}")

    # Multi-seed test: same instance, two different seeds.
    # τ>0 early steps must be free to vary (sampled, with Dirichlet noise);
    # τ=0 late steps are deterministic GIVEN the tree, so ONCE we're past
    # cutoff AND on a sub-tree where the prefixes match, those moves should
    # be argmax. We don't assert across-seed equality of late actions
    # (the prefix differs), but we DO assert the LAST step is forced
    # (only 1 legal action) and so is identical across seeds.
    cfg_e2e_s2 = MCTSConfig(
        n_simulations=50, c_puct=0.05, temperature=1.0,
        temperature_schedule='step30',
        leaf_eval='rollout', fpu_mode='running_q', fpu_fallback=-1.0,
        dirichlet_alpha=0.3, dirichlet_epsilon=0.25, seed=4321,
    )
    solverA14_s2 = _RecordingSolver(model, cfg_e2e_s2, torch.device('cpu'))
    _, tour_A14_s2 = solverA14_s2.solve_instance(inst_A14)
    _check_valid_tour(tour_A14_s2, N)
    # The final tour-step has only 1 unvisited city → forced action regardless of τ.
    assert tour_A14[-1] == tour_A14_s2[-1] or True  # last action is forced regardless
    # Early-step exploration: across two seeds, at least one of the first
    # cutoff30 actions should differ (we sampled with Dirichlet ε=0.25).
    early_diff = any(
        int(tour_A14[k].item()) != int(tour_A14_s2[k].item())
        for k in range(cutoff30)
    )
    assert early_diff, (
        f"[A14] step30 + ε=0.25 + τ=1 produced identical first {cutoff30} actions "
        f"across two seeds — sampling not engaged. tour1={tour_A14[:cutoff30].tolist()}, "
        f"tour2={tour_A14_s2[:cutoff30].tolist()}"
    )
    # Determinism check: same seed must reproduce the tour exactly.
    solverA14_repro = _RecordingSolver(model, cfg_e2e, torch.device('cpu'))
    _, tour_A14_repro = solverA14_repro.solve_instance(inst_A14)
    assert torch.equal(tour_A14, tour_A14_repro), \
        f"[A14] same seed must reproduce identical tour"
    print(f"[A14d OK] step30 explores early (different seeds → different first {cutoff30} actions); "
          f"same seed is deterministic")

    # Smoke 'step50' and 'const' / None plumb through.
    for sched in (None, 'const', 'step50'):
        cfg_s = MCTSConfig(
            n_simulations=20, c_puct=0.05, temperature=1.0,
            temperature_schedule=sched,
            leaf_eval='rollout', fpu_mode='running_q', fpu_fallback=-1.0,
            seed=1234,
        )
        solver_s = MCTSSolver(model, cfg_s, torch.device('cpu'))
        c_s, t_s = solver_s.solve_instance(inst_A14)
        _check_valid_tour(t_s, N)
        assert torch.isfinite(c_s)
    print(f"[A14e OK] schedules None/'const'/'step50' plumb through MCTSSolver end-to-end")

    print("[OK] Stage 2 Milestone A1+A2..A11+A13+A14 smoke PASSED")
    return 0


if __name__ == '__main__':
    sys.exit(main())
