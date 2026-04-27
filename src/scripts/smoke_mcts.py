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

Run:
    PYTHONPATH=src python -m scripts.smoke_mcts
"""
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


def main() -> int:
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
    print("[A9 OK] config validation rejects sqrt_n+value_head and bogus enums")

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
    print("[A11 OK] MCTSConfig() defaults are canonical (rollout + tree_reuse)")

    print("[OK] Stage 2 Milestone A1+A2..A11 smoke PASSED")
    return 0


if __name__ == '__main__':
    sys.exit(main())
