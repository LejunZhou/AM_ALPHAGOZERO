# Stage 2 Plan: MCTS for Routing — Search on Trained AM + Value

**Created:** 2026-04-24
**Last revised:** 2026-04-24 (post per-step value diagnostic: added off-policy R² probe, `value_norm='sqrt_n'` ablation, sharpened Risks #4 with per-step bias mechanism, Stage 4 forward note)
**Predecessor:** Stage 1 (`_plans/stage1_plan.md`) — TSP-20 closed (canonical `xg7t2dlb`, val 3.8424, R²=0.9965); TSP-50 verification runs in flight (`apy5m2lf`, `123x2qr5`)
**Reference:** Proposal §Stage 2 (`proposal.md:75–96`)
**Status:** **Stage 2 substantively complete (2026-04-26).** Phases A–D all closed: smoke A1..A8 green; canonical config locked (`c_puct=0.05`, `tree_reuse=True`, `fpu_running_q`/`-1.0`, `root_select=visits`); TSP-20 K-curve through K=800 (value_head 53.3 % gap reduction; rollout K=400 71.3 %); TSP-50 K-curve through K=400 canonical + K=100 rollout (rollout 59.0 %); decode-step micro-benchmark added (per-call cost flat in N up to 200, overhead-dominated). Outstanding tactical items: off-policy R² probe (Phase D step 13), `value_norm='sqrt_n'` MCTS ablation (Phase D step 14), clean TSP-50 K=100 rollout wall-clock (background task `bln0tv1pg`). None block Stage 3/4. See `_progress/stage2_progress.md` §Stage 2 Conclusions for the full readout.

---

## Context

Stages 0 and 1 gave us a trained dual-head AM (policy + value). Stage 2 builds the **search layer** that consumes this trained network: MCTS adapted from AlphaGo Zero, tailored to sequential node-selection for routing problems. This stage is **purely about the search module** — it does **not** retrain the network and it does **not** yet run the benchmark curve against sampling-1280 (Stage 3) or plug MCTS into training (Stage 4).

Stage 0's refactor already exposed every primitive MCTS needs: `AttentionModel.encode`, `precompute_decoder`, `decode_step(return_glimpse=True)`, `StateTSP.update`, `ValueHead`. **No core-module refactor is required** for Stage 2; only new files under `src/am_baseline/search/` and `src/scripts/`.

---

## Correctness Invariants (asserted in `scripts/smoke_mcts.py`)

These are the non-negotiables the implementation must satisfy. Any regression that breaks one of these is a bug, regardless of what the K-curve looks like.

1. **No edge double-counting at non-terminal leaves.** At a non-terminal leaf with state `s`:
    ```
    total_norm = state.lengths / bl_val + v(s)
    ```
    where `v(s)` is the value head's output and was trained on the V_CURRENT target (`utils.tensor_ops.value_targets_from_edges` — remaining cost from `s` including upcoming edge and closing edge; `state.lengths` = already-traversed cost at `s`). The sum is the **total normalized tour cost estimate**; no edge appears in both terms.

2. **Closing edge included exactly once at terminal leaves.** `state.get_final_cost() = state.lengths + first_to_last_edge`. Terminal leaves use this; non-terminal leaves use `state.lengths + v(s) * bl_val` and the closing edge is embedded in `v`.

3. **Single-agent, no sign flip.** Q is stored as `-total_normalized_cost` (higher Q = lower tour cost = better). Backup propagates this sign unchanged along the selection path. Unlike two-player games, there is no per-depth sign flip.

4. **Priors are a proper distribution over legal actions.** After every `_fill_priors_from_logp` or `_apply_dirichlet` call:
    - `node.P` contains entries for exactly the legal (non-masked) actions.
    - `Σ_a node.P[a] == 1` within fp tolerance.
    - No NaN, no negatives.
    - Fallback: if the softmax underflowed (sum ≤ 0 or non-finite), replace with uniform over legal actions.

5. **K=0, τ=0 matches model greedy decode exactly.** With no simulations, `root.N` is empty; `_pick_root_action` explicitly falls back to `argmax_a P(root, a)` — which is precisely what the decoder's greedy mode does.

---

## Design Summary

### What Stage 2 delivers

A `MCTSSolver` that, given a trained `AttentionModel` checkpoint and a batch of TSP instances, produces feasible tours with quality better than the model's own greedy decode — measured across simulation budgets {50, 100, 200, 400, 800}.

### Tree structure — one tree per instance

- **Node** = partial tour state. Stores:
  - State (StateTSP NamedTuple)
  - Per-action priors `P`, visit counts `N`, total backup `W`, running Q
  - Children dict (lazy)
  - `v_estimate` — normalized remaining cost returned by the value head when this node was expanded (used by `fpu_mode='node_value'`)
- **Root** = initial state for each tour-step. Built fresh each step unless `tree_reuse=True`.
- **Leaf** = unexpanded node (empty P).

### Selection — PUCT

```
a* = argmax_a [ Q(s,a) + c_puct · P(s,a) · √(Σ_b N(s,b)) / (1 + N(s,a)) ]
```

**Critical tuning for routing:** `c_puct=0.05` is canonical for TSP (not the AlphaGo default 1.0). Justification: on a near-optimal trained policy, Q differences between root actions are ~0.01 (normalized cost scale) while PUCT's U term is ~0.2 at c_puct=1.0, so U completely dominates and MCTS collapses to the policy's argmax. Monotone c_puct sweep in §Tuning confirms this.

### FPU — Q_init for unvisited actions (swept hyperparameter)

Three modes, configurable via `cfg.fpu_mode`:

- **`fallback`** (constant): every unvisited action gets `Q_init = cfg.fpu_fallback`. Useful for sweeping {0.0, −0.5, −1.0}.
- **`running_q`** (AlphaZero standard, **default**): if the node has been visited (total_N > 0), `Q_init = sum(W)/sum(N)` at that node (running mean). Else fall back to `cfg.fpu_fallback`. Makes unvisited actions inherit the node's current quality estimate.
- **`node_value`**: `Q_init = -v(node)` — read the node's cached `v_estimate` set at expansion. Leans directly on the value head's signal.

**Default `fpu_fallback = -1.0`** (not 0.0). Realistic Q for TSP completions is `~-1` (normalized cost ≈ 1, negated). Setting `fpu_fallback=0.0` makes unvisited actions look artificially better than any visited one (Q=0 > Q=-1), causing MCTS to spread breadth-first and never deepen — observed early-on as +1% regression vs greedy; fixed by changing the default.

### Expansion — at first visit to a leaf

1. `decode_step(fixed, state, return_glimpse=True)` → `(log_p, mask, glimpse)`.
2. Exponentiate log_p; **mask out infeasibles**; **renormalize** over legal actions with NaN / zero-sum safety (fallback to uniform).
3. Populate `node.P` for legal actions only.
4. For `leaf_eval='value_head'`: `v = value_head(glimpse)` (normalized cost-to-go).
5. For `leaf_eval='rollout'`: greedy rollout from state to terminal; `v = remaining_real / bl_val`.
6. Cache `v` as `node.v_estimate` for `fpu_mode='node_value'`.

### Evaluation + backup — along the selection path

At terminal leaves: `total_norm = state.get_final_cost() / bl_val`.
At non-terminal leaves: `total_norm = state.lengths / bl_val + v(state)`.
Backup: `W[a] += -total_norm; N[a] += 1; Q[a] = W[a]/N[a]` for each edge on the selection path. No depth-based sign flip.

### Root action selection — diagnostic-configurable

- **`visits`** (AlphaGo default, **default**): argmax `N` at root. Robust in the large-K limit. Ties broken by action index.
- **`q`** (diagnostic): argmax `Q` among visited actions. Exposes cases where visit-count heuristic is noisy but Q estimates are informative.

Used to diagnose "prior over-dominance vs FPU vs c_puct" failure modes. If `visits` performs poorly but `q` performs well → prior/c_puct/FPU tuning issue.

### Dirichlet noise at root — hooks only in Stage 2

`P(root, a) ← (1−ε) P(root, a) + ε · η_a, η ~ Dir(α)`. Default `ε=0`. Renormalized after mixing.

### Temperature (τ) — root action sampling

- `τ = 0`: argmax (Stage 2 / Stage 3 test-time default).
- `τ > 0`: sample ∝ N^(1/τ) (Stage 4 training-time exploration).

### Tree reuse (Phase A.5 — configurable)

After picking action `a` at tour-step k, the subtree rooted at `root.children[a]` is already populated with N/W/Q from sims that descended through it. TSP transitions are deterministic, so this subtree is exactly the correct successor state for tour-step k+1.

- Default: **off** (`cfg.tree_reuse=False`) for simplest correctness.
- When on: `root = root.children[a]; root.parent = None; root.action_into_me = None`. Retain statistics below.
- When off: discard the whole tree, rebuild fresh root for the next step.

Trade-off: reuse saves compute AND provides warmer Q estimates at deeper nodes. Cost: prior N counts bias early simulations at the new root. Both are fine in the limit; tune empirically.

---

## Scope — What IS and IS NOT in Stage 2

### IS

- `MCTSNode`, `MCTSSolver`, `MCTSConfig` with all knobs above.
- Prior renormalization with NaN / zero-sum safety.
- FPU modes: `fallback`, `running_q`, `node_value`.
- Root action selection: `visits`, `q`.
- Tree reuse (optional).
- Dirichlet noise hooks (disabled by default).
- CLI `scripts/run_mcts.py` exposing all knobs.
- Smoke test `scripts/smoke_mcts.py` with the 8 correctness checks listed below.
- Sim-budget curve validation at the chosen canonical config.
- Leaf-eval ablation `value_head` vs `rollout`.

### IS NOT (deferred)

- Training with MCTS — **Stage 4**. Stage 4's loss is `value_loss + policy_distillation_loss` where `policy_distillation_loss = -Σ_a π_t(a) log p_θ(a|s_t)` and π is the MCTS visit distribution. This is AlphaGo-Zero-style policy iteration. If we instead use MCTS-weighted REINFORCE (advantages from MCTS), we'll call it *MCTS-enhanced REINFORCE* — not AlphaGo-Zero-style. The naming matters: the distinction is a whole category of design choice.
- Self-play data generation loop — Stage 4.
- Head-to-head vs sampling-1280 benchmark curve — Stage 3.
- Virtual-loss parallel MCTS within a single tree — future optimization.
- Cross-tree leaf batching — future optimization.
- Dirichlet α, ε tuning — Stage 4.
- CVRP — Stage 6.

---

## Normalization and `bl_val`

Value head trained with `value_target_norm='bl'`: targets were `realized_cost_to_go / bl_val` where `bl_val` is per-instance greedy rollout cost from the frozen baseline model. At test time we use **`bl_val = greedy decode cost of the trained model itself`**, computed once per instance in a batched pre-pass.

All internal MCTS math is in normalized space; real units only at final reporting via `state.get_final_cost()`.

Fallback: `--value_norm sqrt_n` → `bl_val = sqrt(graph_size)`.

---

## Architecture

```
src/am_baseline/
  search/                       # NEW
    __init__.py                 # Exports MCTSSolver, MCTSNode, MCTSConfig, select_action
    tree.py                     # MCTSNode (with v_estimate cache)
    mcts.py                     # MCTSConfig, MCTSSolver
    puct.py                     # select_action (takes fpu_value from caller)

src/scripts/
  run_mcts.py                   # CLI with all knobs
  smoke_mcts.py                 # A1..A8 correctness + mechanics checks
```

No changes to `model/`, `problem/`, `training/`, `config.py`.

---

## Key Files to Modify / Create

| File | Status | Change |
|---|---|---|
| `src/am_baseline/search/__init__.py` | NEW | Export `MCTSSolver`, `MCTSNode`, `MCTSConfig`, `select_action` |
| `src/am_baseline/search/tree.py` | NEW | `MCTSNode` with N/W/Q/P/children dicts + `v_estimate` cache + `running_value()` helper |
| `src/am_baseline/search/puct.py` | NEW | `select_action(node, c_puct, fpu_value)` — caller provides fpu |
| `src/am_baseline/search/mcts.py` | NEW | `MCTSConfig` (all knobs), `MCTSSolver` (solve_batch, solve_instance, _simulate, _expand, _populate_priors, _fill_priors_from_logp, _rollout_remaining_real, _pick_root_action, _apply_dirichlet, _fpu_value_for) |
| `src/scripts/run_mcts.py` | NEW | CLI wraps all MCTSConfig knobs + greedy comparison + CSV output |
| `src/scripts/smoke_mcts.py` | NEW | A1..A8 smoke (see below) |
| `src/scripts/dump_mcts_leaves.py` | NEW (Phase D step 13) | Runs `MCTSSolver` instrumented with a leaf-callback; emits JSONL of sampled `(input_idx, partial_pi, v_pred, step_index)` tuples for off-policy R² analysis |
| `src/scripts/eval_value.py` | MODIFY (Phase D step 13) | Add `--off_policy_states <jsonl>` mode: load dumped leaves, replay greedy rollout to terminal as ground-truth, report per-step / bucketed R² on the off-policy distribution |

Reused (no changes): `AttentionModel.{encode, precompute_decoder, decode_step}`, `ValueHead`, `StateTSP.{initialize, update, get_mask, all_finished, get_final_cost}`, `TSP.{make_state, get_costs, make_dataset}`, `utils/misc.py::{torch_load_cpu, load_model}`.

---

## Smoke Tests (`scripts/smoke_mcts.py`)

Eight assertions, each aborts the test on first failure:

| # | Assertion |
|:-:|:----------|
| A1 | All MCTS tours are valid permutations of `[0, N)` |
| A2 | **K=0 with τ=0 matches model greedy decode exactly** (tour equality + cost equality to fp tolerance) |
| A3 | `K=50` with `value_head` runs end-to-end; no NaN |
| A4 | `K=20` with `rollout` fallback runs end-to-end |
| A5 | **Near-terminal backup correctness**: 1 unvisited node → 1 simulation → `root.Q[a] == -(state.lengths + cur_to_last + last_to_start) / bl_val` within 1e-5 |
| A6 | **Prior renormalization invariant**: at a mid-tour node, `Σ_legal P(a) == 1` within 1e-6 |
| A7 | `tree_reuse=True` produces valid tours (strict equality not required — warmer statistics can shift cost slightly) |
| A8 | `root_select='q'` produces valid tours |

---

## Implementation Sequence

### Phase A — core MCTS (single-tree, sequential)

1. `search/tree.py::MCTSNode` — with `v_estimate` cache.
2. `search/puct.py::select_action` — takes `fpu_value` argument; caller's choice.
3. `search/mcts.py::MCTSConfig`, `MCTSSolver` — all knobs.
4. **Milestone A1**: `scripts/smoke_mcts.py` passes (A1..A6).

### Phase A.5 — tree reuse (optional, configurable)

5. Implement `cfg.tree_reuse`: at each tour-step, advance root into `root.children[a]` instead of discarding. Already in `solve_instance` loop.
6. **Milestone A7**: smoke A7 — tree reuse produces valid tours.

### Phase B — throughput optimization (only if measured wall-clock exceeds budget)

Deferred unless needed. Cross-tree leaf batching is the main lever. Requires refactoring `decoder._get_step_context`'s `state.i.item() == 0` check.

### Phase C — CLI

7. `scripts/run_mcts.py` — all knobs exposed; greedy comparison; CSV output.

### Phase D — validation

8. **TSP-20 required.** Stage 1 canonical checkpoint (`outputs/tsp_20/stage1_tsp20_canonical_20260423T103541`). 1000 instances, K ∈ {50, 100, 200, 400, 800}, seed=1234, c_puct=0.05, fpu_mode=running_q, fpu_fallback=-1.0, root_select=visits, tree_reuse=False.
9. **Leaf-eval ablation.** TSP-20 K=200, 1000 instances: `value_head` vs `rollout`.
10. **FPU diagnostic** (new). TSP-20 K=200, 100 instances: `fpu_mode ∈ {fallback(-1.0), running_q, node_value}` to confirm the default choice.
11. **Tree reuse diagnostic** (new). TSP-20 K=200, 500 instances: `tree_reuse ∈ {False, True}` to characterize the tradeoff.
12. **TSP-50 contingent.** Once `stage1_tsp50_with_value` finishes (Modal run `123x2qr5`), run K ∈ {50, 100, 200, 400}.
13. **Off-policy R² probe (NEW, motivated by per-step value diagnostic).** Quantifies the mechanism behind the rollout-vs-value_head gap (≈+15–23pp gap reduction in favor of rollout, observed in step 9 leaf-eval ablation). Plumbing:
    - Run a single MCTS pass at TSP-20 K=200 on 500 instances; while it executes, sample (with reservoir sampling, target ~5000 leaves) the `(state, v_pred, step_index)` triples at non-terminal leaf evaluations. State is captured as `(input_1, partial_pi)` so it can be replayed.
    - For each sampled state, compute the ground-truth normalized cost-to-go `v_true = greedy_rollout(state) / bl_val` (using the same `bl_val` MCTS used for that instance).
    - Report: per-step R²(`v_pred`, `v_true`) on this off-policy distribution; bucketed (early/mid/late); fractional error per step. **Comparison to Stage 1 in-distribution R²=0.9965.**
    - Expected: off-policy R² substantially worse than in-distribution at non-trivial steps (this is what Stage 4 is designed to fix). If the gap is small, the rollout-vs-value_head difference is *not* about distribution shift and we need a different explanation.
    - Implementation: extend `src/scripts/eval_value.py` with a `--off_policy_states <jsonl>` mode plus a new `src/scripts/dump_mcts_leaves.py` that runs `MCTSSolver` instrumented with a leaf-callback. Adds ~80 LoC; no MCTSSolver core changes (callback hook only).
14. **Root-leaf normalization ablation (NEW).** Run TSP-20 K=200 on 500 instances with `--value_norm sqrt_n` (already supported by the CLI, never exercised). Tests whether removing the `bl`-norm degeneracy at `v(s_0) ≈ 1.0` (per-step diagnostic finding: target_std at step 0 = 0.00000 under `bl` norm with greedy decoding) changes MCTS quality. Modest expected effect — root is one evaluation among many — but cheap and answers the question. Decision rule: if `sqrt_n` matches or beats `bl` at K=200, run a 1000-instance comparison at K=400; otherwise log and move on.

---

## Data Contracts

### `MCTSConfig` (dataclass, `search/mcts.py`)

```python
@dataclass
class MCTSConfig:
    n_simulations: int = 200
    c_puct: float = 0.05                # routing sweet spot
    temperature: float = 0.0            # 0 = argmax, >0 = sample from N^(1/τ)
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.0      # 0 = off
    leaf_eval: str = 'value_head'       # 'value_head' | 'rollout'
    value_norm: str = 'bl'              # 'bl' | 'sqrt_n'
    fpu_mode: str = 'running_q'         # 'fallback' | 'running_q' | 'node_value'
    fpu_fallback: float = -1.0          # realistic Q on TSP
    root_select: str = 'visits'         # 'visits' | 'q'
    tree_reuse: bool = False
    seed: Optional[int] = None
```

### `MCTSSolver.solve_batch`

- **Input:** `inputs: Tensor (B, N, 2)`.
- **Output:** `(costs: Tensor (B,), tours: Tensor (B, N))`.
- **Side effect:** none; model is `eval()`'d, queried read-only.

### Determinism

Deterministic when `dirichlet_epsilon == 0` and `temperature == 0`. RNG isolated to `numpy.Generator(seed)` for Dirichlet + temperature sampling.

---

## Success Criteria (Stage 2 done when)

**Hard criteria (gating):**

- [ ] All tours are valid permutations of `[0, N)`, each node visited exactly once (1000 TSP-20 instances; 1000 TSP-50 instances when available).
- [ ] Smoke test A1..A8 passes on a random model (mechanics + correctness, no checkpoint required).
- [ ] K=0 with τ=0 matches model greedy decode exactly (A2 asserts this; extends to full validation dataset).
- [ ] MCTS at some reasonable K ∈ {200, 400, 800} beats model greedy mean cost on the 1000-instance TSP-20 validation set (Δ < 0 statistically).
- [ ] `--leaf_eval rollout` fallback runs and produces valid tours.
- [ ] Full TSP-20 K-curve completes in < 6h on local GPU (or < 3h on A10).

**Soft / diagnostic expectations (not gating):**

- Mean cost **generally improves** with K and does **not systematically collapse** (regression at any K is a diagnostic flag, not a failure). Pure monotone non-increasing is not guaranteed with an imperfect value head and visit-count action selection.
- Gap-to-optimal reduction vs greedy at K=200: target ≥ 10–30% on TSP-20 (the plan's original 30% target was optimistic for a policy already at 0.3% gap; treat as a stretch goal on TSP-20 and a realistic target on TSP-50/100).
- Rollout leaf-eval should be within noise of value-head leaf-eval on Stage 1 quality (Stage 1 R² = 0.9965). Large quality gap → signal of value-head bias at MCTS-explored states.

**Deliverable:** `_progress/stage2_progress.md` with:
- Smoke test results ✅
- c_puct tuning table ✅
- FPU / root_select diagnostic tables ✅
- K-curve at canonical config ✅ (TSP-20 K ∈ {20, 50, 100, 200, 400, 800}; TSP-50 K ∈ {50, 100, 200, 400})
- Leaf-eval ablation ✅ (TSP-20 full rollout K-curve; TSP-50 K=100 rollout)
- Tree-reuse ablation ✅
- Wall-clock breakdown ✅ (TSP-50 wall-clocks partly contaminated by GPU sharing — cost data unaffected)
- TSP-50 results (contingent) ✅
- Decode-step micro-benchmark + rollout-vs-value_head wall-clock decomposition ✅ (added 2026-04-26)
- Stage 2 conclusions section ✅ (added 2026-04-26)

---

## Risks / Unknowns

1. **Normalization consistency.** Invariants #1 and #2 are the load-bearing cost-accounting claims. A1..A8 smoke test (especially A5) asserts these in code.
2. **FPU sensitivity.** Default `fpu_mode='running_q'` with `fpu_fallback=-1.0` is the calibrated choice after early-debugging showed `fpu_fallback=0.0` caused +1% regression. Phase D includes an explicit FPU diagnostic (Phase D step 10) to confirm the default on 100 instances before committing to it for the full curve.
3. **`c_puct` routing-specific.** Default 0.05 (not AlphaGo's 1.0). Swept in Phase D as part of tuning.
4. **Value head bias at off-policy states (sharpened post per-step diagnostic).** Stage 1 R² was measured on the policy's own greedy trajectories — the `eval_value.py` per-step diagnostic added 2026-04-24 surfaced two structural concerns that compound off-policy:
    - **Step-0 / step-1 are degenerate under `bl` norm + greedy decoding.** At eval time the trajectory equals the baseline (both are this model's greedy decode), so `target[0] = greedy_cost / bl_val ≈ 1.0` exactly (measured `target_std = 0.00000` on TSP-20 and TSP-50). The value head learns to emit ≈1.0 trivially at the root; `v(s_0)` carries **zero information about which instance is harder**. Mitigation: instance comparison at the root must un-normalize via `bl_val`. MCTS at the root degrades to "leaf eval contributes the constant `bl_val/bl_val=1`," which is harmless for action ranking *within* an instance but means the value head adds no discriminative signal at the root.
    - **Fractional error grows monotonically through the tour even in-distribution.** TSP-20: 0.4% at step 0 → 16% at step N-1; TSP-50: 0.4% → 32%. Bucketed R² hides this because target variance shrinks late-tour, inflating the R² number even as proportional error gets worse. The value head is *least* trustworthy at the regime that should be easiest, and this is before any off-policy shift.
    - **Diagnosis path:** Phase D step 13 (off-policy R² probe) measures the additional shift on top of these in-distribution issues. Empirical signal already in hand: the leaf-eval ablation (Phase D step 9) showed rollout uniformly beats value_head by +15–23pp gap reduction, consistent with off-policy bias being the dominant Stage 2 ceiling. The probe quantifies it; the fix is Stage 4.
5. **V_CURRENT alignment.** Value head target semantics: `v(s_k)` predicts remaining cost from `s_k` INCLUDING the upcoming edge at step k (V_CURRENT semantics, fixed in Stage 1). MCTS expects exactly this. A5 smoke test asserts alignment at near-terminal states.
6. **Tree reuse stats carryover.** Reused subtrees have N counts from prior sims. Those counts bias early PUCT at the new root. Bound by K anyway — negligible over many tour-steps. Phase D step 11 characterizes the effect.
7. **K-curve plateau.** TSP-20 greedy is already at ~0.3% gap to Gurobi optimal; the value-head ceiling limits MCTS improvement regardless of K. Larger gap-reduction targets are realistic on TSP-50/100 where greedy has more headroom.
8. **TSP-50 checkpoint contingency.** Currently in-flight (Modal `apy5m2lf`, `123x2qr5`, epoch ~57/100 at plan revision time). Stage 2 TSP-20 dev proceeds in parallel.
9. **Checkpoint value-head presence.** `--no_value` checkpoints (e.g., Stage 1 TSP-50 AM-baseline) have no trained value head — MCTS refuses with a clear error when `leaf_eval='value_head'` (`MCTSSolver.__init__` raises ValueError).

---

## Verification Commands

```bash
conda activate AM_AlphaGoZero

# Smoke (A1..A8)
PYTHONPATH=src python -m scripts.smoke_mcts

# Headline TSP-20 K-curve (canonical config)
for K in 50 100 200 400 800; do
  PYTHONPATH=src python -m scripts.run_mcts \
    --model outputs/tsp_20/stage1_tsp20_canonical_20260423T103541/epoch-99.pt \
    --graph_size 20 --val_size 1000 --seed 1234 \
    --n_simulations $K --c_puct 0.05 --temperature 0.0 \
    --leaf_eval value_head --fpu_mode running_q --fpu_fallback -1.0 \
    --root_select visits \
    --output_csv outputs/stage2/tsp20_K${K}_canonical.csv \
    --no_progress_bar
done

# Leaf-eval ablation (Phase D step 9)
PYTHONPATH=src python -m scripts.run_mcts \
  --model outputs/tsp_20/stage1_tsp20_canonical_20260423T103541/epoch-99.pt \
  --graph_size 20 --val_size 1000 --seed 1234 \
  --n_simulations 200 --c_puct 0.05 --leaf_eval rollout \
  --output_csv outputs/stage2/tsp20_K200_rollout.csv \
  --no_progress_bar

# FPU diagnostic (Phase D step 10)
for fpu_mode in fallback running_q node_value; do
  PYTHONPATH=src python -m scripts.run_mcts \
    --model outputs/tsp_20/stage1_tsp20_canonical_20260423T103541/epoch-99.pt \
    --graph_size 20 --val_size 100 --seed 1234 \
    --n_simulations 200 --c_puct 0.05 --fpu_mode $fpu_mode --fpu_fallback -1.0 \
    --output_csv outputs/stage2/tsp20_K200_fpu_${fpu_mode}.csv \
    --no_progress_bar
done

# Tree reuse diagnostic (Phase D step 11)
PYTHONPATH=src python -m scripts.run_mcts \
  --model outputs/tsp_20/stage1_tsp20_canonical_20260423T103541/epoch-99.pt \
  --graph_size 20 --val_size 500 --seed 1234 \
  --n_simulations 200 --c_puct 0.05 --tree_reuse \
  --output_csv outputs/stage2/tsp20_K200_treereuse.csv \
  --no_progress_bar

# Off-policy R^2 probe (Phase D step 13) — two-step:
# (a) dump sampled MCTS leaves
PYTHONPATH=src python -m scripts.dump_mcts_leaves \
  --model outputs/tsp_20/stage1_tsp20_canonical_20260423T103541/epoch-99.pt \
  --graph_size 20 --val_size 500 --seed 1234 \
  --n_simulations 200 --c_puct 0.05 --tree_reuse \
  --reservoir_size 5000 \
  --output_jsonl outputs/stage2/tsp20_K200_offpolicy_leaves.jsonl
# (b) score against greedy-rollout ground truth
python src/scripts/eval_value.py \
  --model outputs/tsp_20/stage1_tsp20_canonical_20260423T103541/epoch-99.pt \
  --off_policy_states outputs/stage2/tsp20_K200_offpolicy_leaves.jsonl \
  --no_cuda

# Root-leaf normalization ablation (Phase D step 14)
PYTHONPATH=src python -m scripts.run_mcts \
  --model outputs/tsp_20/stage1_tsp20_canonical_20260423T103541/epoch-99.pt \
  --graph_size 20 --val_size 500 --seed 1234 \
  --n_simulations 200 --c_puct 0.05 --tree_reuse \
  --value_norm sqrt_n \
  --output_csv outputs/stage2/tsp20_K200_value_norm_sqrtn.csv \
  --no_progress_bar
```

Expected TSP-20 canonical K=200: mean cost ≤ 3.84 (model greedy baseline 3.8424); gap to Gurobi optimal ≤ 0.9%; wall-clock ~25 min on local GPU.

---

## Stage 4 Forward Reference (language discipline)

**AlphaGo-Zero-style training** ≠ **MCTS-enhanced REINFORCE**. The distinction is not cosmetic:

- **AlphaGo-Zero-style (policy iteration via distillation):**
    ```
    L = value_loss + policy_distillation_loss
    value_loss                = || v_θ(s_t) - (normalized cost-to-go)_t ||²
    policy_distillation_loss  = - Σ_a π_t(a) · log p_θ(a | s_t)
    ```
    where `π_t(a)` is the MCTS visit distribution at step `t` — **strictly stronger than the raw network policy**. Training `p_θ → π_t` distills MCTS's improvements back into the network. This is the cycle that powered AlphaGo Zero's Elo gap.

- **MCTS-enhanced REINFORCE** (a different design): keep REINFORCE's policy-gradient estimator, but use MCTS to either refine the sampled trajectory (e.g., sample from π_t instead of p_θ) or refine the baseline. This is a weaker form of improvement and should not be called AlphaGo-Zero-style.

Stage 4's plan must commit to one of these and use the correct language. Stage 2 does not commit.

**Why Stage 4 is well-motivated by Stage 2's findings (added 2026-04-24):** the per-step value diagnostic + leaf-eval ablation together show that the value head's failure mode at search time is **distribution shift onto MCTS-visited states**, not poor in-distribution accuracy (Stage 1 R²=0.9965 is real but measured on the policy's own greedy trajectories). Stage 4's policy-iteration loop is exactly the fix: training the value head against `z` (realized cost-to-go) on MCTS-rolled-out trajectories puts the head on the distribution it failed on in Stage 2. The Phase D step 13 off-policy R² number is the predicted target for "how much should Stage 4 close the gap." If Stage 4 closes it, the value-head leaf-eval should reach (or pass) the rollout leaf-eval ceiling, which retires the +15–23pp deficit measured in Stage 2.
