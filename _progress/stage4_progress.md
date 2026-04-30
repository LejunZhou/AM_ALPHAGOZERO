# Stage 4 Progress: AlphaGo-Zero-Style MCTS Self-Improvement Loop on TSP-20

**Plan:** `_plans/stage4_plan.md`
**Started:** 2026-04-29
**Last updated:** 2026-04-30 — Phases A, B, C, D, E all complete. Wave 1 (A+B), Wave 2 (C+E), and Wave 3 Phase D merged on this worktree.
**Status:** **Phases A, B, C, D, E complete; Phase F (TSP-20 pilot/main CLI) and G (ablations) remain.**

---

## Plan refinement log

**2026-04-30** — External review surfaced five corrections; all folded into plan + spec before Phase A starts:

| # | Concern | Status | Resolution |
|---|---|---|---|
| 1 | **Value-target double-count** — broadcast `z = tour_cost / bl_val` in original plan would double-count path cost at MCTS time, since the leaf evaluator (`mcts.py:1-15`) computes `state.lengths/bl_val + v(state)` assuming v predicts V_CURRENT cost-to-go | **CORRECTED** | Stage 4 now trains on per-state `z_t = (tour_cost − lengths_t) / bl_val`, reusing Stage 1's `value_targets_from_edges` (`utils/tensor_ops.py:57-78`). Spec §3 + plan Phase B.3 + Phase C.2 all updated. AGZ canonical mapping table row 3 marked "Adapt (forced by domain)" |
| 2 | **`step50` vs `step30` inconsistency** — recommended-approach paragraph and Phase C.1 snippet still said `step50`/"first 50%"; later sections + progress all said `step30` | **FIXED** | All three stale references scrubbed in `stage4_plan.md`. Default everywhere is `step30` (closest to AGZ "first 30 of ~250 plies ≈ 12%" scaled to TSP) |
| 3 | **Policy-target temperature ambiguity** — was unclear whether π_t in the buffer used τ-tempered visits (strict AGZ — late-game one-hot) or raw τ=1 normalized | **DECIDED — choice (B): decoupled** | Action selection σ_t uses `step30`; **training target π_t = N/Σ N always (τ=1)**. Spec §4.2 split into σ_t and π_t; AGZ mapping row 3b added with rationale. Strict-AGZ coupling becomes G.4 sub-ablation. Smoke A3 split: σ_t entropy decays at step ⌈0.3N⌉; π_t entropy stays bounded above zero |
| 4 | **Sample-efficiency claim wording** — original criterion 1 was trivially true at x=128M (Stage 4 warm-starts from Stage 1 endpoint) | **REWORDED** | Acceptance criterion split into 1a (marginal: warm-start improvement, ≥0.001 better than starting checkpoint) and 1b (strict total: combined Stage 1+4 curve at matched x). Strict from-scratch sample efficiency moved to Stage 5 stretch goal |
| 5 | **Replay-buffer memory blow-up** — deque-of-Python-objects with millions of small tensors costs ~2.4 GB in tensor headers alone | **REDESIGNED** | Phase B.1 rewritten as flat dict-of-pre-allocated-tensors (KataGo `shuffle.py` pattern). Fixed ~520 MB footprint. Ring-buffer eviction with `inst_idx % capacity_instances` + tuple slot computation |

---

**2026-04-29** — Re-read AGZ paper Methods + Ext. Data Table 3 and walked `ref/KataGo-master/SelfplayTraining.md` + `shuffle.py`. Surfaced four deviations from AGZ-canonical that the original plan had implicit and should be tracked explicitly:

| Knob | Original plan default | AGZ canonical | Action taken |
|---|---|---|---|
| Optimizer | Adam, lr=1e-4 | SGD+momentum 0.9, lr step-anneal {1e-2 → 1e-3 → 1e-4} (Methods §Optimization; Ext. Data Table 3) | **Kept Adam** as F.4 default (matches Stage 1 warm-start; switching optimizers from a converged checkpoint risks destabilizing). Added **G.8 ablation** for SGD+momentum if F.4 plateaus |
| Temperature schedule | `step50` (first 50 % of N steps) | τ=1 for first 30 of ~250 game moves ≈ 12 % (Methods §Self-play) | **Changed default to `step30`** (first ⌈0.3·N⌉ = 6/20 steps for TSP-20) — closer to AGZ proportional. `step50` retained as G.4 ablation lever |
| Replay buffer | 200K instances flat (200-iter window @ 1K/iter) | last 500K games (~20-iter window @ 25K/iter) | **F.3 pilot now uses `buffer_capacity=50_000`** (~50-iter window, AGZ-proportional). Main run scales to 200K only if pilot is stable |
| Symmetry augmentation | (not in plan) | Random dihedral leaf-aug (8-fold) at every NN eval (Methods §Search algorithm) | Added **G.7 ablation**: continuous SO(2) rotation + axis flip on coords pre-encoder |

Other AGZ details that were already correctly handled (no change): equal-weighted MSE+CE loss (eq. 1), c=1e-4 L2, c_puct=0.05, ε=0.25 Dirichlet noise, best-player tracking with continuous-trainer optimizer, no-rollback-on-reject (scope decision 3 matches Methods §Self-play training pipeline). Full canonical mapping is in plan's "AGZ canonical mapping" section with paper-page citations.

KataGo cross-references added to plan's footer: `MAX_TRAIN_PER_DATA=8` cap (we sit at 5.1×), power-law windowing in `shuffle.py` for Stage 5, and confirmation that gating is optional (supports scope decision 3 and G.5.c ablation).

---

## Implementation Progress

### Phase A — Visit-distribution exposure (foundation; no GPU) — **COMPLETE 2026-04-30**

- [x] **A.1** Python visit hook in `src/am_baseline/search/mcts.py` — `MCTSConfig.return_root_visits` flag + `MCTSSolver.root_visit_dists` side-effect attribute populated per tour-step. Snapshot taken AFTER `_pick_root_action` and BEFORE the tree-reuse advance via `dict(root.N)` (immutable copy).
- [x] **A.2** C++ sequential — `Config::return_root_visits` flag in `mcts.hpp`; `Solver::solve_instance` collects `root->n_visits` per step into `std::vector<std::vector<std::pair<int,int>>>` and emits as `result["root_visit_dists"]` py::list. `CppMCTSSolver.solve_instance` plumbs back as `self.root_visit_dists: list[dict[int,int]]`. No bindings change needed (the flag rides through `Config::from_python` and the new field rides through the existing result dict).
- [x] **A.3** C++ batched — same pattern in `BatchInstance::collect_request` per-instance; `BatchSearch::results()` emits `root_visit_dists_per_instance: list[list[list[(int,int)]]]`. `CppBatchMCTSSolver._solve_chunk` returns `stats["root_visit_dists"]: list[list[dict]]`; `solve_batch` populates `self.root_visit_dists_per_instance`. `solve_instance` (single-instance entry) also exposes the chosen instance's dists via `self.root_visit_dists`.
- [x] **A.4** Validation: A13 added to `src/scripts/smoke_mcts.py` covering:
  - **A13.a** legality invariants (i)-(vi) on `solver.root_visit_dists` for `value_head` and `rollout` leaf eval, on python / cpp / cpp_batch (production config: tree_reuse=True, K=200, ε=0, τ=0).
  - **A13.b** exact-count `Σ_a N(s_t, a) == K` at every step under `tree_reuse=False, K=200`, on python and cpp.
  - **A13.c** deterministic-clamp bit-equivalence: python vs cpp and cpp vs cpp_batch produce identical per-step visit dicts (integer counts, no fp drift) under ε=0, τ=0, K=200.
- All three backend smokes pass (`smoke_mcts.py --backend python|cpp|cpp_batch`).
- C++ extension rebuilt cleanly; pybind11 wire format only adds an optional dict key — Stage 2/3 callers see no behavioral or memory change with `return_root_visits=False` (default).
- **Implementation note (sparse storage):** Both Python and C++ store only actions with N>0 (sparse `{action: count}`). Phase C must zero-init `(N,)` arrays and scatter when building dense `pi_t`. Visited cities are never keys (legality invariant ii from A13.a).
- **Implementation note (tree reuse):** Under tree_reuse=True the per-step total exceeds K (root inherits subtree visits) — this is the correct π_t to distill. Phase C sanity assertions should not assume `Σ N == K` outside `tree_reuse=False`.

### Phase B — Replay buffer + distillation training step (no GPU) — **COMPLETE 2026-04-30**

- [x] **B.1** `MCTSReplayBuffer` class in `src/am_baseline/training/coach.py` — flat dict-of-pre-allocated-tensors (per-instance: coords, bl_val FROZEN, tour_cost; per-step: pi raw τ=1, visited, first/prev_a, lengths, cost_to_go, inst_idx). ~520 MB fixed footprint. Ring-buffer eviction.
- [x] **B.2** State-tensor reconstruction utility (mirrors `mcts_cpp/solver.py:_state_from_snapshot`).
- [x] **B.3** `train_step_alphazero` in `src/am_baseline/training/trainer.py` — per-state π_t distillation (raw τ=1) + per-state z_t = cost_to_go/bl_val (V_CURRENT shape, matches `value_targets_from_edges`). Squeeze log_p/mask from (B,1,N) to (B,N) before CE.
- [x] **B.4** Smoke A1 in `src/scripts/smoke_alphazero.py`. A1.5/A1.6 buffer invariants + save/load round-trip + state-reconstruction round-trip all green.

(Detailed closure note in **Results → Phase B** below.)

### Phase C — Self-play data generator (no GPU) — **COMPLETE 2026-04-30**

- [x] **C.1** `make_self_play_config(graph_size, n_simulations) -> MCTSConfig` in `coach.py`. AGZ-canonical preset: `leaf_eval='value_head'` + `value_norm='bl'` (Stage 3 E.2 explicitly allows this combo), `c_puct=0.05`, base τ=1, `temperature_schedule='step30'` (Phase E), `dirichlet_alpha=10/N`, `dirichlet_epsilon=0.25`, `fpu_mode='running_q'`, `tree_reuse=True`, `return_root_visits=True`.
- [x] **C.2** `generate_self_play_batch(model, M, graph_size, cfg, device) -> list[InstanceRecord]` in `coach.py`. Drives `CppBatchMCTSSolver` with `return_root_visits=True`. `bl_val` is computed once via greedy decode under θ★ and frozen across the batch (passed to `solve_batch(bl_vals=...)` so the solver does not silently recompute). cost-to-go targets derived from realized edge costs through Stage 1's `value_targets_from_edges` (V_CURRENT convention; closing edge counted exactly once). Pack matches `MCTSReplayBuffer.push_instance`'s per-step schema verbatim. Helper functions `_compute_edge_costs`, `_mask_from_tour`, `_normalize_visit_dict` (the last one asserts on `total == 0` rather than fabricating a uniform fallback).
- [x] **C.3** Smoke A2 in `src/scripts/smoke_alphazero.py`: M=10, N=20, K=20, `temperature=0`, `dirichlet_epsilon=0`. Verifies (i) `bl + value_head` accepted by `_validate_config`, (ii) per-step π_t sums to 1 and is non-negative, (iii) zero mass on visited cities, (iv) `argmax(π_t) == tour[t]` (recovered from `visited[t+1] \ visited[t]`), (v) `cost_to_go[0] == tour_cost`. Records also push cleanly into a `MCTSReplayBuffer` confirming end-to-end schema compatibility. Wall-clock: ~2 s for the full A2 case on CPU.

### Phase D — `MCTSCoach.learn` orchestrator (no GPU) — **COMPLETE 2026-04-30**

- [x] **D.1** `MCTSCoach` class in `src/am_baseline/training/coach.py`. Wires `generate_self_play_batch` → `MCTSReplayBuffer.push_instance` → `train_step_alphazero` → `validate` → `RolloutBaseline.epoch_callback` per plan lines 322-378. Uses `Adam(lr=opts.lr_model, weight_decay=opts.weight_decay)` over the working copy; `best_model = copy.deepcopy(model)` is the self-play / gating reference. **Init-order trap (caught at review)**: `RolloutBaseline.__init__` is constructed *after* `opts.val_size` is captured in opts; verified experimentally that mutating `opts.val_size` later is silently ignored, so the coach must not depend on post-init mutations. Per scope decision 3, no rollback on reject. Buffer add path is the per-record `for r in records: self.buffer.push_instance(r.coords, r.bl_val, r.tour_cost, r.per_step)` loop because Phase B's API is per-instance, not a batched `add(...)`.
- [x] **D.2** Logging extensions in `src/am_baseline/training/logging.py`. New `iterations.csv` with the columns required by the plan: `iter, total_instances, val_avg_cost, policy_loss_mean, value_loss_mean, mean_entropy_pi, gated, accepted, mcts_wall_s, train_wall_s, buffer_size`. Two new methods: `log_alphazero_step(metrics, iter_idx, step)` accumulates running means within an iteration; `log_iteration(...)` flushes them to CSV + W&B + console. New W&B step axis `iteration` plus a sample-efficiency series (`val_avg_cost_vs_instances`) keyed off `total_instances`. CSV is opened lazily on first call so Stage 1 callers (which don't use `log_iteration`) see no behavioral change. Honors `opts.no_wandb` / `wandb_mode='disabled'` by passing `wandb_project=None`.
- [x] **D.3** Checkpoint format + resume. `MCTSCoach.save_checkpoint(tag)` writes `{model, best_model, optimizer, iter_idx, total_instances_seen, rng_state}` to `outputs/.../iter-{tag}.pt`; the (large) replay buffer is written separately to `outputs/.../buffer.pt` (overwritten each iteration). `MCTSCoach.load_checkpoint(path)` restores all of the above plus the buffer (best-effort — warns and continues if `buffer.pt` is missing, since the buffer refills in O(1 iter)). The stored `iter_idx` is the LAST COMPLETED iteration; `load_checkpoint` advances it by one so a follow-up `learn(...)` resumes at the next integer.
- [x] **D smoke** A5 + A6 added in `src/scripts/smoke_alphazero.py`:
  - **A5**: `gate_every=10, n_iterations=3` — patches `epoch_callback` with a counting stub; asserts call count == 0 + verifies `iterations.csv` reports `gated=0, accepted=''` for all 3 rows.
  - **A6**: `M=10, K=20, gate_every=2, n_iterations=3` — verifies (i) no NaN in any iterations.csv row, (ii) val_avg_cost is finite, (iii) at least one gating decision logged (iter 1), (iv) checkpoint round-trip restores model + best_model + iter_idx + total_instances_seen to bit-identical state and a follow-up `learn(1)` advances to the next iter cleanly.
- Wall-clock for the full Phase B + C + D smoke (B0, B1, A1b, A1, A2, A5, A6): ~7 s on CPU.

### Phase E — Temperature schedule + Dirichlet noise (no GPU)

- [x] **E.1** `MCTSConfig.temperature_schedule` (`None` | `'const'` | `'step30'` | `'step50'`); Python + C++ wiring. Default kept at `None` (≡ `'const'` ≡ existing scalar `cfg.temperature`) to preserve Stage 2/3 caller behavior; `'step30'` becomes the documented self-play default and is opted into by the Phase C self-play preset.
- [ ] **E.2** Dirichlet noise CLI flags exposed (ε, α via `--dirichlet_epsilon`, `--dirichlet_alpha_factor`). *(Deferred to Phase G — wiring already verified via existing `MCTSConfig.dirichlet_*` fields and confirmed by A14.c: ε=0.25 + step30 produces seed-divergent early-step actions on TSP-20.)*
- [x] **E.3** Smoke A14 added in `src/scripts/smoke_mcts.py`. Sub-checks:
    - **A14a** unit-tests `MCTSSolver._resolve_tau` for None/'const'/'step30'/'step50' on N=20 (cutoffs 6 and 10).
    - **A14b** `temperature_schedule='garbage'` raises `ValueError`.
    - **A14c** TSP-20 K=50 self-play with τ=1, ε=0.25, schedule='step30' instruments `_pick_root_action` and confirms τ=1 for steps 0–5 and τ=0 for steps 6–19.
    - **A14d** Two seeds on the same instance produce different first-6 actions (sampling engaged); same seed is bit-exactly reproducible.
    - **A14e** None/'const'/'step50' plumb through end-to-end on the Python solver.
    - **CPP A14** mirror smoke: all four schedule values plumb cleanly through `CppMCTSSolver` and `CppBatchMCTSSolver` on a 4-instance batch.

**Phase E completion note (2026-04-30):** Edits landed in this worktree:
| File | LOC delta | Change |
|---|---|---|
| `src/am_baseline/search/mcts.py` | +49 | Added `MCTSConfig.temperature_schedule`, `VALID_TEMPERATURE_SCHEDULE`, validation, `MCTSSolver._resolve_tau` static helper, per-step τ lookup in `_pick_root_action`. |
| `src/am_baseline/search/mcts_cpp/mcts.hpp` | +9 | Added `Config::temperature_schedule` (int 0/1/2 encoding) and `Solver::tau_per_step_` member. |
| `src/am_baseline/search/mcts_cpp/mcts.cpp` | +35 | Anonymous-namespace `build_tau_per_step` helper; populated `tau_per_step_` at solve_instance entry; `pick_root_action` and `batch_pick_root_action` now index `tau_per_step` for τ; `BatchInstance` carries its own `tau_per_step`. |
| `src/am_baseline/search/mcts_cpp/solver.py` | +18 | `_SCHEDULE_TO_INT` map + translation in `_cfg_dict`. |
| `src/scripts/smoke_mcts.py` | +135 | A14 docstring + 5 sub-checks (Python) and CPP/CPP_BATCH A14 schedule plumbing. |

Defaults are unchanged: `MCTSConfig.temperature_schedule = None` ≡ `'const'` ≡ existing scalar `cfg.temperature` behavior — Stage 2/3 callers see no behavioral change. Dirichlet wiring (E.2) was verified end-to-end via A14.c (ε=0.25 + step30 + τ=1 yields seed-divergent first-6 actions).

### Phase F — TSP-20 pilot + main run (~2.5 h compute) — **dev portion (F.1, F.2, F.5) COMPLETE 2026-04-30**

- [x] **F.1** `src/scripts/train_alphazero.py` CLI (~225 LOC). All required flags landed: `--load_path` (required), `--n_iterations`, `--M_instances`, `--n_simulations_train`, `--buffer_capacity`, `--train_steps_per_iter`, `--batch_size`, `--gate_every`, `--temperature_schedule {const,step30,step50}`, `--dirichlet_epsilon`, `--dirichlet_alpha_factor`, `--lambda_v`, `--weight_decay`, `--lr_model`, `--leaf_eval {value_head,rollout}`, `--resume_from`, `--graph_size`, `--val_size`, `--no_wandb`, `--run_name`. Output dir convention: `outputs/tsp_<graph_size>/<run_name>_<timestamp>/`. Construction order is faithful to plan F.1: parse → finalize opts → load Stage 1 ckpt via `torch_load_cpu` → make `val_dataset` → construct `MCTSCoach(model, problem, opts, val_dataset, device)` (after `opts.val_size` is finalized — init-order trap honored) → optional `coach.load_checkpoint(opts.resume_from)` → `coach.learn(opts.n_iterations)`. The Stage 1 architecture knobs (`embedding_dim`, `n_encode_layers`, `n_heads`, `value_hidden_dim`, etc.) are exposed so the warm-start `state_dict` matches the constructed `AttentionModel`.
- [x] **F.2** Smoke battery in `src/scripts/smoke_alphazero.py` — A1, A2, A3, A5, A6 all green. **A3 added** as a Phase-F-specific check on the *training-target* π_t entropy under `temperature_schedule='step30'` (spec §4.2 choice (B)). A14 in `smoke_mcts.py` already covers σ_t entropy decay; A3 here exercises the orthogonal invariant — that π_t in the buffer is decoupled from σ_t, i.e. entropy(π_t) > 0 except at the forced last step (N-1). **A4 explicitly skipped** (documented in the smoke harness docstring): legality / support / finiteness on `pi_t` is already covered by A2 here and by A13 in `smoke_mcts.py`. Full smoke wall-clock: ~10 s on CPU.
- [x] **F.3 attempt 1 (FAILED 2026-04-30)** — Plan-canonical recipe (K=50, value_head, ε=0.25, step30) on RTX 4060. **20 iter × ~50s/iter ≈ 17 min wall-clock. Output dir: `outputs/tsp_20/stage4_pilot_20260430T024046/`.**
  - **Failed pilot conditions:** (1) val_avg_cost(iter 19) = **3.84136**, threshold was 3.83843 → fail by 0.003. (4) val_avg_cost trend was non-monotone (range 3.8414-3.8475 across iters, 6× the ±0.001 noise band).
  - **Passed pilot conditions:** (2) no NaN losses across 20 iters. (3) gating fired 4× (iters 4/9/14/19) but ALL **rejected** at α=0.05 — candidate consistently 0.0016-0.0029 *worse* than baseline.
  - **Smoking gun**: iter 0's val_avg_cost was 3.8454 — *one* MCTS-driven training iteration immediately degraded the warm-started Stage 1 policy by ~0.006. The loop spent the next 20 iters partially recovering (down to 3.8414) but never crossed back to Stage 1's level.
  - **Root cause (probe verified):** From `buffer.pt` (20K self-play instances), MCTS-K=50-vh-step30+ε=0.25 produced tours **0.077 *worse* than greedy θ★ on average** (mean MCTS = 3.92 vs mean bl_val = 3.84). Only 26.2% of MCTS tours strictly beat greedy; 27.2% strictly worse; 46.7% tied. The "improvement engine" was actively producing a *worse-than-greedy policy* to distill, which is exactly why the model degraded.
  - **Mechanism:** `dirichlet_epsilon=0.25` (with α=10/N=0.5 — sparse Dirichlet) injects 25% noise into the root prior, and `temperature_schedule='step30'` with τ=1 then *samples* the action from the visit distribution for the first 6 of 20 steps. With c_puct=0.05 and only K=50 simulations, the search can't recover from this noise before action selection. Compounded over 6 early steps, this gives a high probability of committing to a disastrous prefix that the rest of the tour can't fix. **AlphaGo-Zero-canonical exploration is not safe when bootstrapping from an already-converged policy** — in Go the policy starts uniform-random, so 25% Dirichlet is barely a perturbation; here it's effectively a 25%-random-action substitution.
  - **MCTS quality probe** (`src/scripts/probe_mcts_quality.py`, 200 fresh TSP-20 instances; full output saved at `outputs/tsp_20/stage4_pilot_20260430T024046/mcts_quality_probe.txt`):

    | Config | mean cost | gap vs greedy | frac better | frac worse |
    |---|---|---|---|---|
    | greedy θ★ | 3.86171 | — | — | — |
    | F.3 default (K=50 vh step30+ε=0.25) | 4.10767 | **+0.24596** ❌ | 19.5% | 38.5% |
    | K=50 vh step30+ε=0.10 | 3.86146 | -0.00025 | 26.5% | 16.5% |
    | K=50 vh step30+ε=0.05 | 3.85900 | -0.00271 ✓ | 26.5% | 15.5% |
    | K=50 vh no-explore | 3.85753 | -0.00418 ✓ | 28.0% | 13.0% |
    | K=50 rol no-explore | 3.85623 | -0.00548 ✓ | 31.5% | 11.5% |
    | K=100 vh no-explore | 3.85684 | -0.00487 ✓ | 31.0% | 11.5% |
    | **K=100 rol step30+ε=0.05** | **3.85528** | **-0.00643 ✓** | **35.5%** | 12.0% |
    | K=100 rol no-explore | 3.85572 | -0.00599 ✓ | 34.5% | 11.0% |
    | K=200 rol no-explore | 3.85384 | -0.00786 ✓ | 40.0% | 9.5% |

    The big takeaways are: (a) `ε=0.25` is catastrophic regardless of K or leaf eval; (b) `ε ≤ 0.05` is the safe regime for warm-started training; (c) `leaf_eval='rollout'` slightly beats `value_head` at every K and is more robust to residual exploration noise; (d) MCTS quality scales with K (K=200 rollout no-explore is the strongest), but K=100 rollout step30+ε=0.05 is the best **wall-clock-vs-quality** trade-off and is the recommended F.3 retry config.
- [x] **F.3 attempt 2 (FAILED 2026-04-30; bug discovered)** — Same CLI as attempt 3 below, but a hidden bug in `make_self_play_config` hardcoded `leaf_eval='value_head', dirichlet_epsilon=0.25, temperature_schedule='step30', dirichlet_alpha=10/N`, ignoring all four corresponding CLI flags. So v2 actually ran with EXACTLY the canonical-AGZ recipe (only K differed: 100 instead of 50). Buffer mean(bl − mcts) = +0.114 (worse than v1's +0.077 — more K on a corrupted prior amplifies the bad signal). val_avg_cost(iter 19) = 3.84407, fail by 0.0056. Output: `outputs/tsp_20/stage4_pilot_v2_20260430T031137/`.
  - **Bug fix (commit 419a857):** `make_self_play_config(graph_size, n_simulations, leaf_eval=…, dirichlet_epsilon=…, dirichlet_alpha_factor=…, temperature_schedule=…)` now takes all four as kwargs; `MCTSCoach.learn` reads them from `opts`. Smokes B0/B1/A1b/A1/A2/A3/A5/A6 still pass.
- [x] **F.3 attempt 3 (PARTIAL 2026-04-30)** — Bug-fixed recipe: `K=100, leaf_eval='rollout', dirichlet_epsilon=0.05, temperature_schedule='step30'`. Output: `outputs/tsp_20/stage4_pilot_v3_20260430T034502/`. ~42 min wall-clock (~125 s/iter; rollout K=100 is ~2.5× the plan's K=50-vh estimate).
  - **Buffer signal verified clean:** mean(bl − mcts) = **+0.00609** (MCTS *better* than greedy by 0.006), 36.1% strict-better, 48.6% tied, 15.2% worse. Probe prediction was -0.006; actual is +0.006 (same magnitude, sign flip is just my probe's "MCTS − greedy" vs buffer's "bl − mcts" convention). The bug fix took effect.
  - **Pilot conditions:** (1) val_avg_cost(iter 19) = **3.84127**, threshold 3.83843 → **fail by 0.00284**. (2) no NaN ✓. (3) gating fired 4× all rejected ✓ literal; 0 accepts. (4) val_avg_cost spread 3.8404-3.8423 (range 0.0019, **within ±0.001 noise band — partial pass**); but trajectory is FLAT, not monotone-decreasing.
  - **The flat-val plateau:** val_avg_cost dropped from 3.84154 (iter 0) to a min of 3.84037 (iter 9), then bounced back to 3.84127 (iter 19). Best ever was 3.84037. The policy is wandering around a local optimum near the MCTS-K=100-rollout argmax solution, with random fluctuations.
  - **Gating diff trend (candidate − baseline on gating set):** +0.0014, +0.0003, +0.0015, +0.0011 — oscillating, not converging toward acceptance. Without a gate-accept, `best_model` (the MCTS prior) never updates, so MCTS never gets stronger, so the policy iteration cycle is stuck.
  - **Catch-22 diagnosed:** Stage 4's improvement loop is paused at the AGZ "policy iteration" cycle. To break out, MCTS needs a stronger prior (gating accept), but gating won't accept without a candidate that's reproducibly better — which requires either (a) more search depth (K) so MCTS targets are decisively better than the greedy baseline, (b) a longer warm-up period before gating, or (c) a softer gating threshold. Phase G.5 (gating cadence sweep) and G.1 (K sweep) become load-bearing rather than optional.
- [x] **F.3 v3 resumed (FAILED 2026-04-30)** — Same recipe, iters 20-39 via `--resume_from`. Plateau confirmed: best val 3.84006 (iter 27), final 3.84059. 30 more iters bought 0.00031 of improvement — at that rate ~200 more iters to close the 0.002 gap (well beyond F.4's 100). Gating diff drifted slowly toward zero (+0.0011 → +0.0005) but never crossed.
- [x] **F.3 v4 K=200 (FAILED 2026-04-30)** — `K=200, leaf_eval='rollout', dirichlet_epsilon=0.05, temperature_schedule='step30'`. Buffer signal stronger (mean(bl-mcts) = +0.00749 vs v3's +0.00609; 39% strict-better vs 36%) but val_avg_cost still flat. Final 3.84102, min 3.84016. Output: `outputs/tsp_20/stage4_pilot_v4_K200_20260430T052113/`.
- [x] **Apples-to-apples comparison (Stage 1 vs Stage 4 v4, fixed seed=42 10K val):**
  | Model | val_avg_cost | Δ vs Stage 1 |
  |---|---|---|
  | Stage 1 (loaded) | 3.83622 | — |
  | Stage 4 v4 working | **3.83757** | **+0.00135 worse**, p_one_sided(S4<S1)=1.0 |
  | Stage 4 v4 best_model | 3.83622 | identical (gating never accepted, frozen at Stage 1) |
  Hard truth: Stage 4 with bug-fixed K=200 recipe **degrades Stage 1 by 0.00135** with strong significance. The +0.0075 buffer-level MCTS-better-than-greedy signal does NOT translate into a better greedy policy — distillation noise (cases where MCTS picks worse moves than greedy) overwhelms the small mean improvement.
- [x] **F.3 v5 — `--gate_mode=always` test (Phase G.5.c) (FAILED 2026-04-30)** — Same K=200 recipe + new `--gate_mode always` CLI flag (and `MCTSCoach` opt) that bypasses the t-test and forces best_model = working_model every iter. Diagnostic intent: break the catch-22 where gating freezes best_model = Stage 1, so MCTS prior never strengthens. Output: `outputs/tsp_20/stage4_pilot_v5_alwaysgate_K200_*/`.
  - Result: best_model now matches working_model (3.83732, both worse than Stage 1 by +0.00110). 0.00025 better than v4 but still net-worse.
  - **The plateau is NOT a gating artifact.** Letting MCTS prior evolve via always-accept doesn't escape the plateau — the working model itself is the limit.
- [ ] **F.4 TSP-20 main run (100 iter × K=100 × 200 train_steps)** — **NEGATIVE-RESULT BRANCH.** F.3 ran 5 attempts and could not achieve the F.3 pilot pass conditions even with the corrected recipe. Decision points pending user input: (a) run F.4 anyway with corrected recipe to document the longer-horizon trajectory, (b) try architecture-level interventions (lower lr, freeze encoder, or fresh value head) to break the plateau, (c) move to Stage 5 from-scratch where AGZ has a cleaner test, (d) test on TSP-50/TSP-100 where Stage 1 is weaker and there's more headroom for AGZ to add value.

### Phase F — TSP-20 pilot diagnosis summary (2026-04-30)

**Hypothesis under test:** AGZ-style MCTS self-improvement can lift the converged Stage 1 policy on TSP-20.

**Result: hypothesis NOT supported in this regime.** Five pilot attempts (v1-v5) across canonical AGZ exploration (ε=0.25 step30), warm-start-adapted exploration (ε=0.05), strong MCTS (K=100 → K=200), and gating-off (always-accept). All five end at val_avg_cost in [3.840, 3.844] range, ~0.001-0.005 *worse* than Stage 1 on apples-to-apples comparison.

**Two actionable findings transferable beyond Stage 4:**

1. **AGZ's canonical exploration (ε=0.25 + step30 with τ=1) is unsafe on a converged warm-start.** With a sharp prior, 25% Dirichlet noise + temperature-1 sampling commits to disastrous prefixes within the first ⌈0.3·N⌉ steps. Probe (`src/scripts/probe_mcts_quality.py`) shows mean MCTS tour cost is **+0.246 *worse* than greedy** under this config. Safe regime for warm-started training: **ε ≤ 0.05** with rollout leaf eval. Memory note added so future sessions don't repeat the trap.

2. **Bug fix `419a857`:** `make_self_play_config` was hardcoding leaf_eval, dirichlet, schedule kwargs and ignoring CLI flags. Fixed to read all four from opts. This is what let v3+ actually run a different recipe than v1.

**Where the plateau comes from (best current explanation):**

- The Stage 1 policy is already at the architectural ceiling for this AM model + TSP-20 distribution. Greedy decoding gets val_avg_cost ≈ 3.836-3.840.
- Even strong MCTS (K=200 rollout) only beats greedy by ~0.008 on average — and only on 39% of instances. The other 61% are tied or worse.
- Distillation captures the *mean* MCTS distribution (loss converges to ~0.10), but the per-instance "wins" and "losses" of MCTS-vs-greedy roughly cancel — leaving the distilled policy approximately tied with greedy, slightly worse on average due to gradient noise from the shared encoder.
- Gating correctly detects "candidate is reproducibly slightly worse" and rejects (catch-22: best_model never updates → MCTS prior never strengthens → loop converges to local optimum).
- Always-accept gating doesn't help because the candidate's MCTS-prior-evolution is too noisy to compound into improvement.

**What WOULD likely help (if we want to keep trying):**
- Stronger MCTS at the leaf (K=1000+, AGZ-paper-scale) to get a cleaner improvement signal. Probe didn't extend that far; would need a separate experiment.
- Filter distillation targets: only train on instances where MCTS strictly beats greedy (drop the noise). Plan deviation, not in spec.
- Lower lr (1e-5) + many more iters (500+) for slower, more careful convergence. Potentially weeks of compute.
- Wider AM model (more encoder layers / embed dim) — Stage 1's architectural capacity may be the bottleneck.
- TSP-50 or TSP-100 — Stage 1 is weaker on larger N so there's more headroom for MCTS to add value (Phase G already wanted this; should hoist to a Stage 5 priority).
- [x] **F.5** `src/scripts/plot_stage4.py` (~225 LOC) — reads Stage 1 `epochs.csv` + Stage 4 `iterations.csv`, projects each row to `(total_instances, val_avg_cost)`, log-x scatter with two curves + three reference horizontal lines (Gurobi opt 3.8279, Stage 1 final 3.83943, Stage 3 K=400 rollout 3.8312). CLI: `--stage1_dir`, `--stage4_dir`, `--out`. Stage-1 cumulative-instances axis read from `args.json:epoch_size` (fallback `1_280_000`). Self-test mode `--smoke` fabricates synthetic 5-row CSVs into a temp dir and verifies the PNG is written and non-empty (~96 KB). Smoke green on this dev box.

**Phase F dev-portion completion note (2026-04-30):** Edits landed in this worktree:
| File | LOC delta | Change |
|---|---|---|
| `src/scripts/train_alphazero.py` | +225 (new) | Stage 4 launcher CLI mirroring `train.py`. |
| `src/scripts/plot_stage4.py` | +225 (new) | Sample-efficiency plotter + `--smoke` self-test. |
| `src/scripts/smoke_alphazero.py` | +110 | A3 (π_t entropy under `step30` schedule); docstring updated to explain why A4 is skipped. |

**Decisions on optional smokes:**
- **A3 (π_t entropy under schedule):** *Added.* Distinct from A14 in `smoke_mcts.py` (which checks σ_t collapse). This guards spec §4.2 choice (B) — the decoupling between action-selection σ_t and training-target π_t — and would catch a future regression where π_t accidentally gets coupled to σ_t.
- **A4 (legality/support/finiteness on pi_t):** *Skipped.* Fully covered by A13 in `smoke_mcts.py` (raw `solver.root_visit_dists`) and A2 here (post-`generate_self_play_batch` `pi_t`). Adding A4 would only re-run the same checks under the same code path. Documented in the smoke harness docstring instead of duplicating.

F.3/F.4 are explicitly out of scope for this dev pass and are queued for the user's GPU-equipped box.

### Phase G — Ablations (optional)

- [ ] **G.1** Leaf eval `rollout` vs `value_head`.
- [ ] **G.2** Buffer capacity sweep.
- [ ] **G.3** Dirichlet ε sweep.
- [ ] **G.4** Temperature schedule comparison.
- [ ] **G.5** Gating cadence sweep.
- [ ] **G.6** Best-so-far per-instance value normalization ($z_t = (\text{tour\_cost} - \text{lengths}_t)/\min_\text{seen}\text{tour\_cost}(x)$) — replaces original "cost-to-go vs broadcast z" since cost-to-go is now F.4 default.
- [ ] **G.7** Symmetry augmentation at leaf eval (random 2D rotation+flip on coords) — AGZ Methods §Search algorithm analog.
- [ ] **G.8** Optimizer ablation: SGD+momentum 0.9, lr=1e-3 (AGZ canonical) vs Adam lr=1e-4 (F.4 default).

---

## Results

### Phase A (2026-04-30)

- **Files edited:**
  - `src/am_baseline/search/mcts.py` (+29 LOC) — `MCTSConfig.return_root_visits`, `MCTSSolver.root_visit_dists` init/reset/append.
  - `src/am_baseline/search/mcts_cpp/mcts.hpp` (+4 LOC) — `Config::return_root_visits`.
  - `src/am_baseline/search/mcts_cpp/mcts.cpp` (+81 LOC) — sequential and batched per-step visit dumps + result-dict marshalling.
  - `src/am_baseline/search/mcts_cpp/solver.py` (+40 LOC) — `CppMCTSSolver.root_visit_dists`, `CppBatchMCTSSolver.root_visit_dists_per_instance` plumbing.
  - `src/scripts/smoke_mcts.py` (+293 LOC) — A13 helper (`_run_a13_visit_dists`) plus call sites in each backend entrypoint and main.
- **Smokes:** `python -m scripts.smoke_mcts --backend python|cpp|cpp_batch` all pass; A13 a/b/c sub-cases all green.
- **Bit-equivalence (A13.c):** under deterministic clamp (ε=0, τ=0) python and cpp produce identical visit-count dicts at every tour-step on a TSP-20 K=200 instance; cpp and cpp_batch also identical.

### Phase B (2026-04-30)

- **Files created/edited:**
  - `src/am_baseline/training/coach.py` (+~430 LOC, NEW) — `MCTSReplayBuffer` (push/sample/sample_step/save/load/_rebuild_step_index/step_counts) + `reconstruct_state(batch, device)`.
  - `src/am_baseline/training/trainer.py` (+~115 LOC) — `train_step_alphazero(model, optimizer, batch, opts) -> metrics dict`.
  - `src/scripts/smoke_alphazero.py` (+~280 LOC, NEW) — A1 (gradient flow), B0 (buffer invariants), B1 (save/load + step_index rebuild), A1b (reconstruct_state -> decode_step round-trip).
- **Smoke run** (CPU, AM_AlphaGoZero env, N=8, 5 instances, embedding_dim=32):
  ```
  [A1] metrics = {policy_loss: 1.81, value_loss: 0.66, total_loss: 2.47,
                  mean_entropy_pi: 1.53, mean_z: 0.86, gradient_norm: 2.90}
  PASS — all Phase B smoke checks succeeded.
  ```
  Encoder, decoder, value_head, and init_embed parameters all moved (gradient flow confirmed end-to-end). Ring-buffer eviction sanity-checked (push 5 into capacity 3 → wrapped to 3 instances, step_counts == [3]*N).
- **Stubs / TODOs for Phase C/D:**
  - `_synth_instance_records` in the smoke harness fabricates `pi` from a Dirichlet draw. Phase C will replace with `generate_self_play_batch(...)` output.
  - `MCTSCoach` orchestrator (Phase D) consumes `MCTSReplayBuffer.sample()` directly; no buffer-API changes anticipated.
- **Decoder-graph correctness:** `train_step_alphazero` calls `model.encode` and `model.precompute_decoder` *inside* the train step (not cached), so gradients flow through the same compute graph as self-play. `decode_step` returns `(B,1,N)` tensors that are squeezed to `(B,N)` before the CE — A1b confirms that `mask` aligns with visited cities (every visited entry has `log_p == -inf` after the squeeze). The `0 * -inf = NaN` corner is handled with `torch.where(mask, zeros, log_p)` before the multiply.

---

## Wall-clock / Resource Accounting

(To be populated as phases close.)

---

## Known Issues

(None yet.)

---

## Notes

- Plan file mirrored here: `_plans/stage4_plan.md`
- Original plan (Claude Code plans dir): `C:\Users\Jun18\.claude\plans\i-think-we-have-snazzy-trinket.md`
- Stage 4 reuses Stage 1 canonical TSP-20 checkpoint: `outputs/tsp_20/stage1_tsp20_canonical_20260423T103541/epoch-99.pt`.
- Stage 0 Gurobi reference for TSP-20: 3.8279 mean (1000 instances, seed=1234).
- Stage 1 reference val_avg_cost (TSP-20 canonical bs=512): 3.83943; bs=2048: 3.84443.
- Stage 3 reference test-time MCTS K=400 rollout: 3.8312 (gap 0.087% vs Gurobi).
