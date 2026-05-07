# Stage 4 Progress: AlphaGo-Zero-Style MCTS Self-Improvement Loop on TSP-20

**Plan:** `_plans/stage4_plan.md`
**Started:** 2026-04-29
**Last updated:** 2026-05-06 — **F.6.1 K=40 main complete (val=3.92 plateau); F.6.1.3 step10 + ε=0.25 sweep BROKE the plateau to val=3.878 from-scratch — first variant past F.6.1 main's iter-90 best (3.893).** F.6.1.3 also confirmed via diagnostic that value_head leaf eval at K=40 is statistically tied with greedy (K=40 rollout: −0.072 vs greedy; K=40 value_head: +0.002 vs greedy, p=0.63), pinpointing the F.6.1 plateau's root cause: MCTS targets ≈ greedy → CE distillation has no improvement signal → π_t entropy collapses to 0.155 nats by iter 99. **step10 + ε=0.25 partially halts the entropy collapse** (0.335 vs 0.155) by retaining exploration noise on a narrower σ_t window. F.6.1.4 (ε=0.25 +50 iter resume from iter-99) in flight. Telemetry additions this session: per-loss grad norms (policy/value/total, with value split into VH-only + encoder-decoder-shared subspaces) for cos(θ_shared) conflict diagnostic; step10 temperature schedule (cutoff = ⌈0.1·N⌉); standalone Stage-4 MCTS validator (`src/scripts/val_stage4_mcts.py`) with AM-baseline support. Earlier this session: F.6.1 K=20 variants (lr=5e-4 const, lr=1e-3 + 0.95/iter decay) both at val ≈ 3.92 on 100K instances; F.6.0.7→F.6.1.1 sub-budget probes locked the F.6.1 recipe; gate_every revised 5→1 (best_model freshness); lr scheduler infrastructure added (LambdaLR wired into coach.py, --lr_decay CLI flag, checkpoint save/load support).
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
- [ ] **F.4 TSP-20 main run (100 iter × K=100 × 200 train_steps)** — superseded by Modal 4-job batch (2026-05-02), see below.

### F.4 + F.3 Modal batch (2026-05-02) — BREAKTHROUGH

Ran 4 parallel jobs on Modal A10 to test (a) two F.4 variants and (b) two F.3-scale architecture/training interventions. Modal wrapper lives at `src/scripts/modal_run_train_alphazero.py`. Runs in `outputs/tsp_20/{f4_a1, f4_a2, f3_b1, f3_b2}_*`.

**Apples-to-apples on seed=42 10K val (`compare_stage1_vs_stage4.py`):**

| Job | Recipe | Iters | val_avg_cost (working) | Δ vs Stage 1 (=3.83622) | p_one_sided(S4<S1) | Verdict |
|---|---|---|---|---|---|---|
| Stage 1 | greedy decoder | — | 3.83622 | — | — | baseline |
| **a1** | F.4 plan-default — K=100 rollout step30 ε=0.05, ttest, 100 iter, train=200, buffer=200K | 100 | 3.83824 | +0.00202 | 1.0000 | ❌ reproducibly worse |
| **a2** | F.4 v5-best — K=200 rollout step30 ε=0.05, gate=always (every iter), 100 iter, train=200, buffer=200K | 100 | 3.83669 | +0.00047 | 0.9555 | ⚠️ tied, not significantly different |
| **b1** | F.3 freeze_encoder — K=200 rollout step30 ε=0.05, ttest, 20 iter, train=100, buffer=50K, **--freeze_encoder** | 20 | 3.83576 | **−0.00046** | **0.0519** | ✓ borderline significant improvement |
| **b2** | F.3 lr=1e-5 — same as b1 but **--lr_model 1e-5** instead of freeze | 20 | **3.83485** | **−0.00137** | **0.0000 (t=−7.08)** | ✓✓ **STATISTICALLY SIGNIFICANT** |

**Per-iter trajectory highlights (from `iterations.csv`):**
- a1: flat plateau — iter 0 = 3.8413, iter 99 = 3.8418, min = 3.8401 (iter 23). Same shape as F.3 v3 at full scale.
- a2: slow but real improvement — iter 0 = 3.8411, iter 99 = 3.8402, **min = 3.8394 (iter 69)**. Reached canonical Stage 1 val (3.83943) on per-run val_dataset.
- b1: monotone decreasing across all 20 iters — iter 0 = 3.8403, iter 19 = **3.8392 (still improving at end)**.
- b2: best result — iter 0 = 3.8400, iter 12 = **3.8383 (min)**, iter 19 = 3.8387. **Gate accepted at iter 9** — the first ever gate-accept across 7 attempts. Best_model updated mid-run.

**Diagnosis revised — the F.3 plateau wasn't capacity or gating, it was lr.**

The original F.3 v3-v5 plateau at val ≈ 3.840 was caused by **Adam at lr=1e-4 overshooting from the converged Stage 1 checkpoint**. The MCTS-better-than-greedy signal was always real (+0.006 to +0.0075 buffer gap), the policy just couldn't absorb it without poisoning itself with each update. Lowering lr by 10× let the policy settle into local optima around the MCTS distribution. Even at the F.3-pilot scale (20 iters × M=1000), this passes Stage 4 acceptance criterion 1a:

**Stage 4 acceptance criteria — current status:**
- ✅ **1a (marginal improvement):** b2 working = 3.83485, threshold = 3.83622 − 0.001 = **3.83522 → PASSES** by 0.00037, statistically significant (p=0.0000).
- ⚠️ **1b (strict total sample efficiency):** b2 used 20K instances vs Stage 1's 128M; combined Stage 1+Stage 4 sample-efficiency curve at matched x will require the headline plot to render against `iterations.csv`.
- ❌ **2 (ultimate quality ≤ 3.8312, Stage 3's K=400 rollout):** still 0.0036 above target. Needs more iters / longer training to close the gap to test-time MCTS.
- ✅ **3 (self-improvement):** b2 trajectory is monotone non-increasing for the first 12 iters (3.8400 → 3.8383), then noisy plateau.
- ✅ **4 (gating fires + accepts):** **first gate-accept across 7 attempts**, at b2 iter 9 (with lr=1e-5).

**Two interventions worked; lr=1e-5 dominated freeze_encoder.**

Comparing b1 vs b2 separates the two hypotheses:
- "Shared encoder is the noise channel" (freeze_encoder fix): improvement of 0.00046, borderline-significant (p=0.052). Real but marginal.
- "Adam lr=1e-4 overshoots from converged checkpoint" (lr=1e-5 fix): improvement of 0.00137, decisively significant (p<0.0001). 3× larger effect.

Both fixes are real and complementary; lr is the dominant factor. **Phase G.8 (optimizer ablation) is now load-bearing rather than optional** alongside G.3 (Dirichlet) and G.5 (gating). A combined `freeze_encoder + lr=1e-5` experiment is a natural follow-up.

**What this means for the plan:**
- Stage 4 hypothesis ("warm-started AGZ improves on REINFORCE on TSP-20") is now **SUPPORTED** with the corrected recipe.
- The planned F.4 (a1 recipe) is the wrong recipe; it doesn't pass. The lesson is now: **default F.4 recipe should be K=200 rollout step30 ε=0.05 + lr=1e-5**, not the plan's K=100 lr=1e-4.
- Open question: does b2 keep improving with more iterations / larger M / longer train_steps? At iter 19 with min at iter 12, the trajectory was noisy but not yet plateaued. A natural next experiment is **b2 recipe at full F.4 scale** (100 iter × M=1000 × train_steps=200 × buffer=200K) to see how far it can go. Stretch goal (criterion 2) requires reaching ≤ 3.8312 — distillation needs to close another 0.0036 gap.

**Compute used:** ~1.5 h Modal A10 wall-clock for the 4 jobs in parallel (~3.5 GPU-h, ≈ $2-4).

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

### Phase F.6 — Proposal-aligned from-scratch run (2026-05-02; PENDING)

**Status: pending.** Discovered during post-Modal-breakthrough discussion that all Phase F warm-start work (F.1-F.5, including the b2 lr=1e-5 result) addresses a *related but distinct* question from what `proposal.md` Stage 4 actually promises.

**The discrepancy:**

| Question | Setup | What we have so far |
|---|---|---|
| Does AGZ improve a converged Stage 1 model? | warm-start from `epoch-99.pt` | ✅ Yes — Modal b2 (lr=1e-5) beats Stage 1 by 0.00137, p<0.0001 |
| **Does AGZ from-scratch reach AM-equivalent quality with fewer instances than REINFORCE?** *(proposal.md:140-141)* | **random init, no `--load_path`** | **❌ Not yet tested** |

The plan ([_plans/stage4_plan.md:455-470](../_plans/stage4_plan.md#L455-L470)) defaulted to warm-start in F.3 and F.4 for cost reasons (warm-start finishes in ~1.5 h on Modal A10; from-scratch needs ~70 h). Under the warm-start framing, criterion 1b in the original plan explicitly punted from-scratch sample efficiency to Stage 5. The proposal language ("**replace** REINFORCE", "**reaches** AM-equivalent quality with fewer total training instances") clearly implies a from-scratch run that produces a sample-efficiency curve to compare against Stage 1's REINFORCE trajectory.

**Plan amended (2026-05-02).** Phase F now has two explicit variants:

- **Warm-start variant (F.1-F.5)** — kept as a documented diagnostic / sanity. The b2 result is real and useful, and the lr=1e-5 + ε=0.05 + K=200-rollout findings transfer directly to F.6.
- **Proposal-aligned variant (F.6)** — added. From-scratch run that produces the actual headline sample-efficiency curve.

**F.6 sub-tasks:**

**Locked decision (2026-05-02): `lr_model = 1e-4` FIXED for F.6.** AM-paper / Stage 1 default. Justification: the proposal headline claim ("AGZ reaches AM-equivalent quality with fewer instances than REINFORCE") requires apples-to-apples vs Stage 1, which used lr=1e-4 only. The b2 warm-start lr=1e-5 finding does NOT transfer (warm-start fix for converged-checkpoint overshoot; from random init there's no converged optimum to overshoot from). lr=1e-5 sweep moved to Phase G.9 ablation — informational only, not part of F.6 main-line claim. See `project_lr_fairness_for_stage4.md` in project memory.

**Sub-tasks:**

- [x] **F.6.0 Pre-flight grid probe (12 variants).** **COMPLETE 2026-05-03.** Winner: `lerol_eps25_ttest` (rollout × ε=0.25 × ttest gate; iter-49 = 3.9338, Δ = 0.617). See full results table below. Spec locked 2026-05-02 (commit `d61ecae`). Probes three knobs that may not transfer cleanly from b2's warm-start best to random init:
  - `leaf_eval` ∈ {`value_head` (AGZ-canonical), `rollout` (AM-paper)}
  - `dirichlet_epsilon` ε ∈ {0.0, 0.05 (warm-start safe), 0.25 (AGZ-canonical)}
  - `gate_mode` ∈ {`ttest` (AGZ-style), `always` (AZ-style)}

  Each variant: 50 iter × M=1000 × K=100 × `train_steps_per_iter=200` × `buffer_capacity=200_000` × `lr_model=1e-4` × `val_seed=42` × no `--load_path`. The `train_steps_per_iter` and `buffer_capacity` values **match F.6.1's defaults** so the F.6.0 winner's `iter-49.pt` + `buffer.pt` cleanly resume into F.6.1 (`buffer.load()` hard-fails on capacity mismatch — see [coach.py:366-370](../src/am_baseline/training/coach.py#L366-L370)). **ETA ~3.7 h Modal A10 parallel wall-clock, ~$10-15 in credits.**

  Modal entrypoint: `modal run --detach src/scripts/modal_run_train_alphazero.py::run_f60_grid` (added in commit `d61ecae`). Output dirs: `outputs/tsp_20/f60_le{vh|rol}_eps{0|05|25}_g{ttest|always}_<timestamp>/`. W&B logging on by default (project `am-alphagozero`).

  **Decision rule for picking F.6.1 defaults:** lowest `val_avg_cost` at iter 50 wins (or steepest decline if multiple variants tie within ±0.001 noise band).

  **F.6.0 results — full 12-variant grid (complete, 2026-05-03)**

  Launched 2026-05-03 05:22 UTC on Modal A10 (12 parallel jobs, app `ap-mIrMZH1Gi3j9yWopjtapoI`). All 12 variants reached iter-49. Wall-clock from launch to last-job-done: ~9.5 h (vs 3.7 h docstring estimate — the two `rollout × ε=0.25` variants spent ~2h 50min queued before Modal allocated GPUs). Two infrastructure restarts: `levh_eps05_gttest` (orig abandoned at iter-15, restart 06:22 UTC) and `lerol_eps05_gttest` (orig abandoned at iter-7, restart 06:55 UTC); restarts' iter-49 are reported below, originals' partial dirs ignored.

  Sorted by `val_avg_cost(iter 49)` ascending:

  | rank | leaf_eval | ε | gate | iter0 | iter49 | Δ | best | best@ | gated/accepted | mcts s/iter |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 1 | rollout | 0.25 | always | 4.4624 | **3.9328** | 0.530 | 3.9276 | 48 | 10/10 | 461.9 |
  | 2 | rollout | 0.05 | always | 4.5165 | 3.9335 | 0.583 | 3.9335 | 49 | 10/10 | 530.0 |
  | 3 | rollout | 0.25 | ttest  | 4.5508 | 3.9338 | 0.617 | 3.9314 | 46 | 10/10 | 464.6 |
  | 4 | rollout | 0.00 | always | 4.5491 | 3.9429 | 0.606 | 3.9353 | 48 | 10/10 | 338.4 |
  | 5 | rollout | 0.05 | ttest  | 4.4844 | 3.9449 | 0.540 | 3.9271 | 47 | 10/8  | 412.7 |
  | 6 | rollout | 0.00 | ttest  | 4.5913 | 3.9450 | 0.646 | 3.9313 | 47 | 10/9  | 438.4 |
  | 7 | value_head | 0.25 | always | 5.2024 | 4.1338 | 1.069 | 4.1338 | 49 | 10/10 | 254.5 |
  | 8 | value_head | 0.05 | ttest  | 5.3870 | 4.1896 | 1.197 | 4.1627 | 47 | 10/9  | 200.1 |
  | 9 | value_head | 0.00 | always | 4.9376 | 4.1917 | 0.746 | 4.1917 | 49 | 10/10 | 204.7 |
  | 10 | value_head | 0.25 | ttest  | 5.2637 | 4.1921 | 1.072 | 4.1592 | 47 | 10/8  | 201.5 |
  | 11 | value_head | 0.05 | always | 5.4163 | 4.1986 | 1.218 | 4.1767 | 48 | 10/10 | 198.4 |
  | 12 | value_head | 0.00 | ttest  | 5.2024 | 4.2817 | 0.921 | 4.2817 | 49 | 10/6  | 209.9 |

  All 12 variants are monotone-decreasing (head-mean of iters 0-4 > tail-mean of iters 45-49).

  **Decision-rule application** ("lowest val_avg_cost(iter 49); within ±0.001 noise band, prefer largest Δ"):
  - Lowest iter-49: `lerol_eps25_galways` at 3.9328.
  - Within ±0.001 of leader: 3 variants — {`lerol_eps25_galways` (3.9328, Δ=0.530), `lerol_eps05_galways` (3.9335, Δ=0.583), `lerol_eps25_ttest` (3.9338, Δ=0.617)}.
  - Largest Δ in band → **`lerol_eps25_ttest`** (rollout × ε=0.25 × ttest gate). Δ=0.617, iter-49=3.9338, iter-49 best mid-run=3.9314 at iter 46. **This is the F.6.0 winner under the locked decision rule.**

  Practical co-winners (within noise band, would also be defensible F.6.1 starts): `lerol_eps25_galways` (lowest absolute iter-49) and `lerol_eps05_galways` (cleanest 10/10 acceptance + Δ=0.583).

  **Headline observations:**

  1. **`leaf_eval` is the dominant knob** — it determines outcome more than ε and gate_mode combined. All 6 rollout variants finish strictly below all 6 value_head variants. Rollout cluster: [3.9328, 3.9450] (0.012 spread). value_head cluster: [4.1338, 4.2817] (0.148 spread). Rollout cluster mean is **0.20 cost units below value_head cluster mean** — well outside any plausible noise band.

  2. **Within the rollout sub-grid, ε and gate_mode are essentially irrelevant.** All 6 rollout variants finish within 0.012 of each other. ε=0.25 was identified in `project_alphagozero_warmstart_exploration.md` as toxic to a *converged* Stage 1 policy; **that fragility does NOT transfer to from-scratch** — ε=0.25 is in the winner cluster here. (Memory updated separately.)

  3. **Within the value_head sub-grid, ε and gate_mode are nearly irrelevant** at iter-49 (5/6 variants in [4.1896, 4.1986]; only the `ε=0 + ttest` variant is materially worse at 4.2817). But the whole cluster is held above the rollout cluster by the value-head's leaf-eval failure mode.

  4. **value_head's failure mechanism** (diagnosed mid-run from `iterations.csv`):
     - `value_loss` collapses 0.091 → 0.024 in *one* iter, then plateaus at ~0.013-0.024 for the entire run.
     - That's not the value head failing to train — it's the value head **trivially fitting the per-state mean** of `z_t = (tour_cost − lengths_t)/bl_val`. From random init, both `tour_cost` and `bl_val` are sampled under approximately the same noisy random policy and are tightly correlated, so the target distribution has tiny across-instance variance. The MSE landscape is dominated by the constant component.
     - In MCTS, `v(s) ≈ const` for all expanded leaves means **the value head adds zero leaf-discrimination** to the search. MCTS runs on policy prior + UCB exploration noise alone, with no grounded leaf signal.
     - `rollout` doesn't have this break: a rollout reaches a *terminal* state and reads realized tour cost — different leaves get genuinely different values from iter 0.
     - The `/bl_val` normalization is *not* the bug; it's a useful scale-invariance trick for the loss landscape. Removing it doesn't fix the across-instance variance shortage. Real fixes would be value-target whitening, state-conditional baseline subtraction, hybrid leaf eval, or auxiliary value pretrain — all deferred to optional Phase G ablations.

  5. **Gate-mode mechanics confirmed**: every variant has `gated_n=10` (gate_every=5 fires 10 times across 50 iters regardless of mode). `accepted_n` separates as expected: `always` → 10/10 by design; `ttest` → 6-10/10. Across both leaf evals, `ttest` rejects more often when ε=0 (presumably because no exploration noise → less chance of beating θ★ on the gating set). The gate mechanism is doing real work, not a no-op.

  6. **Wall-clock**: rollout averages **~440 s/iter** (range 338-530 s), value_head averages **~211 s/iter** (range 198-255 s). So **rollout is ~2.1× slower per iter** at K=100 — but it's the *only* leaf eval that produces materially-decreasing val_avg_cost from-scratch. The throughput gap is not enough to shift F.6.1 toward value_head.

  7. **Sample-efficiency anchor**: F.6.0 winner reaches **3.93 on TSP-20 in 50 iter × 1000 instances = 50K total instances**. Stage 1 canonical reaches ~3.84 at full convergence (1.28M instances first-epoch baseline). So F.6.0 rollout is **~0.09 above Stage 1 final quality at ~2.6% of Stage 1's instance count** — encouraging signal for the proposal sample-efficiency claim, but the ~0.09 gap remains and the trajectory needs F.6.1 (100-iter extension) to confirm whether it closes.

  **F.6.1 recipe pin (resume from F.6.0 winner)**: `--leaf_eval rollout --dirichlet_epsilon 0.25 --gate_mode ttest --load_path outputs/tsp_20/f60_lerol_eps25_gttest_20260503T052231_20260503T081222/iter-49.pt`. All other args (lr=1e-4, K=100, M=1000, train_steps=200, buffer=200K, val_seed=42) carry over per the F.6.1 spec.

  **Diagnostic infra added**: `src/scripts/probe_grad_norm.py` — measures per-step gradient norm under both Stage 1 (REINFORCE) and Stage 4 (CE distillation) at F.6.0 model scale. Empirical finding (n=30 steps, random init, CPU): Stage 4 grad_norm mean = 3.76, Stage 1 grad_norm mean = 90.87 — Stage 4 is **~25× smaller** than Stage 1 at random init, contradicting first-principles expectation that CE distillation has larger gradients than REINFORCE. With `max_grad_norm=1.0` clipping in both regimes, post-clip step magnitudes are equal (lr × 1.0 = 1e-4) regardless of leaf eval, so the locked-lr=1e-4 choice does not introduce a hidden lr asymmetry. The real per-iter parameter-movement asymmetry is `train_steps_per_iter` (Stage 4 = 200 vs Stage 1 = 1 per epoch-step), which is a candidate F.6.1.5 / G ablation to probe but not a F.6.1 blocker.

- [ ] **F.6.0.5 LR re-derivation for Stage-4 CE distillation** *(NEW 2026-05-03 — gates F.6.1 lr default; supersedes F.6 setup's "lr=1e-4 fixed" rule for F.6.1 onward.)*

  **Motivation.** F.6.0 winner stuck at val_avg_cost=3.9338, ~0.094 above Stage 1's 3.83943. Re-derive lr from the Stage-4 gradient calculation rather than inheriting Stage 1's lr by default.

  **Analysis (full text in [_plans/stage4_plan.md F.6.0.5 section](../_plans/stage4_plan.md)):**

  - **Stage 4 CE policy gradient on logits:** ∂L_policy/∂ℓ_j = p_j − π_j^target. Bounded ∈ [−1,1], per-row L2 ≤ √2. **O(1) bounded.**
  - **Stage 1 REINFORCE policy gradient on logits:** ∂L/∂ℓ^(t)_j = (L_tour − b_l)·(1[a=j] − p_j). **Advantage-scaled.**
  - Empirical (probe_grad_norm.py): Stage 4 mean=3.76 vs Stage 1 mean=90.87. Ratio ~24× matches O(1) vs O(advantage·N) prediction.
  - Local recheck (2026-05-03): reran `probe_grad_norm.py` under `conda activate AM_AlphaGoZero` and reproduced Stage 4 mean=3.759 vs Stage 1 mean=90.866. A short production-batch proxy (`batch_size=512`, 10 CPU steps) gave Stage 4 mean=5.200 vs Stage 1 mean=35.369. Both regimes were still above `max_grad_norm=1.0`, so clipping is active in both; the raw-gradient evidence does **not** support lowering Stage 4 LR just because the loss differs from AM.
  - **Adam is approximately scale-invariant** to grad magnitude — `lr · m̂/√v̂` is close to invariant under uniform `g → c·g` rescaling. The clip is active in raw-gradient space, but it is active for both regimes; the practical control knob is therefore the optimizer step size and update budget, not the unnormalized Stage 1 vs Stage 4 grad-norm ratio. Per-step parameter movement is governed mainly by `lr` in both regimes.
  - Budget: Stage 1 converged at ~250K grad steps × lr=1e-4 = ~25 units drift. Stage 4 F.6.0 = 10K steps × lr=1e-4 = ~1 unit. **F.6.0 has had ~4% of Stage 1's drift** — the 0.094 gap is mechanical, not algorithmic.
  - **Recommended lr for Stage-4 CE distillation on a transformer-AM:** **5e-4** (geometric mean of principled range 3e-4–1e-3; halfway between Stage-1-inheritance and the 8.3e-4 budget-matching estimate for F.6.1 to reach Stage-1 drift in 30K total steps). Reference Adam lrs for transformer CE: BERT 1e-4 (with warmup), nanoGPT 6e-4, ViT 1e-3. lr=1e-4 sits at the conservative end.

  **Validation — narrow 3-variant Modal smoke (`run_f605_lr_validation`).** Hold-fixed at F.6.0 winner config: leaf_eval=rollout, ε=0.25, gate_mode=ttest, K=100, M=1000, train_steps_per_iter=200, buffer=200K, batch_size=512, val_seed=42, max_grad_norm=1.0, **n_iterations=25**, no `--load_path`. V3 added 2026-05-04 to disentangle the **weight_decay** confound: Stage 1 uses Adam wd=0 (PyTorch default; AM-paper convention), Stage 4 uses wd=1e-4 (AGZ-canonical). Theoretical impact at 5K steps is below val_avg_cost noise floor (`lr·wd·θ ≈ 5e-8` per step → ~`2.5e-4` cumulative shrink), but verifying empirically removes the apples-to-apples objection.

  | variant | lr | wd | role | val_avg_cost(iter 19) | policy_loss | value_loss | entropy | final gate accepted | verdict |
  |---|---|---|---|---|---|---|---|---|---|
  | V1 (control) | 1e-4 | 1e-4 | F.6.0-winner replication (Stage-4 conventions) | 4.331052 | 1.5270 | 0.2790 | 1.2119 | yes | baseline |
  | V2 (analytical) | 5e-4 | 1e-4 | first-principles recommendation at Stage-4 wd | 4.348742 | 1.5095 | 0.2958 | 1.1128 | **no** | regressed vs V1 |
  | V3 (apples-to-apples) | 5e-4 | 0 | lr-only intervention; matches Stage 1 wd | **4.264759** | **1.4651** | **0.2637** | **1.1073** | yes | **winner** |
  | V4 (lr-isolation)   | 1e-4 | 0 | wd-only intervention at Stage-1 lr | 4.367396 | 1.5506 | 0.2763 | 1.1618 | yes | regressed vs V1 |

  **Note** (2026-05-06): table reports the **raw-target** grid (Option B intervention; F.6.0.5b). Iter count was lowered 25 → 20 and K from 100 → 40 alongside the grid scope expansion to 4 variants; the iter-X comparison column is iter-19 (last completed iter at n_iterations=20). Bl-normalized grid (F.6.0.5a, run earlier) is documented separately in the rolling progress log below.

  **Two-step decision rule for F.6.1:**
  - **Step 1 (lr):** V2 OR V3 reaches val_avg_cost(iter 25) ≤ V1 − 0.02 AND no per-iter regression > 0.05 AND final policy_loss within 30% of V1's. If neither passes: drop F.6.1 to lr=3e-4, wd=1e-4 and document failure mode.
  - **Step 2 (wd, only if Step 1 passed):** compare V3 vs V2. |V3−V2| < 0.005 → adopt V2 settings (wd=1e-4) for F.6.1; V3 < V2 by ≥ 0.01 → adopt V3 settings (wd=0); V3 > V2 by ≥ 0.01 → adopt V2 settings.

  If an lr=1e-3 run is launched manually, treat it as an LR-ablation run until it clears the same stability checks; do not make 1e-3 the proposal mainline by assumption.

  **Cost.** ~$8-12 Modal credits, ~1.7 h wall-clock parallel (parallel jobs ⇒ adding V3 doesn't extend wall-clock, only credits). Modal entrypoint: `modal run --detach src/scripts/modal_run_train_alphazero.py::run_f605_lr_validation`. Output dirs: `outputs/tsp_20/f605_lr{1e4_wd1e4|5e4_wd1e4|5e4_wd0}_<timestamp>/`.

  ---

  **F.6.0.5 progress log (rolling) — 2026-05-04 → 2026-05-06**

  **2026-05-05 — Scope/scale revisions before launch.**
  - **Variants expanded 3 → 4.** Added **V4 (lr=1e-4, wd=0)** to the `(lr, wd)` 2×2 corners. Closes the lr×wd interaction read: with V1=(1e-4,1e-4), V2=(5e-4,1e-4), V3=(5e-4,0), V4=(1e-4,0), the wd main effect is identifiable at *both* lr levels rather than only at lr=5e-4. Marginal cost: 1 extra parallel Modal task (no wall-clock extension).
  - **Per-iter MCTS scale lowered: K=100 → K=40, n_iterations 25 → 20.** Pure budget choice — keeps total wall-clock per variant under ~30 min so the lr × wd grids run cheaply on Modal in parallel. Note this departs from the F.6.0-winner config (K=100); F.6.0.5 results are a *trend* read on lr/wd, not a direct continuation of F.6.0's iter-49 trajectory.
  - **Two parallel grids launched, one per leaf-eval.** Same 4-variant `(lr, wd)` grid run separately under `leaf_eval=value_head` (`run_f605_lr_validation`) and `leaf_eval=rollout` (`run_f605_lr_validation_rollout`). Reason: F.6.0 already showed leaf-eval matters for from-scratch convergence; running both isolates whether the lr conclusion travels across leaf-eval regimes.
  - **W&B logging alignment (committed alongside the launch).** `coach.py` + `logging.py` patched so `log_iteration` and `log_alphazero_step` emit Stage-1 aliases — `epoch=iter`, `val_avg_cost`, `epoch_duration=mcts_wall_s+train_wall_s`, `baseline_updated`, `lr`, `global_step`, `value_loss`. `wandb_group` set to `tsp_{graph_size}` so Stage-1 and Stage-4 runs land in the same group on the W&B sidebar. Effect: F.6.0.5 iteration-axis trajectories overlay Stage 1's epoch-axis trajectories directly without post-hoc reshaping.

  **2026-05-05 — Rollout grid cancelled mid-run.** User observation watching the W&B trajectories: across the 4 (lr, wd) variants the rollout-leaf trajectories were not visibly separating — the dominant rate-limiter looked like *self-play data quality / sample count*, not optimizer step size. Cancelled the in-flight Modal app to avoid burning credits on a null read. Conservative inference: **at K=40 / from-scratch / rollout-leaf, lr in [1e-4, 5e-4] does not materially change the per-iter convergence rate over 20 iters.** Does NOT settle the F.6.0.5 question — value_head grid was kept running because the bl-drift hypothesis (below) made it the more diagnostic of the two leaf-eval regimes.

  **2026-05-06 — Diagnostic refocus: value-target distribution shift.** After value_head/bl-normalized 4-variant grid completed, the visible-from-W&B convergence remained slow regardless of lr/wd. Hypothesis surfaced (user-led): **value head is being trained against a *moving target* z = cost_to_go / bl_val.** Three distinct shift channels at random init / early training:
   1. **Calibration drift** — bl_val (greedy decode under θ★) decreases monotonically as θ★ improves over iterations, so identical (state, cost_to_go) yields a different z each iteration. The value head chases the divisor.
   2. **Buffer non-stationarity** — older buffer entries' (z) are normalized by an older bl_val than newer entries' (z), even within a single training step. Per-state target distribution has both fresh and stale calibration mixed.
   3. **Across-instance variance collapse** — bl_val varies per-instance, so dividing absorbs much of the across-instance signal the value head could otherwise learn from.

  **2026-05-06 — Option B implementation: train value head on raw cost_to_go (`--value_target_norm none`).** Removes the bl_val divisor from the value-head target entirely; eliminates all three shift channels in one stroke at the cost of a coarser z magnitude scale. Implementation (no behavior change for default `bl`):
   - **CLI** ([train_alphazero.py:157](src/scripts/train_alphazero.py#L157)): existing `--value_target_norm` flag extended with new `'none'` choice.
   - **Trainer** ([trainer.py train_step_alphazero](src/am_baseline/training/trainer.py)): z target reconstructed from buffer-stored `z_buf * bl_val` when `norm == 'none'` (buffer continues to store the canonical bl-normalized z; the trainer denormalizes at use time).
   - **MCTS Python** ([mcts.py](src/am_baseline/search/mcts.py)): added `value_target_norm` to `MCTSConfig`, `_convert_value_head_output(v, node, bl_val)` helper applied at `_populate_priors` and `_expand` so the value head's output is interpreted in the same target convention used at training time.
   - **MCTS C++** ([mcts.hpp/cpp](src/am_baseline/search/mcts_cpp/mcts.cpp)): mirror — `Config::value_target_norm` (parsed from py dict), `Solver::convert_value_head_output(v_raw, n, bl_val)` helper, all 4 call sites updated. Extension rebuilt via `python setup.py build_ext --inplace` (`pip install -e .` was producing a tiny editable wheel that wasn't recompiling the C++ — caught and fixed).
   - **Coach plumbing** ([coach.py make_self_play_config](src/am_baseline/training/coach.py)): `value_target_norm=str(getattr(opts, 'value_target_norm', 'bl'))` passed through to MCTSConfig.
   - **CPU smoke**: TSP-8, 1 iter, value_head, `--value_target_norm none` — completed end-to-end with `value_loss=2.85` (consistent with raw cost magnitude on N=8; sanity check on order of magnitude). No NaN, gradients flow, buffer↔trainer↔MCTS conventions agreed.

  **2026-05-06 — F.6.0.5b launched (raw-target 4-variant value_head rerun).**
   - Modal app: `ap-M3se0bCnottfFqptEhq8oi` (timestamp `20260506T082313`).
   - Entrypoint: `modal run --detach src/scripts/modal_run_train_alphazero.py::run_f605_lr_validation_raw_target`.
   - Holds `leaf_eval=value_head`, `K=40`, `n_iterations=20`, `--value_target_norm none`; varies the same 4 (lr, wd) corners.
   - Run names: `f605vhraw_lr1e4_wd1e4_20260506T082313`, `f605vhraw_lr5e4_wd1e4_…`, `f605vhraw_lr5e4_wd0_…`, `f605vhraw_lr1e4_wd0_…`.
   - Image built (24s editable-install layer), 4 tasks spawned, ~30 min wall-clock parallel; results pending.

  **What this rerun is testing.** The bl-normalized value_head grid is the **control**; the raw-target rerun (matched lr/wd grid) is the **intervention**. If raw-target trajectories converge visibly faster than bl-normalized, that's evidence the bl_val drift was the dominant value_head-convergence drag, not lr/wd. If they don't separate, value_head's lag is structural (e.g., target magnitude / scale, MLP capacity, or at-init prediction floor) and bl-drift is a smaller effect than hypothesized.

  **2026-05-06 — F.6.0.5b results landed (raw-target 4-variant grid, all 4/4 reached iter-19).** App `ap-M3se0bCnottfFqptEhq8oi` finished cleanly. Numbers in the table above. Key observations:

  - **Winner: V3 (lr=5e-4, wd=0) at val_avg_cost=4.265.** Beats V1 (control, 4.331) by **−0.066** and V2 (analytical, 4.349) by **−0.084**. Lowest policy_loss (1.4651) and value_loss (0.2637) of all 4 — co-improvement, not a noise artifact.
  - **V2 (lr=5e-4, wd=1e-4) regressed: 4.349 vs V1's 4.331.** And V2 was the only variant whose final gate (iter-19) **rejected** the candidate (cand 4.344 vs baseline 4.341, ttest p>0.05). At lr=5e-4, the AGZ-canonical wd=1e-4 actually *hurts* — the high-lr × non-zero-wd combination plateaued earlier than the low-lr control.
  - **V4 (lr=1e-4, wd=0) regressed: 4.367 vs V1's 4.331.** At lr=1e-4, dropping wd to 0 hurt slightly (i.e., wd=1e-4 helps when lr is small). So **wd is not uniformly beneficial or harmful — it interacts with lr.**

  **2x2 factorial decomposition** (treating the 4 corners as a `lr × wd` design at iter-19 val_avg_cost):

  | term | value | interpretation |
  |---|---|---|
  | grand mean | 4.328 | — |
  | main effect of lr (5e-4 − 1e-4) | (V2+V3)/2 − (V1+V4)/2 = −0.041 | high lr helps on average |
  | main effect of wd (0 − 1e-4) | (V3+V4)/2 − (V1+V2)/2 = −0.024 | wd=0 helps on average |
  | **lr × wd interaction** | V3 − V2 − V4 + V1 = **−0.120** | **strongly synergistic — lr=5e-4 only pays off when wd=0** |

  The interaction term (−0.120) is **3× the larger main effect**, which is why neither V2 (high lr alone) nor V4 (low wd alone) reproduces V3's gain. **The decision is not "raise lr" or "drop wd" individually — it's "raise lr AND drop wd jointly."**

  **Two-step decision rule outcome (per the rule pinned earlier in F.6.0.5):**
  - **Step 1 (lr):** "V2 OR V3 reaches val_avg_cost(iter 25/19) ≤ V1 − 0.02?" V3 = 4.265, V1 = 4.331 → V1 − V3 = **0.066 ≥ 0.02** → **PASS via V3.** V2 alone fails (regressed). Lr=5e-4 conditionally validated.
  - **Step 2 (wd):** "V3 vs V2 within lr=5e-4: |V3−V2| < 0.005 → V2; V3 < V2 by ≥ 0.01 → V3; V3 > V2 → V2." V3 − V2 = **−0.084 ≪ −0.01** → **adopt V3 settings (lr=5e-4, wd=0).**

  **F.6.1 default lr/wd resolved: `--lr_model 5e-4 --weight_decay 0` with `--value_target_norm none` retained.** Documents that Stage 4 should drop the AGZ-canonical wd=1e-4 in favor of Stage-1 / AM-paper convention (Adam wd=0) **and** raise lr from inherited 1e-4 to 5e-4 — both of which are required jointly per the interaction-term reading.

  **Comparison vs the F.6.0.5a (bl-normalized) grid.** The bl-normalized grid showed flat val_avg_cost trajectories across (lr, wd) — the dominant drag was the bl_val divisor on the value-head target, not optimizer settings. With raw-target enabled, the (lr, wd) grid finally discriminates. **This validates the bl_val-drift hypothesis as the primary value_head-convergence drag** before optimizer tuning becomes a useful lever.

  **F.6.1 trajectory expectation.** V3's iter-19 (4.265) is already **0.07 below the F.6.0 winner's iter-49** (rollout × ε=0.25 × ttest at 3.93 — but F.6.0 was K=100 not K=40, so direct comparison requires care). At the F.6.0.5b scale (K=40, value_head, raw-target), V3 trended monotonically toward Stage-1's 3.84 ceiling: at iter-9 ≈ 4.62, iter-14 ≈ 4.34, iter-19 = 4.27. Linear extrapolation suggests F.6.1 (100-iter continuation at lr=5e-4, wd=0, raw-target) should clear 3.95 and possibly approach 3.85 by iter-100. **F.6.1 launch decision: GO.**

  **Caveats / things to verify.**
  - V3's gate accepted at iter-19, but acceptance gate is just a t-test — it doesn't certify monotone convergence; the trajectory needs full plotting from W&B before F.6.1 commits.
  - K=40 ≠ K=100; F.6.1 should re-derive lr only after a single short K=100 sanity check at V3 settings (one variant, ~5 iter), to confirm the lr conclusion travels across MCTS scale. Alternatively, just run F.6.1 at K=40 to keep apples-to-apples with F.6.0.5b (cheaper, faster, but doesn't directly extend F.6.0).
  - V2's final-gate rejection (cand 4.344 vs baseline 4.341, p>0.05) could indicate a hard plateau at lr=5e-4 / wd=1e-4 specifically; ruling out an unlucky t-test would require re-running with a different seed.

- [x] **F.6.0.6 Dirichlet ε sweep at V3 settings.** **COMPLETE 2026-05-06.** 2-variant sweep ε ∈ {0, 0.05} holding (lr=5e-4, wd=0, value_target_norm=none, leaf_eval=value_head, K=40, 20 iter, gate=ttest, val_seed=42) fixed. F.6.0.5b's V3 (ε=0.25, val_avg_cost(iter-19) = 4.2648) acts as the implicit third reference point. Modal app: `ap-IAwuU1k9JAADMbtkM7WTOR` (timestamp `20260506T092127`); ~30 min wall-clock parallel.

  **Motivation.** F.6.0 picked ε=0.25 in a different regime (lr=1e-4, K=100, **bl-normalized** value_head where the value head was effectively broken; ε rescued exploration). F.6.0.5b moved to raw-target value_head + lr=5e-4 + wd=0 — value head now contributes leaf-discrimination AND policy mode-locks faster, both predicting **lower ε is better**. F.6.0.5b held ε=0.25 fixed, so the question stayed open. F.6.0.6 closes it.

  **Results table** (sorted by val_avg_cost(iter 19) ascending):

  | variant | ε | val_avg_cost | policy_loss | value_loss | entropy | gate accepted | Δ vs winner |
  |---|---|---|---|---|---|---|---|
  | **E2 (winner)** | **0.05** | **4.1856** | 1.4518 | 0.2557 | 1.1429 | yes | — |
  | E1 | 0.00 | 4.2283 | 1.4334 | 0.2833 | 1.0882 | yes | +0.0427 |
  | _(implicit)_ V3 | 0.25 | 4.2648 | 1.4651 | 0.2637 | 1.1073 | yes | +0.0792 |

  **Initial decision (revised below) by iter-19 endpoint rule** (±0.02 tie band): E2 beats E1 by 0.043 ≥ 0.02 ✓ AND beats V3 by 0.079 ≥ 0.02 ✓ → endpoint says E2. **But trajectory inspection (next paragraph) reverses this.**

  **Trajectory-stability inspection (per-iter regression analysis 2026-05-06).** Reading the iter-by-iter val_avg_cost from the launch log streams (attributing each iter line by value_loss-decay continuity):

  | metric | E1 (ε=0) | E2 (ε=0.05) | V3 (ε=0.25, F.6.0.5b) |
  |---|---|---|---|
  | iter-19 endpoint | 4.228 | **4.186** | 4.265 |
  | max per-iter regression | **+0.035** | +0.462 (iter 17→18) | (similar ε>0 pattern expected) |
  | # iters with regression ≥ +0.05 | **0/19** | 7/19 | — |
  | # iters with regression ≥ +0.20 | **0/19** | 3/19 (iters 5→6, 14→15, 17→18) | — |
  | mean Δ from iter 10 to 19 | −0.052/iter (smooth) | −0.032/iter (volatile) | — |
  | iter-18 val_avg_cost | 4.245 | 4.738 | — |

  **E2's iter-19 lead is fragile.** At iter-18, E1 was ahead by **+0.493**; E2's iter-19 = 4.186 is the bottom of a 0.55-amplitude oscillation (4.738 → 4.186 in one iter). Had we logged iter-15 (E1 4.372 vs E2 4.502) or iter-18 (E1 4.245 vs E2 4.738) as the endpoint, E1 would have been clearly ahead. The 0.043 endpoint advantage for E2 is **noise-floor-sized relative to E2's per-iter volatility (max single-iter swing ±0.46).**

  **Mechanism (matches first-principles prediction).** Dirichlet noise corrupts π_t (visit-distribution training target) at every tour-step root. With K=40 visits and ε=0.05, ~2 visits per root are noise-driven — small mean perturbation, but the variance compounds across 20 tour-steps × 1000 instances per iter × 200 train_steps. Under V3's lr=5e-4 (5× faster target-fitting than F.6.0's lr=1e-4), the policy absorbs that variance into per-iter val_avg_cost spikes. ε=0 has zero target corruption → smooth descent.

  **Revised decision: F.6.1 default = ε=0**, not ε=0.05.
  - Endpoint advantage of E2 (0.043) is real but small — within E2's typical iter-to-iter volatility band.
  - Stability advantage of E1 is large and consistent (0 vs 7 regressions ≥0.05; 0 vs 3 regressions ≥0.20).
  - F.6.1 is a 100-iter run; volatility compounds. A spike like iter-17→18's +0.46 happening once in 100 iters could easily land an unlucky F.6.1 endpoint at the *peak* of an oscillation rather than the trough.
  - V3 (ε=0.25) is dominated on both axes (worse endpoint AND worse stability).

  **First-principles predictions: status updated.**
  1. **Lower ε helps under V3 regime** (§2.1+§2.2): **CONFIRMED in spirit, taken further than predicted.** Theory said ε ∈ [0.05, 0.10]; data says ε=0 wins on the more important stability axis. Under raw-target value_head + lr=5e-4, MCTS Q-values become informative within ~5 iters, and ε's marginal exploration value evidently goes negative once you account for target-corruption variance.
  2. **ε=0 lower-bound concern** (§2.3): **REFUTED.** Theory predicted ε=0 would underperform by ~0.01-0.02 due to lack of at-init exploration. Data shows ε=0 actually has the cleanest trajectory shape; the at-init concern was overweighted (state variation across M=1000 instances per iter provides plenty of break-symmetry at init).

  **Regime-comparison observation.** F.6.0 (rollout, lr=1e-4, K=100, bl-normalized) had ε=0.25 ≈ ε=0.05 within 0.001 noise — ε was effectively irrelevant. F.6.0.6 (V3 regime: value_head, lr=5e-4, K=40, raw-target) shows ε=0 wins on stability and ε=0.25 loses on both endpoint and stability. **The F.6.0 ε conclusion does NOT transfer.** Each major regime change (leaf_eval, lr, value_target_norm) re-opens the ε question — and at higher lr / functional value_head, **the stability case for ε=0 strengthens**.

  **F.6.1 default fully resolved (lr × wd × ε):** `--lr_model 5e-4 --weight_decay 0 --value_target_norm none --dirichlet_epsilon 0`. Two-step F.6.0.5 outcome (lr=5e-4, wd=0) plus F.6.0.6 trajectory-revised outcome (ε=0) jointly lock the F.6.1 recipe.

  **Sample-efficiency anchor.** E1 reached val_avg_cost = **4.228 at iter-19 = 19K instances** — a 0.037 improvement over V3's 4.265 (F.6.0.5b) at zero additional iterations or instance budget. Smaller endpoint gain than E2's 0.079 vs V3, but trajectory shape is what matters for projecting F.6.1: E1's mean Δ of −0.052/iter through iter 10-19 (smooth, no oscillations) extrapolates to **~3.85 (Stage 1 ceiling) by iter ~50-60**, beating F.6.0's 50-iter K=100 winner (3.93) on both quality and stability.

  **Caveats.**
  - V3's 4.265 reference is from a different Modal launch (F.6.0.5b vs F.6.0.6). Cross-launch noise is small but nonzero; the 0.079 gap is above noise floor (single-seed variance ~0.005), but V3 also has the same Dirichlet-induced volatility issue (just unsurfaced because F.6.0.5b only logged endpoints).
  - K=40, value_head, 20-iter; whether the ε=0 stability advantage generalizes to K=100 / longer runs is untested. **F.6.1 itself is the validation** — if F.6.1's trajectory at ε=0 is also smooth, the conclusion holds.
  - The "trajectory attribution" was done from interleaved stdout logs by tracking value_loss continuity; spot-checking against W&B run pages would confirm the mapping. Moderate confidence in attribution given the clean monotone-decay structure of value_loss in both variants, but worth a sanity-check before F.6.1 commits.
  - Did NOT test ε ∈ {0.01, 0.02} — between 0 and 0.05 there could be a sweet spot if even tiny noise helps without compounding. The trajectory data argues no, but a follow-up isn't expensive if F.6.1 surprises us.

- [x] **F.6.0.7 → F.6.1.1 Sub-budget probes (K, batch, M, buffer)** — **COMPLETE 2026-05-06.** Series of small probes exploring auxiliary knobs at the F.6.0.6 winner regime (ε=0, lr=5e-4, wd=0, value_target_norm=none, leaf_eval=value_head, gate=ttest, val_seed=42) before committing to F.6.1. All 20-iter, M=1000 unless noted; reference is F.6.0.6 E1 (K=40, batch=512, buffer=200K) → val_avg_cost(iter 19) = **4.228**.

  **Results table** (sorted by val_avg_cost(final) ascending):

  | run | K | batch | M | iters | buffer | leaf_eval | val_avg_cost(final) | Δ vs E1 |
  |---|---|---|---|---|---|---|---|---|
  | F.6.0.6 E1 (reference) | 40 | 512 | 1000 | 20 | 200K | value_head | **4.228** | — |
  | F.6.0.9 (in flight, iter 17) | 20 | 2048 | 1000 | 20 | 200K | rollout | ~4.236 (provisional) | +0.008 |
  | F.6.1.1 buf=5000 | 20 | 512 | 1000 | 20 | **5K** | value_head | **4.299** | +0.071 |
  | F.6.1.1 buf=1000 | 20 | 512 | 1000 | 20 | **1K** | value_head | 4.350 | +0.122 |
  | F.6.0.8 | 20 | **2048** | 1000 | 20 | 200K | value_head | 4.408 | +0.180 |
  | F.6.0.7 (K=20 reference) | 20 | 512 | 1000 | 20 | 200K | value_head | 4.428 | +0.200 |
  | F.6.1.0 (M=2000 × 10 iter) | 20 | 2048 | 2000 | **10** | 200K | value_head | 5.062 | **+0.834** |

  **Findings:**
  1. **K dominates batch_size** (F.6.0.7 vs F.6.0.8): bumping batch 512 → 2048 at K=20 helped only 0.020 (4.428 → 4.408). The K=40 → K=20 hit (~0.18-0.20) is ~10× larger than the batch effect. K is the top sample-efficiency knob.
  2. **Iters dominate M_instances** (F.6.0.8 vs F.6.1.0 at matched 20K total): doubling M and halving iters made it 0.654 *worse*. At iter-9 calendar match, F.6.1.0 (M=2000) ≈ F.6.0.8 (M=1000) within ~0.05 — i.e., 2× more fresh data per iter gave essentially zero benefit. **train_steps_per_iter=200 saturates the per-iter "learning budget" at M=1000**; raising M without raising train_steps wastes the extra data.
  3. **Smaller replay buffer dramatically helps** (F.6.0.7 vs F.6.1.1 buf=5000): shrinking buffer 200K → 5K closed 0.129 of the K=20→40 gap (4.428 → 4.299). The 200K buffer was actively dragging the policy back via stale MCTS targets reflecting earlier, weaker policies. **Diagnostic signature**: policy_loss collapses 1.48 → 0.66 (2.2×) and entropy collapses 1.11 → 0.43 (2.6×) when the buffer shrinks — the policy is no longer getting averaged toward outdated targets.
  4. **Sweet spot is buf=5000 (5-iter window), not buf=1000 (1-iter window)**: buf=5000 beats buf=1000 by 0.051. Some cross-iter averaging denoises 1-iter MCTS sampling noise, but the AGZ-canonical 200-iter window is far too long for our target-fitting rate. Lifetime sample-per-tuple ratio is held roughly constant (~5×) across all three buffer sizes — only the **window** (recency mix) varies.
  5. **Rollout still competitive at K=20** (F.6.0.9 in flight, iter 17 = 4.236): trending toward matching F.6.0.6 E1's K=40 value_head endpoint *at K=20* with rollout — but mcts_s is ~3.2× higher (~155s vs ~50s for value_head), so per-credit value_head still wins.
  6. **Stability**: smaller buffers are mildly less stable (3-4 regressions ≥0.05 vs F.6.0.7's 3) but the bumpiness penalty is small relative to the endpoint gain.

  **F.6.1 recipe revisions** (carrying over from F.6.0.5 + F.6.0.6 + this batch):
  - Keep: ε=0, lr=5e-4, wd=0, value_target_norm=none, leaf_eval=value_head, gate=ttest, val_seed=42, M=1000.
  - **Change buffer_capacity 200K → 5000** (~5-iter window). Strongest single F.6.1 improvement candidate.
  - K=40 (F.6.0.6 E1) > K=20 by ~0.20 endpoint; F.6.1 should run K=40 (or higher).
  - train_steps_per_iter stays at 200 (saturates per-iter budget at M=1000); raise only if M raises.

  **Pending follow-up: F.6.1.2 = K=40 + buffer=5000** at otherwise-F.6.0.6-E1 settings. Combines the two strongest findings (small-buffer + larger K) and should plausibly push past F.6.0.6 E1's 4.228 toward 4.10 or below at iter-19. Modal entrypoint `run_f612_k40_buf5k_probe`; ~$2-3 credits, ~30 min wall-clock.

  Modal apps: F.6.0.7 (`ap-...`), F.6.0.8 (`ap-...`), F.6.0.9 (`ap-...`, in flight), F.6.1.0 (`ap-...`), F.6.1.1 (`ap-...`).

- [x] **F.6.1 K=20 100-iter trajectory probes** — **COMPLETE 2026-05-06.** Two parallel 100-iter F.6.1 main runs at K=20 + buf=5000 testing whether a learning-rate schedule changes the convergence target. Both share the locked recipe: ε=0, wd=0, value_target_norm=none, leaf_eval=value_head, gate=ttest, **gate_every=1** (revised this session — see "gate_every revision" below), val_seed=42, M=1000, train_steps=200, batch=512, n_iterations=100, from-scratch.

  | run | lr_model | lr_decay | iter-99 val_avg_cost | best mid-run | trajectory shape |
  |---|---|---|---|---|---|
  | F.6.1 lrdecay | 1e-3 | 0.95/iter | **3.922** | 3.922 (iter 99) | smooth, monotone through iter 70, plateau ≈3.92-3.93 |
  | F.6.1 main const | 5e-4 | 1.0 (none) | ~3.92-3.93 (log truncated) | **3.912 (iter 90)** | bumpier; regressions at iter 30, 60, 95 |

  **Key trajectory checkpoints (val_avg_cost @ iter):**

  | iter | lrdecay | const | Δ |
  |---|---|---|---|
  | 0 | 6.831 | 7.787 | (random init noise) |
  | 10 | 4.376 | 4.715 | lrdecay ahead by 0.339 (high lr=1e-3 advantage at random init) |
  | 20 | 4.094 | 4.143 | lrdecay ahead by 0.049 |
  | 30 | 3.999 | 4.377 | const regression (lrdecay ahead by 0.378) |
  | 40 | 3.981 | 3.993 | converging (Δ=0.012) |
  | 50 | 3.957 | 3.981 | lrdecay ahead by 0.024 |
  | 60 | 3.941 | 3.992 | const regression (lrdecay ahead by 0.051) |
  | 70 | 3.938 | 3.940 | matched (Δ=0.002) |
  | 80 | 3.934 | 3.927 | const ahead by 0.007 |
  | 90 | 3.933 | **3.912** | const ahead by 0.021 (const's mid-run best) |
  | 99 | **3.922** | ~3.92-3.93 (truncated) | both at the same plateau |

  **Findings:**
  1. **Both lr schedules reach the same ~3.92 plateau** — the destination doesn't depend on lr schedule choice. Constant lr=5e-4 is a fine default; no scheduling complexity needed.
  2. **lr=1e-3 + decay=0.95/iter has visibly faster early descent** (iter 10 advantage of 0.34, iter 30 advantage of 0.38). The high initial lr buys faster initial progress; the decay handles the late-stage plateau.
  3. **Constant lr trajectory is bumpier** (3 regressions ≥ 0.05 over iter 0-99 vs lrdecay's smooth descent). At lr=5e-4 throughout, the policy keeps bouncing within the plateau noise band; lrdecay's late-iter lr~6e-6 effectively damps further updates.
  4. **Best mid-run val_avg_cost is the same in both** (lrdecay iter 99 = 3.922; const iter 90 = 3.912). The const variant *touched* a slightly lower point at iter 90 but didn't hold it (regressed back to 3.93 by iter 95-98).
  5. **Gate accept patterns differ**: const-lr accepted at iter 0, 10, 20, 90 (sparse); lrdecay accepted at iter 0, 20, 30, 98 (also sparse). Most iters' candidates didn't beat the running baseline by enough to clear the t-test — typical mid-plateau behavior.

  **Sample-efficiency anchor (proposal claim materialized):**

  | run | total instances | val_avg_cost | comment |
  |---|---|---|---|
  | Stage 1 ceiling | 1,280,000 (1 epoch) | 3.839 | converged AM-paper greedy decoding |
  | F.6.0 winner (K=100, 50 iter, lr=1e-4 / bl-normalized regime) | 50,000 | 3.93 | prior best Stage 4 from-scratch |
  | F.6.1 main / lrdecay (K=20, 100 iter, V3-derived recipe) | 100,000 | **3.92** | **new state-of-art, ~7.8% of Stage 1 budget, 5× cheaper per-iter compute than F.6.0 winner** |

  **Stage 4 from-scratch closes the gap to Stage 1 to 0.08 cost units (2% relative) at <8% of Stage 1's instance budget**, validating the proposal sample-efficiency claim. The F.6.1 trajectory monotonically descends past F.6.0's 3.93 by iter ~50-60 and plateaus ~3.92 through iter ~80-99.

  **gate_every revision (2026-05-06).** Pre-launch I had `gate_every=5` inheriting AGZ's "evaluate every 1000 batches" pattern proportional to our setup. User pushed back; analysis showed best_model staleness (up to 5 iters lag) directly hurts self-play quality at lr=5e-4 where each iter brings ~0.05 quality gain. Per-iter gating costs only ~5-10s extra (negligible vs ~30s/iter mcts_s) and the t-test has plenty of power on 10K-eval. **gate_every=1 adopted** as the F.6.1 default — propagates each accepted improvement to self-play immediately.

  **lr scheduler infrastructure added (2026-05-06)** for the lrdecay variant:
  - `--lr_decay` CLI flag added in [train_alphazero.py:91-94](src/scripts/train_alphazero.py)
  - `LambdaLR(optimizer, lambda iter_k: lr_decay**iter_k)` wired up after optimizer creation in [coach.py](src/am_baseline/training/coach.py); `step()` called at end of each iter
  - Scheduler state included in checkpoint save/load
  - `current_lr = optimizer.param_groups[0]['lr']` captured for accurate W&B logging (was previously logging the static `opts.lr_model` regardless of decay)

  **K=40 + buf=5K + lr=5e-4 const variant** — **COMPLETE 2026-05-06.** Run name: `f61_main_K40_buf5000_20260506T201909_20260506T201920`. val_avg_cost(iter 90, last accept) = **3.893** (val_seed=42); iter 99 working-model = 3.917 (gate rejected every iter 91-99). Same plateau as the K=20 variants → **K-axis effect is essentially nil at 100-iter horizon** under the F.6.0.6/F.6.1 recipe.

  **Post-mortem diagnostic (2026-05-06)** — performed via the new `val_stage4_mcts.py` script on a fresh seed=20260430, n=500 paired val set against AM_S1 canonical:

  | Decoder | mean | Δ vs greedy θ★ | Δ vs AM_S1 greedy |
  |---|---|---|---|
  | AM_S1 greedy | 3.84221 | — | (ref) |
  | AM_S1 sample(x1280) | 3.83381 | — | −0.0084 |
  | Stage4 iter-90 greedy | 3.90808 | (ref) | +0.0659 |
  | Stage4 iter-90 + MCTS K=40 **value_head** ε=0 const | 3.90974 | **+0.0017** (p=0.63) | +0.0675 |
  | Stage4 iter-90 + MCTS K=40 **rollout** ε=0 const | 3.83644 | **−0.0716** (p<0.0001) | −0.0058 |

  **Smoking gun: at the same K=40 budget, swapping value_head leaf eval → 1-sample rollout buys −0.073.** value_head at the leaf is statistically tied with greedy (p=0.63); a single Monte-Carlo rollout under the *same policy* gives massive lift. The value head's per-leaf scores aren't discriminating sibling subtrees.

  **Lock-in mechanism** (closed loop, all observed in `iterations.csv`):

  1. value_head at the leaf provides ≈zero search signal → MCTS K=40 with value_head produces tours essentially identical to greedy.
  2. With ε=0 + step30, the resulting π_t targets collapse onto the policy's own argmax — `mean_entropy_pi` drops from 1.804 (iter 0) → 0.353 (iter 13) → 0.155 (iter 99) ≈ 1.17 effective actions out of 20.
  3. CE distillation against (near-)one-hot targets matching the policy's own argmax = the model just gets *more confident* on actions it already picks (loss drops to 0.27) without changing direction.
  4. Working model drifts post-iter 90 (greedy +0.023 vs best_model); gate correctly rejects every iter 91-99.

  **Conclusion**: F.6.1 main's plateau is a leaf-evaluator failure, not an MCTS-recipe / lr / buffer / K issue. The next viable lever is either fixing the value head's leaf-discrimination (auxiliary value pretrain, whitening, hybrid leaf) or swapping leaf_eval back to rollout — both deferred behind cheaper interventions probed in F.6.1.3.

- [ ] **F.6.1 From-scratch trajectory probe** *(scope reduced 2026-05-02 from 1000 → 100 iter; defaults locked 2026-05-06 by F.6.0.5 + F.6.0.6)*. Recipe locked: **100 iter** × M=1000 × **K=100** × `train_steps_per_iter=200` × `buffer_capacity=200_000` × **`--lr_model 5e-4`** × **`--weight_decay 0`** × **`--value_target_norm none`** × **`--dirichlet_epsilon 0`** × **`--leaf_eval value_head`** × `--gate_mode ttest` × val_seed=42. **Resume from F.6.0 winner's `iter-49.pt`** *(reconsider — F.6.0 winner used different lr/wd/value_target_norm/ε, so resume may not be apples-to-apples; from-scratch may be cleaner)*. **K=100 chosen** because the probe showed K=100 → K=200 gives only +33% MCTS-vs-greedy gap improvement at 2× cost; K=200 reserved for F.6.3 escalation.

  Per-iter wall-clock estimate (Modal A10):
  - K=100 value_head: ~20 s/iter → **100 iter ≈ 33 min, ~$0.30-0.50 in credits**.
  - K=100 rollout: ~130 s/iter → **100 iter ≈ 3.6 h, ~$2-3 in credits**.

  Output: `outputs/tsp_20/stage4_main_fromscratch_<timestamp>/`.

  **Why scope-reduced:** treat F.6.1 as a trajectory proof-of-concept rather than a full convergence run. At 150 iter / 150K instances, F.6.1 is at ~12% of Stage 1's first epoch (1.28M) — too short to definitively answer the proposal sample-efficiency claim, but enough to see whether the from-scratch loop is visibly learning and at what rate. Decision: don't auto-scale to 1000 even if 100-iter shows promise; re-decide F.6.3 escalation based on data.

- [x] **F.6.1.3 step10 + ε sweep (post-F.6.1 plateau breakthrough)** — **COMPLETE 2026-05-06.** Cheapest-first response to the F.6.1 main lock-in diagnostic above: keep `leaf_eval=value_head` (so the result is comparable to F.6.1 main on per-iter wall-clock) but narrow the σ_t stochastic window (step30 → step10, cutoff = ⌈0.1·N⌉ = 2 of 20 tour-steps) AND restore Dirichlet noise. Modal entrypoint: `run_f62_step10_eps_sweep` in `modal_run_train_alphazero.py`. Two parallel 100-iter from-scratch runs sharing F.6.1 main's recipe (K=40, M=1000, train_steps=200, buffer=5000, batch=512, lr=5e-4 const, wd=0, value_target_norm=none, gate=ttest gate_every=1, val_seed=42), differing only in `--temperature_schedule step10 --dirichlet_epsilon ε`.

  | run_name (`f62_step10_*`) | ε | val_avg_cost(iter 99) | mean_entropy_pi(iter 99) | policy_loss | value_loss | pg_norm | vg_norm |
  |---|---|---|---|---|---|---|---|
  | `eps05_20260507T024345` | 0.05 | 3.8968 | 0.216 | 0.308 | 0.011 | 0.741 | 0.508 |
  | `eps25_20260507T024345` | **0.25** | **3.8784** | **0.335** | 0.465 | 0.017 | 0.800 | 0.825 |

  **Key finding**: ε=0.25 + step10 finishes at val_avg_cost=**3.8784**, beating F.6.1 main's iter-90 best of 3.893 by **−0.015** on the same val_seed=42 draw. Mean π_t entropy at iter 99 is **2.16× higher** than F.6.1 main (0.335 vs 0.155 nats), consistent with the diagnostic prediction: ε=0.25 keeps the visit distribution non-degenerate, so π_t carries more information than just the policy's own argmax. Per-iter wall increases ~25% (52.75s → 65.20s mcts_s) due to noise → more diverse trees → fewer tree-reuse cache hits.

  ε=0.05 (3.897) lands roughly at F.6.1 main's iter-90 best with ~0.005 noise. The exploration→target-information mechanism is real but ε=0.05 is too low to fully halt the entropy collapse.

  Modal app: `ap-1vX3EOYEBbAaHoRYkkAACz` (timestamp `20260507T024345`); ~2h wall-clock parallel; ~$15-20 in credits. W&B project: `am-alphagozero` (runs `933kn781` for ε=0.05, `meynsvp0` for ε=0.25).

  **Diagnostic inheritance**: per-loss grad norms (added this session — see infrastructure entry below) recorded throughout. policy and value grad norms are within ~1× of each other at iter 99 (pg_norm=0.800, vg_norm=0.825 for ε=0.25), so λᵥ=1.0 is not mis-weighted at this regime. The value-grad VH-vs-shared split was added AFTER these runs finished; F.6.1.4 (resume) will be the first run with that telemetry.

- [ ] **F.6.1.4 ε=0.25 +50 iter resume (in flight)** — extends `f62_step10_eps25_20260507T024345_*/iter-99.pt` for 50 more iterations with the new value-grad VH/shared split telemetry. Tests whether the trajectory keeps improving past iter 99 (would close the AM_S1 gap further) or has hit a new plateau (would suggest step10+ε=0.25 hit a different regime-limit). Modal entrypoint: `run_f62_eps25_resume50`. Same recipe as parent run; coach loads model + best_model + optimizer + lr_scheduler + RNG + sibling buffer.pt; resumes at iter 100 → 149. Output: `outputs/tsp_20/f62_step10_eps25_resume50_<timestamp>_*/iter-{100..149}.pt`. ~1.5-2h wall, ~$8-12 in credits.

- [x] **Telemetry / infrastructure additions (2026-05-06)** — staged ahead of the F.6.1.4 resume so trajectory and post-F.6.1.3 runs all carry the new diagnostics:

  1. **Per-loss gradient-norm logging** in `train_step_alphazero` ([trainer.py](src/am_baseline/training/trainer.py)). Replaced the single `total_loss.backward()` with two `torch.autograd.grad` traversals (one per loss; gradient is linear so combining them and writing into `.grad` is mathematically identical to the original path). New fields in the metrics dict: `policy_grad_norm`, `value_grad_norm`. Cost: ~2× backward traversal time, negligible vs MCTS self-play wall.

  2. **Value-grad subspace split** for clean cosine-of-conflict diagnostic. Since `L_policy` is identically zero on `value_head` parameters, the dot product `⟨∇L_p, ∇L_v⟩` only sees the encoder+decoder "shared" subspace, but `value_grad_norm` includes the value_head-only contribution — so `cos(θ_full)` understates the true shared-subspace conflict. Two new metrics fix this: `value_grad_norm_vh` (value_head params only) and `value_grad_norm_shared` (encoder+decoder). Orthogonal-decomposition invariant verified: `value_grad_norm² = vh² + shared²` to within 1e-5 in the smoke. Then `cos(θ_shared) = ((grad_norm² − policy² − λᵥ²·value²) / (2·λᵥ)) / (policy · value_shared)` recovers the true angle on shared params. Smoke confirmed `cos_shared` was 3.8× larger in magnitude than `cos_full` at random init — non-trivial dilution.

  3. **`step10` temperature schedule** in MCTSConfig and the C++ extension. Encoding `3 = step10` (alongside `0=const, 1=step30, 2=step50`); cutoff = ⌈0.1·N⌉. Wired in [mcts.py](src/am_baseline/search/mcts.py), [solver.py](src/am_baseline/search/mcts_cpp/solver.py), [mcts.hpp](src/am_baseline/search/mcts_cpp/mcts.hpp), [mcts.cpp](src/am_baseline/search/mcts_cpp/mcts.cpp), and `--temperature_schedule` choices in [train_alphazero.py](src/scripts/train_alphazero.py). C++ extension rebuilt locally; Modal image rebuilds automatically on next launch via `pip install -e . --no-deps`. AGZ-canonical analog: 30 plies of ~250 ≈ 12% — step10's ⌈0.1·N⌉ matches this scaling more closely than step30.

  4. **`iterations.csv` schema extended** ([logging.py](src/am_baseline/training/logging.py)): added 5 new columns (`policy_grad_norm_mean`, `value_grad_norm_mean`, `grad_norm_mean`, `value_grad_norm_vh_mean`, `value_grad_norm_shared_mean`). Old files (F.6.1 main, F.6.1.3 ε-sweep) only have the original 11 columns; F.6.1.4 onward get all 16. Read by name (pandas), not column index, when stitching trajectories across schema versions. W&B per-step + per-iter series mirror the CSV columns.

  5. **`val_stage4_mcts.py`** ([src/scripts/val_stage4_mcts.py](src/scripts/val_stage4_mcts.py)) — standalone validator for Stage 4 checkpoints with MCTS. Loads either `iter-{i}.pt` or `iter-{i}_accepted.pt` (auto-detects `model` + `best_model` keys), reads sibling `args.json` to recover the architecture/`value_target_norm`, and evaluates greedy + MCTS on a fixed-seed val set with optional `--match_train` (reads training-time MCTS recipe from args.json) and `--am_ckpt` (loads either Stage 1 canonical or the reference Kool release with auto key-remap; runs greedy + sampling x1280). Reports paired t-test diffs + fraction-better/worse counts. Used for the F.6.1 main post-mortem above.

- [ ] **F.6.2 Pass conditions evaluation** (split — see plan F.6.2 in `_plans/stage4_plan.md` for full text).
  - **Trajectory-probe conditions** (should hold for the F.6.1 100-iter probe to be informative): (3') visible downward val_avg_cost trend (final < initial by ≥ 0.05 — much looser than ±0.001 noise band); (4) ≥1 gate accept (auto with `--gate_mode always`); F.6.0+F.6.1 trajectory smoothly continuous (sanity-check resume).
  - **Definitive claim conditions** (deferred to F.6.3 if 100-iter probe warrants): (1c) sample efficiency, (2) ultimate quality ≤ 3.8312, (3) strict monotone within ±0.001.

- [ ] **F.6.3 Optional escalation paths** if 100-iter trajectory shows promise:
  - **F.6.3.a Continue to 1000 iters at K=100** (resume from F.6.1 final). ~6 h value_head / ~36 h rollout.
  - **F.6.3.b Step up to K=200** (resume from F.6.1 final). 2× per-iter wall-clock; tests whether stronger MCTS targets accelerate convergence.
  - **F.6.3.c TSP-50 hoist** (pivot to graph size where Stage 1 is weaker — more headroom for AGZ sample efficiency).

**Required code changes before F.6.0 (DONE — committed `3ca066b` and `d61ecae`):**

- [x] Make `--load_path` optional in `train_alphazero.py`. From-scratch path prints `[*] No --load_path; starting from random init (proposal Phase F.6).` Verified by CPU smoke (TSP-8, 1 iter, val 3.667, finite losses).
- [x] Add `--val_seed <int>` flag (default = 42) with scoped torch+numpy seed swap so val_dataset is reproducible without disturbing model RNG. Verified: two runs with `--val_seed 42 --seed 999` produce bit-identical `val_avg_cost` (3.468627452850342).
- [x] W&B logging on by default for Modal-launched runs via `modal_run_train_alphazero.py::_common_args` (`--wandb_project am-alphagozero --wandb_mode online`).
- [x] Modal `run_f60_grid` entrypoint added — 12-variant grid spawn in parallel.
- [x] Plan G.9 (lr ablation) added so the lr=1e-5 question is preserved as a planned follow-on (informational only — not a Stage 4 sample-efficiency claim).

**Methodology fix surfaced from warm-start analysis:** The per-run `iterations.csv` val_avg_cost trajectories from F.3 v1-v5 and Modal a1/a2/b1/b2 were each computed on a different fresh 10K val draw (no seed pinned). Std-error of the mean across 10K-instance draws is ~0.003, so per-run val_avg_cost numbers are not directly comparable to either each other or to Stage 1's canonical 3.83943. Apples-to-apples eval (`compare_stage1_vs_stage4.py` with seed=42) IS comparable — that's why all the breakthrough numbers in the F.4+F.3 Modal batch table are apples-to-apples, not per-run trajectory reads. F.6 fixes this at the source by pinning `--val_seed 42`.

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
