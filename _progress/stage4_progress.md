# Stage 4 Progress: AlphaGo-Zero-Style MCTS Self-Improvement Loop on TSP-20

**Plan:** [`_plans/stage4_plan.md`](../_plans/stage4_plan.md)
**Started:** 2026-04-29
**Closed:** 2026-05-13 (proposal-aligned "self-improvement loop converges" claim satisfied via Phase F.6.0 + F.6.1 from-scratch runs).
**Status:** **Phases A–E complete; Phase F.1–F.5 closed (warm-start variant, superseded); Phase F.6.0 + F.6.1 demonstrate the proposal Stage 4 convergence claim. Deeper recipe-tuning, value-head ablation, lv0, TSP-50 scaling, and Track A wall-time optimizations are partitioned to Stage 5** — see [`_progress/stage5_progress.md`](stage5_progress.md) and [`_plans/stage5_plan.md`](../_plans/stage5_plan.md).

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
- All three backend smokes pass (`smoke_mcts.py --backend python|cpp|cpp_batch`); A13 a/b/c sub-cases all green.
- **Bit-equivalence (A13.c):** under deterministic clamp (ε=0, τ=0) python and cpp produce identical visit-count dicts at every tour-step on a TSP-20 K=200 instance; cpp and cpp_batch also identical.
- C++ extension rebuilt cleanly; pybind11 wire format only adds an optional dict key — Stage 2/3 callers see no behavioral or memory change with `return_root_visits=False` (default).

**Phase A completion note (2026-04-30):** Edits landed in this worktree:
| File | LOC delta | Change |
|---|---|---|
| `src/am_baseline/search/mcts.py` | +29 | `MCTSConfig.return_root_visits`, `MCTSSolver.root_visit_dists` init/reset/append. |
| `src/am_baseline/search/mcts_cpp/mcts.hpp` | +4 | `Config::return_root_visits`. |
| `src/am_baseline/search/mcts_cpp/mcts.cpp` | +81 | sequential and batched per-step visit dumps + result-dict marshalling. |
| `src/am_baseline/search/mcts_cpp/solver.py` | +40 | `CppMCTSSolver.root_visit_dists`, `CppBatchMCTSSolver.root_visit_dists_per_instance` plumbing. |
| `src/scripts/smoke_mcts.py` | +293 | A13 helper (`_run_a13_visit_dists`) + call sites in each backend entrypoint and main. |

- **Implementation note (sparse storage):** Both Python and C++ store only actions with N>0 (sparse `{action: count}`). Phase C must zero-init `(N,)` arrays and scatter when building dense `pi_t`. Visited cities are never keys (legality invariant ii from A13.a).
- **Implementation note (tree reuse):** Under tree_reuse=True the per-step total exceeds K (root inherits subtree visits) — this is the correct π_t to distill. Phase C sanity assertions should not assume `Σ N == K` outside `tree_reuse=False`.

### Phase B — Replay buffer + distillation training step (no GPU) — **COMPLETE 2026-04-30**

- [x] **B.1** `MCTSReplayBuffer` class in `src/am_baseline/training/coach.py` — flat dict-of-pre-allocated-tensors (per-instance: coords, bl_val FROZEN, tour_cost; per-step: pi raw τ=1, visited, first/prev_a, lengths, cost_to_go, inst_idx). ~520 MB fixed footprint. Ring-buffer eviction.
- [x] **B.2** State-tensor reconstruction utility (mirrors `mcts_cpp/solver.py:_state_from_snapshot`).
- [x] **B.3** `train_step_alphazero` in `src/am_baseline/training/trainer.py` — per-state π_t distillation (raw τ=1) + per-state z_t = cost_to_go/bl_val (V_CURRENT shape, matches `value_targets_from_edges`). Squeeze log_p/mask from (B,1,N) to (B,N) before CE.
- [x] **B.4** Smoke A1 in `src/scripts/smoke_alphazero.py`. A1.5/A1.6 buffer invariants + save/load round-trip + state-reconstruction round-trip all green.

**Smoke run** (CPU, AM_AlphaGoZero env, N=8, 5 instances, embedding_dim=32):
```
[A1] metrics = {policy_loss: 1.81, value_loss: 0.66, total_loss: 2.47,
                mean_entropy_pi: 1.53, mean_z: 0.86, gradient_norm: 2.90}
PASS — all Phase B smoke checks succeeded.
```
Encoder, decoder, value_head, and init_embed parameters all moved (gradient flow confirmed end-to-end). Ring-buffer eviction sanity-checked (push 5 into capacity 3 → wrapped to 3 instances, step_counts == [3]*N).

**Phase B completion note (2026-04-30):** Edits landed in this worktree:
| File | LOC delta | Change |
|---|---|---|
| `src/am_baseline/training/coach.py` | +~430 (new) | `MCTSReplayBuffer` (push/sample/sample_step/save/load/_rebuild_step_index/step_counts) + `reconstruct_state(batch, device)`. |
| `src/am_baseline/training/trainer.py` | +~115 | `train_step_alphazero(model, optimizer, batch, opts) -> metrics dict`. |
| `src/scripts/smoke_alphazero.py` | +~280 (new) | A1 (gradient flow), B0 (buffer invariants), B1 (save/load + step_index rebuild), A1b (reconstruct_state -> decode_step round-trip). |

- **Decoder-graph correctness:** `train_step_alphazero` calls `model.encode` and `model.precompute_decoder` *inside* the train step (not cached), so gradients flow through the same compute graph as self-play. `decode_step` returns `(B,1,N)` tensors that are squeezed to `(B,N)` before the CE — A1b confirms that `mask` aligns with visited cities (every visited entry has `log_p == -inf` after the squeeze). The `0 * -inf = NaN` corner is handled with `torch.where(mask, zeros, log_p)` before the multiply.
- **Stubs / TODOs that were resolved by Phase C/D:**
  - `_synth_instance_records` in the smoke harness fabricated `pi` from a Dirichlet draw. Phase C replaced with `generate_self_play_batch(...)` output.
  - `MCTSCoach` orchestrator (Phase D) consumed `MCTSReplayBuffer.sample()` directly; no buffer-API changes needed.

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

### Phase F.1–F.5 — Warm-start TSP-20 pilots — **CLOSED 2026-05-02** (superseded by F.6 from-scratch)

**Reframing (2026-05-02):** all F.1–F.5 work assumed warm-start from Stage 1's `epoch-99.pt`. Subsequent reading of `proposal.md:140-141` ("AGZ reaches AM-equivalent quality with fewer total training instances") clarified the proposal claim requires a **from-scratch** sample-efficiency curve. Warm-start became a documented diagnostic, not the proposal mainline. F.6 (below) is the proposal-aligned line.

**Work done in this arc:** infrastructure build-out (F.1 CLI, F.2 smoke battery, F.5 plotter) + 5 local pilot attempts (F.3 v1–v5) + 4-job Modal batch (F.4 a1/a2/b1/b2). Only `b2 (--lr_model 1e-5)` beat Stage 1 (Δ=−0.00137, p<0.0001) — the first ever gate-accept across 7 attempts. All other variants tied with or degraded Stage 1 by 0.001–0.005.

**Transferable findings worth keeping:**

1. **AGZ-canonical exploration (ε=0.25 + step30 with τ=1) is toxic on a converged warm-start.** With a sharp prior, 25% Dirichlet + first-30%-steps τ=1 sampling commits to disastrous prefixes. Probe ([probe_mcts_quality.py](src/scripts/probe_mcts_quality.py), 200 fresh TSP-20 instances) showed MCTS-K=50-vh-step30+ε=0.25 produces tours **+0.246 worse than greedy θ★ on average** — the "improvement engine" was producing a worse-than-greedy distillation target. Safe regime for warm-started training: **ε ≤ 0.05** with rollout leaf eval. Memory: [project_alphagozero_warmstart_exploration.md](C:\Users\Jun18\.claude\projects\c--Users-Jun18-Desktop-AM-ALPHAGOZERO\memory\project_alphagozero_warmstart_exploration.md). Note: this fragility does NOT transfer to from-scratch — F.6.0's 12-variant grid found ε=0.25 in the rollout winner cluster from random init.

2. **Bug fix commit `419a857`:** `make_self_play_config` was hardcoding `leaf_eval`/`dirichlet_*`/`temperature_schedule` and silently ignoring CLI flags. F.3 v1 and v2 ran the canonical-AGZ recipe regardless of CLI args. Production bug; would have bitten F.6 if undiscovered. Fix: take all four as kwargs; `MCTSCoach.learn` reads from `opts`.

3. **Stage 1 is at the architectural ceiling on TSP-20.** Even strong MCTS (K=200 rollout) only beats greedy by ~0.008 on 39% of instances; on the other 61% MCTS ties or loses. Distillation captures the *mean* of MCTS-vs-greedy outcomes — wins and losses approximately cancel, slightly net-negative due to shared-encoder gradient noise. Mechanistic implication: **for the AGZ self-improvement claim to have meaningful headroom, the underlying model must not already be at the ceiling** — TSP-50 (where Stage 1 is weaker) is the natural test bed. (This is why we later moved to TSP-50 in Track 4.)

4. **lr=1e-4 overshoots from converged checkpoints; lr=1e-5 fixes it — WARM-START ONLY.** Modal b2 (lr=1e-5, K=200 rollout, ε=0.05, ttest, 20 iter): val_avg_cost = 3.83485, Δ vs Stage 1 = **−0.00137 (p<0.0001)**. **Critical caveat: this finding did NOT transfer to from-scratch.** F.6.0.5 re-derived lr from CE-distillation gradient analysis and concluded lr=5e-4 (random-init has no converged optimum to overshoot). Treat lr=1e-5 strictly as a warm-start recipe. Memory: [project_lr_fairness_for_stage4.md](C:\Users\Jun18\.claude\projects\c--Users-Jun18-Desktop-AM-ALPHAGOZERO\memory\project_lr_fairness_for_stage4.md).

5. **Pin `--val_seed` for cross-run comparison.** Pre-F.6 runs each rolled a fresh 10K val draw with no seed pinned; SEM across draws is ~0.003 so per-run val_avg_cost numbers are not directly comparable. The apples-to-apples breakthroughs in this arc all used `compare_stage1_vs_stage4.py` with seed=42 paired-eval. F.6 pins `--val_seed 42` at the source.

**Infrastructure built (all carried into F.6 and beyond):**

| file | role | reuse |
|---|---|---|
| [src/scripts/train_alphazero.py](src/scripts/train_alphazero.py) | Stage 4 launcher CLI (F.1) | F.6+ — `--load_path` made optional for from-scratch |
| [src/scripts/smoke_alphazero.py](src/scripts/smoke_alphazero.py) | A1–A6 smoke (F.2) | Every Phase F.6 launch validated via this battery; A3 added here guards spec §4.2 choice (B) decoupling between σ_t and π_t |
| [src/scripts/probe_mcts_quality.py](src/scripts/probe_mcts_quality.py) | MCTS buffer-quality probe | Reused for F.6.0 mechanism analysis + F.6.1.6 vh-bias probe chain |
| [src/scripts/plot_stage4.py](src/scripts/plot_stage4.py) | Sample-efficiency plotter (F.5) | Headline-plot generator (log-x stage1 vs stage4 + Gurobi/Stage1/Stage3-K400 horizontals) |
| [src/scripts/compare_stage1_vs_stage4.py](src/scripts/compare_stage1_vs_stage4.py) | Apples-to-apples seed=42 paired-eval comparator | Built for v4 + Modal batch; standard tool to disambiguate per-run val noise from real Δ |
| [src/scripts/modal_run_train_alphazero.py](src/scripts/modal_run_train_alphazero.py) | Modal launcher | All later Modal runs (F.6.0 grid through Track A) extended this |

**What the arc is NOT useful for going forward:**
- The lr=1e-5 recipe (warm-start-specific; F.6.0.5 re-derived lr=5e-4 for from-scratch).
- Detailed v1–v5 trajectory numbers and gating dynamics (the headline takeaways above + the two memory entries capture everything load-bearing; the per-run iter-by-iter details are not transferable).
- The "Phase F dev-portion completion note" file-edits table (redundant with the infrastructure table above).

### Phase F.6 — Proposal-aligned from-scratch run — **COMPLETE 2026-05-06** (convergence claim)

**Status:** F.6.0 + F.6.1 demonstrate the proposal Stage 4 convergence claim. F.6.0.5 → F.6.1.6 + lv0 recipe-tuning and ablation work migrated to Stage 5. Discovered during post-Modal-breakthrough discussion that all Phase F warm-start work (F.1-F.5, including the b2 lr=1e-5 result) addresses a *related but distinct* question from what `proposal.md` Stage 4 actually promises.

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

- [x] **F.6.0 Pre-flight grid probe (12 variants).** **COMPLETE 2026-05-03.** 12 parallel Modal jobs (app `ap-mIrMZH1Gi3j9yWopjtapoI`, ~9.5 h elapsed) cross `leaf_eval ∈ {value_head, rollout}` × `ε ∈ {0, 0.05, 0.25}` × `gate_mode ∈ {ttest, always}`. All from-scratch, 50 iter × M=1000 × K=100 × train_steps=200 × buffer=200K × lr=1e-4 × val_seed=42. Modal entrypoint: `run_f60_grid` (commit `d61ecae`).

  **Headline: `leaf_eval` is the dominant knob.** All 6 rollout variants finished in [3.9328, 3.9450]; all 6 value_head variants in [4.1338, 4.2817] — 0.20 cluster separation, well outside any noise band. Within either cluster, ε and gate_mode are essentially irrelevant (rollout spread 0.012, value_head spread 0.148 driven by a single ε=0 outlier).

  **Winner: `lerol_eps25_ttest`** (rollout × ε=0.25 × ttest gate) — iter-49 val_avg_cost = **3.9338**, Δ = 0.617 from iter 0, best mid-run = 3.9314 (iter 46), 10/10 gate accepts, 464.6 s/iter (rollout ~2.1× slower per iter than value_head at K=100, but the only leaf eval that descends materially from random init).

  **value_head failure mechanism** (diagnosed from `iterations.csv`): `value_loss` collapses 0.091 → 0.024 in one iter then plateaus. Not the head failing to train — it's the head **trivially fitting the per-state mean** of `z_t = (tour_cost − lengths_t)/bl_val`. At random init `tour_cost` and `bl_val` come from the same noisy policy and are tightly correlated → tiny across-instance variance → MSE landscape dominated by the constant component → **`v(s) ≈ const` for all expanded leaves → zero leaf discrimination in MCTS**. Rollout doesn't have this break (terminal reads real cost). The `/bl_val` normalization isn't the bug per se but it's the channel via which the variance collapse happens — F.6.0.5 below fixes this with `--value_target_norm none`.

  **Notable: ε=0.25 is in the winner cluster from random init**, reversing the F.1-F.5 finding that ε=0.25 is toxic on a converged warm-start. AGZ-canonical exploration is safe when the prior is uniform.

  **Sample-efficiency anchor**: F.6.0 winner reaches 3.93 at 50K instances vs Stage 1 final 3.84 at 1.28M — **0.09 above Stage 1 quality at ~2.6% of Stage 1's instance budget**. Encouraging signal but the 0.09 gap drove the F.6.0.5 → F.6.1.4.c recipe chain below.

  Diagnostic infra: [probe_grad_norm.py](src/scripts/probe_grad_norm.py) (Stage 4 CE grad mean 3.76 vs Stage 1 REINFORCE 90.87 — ~24× smaller, matches O(1) vs O(advantage·N) prediction; motivates the F.6.0.5 lr re-derivation).

- [x] **F.6.0.5 → F.6.0.6 → F.6.0.7-1.1 recipe lockdown** *(2026-05-03 → 2026-05-06)* — Moved to Stage 5 §A. Headline: V3 (lr=5e-4, wd=0, value_target_norm=none, ε=0, K=40, M=1000, buffer=5000) replaces the original lr=1e-4 setting. See [`stage5_progress.md`](stage5_progress.md) §A for the lr×wd factorial, ε sweep, and K/buffer/batch/M sub-budget probes.

- [x] **F.6.1 K=20/K=40 100-iter trajectory probes** — **COMPLETE 2026-05-06.** Three parallel 100-iter from-scratch runs at the F.6.0.7-1.1 locked recipe (gate_every=1 adopted post-launch; lr scheduler infra added). **This is the proposal Stage 4 convergence demonstration — "tour quality improves over successive iterations".**

  | run | K | lr schedule | iter-99 val | best mid-run |
  |---|---|---|---:|---:|
  | F.6.1 lrdecay | 20 | 1e-3 × 0.95^iter | **3.922** | 3.922 (iter 99) |
  | F.6.1 main const | 20 | 5e-4 const | ~3.92-3.93 | **3.912** (iter 90) |
  | F.6.1 K=40 const | 40 | 5e-4 const | 3.917 | **3.893** (iter 90) |

  **K and lr-schedule are essentially neutral at this horizon.** lrdecay buys faster early descent (iter 10 advantage of 0.34) but same destination. const-lr is bumpier (3 regressions ≥0.05). All hit ~3.92 plateau by iter 70-80.

  **Sample-efficiency anchor: 3.92 at 100K instances vs Stage 1's 3.84 at 1.28M — 0.08 gap at ~7.8% of Stage 1's instance budget.** New Stage 4 from-scratch best, 5× cheaper per-iter compute than F.6.0 winner (K=20 vs K=100).

  **F.6.1 lock-in diagnostic** (post-mortem on iter-90 checkpoint via [val_stage4_mcts.py](src/scripts/val_stage4_mcts.py), seed=20260430 n=500):

  | decoder | val_avg_cost | Δ vs Stage 4 greedy | Δ vs AM_S1 greedy (3.842) |
  |---|---:|---:|---:|
  | AM_S1 greedy | 3.84221 | — | (ref) |
  | Stage 4 iter-90 greedy | 3.90808 | (ref) | +0.066 |
  | + MCTS K=40 **value_head** ε=0 | 3.90974 | **+0.0017 (p=0.63, tied)** | +0.068 |
  | + MCTS K=40 **rollout** ε=0 | **3.83644** | **−0.0716 (p<0.0001)** | **−0.006** |

  **Smoking gun: vh leaf eval is statistically tied with greedy.** A single-sample rollout under the same policy gives −0.072 lift at the same K=40 budget — the value head adds zero leaf-discrimination signal. Lock-in mechanism: vh ≈ const at leaves → MCTS runs on policy prior alone → π_t collapses to one-hot on the policy's own argmax (mean_entropy_pi 1.80 → 0.16 over 99 iters) → CE distillation just makes the policy more confident on its own argmax. **The 3.92 plateau is a leaf-evaluator failure, not a recipe issue.** This diagnostic motivated the Stage 5 §A-D recipe-tuning + bottleneck-probe + lv0 ablation chain — see [`stage5_progress.md`](stage5_progress.md).

- [x] **F.6.2 Trajectory-probe pass conditions (proposal Stage 4 convergence claim)** — **SATISFIED 2026-05-06.** F.6.1 100-iter trajectory shows (3') visible downward val_avg_cost trend (final < initial by ≥ 0.05 — F.6.1 K=40 const went from random-init val to 3.917 = ≥3 unit drop); (4) ≥1 gate accept (multiple under `gate_every=1`); F.6.0+F.6.1 trajectory smoothly continuous. **The proposal Stage 4 expected outcome — "the self-improvement loop converges: tour quality improves over successive iterations" — is demonstrated.**

- **Definitive claim work moved to Stage 5** (sample efficiency, ultimate quality ≤ 3.8312, strict monotone within ±0.001): F.6.1.3 step10+ε=0.25 sweep, F.6.1.4 → F.6.1.4.b → F.6.1.4.c lr=1e-4 unlock chain (best val 3.8498 at iter 249), F.6.1.6 step-decay 400-iter, F.6.1.6 bottleneck probes (capacity / aleatoric / leaf-eval bypass — diagnosed structural value-head bias), lv0 ablation (rollout + λᵥ=0; best 3.8486 — beats Stage 1 by 0.01), mcts_batch_size sweep (5× wall reduction), TSP-50 K-comparison + lv0 scaling, Track A wall-time optimizations. **All retained in [`stage5_progress.md`](stage5_progress.md) §A–G.** F.6.3 optional escalation paths (1000-iter / K=200 / TSP-50 hoist) likewise migrate.

**Required code changes before F.6.0 (DONE — committed `3ca066b` and `d61ecae`):**

- [x] Make `--load_path` optional in `train_alphazero.py`. From-scratch path prints `[*] No --load_path; starting from random init (proposal Phase F.6).` Verified by CPU smoke (TSP-8, 1 iter, val 3.667, finite losses).
- [x] Add `--val_seed <int>` flag (default = 42) with scoped torch+numpy seed swap so val_dataset is reproducible without disturbing model RNG. Verified: two runs with `--val_seed 42 --seed 999` produce bit-identical `val_avg_cost` (3.468627452850342).
- [x] W&B logging on by default for Modal-launched runs via `modal_run_train_alphazero.py::_common_args` (`--wandb_project am-alphagozero --wandb_mode online`).
- [x] Modal `run_f60_grid` entrypoint added — 12-variant grid spawn in parallel.
- [x] Plan G.9 (lr ablation) added so the lr=1e-5 question is preserved as a planned follow-on (informational only — not a Stage 4 sample-efficiency claim).

**Methodology fix surfaced from warm-start analysis:** The per-run `iterations.csv` val_avg_cost trajectories from F.3 v1-v5 and Modal a1/a2/b1/b2 were each computed on a different fresh 10K val draw (no seed pinned). Std-error of the mean across 10K-instance draws is ~0.003, so per-run val_avg_cost numbers are not directly comparable to either each other or to Stage 1's canonical 3.83943. Apples-to-apples eval (`compare_stage1_vs_stage4.py` with seed=42) IS comparable — that's why all the breakthrough numbers in the F.4+F.3 Modal batch table are apples-to-apples, not per-run trajectory reads. F.6 fixes this at the source by pinning `--val_seed 42`.

### Phase G — Ablations (moved to Stage 5)

Most of Phase G was absorbed into Phase F.6.0/F.6.1.3-1.6/lv0 (now Stage 5 §A-D). Remaining open items (G.4 partial strict-AGZ coupling, G.6 best-so-far per-instance norm, G.7 symmetry aug, G.8 SGD+momentum, G.9 partial lr=1e-3 / lr=1e-5 from-scratch) are tracked in [`_plans/stage5_plan.md` §Remaining open items](../_plans/stage5_plan.md).

---

## Notes

- Plan file: [`_plans/stage4_plan.md`](../_plans/stage4_plan.md). Stage 5 follow-on: [`_plans/stage5_plan.md`](../_plans/stage5_plan.md) / [`_progress/stage5_progress.md`](stage5_progress.md).
- Original plan (Claude Code plans dir): `C:\Users\Jun18\.claude\plans\i-think-we-have-snazzy-trinket.md`.
- Stage 4 reuses Stage 1 canonical TSP-20 checkpoint: `outputs/tsp_20/stage1_tsp20_canonical_20260423T103541/epoch-99.pt`.
- Stage 0 Gurobi reference for TSP-20: 3.8279 mean (1000 instances, seed=1234).
- Stage 1 reference val_avg_cost (TSP-20 canonical bs=512): 3.83943; bs=2048: 3.84443.
- Stage 3 reference test-time MCTS K=400 rollout: 3.8312 (gap 0.087% vs Gurobi).
