# Stage 4 Progress: AlphaGo-Zero-Style MCTS Self-Improvement Loop on TSP-20

**Plan:** `_plans/stage4_plan.md`
**Started:** 2026-04-29
**Last updated:** 2026-04-30 — Phases A, B, C, E all complete. Wave 1 (A+B) and Wave 2 (C+E) merged. Wave 3 (D, F, G) unblocked.
**Status:** **Phases A, B, C, E complete; Phase D (MCTSCoach orchestrator) ready to start. Phases F (TSP-20 pilot/main) and G (ablations) follow.**

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

### Phase D — `MCTSCoach.learn` orchestrator (no GPU)

- [ ] **D.1** `MCTSCoach` class.
- [ ] **D.2** Logging extensions in `logging.py` (`iterations.csv`).
- [ ] **D.3** Checkpoint format + resume.

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

### Phase F — TSP-20 pilot + main run (~2.5 h compute)

- [ ] **F.1** `src/scripts/train_alphazero.py` CLI.
- [ ] **F.2** Smoke battery A1-A6.
- [ ] **F.3** TSP-20 pilot (20 iter × M=1000 × K=50 × 100 train_steps × **buffer_capacity=50K** — AGZ-proportional ~50-iter window).
- [ ] **F.4** TSP-20 main (100 iter × M=1000 × K=100 × 200 train_steps).
- [ ] **F.5** Headline plot.

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
