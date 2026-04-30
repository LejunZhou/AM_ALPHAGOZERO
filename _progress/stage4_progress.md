# Stage 4 Progress: AlphaGo-Zero-Style MCTS Self-Improvement Loop on TSP-20

**Plan:** `_plans/stage4_plan.md`
**Started:** 2026-04-29
**Last updated:** 2026-04-30 — Phase E (temperature schedule) landed independently of Phase A in this worktree.
**Status:** **Phase E (temperature schedule) complete; Phases A/B/C/D/F/G remaining.** (Phase E intentionally adds no dependency on Phase A: the new `MCTSConfig.temperature_schedule` field defaults to `None`, which preserves Stage 2/3 behavior bit-for-bit.)

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

### Phase A — Visit-distribution exposure (foundation; no GPU)

- [ ] **A.1** Python visit hook in `src/am_baseline/search/mcts.py` — `MCTSConfig.return_root_visits` flag + `MCTSSolver.root_visit_dists` side-effect attribute populated per tour-step.
- [ ] **A.2** C++ sequential — emit per-step `root_visit_dists` via `Solver::solve_instance` → bindings → `CppMCTSSolver`.
- [ ] **A.3** C++ batched (`cpp_batch`) — emit per-instance per-step visit dists via `BatchSearch` → `CppBatchMCTSSolver`.
- [ ] **A.4** Validation: smoke A13 covering all three backends; bit-equivalence python vs cpp visit counts at fixed seed.

### Phase B — Replay buffer + distillation training step (no GPU)

- [ ] **B.1** `MCTSReplayBuffer` class in `src/am_baseline/training/coach.py` — flat dict-of-pre-allocated-tensors (per-instance: coords, bl_val FROZEN, tour_cost; per-step: pi raw τ=1, visited, first/prev_a, lengths, cost_to_go, inst_idx). ~520 MB fixed footprint. Ring-buffer eviction.
- [ ] **B.2** State-tensor reconstruction utility (mirrors `mcts_cpp/solver.py:_state_from_snapshot`).
- [ ] **B.3** `train_step_alphazero` in `src/am_baseline/training/trainer.py` — per-state π_t distillation (raw τ=1) + per-state z_t = cost_to_go/bl_val (V_CURRENT shape, matches `value_targets_from_edges`). Squeeze log_p/mask from (B,1,N) to (B,N) before CE.
- [ ] **B.4** Smoke A1 in `src/scripts/smoke_alphazero.py`. A1.5: cost-to-go consistency vs `value_targets_from_edges` on a known tour.

### Phase C — Self-play data generator (no GPU)

- [ ] **C.1** `make_self_play_config` preset in `coach.py`.
- [ ] **C.2** `generate_self_play_batch` function.
- [ ] **C.3** Smoke A2.

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

(To be populated as phases close.)

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
