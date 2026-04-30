# Stage 4 Progress: AlphaGo-Zero-Style MCTS Self-Improvement Loop on TSP-20

**Plan:** `_plans/stage4_plan.md`
**Started:** 2026-04-29
**Last updated:** 2026-04-29 — Plan refined against AGZ paper (Methods + Ext. Data Table 3) and `ref/KataGo-master`. Phase A in flight.
**Status:** **Phase A.1 (Python visit hook) in progress.**

---

## Plan refinement log

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

- [ ] **B.1** `MCTSReplayBuffer` class in `src/am_baseline/training/coach.py` (deque-of-instances, instance-keyed).
- [ ] **B.2** State-tensor reconstruction utility (mirrors `mcts_cpp/solver.py:_state_from_snapshot`).
- [ ] **B.3** `train_step_alphazero` in `src/am_baseline/training/trainer.py` (per-step π distillation + per-step z target).
- [ ] **B.4** Smoke A1 in `src/scripts/smoke_alphazero.py`.

### Phase C — Self-play data generator (no GPU)

- [ ] **C.1** `make_self_play_config` preset in `coach.py`.
- [ ] **C.2** `generate_self_play_batch` function.
- [ ] **C.3** Smoke A2.

### Phase D — `MCTSCoach.learn` orchestrator (no GPU)

- [ ] **D.1** `MCTSCoach` class.
- [ ] **D.2** Logging extensions in `logging.py` (`iterations.csv`).
- [ ] **D.3** Checkpoint format + resume.

### Phase E — Temperature schedule + Dirichlet noise (no GPU)

- [ ] **E.1** `MCTSConfig.temperature_schedule` (`'const'` | `'step30'` | `'step50'`); Python + C++ wiring. Default `'step30'` (closest to AGZ Methods §Self-play scaled to TSP plies).
- [ ] **E.2** Dirichlet noise CLI flags exposed (ε, α via `--dirichlet_epsilon`, `--dirichlet_alpha_factor`).
- [ ] **E.3** Smoke A3 — TSP-20 K=50 self-play with `step30`; verify per-step root entropy decays sharply at step 6.

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
- [ ] **G.6** Per-step cost-to-go target ablation.
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
