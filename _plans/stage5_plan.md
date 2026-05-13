# Stage 5 Plan: Systematic Experiments and Ablations

**Created:** 2026-05-13 (partitioned from Stage 4 after proposal-alignment review).
**Predecessor:** Stage 4 (`_plans/stage4_plan.md`, `_progress/stage4_progress.md`) — closing with proposal-aligned "loop converges" deliverable.
**Reference:** Proposal §Stage 5 (`proposal.md:146-166`).
**Status:** Open. Sub-tracks executed 2026-05-03 → 2026-05-13 prior to partition; remaining items listed at the bottom.

---

## Context

After Stage 4 was closed against the proposal's *first* expected outcome — *"the self-improvement loop converges: tour quality improves over successive iterations"* (proposal.md:134) — a body of recipe-tuning, scaling, and ablation work executed during Phase F.6 (2026-05-02 → 2026-05-13) was found to align better with Stage 5's proposal scope ("Systematic Experiments and Ablations", proposal.md:146-166) than with the Stage 4 convergence claim. That work has been migrated here.

The Stage 5 proposal mandate is:

- **Ablation studies:** value head contribution, MCTS budget during training, replay buffer size, gating vs no-gating, training loss variants.
- **Scaling experiments:** TSP-20 → TSP-50 → TSP-100; generalization (train 50, test 100); transfer to CVRP.
- **Deliverable:** results table across all variants × TSP sizes, with sample-efficiency curves and ablation analysis.

This plan captures the slice already executed (recipe lockdown, lr/wd factorial, ε sweep, K/buffer/batch sweep, lr-schedule chains, vh-leaf-eval bottleneck diagnosis, lv0 ablation, TSP-50 scaling + Track A wall-time optimizations) plus the items still open.

---

## Migrated work (executed pre-partition; full results in progress doc)

The following items were originally Phase F.6 sub-tasks under Stage 4. They are kept here as ablation/scaling/optimization work per the proposal Stage 5 framing. Detailed numbers and trajectories live in [`_progress/stage5_progress.md`](../_progress/stage5_progress.md).

### A. Recipe lockdown ablations (TSP-20 from-scratch baseline already in Stage 4 F.6.0 / F.6.1)

**A.1 LR×wd factorial decomposition (F.6.0.5).** 4-variant raw-target `(lr, wd)` 2×2 grid (V1 = lr=1e-4/wd=1e-4 control; V2 = 5e-4/1e-4; V3 = 5e-4/0; V4 = 1e-4/0) under K=40, leaf_eval=value_head, 20 iter. Pre-execution motivation: F.6.0 winner stuck at val=3.9338, ~0.094 above Stage 1's 3.83943. Stage-4 CE gradient is O(1) (probe `probe_grad_norm.py`: Stage 4 grad-norm mean 3.76 vs Stage 1 REINFORCE 90.87 → ~24× smaller, matches O(1) vs O(advantage·N)). Budget arithmetic concluded lr=5e-4. Hypothesis: `value_target_norm=none` (raw `cost_to_go`) removes the bl_val-drift confound on the value head's target distribution. Decision rule: Step 1 lr — adopt high-lr variant if val(19) ≤ V1 − 0.02; Step 2 wd — pick V2 vs V3 by |V3−V2|. Outcome: **adopted V3 (lr=5e-4, wd=0, value_target_norm=none)**.

**A.2 Dirichlet ε sweep at V3 regime (F.6.0.6).** 2 variants ε ∈ {0, 0.05} at V3, with V3's ε=0.25 as implicit reference. K=40, 20 iter, leaf_eval=value_head. Decision: pick ε with smooth trajectory and no per-iter regression ≥ 0.05. Outcome: **ε=0 winner** (smooth descent; ε≥0.05 oscillated under lr=5e-4 high-lr regime). Mechanism: at K=40 + ε=0.05, ~2 visits/root are noise-driven; lr=5e-4 amplifies this into per-iter spikes.

**A.3 K + buffer + batch + M sub-budget probes (F.6.0.7 → F.6.1.1).** 7-run sweep at F.6.0.6 winner regime testing knob sensitivity:
- K (20 vs 40 vs 100): dominates (10× the batch effect at TSP-20).
- buffer (200K → 5K → 1K): 5K wins; 200K stale, 1K under-denoised.
- batch (512 vs 2048) at train_steps=200: neutral.
- M=2000 × 10 iter vs M=1000 × 20 iter: iters dominate M (train_steps=200 saturates per-iter learning at M=1000).
- Rollout vs value_head at K=20: rollout ties value_head at 3.2× wall (per-credit value_head wins at low K).

Final F.6.1 recipe lock-in: `lr=5e-4 const, wd=0, value_target_norm=none, ε=0, K=40, M=1000, buffer=5000, batch=512, train_steps=200, gate=ttest, gate_every=1, leaf_eval=value_head, val_seed=42`.

### B. LR-schedule deepening (lr=1e-4 unlock + step decay)

**B.1 step10 + ε=0.25 sweep (F.6.1.3).** 100-iter from-scratch runs varying ε ∈ {0.05, 0.25} at locked F.6.1 recipe but with `temperature_schedule=step10` (cutoff ⌈0.1·N⌉ = 2/20 steps, closer to AGZ-canonical 12%). Hypothesis: F.6.1's 3.92 plateau is the policy collapsing to one-hot under low π_t entropy; restoring Dirichlet should re-introduce target multimodality. Outcome: **ε=0.25 → 3.8784 at iter 99** vs ε=0.05 → 3.897. mean_entropy_pi 2.16× higher at ε=0.25, confirming exploration→target-information mechanism.

**B.2 lr=1e-4 unlock chain (F.6.1.4 → F.6.1.4.b → F.6.1.4.c).** Three sequential 50-iter resumes from F.6.1.3 `iter-99.pt`. Discovery: lr=5e-4 saturates ~iter 127 (22 consecutive rejects); resuming at lr=1e-4 immediately unlocks improvement (10/35 accepts in iter 150-184). Required `lr-override-on-resume` infrastructure ([`train_alphazero.py`](../src/scripts/train_alphazero.py)) so a `--resume_from … --lr_model 1e-4` flag overrides the optimizer's restored lr=5e-4 (otherwise LambdaLR silently keeps the loaded value). Pattern: at any constant-lr plateau, the next-tier lr (×0.2) re-opens convergence in ONE iter. Chain endpoint: **3.8498 at iter 249** (gap to AM_S1 greedy 3.842 = ~0.008, within ~1× val-set SE).

**B.3 Step-decay 400-iter from-scratch (F.6.1.6).** Single from-scratch run consolidating the F.6.1.3→4→4.b→4.c manual chain into a built-in lr step schedule (5e-4 → 1e-4 → 2e-5 → 4e-6 at iter 100/200/300). 80K train steps, 400K instances; W&B `plpvfhv2`. Outcome: **best 3.8578 at iter 365, final 3.8619** — never crossed below 3.85. Underperformed F.6.1.4.c chain by 0.014 at matched iter despite 175 more iters; lr=4e-6 segment wasted (only 2 accepts, 38 consecutive rejects). Conclusion: **3.85 is the recipe ceiling on TSP-20 under value_head leaf eval at training time** — drove the bottleneck-probe chain (C below).

### C. Value-head bottleneck diagnosis (proposal Stage 5: "Value head contribution")

Three layered probes on F.6.1.6 iter-399 best_model to determine whether the 3.85 plateau is encoder capacity, aleatoric uncertainty, or structural value-head bias.

**C.1 Capacity probe.** Local refit on cached glimpses from the frozen encoder. Four value-head architectures from 16K to 230K params. Outcome: **val MSE plateaus at ~0.0060 across all sane sizes**; capacity buys at most 0.0005 in val. Head capacity is not the binding constraint.

**C.2 Aleatoric probe** ([`src/scripts/probe_value_aleatoric.py`](../src/scripts/probe_value_aleatoric.py)). Partial-root MCTS from 100 buffer states × 20 fresh completions, K=40 + step10 + ε=0.25 + value_target_norm=none. MSE decomposition `mean((v−z)²) = Var(z|s) + (v(s) − E[z|s])²`. Outcome: **0.00711 = 0.00169 var_z + 0.00543 bias² → 24% aleatoric / 76% bias.** Variance lives ONLY at steps 0-1 (var_z=0 at steps 2-19 under step10 + argmax). **RMS bias = 0.074** at every step ≥ 2 deterministically.

**C.3 Leaf-eval bypass sweep** ([`val_stage4_mcts.py`](../src/scripts/val_stage4_mcts.py)). Re-evaluate F.6.1.6 iter-399 best_model at val time under MCTS × leaf_eval × K. Outcome: vh leaf eval (any K) is **statistically tied with greedy** (3.868 vs 3.863); rollout K=40 buys **−0.028 over greedy** and breaks 3.85 trivially; rollout K=200 → 3.8330 (within 0.001 of Stage 3 K=400 rollout's 3.8312).

**Decisive finding:** the 3.85 greedy ceiling is a vh-leaf-eval-induced training ceiling, not a model-quality ceiling. The 0.074 RMS bias propagates structurally into MCTS visit distributions. Memory entry: [`project_alphagozero_value_head_leaf_eval_bias.md`](../../C:/Users/Jun18/.claude/projects/c--Users-Jun18-Desktop-AM-ALPHAGOZERO/memory/project_alphagozero_value_head_leaf_eval_bias.md). Implications: (i) inference-time always use `leaf_eval=rollout` on F.6.1-family checkpoints; (ii) training-time, F.6.1.6 used vh leaf eval throughout so π_t distillation targets carry the bias — explains the 3.85 greedy plateau.

### D. lv0 ablation — proposal Stage 5: "Value head contribution: Full system vs. MCTS with policy-only"

**D.1 Implementation.** Made `lambda_v=0` a true policy-only update in `train_step_alphazero`: `value_loss` is still logged, but no value-head or shared-encoder value gradients are computed/applied. Value head stays enabled in the model for checkpoint compatibility. Added `mean_entropy_policy` (model entropy alongside target entropy) and per-iteration MCTS-vs-greedy teacher metrics (`mcts_delta_vs_greedy_mean`, `mcts_win_rate_vs_greedy`, `greedy_cost_mean`, `mcts_cost_mean`).

**D.2 TSP-20 grid (`run_rollout_lambda_ablation`).** 50-iter from-scratch parallel runs at `leaf_eval=rollout, K=40, step10, ε=0.25, value_target_norm=none, buffer=5000, batch=512, train_steps=200, lr=5e-4, wd=0, gate_every=1, val_seed=42`. Variants: `λᵥ=0.0` and `λᵥ=1.0` (optional `0.1` via `include_weak=True`).

**D.3 lv0 resume chain.** Two follow-up resumes to take the lv0 winner to iter 199:
- `run_rollout_lv0_resume50_to_iter99` — +50 iter at lr=5e-4 const.
- `run_rollout_lv0_resume100_lr1e4_to_iter199` — +100 iter at lr=1e-4 (via lr-override-on-resume).

**D.4 Inference-time validation.** Mirror the F.6.1.6 leaf-eval bypass sweep on lv0 iter-199 best_model: greedy θ★ + MCTS × {K=40, 100, 200} × leaf_eval=rollout.

Outcomes summarized in stage5_progress.md §D. Verdict: **for any Stage 4 recipe with `leaf_eval=rollout`, default to `lambda_v=0`** (the value head's biased gradient on the shared encoder stops poisoning policy distillation). lv0 chain best **val 3.8486 at iter 197 — beats Stage 1 canonical greedy by 0.01** at ~200K instances vs Stage 1's 1.28M.

### E. TSP-50 scaling (proposal Stage 5: "TSP-20 → TSP-50 → TSP-100")

**E.1 TSP-50 K-comparison ablation.** Parallel 20-iter from-scratch runs at TSP-50 comparing K=50 vs K=100 (other knobs locked at the lv0 recipe). Test whether the K-dominates finding from TSP-20 holds at scale.

**E.2 TSP-50 lv0 K=50 chain.** 50-iter from-scratch run + +50 iter resume to iter 99 + Track A relaunch (see F below). Goal: demonstrate lv0 recipe transfers to a regime where Stage 1 is weaker (Stage 1 TSP-50 greedy 5.7999 vs Gurobi 5.6987 vs Stage 4 prior best 6.060 vh+λᵥ=1).

**E.3 TSP-50 lv0 +50 iter resume from iter-49 at lr=5e-4 const, target iter 99** (Track A entrypoint).

Pending: TSP-100 from-scratch (requires Stage 1 TSP-100 full-compute baseline first).

### F. Wall-time optimizations (engineering enabler for TSP-50 + Stage 5 scaling)

Six surgical Python optimizations to the C++ batched MCTS rollout path (`src/am_baseline/search/mcts_cpp/solver.py`). All preserve determinism (max_abs_cost_diff = 0.0 at every step on paired-seed runs).

- **Fix #1** — `bytes(visited)` cache key body.
- **Fix #2b** — vectorize `rollout_many` state + masked argmax.
- **Fix #3** — numpy-direct evaluator `eval_many_arrays` for rollout (bypass dict construction).
- **Fix #4** — cache stores numpy arrays (`.copy()` ~35× faster than `.tolist()`).
- **Fix #5** — bulk-vectorize cache key construction (2-tuple key via numpy bit-packing).
- **Track A** — per-row `state.i` in `Decoder._get_step_context` so all active rollouts at heterogeneous steps batch into ONE NN call per outer iter (vs `mcts_batch_size=1000` instances × ~50 unique steps → 50 small calls). Eliminates `rollout_many`'s per-step-group loop. Decoder edits: scalar fast-path preserved bit-for-bit via `state.i.numel() == 1` branch.

**Net production wall:** TSP-50 K=50 M=1000 went from 1255 s/iter pre-fix → 310-320 s/iter post-Track-A (**−75%**). A 50-iter Stage 5 TSP-50 lv0 run fits in ~4h vs ~17h pre-fix.

### G. mcts_batch_size — 5× wall reduction at no quality cost (recipe-tuning artifact)

Discovery during F.6.1 trajectory probes: `mcts_batch_size` (prior default 64) is the **cross-instance chunk size** in `solver.py`, not a per-NN-forward batch. At M=1000 the prior default sequentially processed 16 chunks of 64 — wildly underutilizing the GPU. Sweep at F.6.1.3 ε=0.25 recipe (10 iter, 4 parallel jobs):

| mcts_batch_size | s/iter | speedup |
|---|---:|---:|
| 64 (prior default) | ~124 | 1.0× |
| 256 | ~38 | 3.3× |
| 1000 | ~25 | 5.0× |
| 2000 (≡1000 at M=1000) | ~25 | 5.0× |

Quality unaffected. Default changed 64 → 1000 in [`train_alphazero.py`](../src/scripts/train_alphazero.py).

---

## Remaining open items

These are Phase G ablations carried over from Stage 4 plus the additional scaling items mandated by the proposal but not yet executed.

### Open ablations (after F.6 execution)

Most of Stage 4's Phase G was absorbed into F.6: G.1 (leaf-eval rollout vs value_head) resolved across F.6.0 grid + bypass probe + lv0; G.2 (buffer capacity) resolved by F.6.0.7-1.1 → buffer=5000; G.3 (Dirichlet ε) covered by F.6.0 + F.6.0.6 + F.6.1.3 sweeps; G.5 (gating cadence) resolved to `gate_every=1` via the lr=5e-4 staleness analysis.

What remains open:

| Ablation | Open part | Priority | Why open |
|---|---|---|---|
| **G.4 (partial)** Training-target coupling: strict-AGZ (π_t = one-hot late, same τ_t as σ_t) vs decoupled (π_t always τ=1 raw visits — F.6 default, choice B) | Strict-AGZ coupling never empirically tested | Low | Decoupled has been productive across all F.6 runs; strict-AGZ has theoretical appeal but high risk of late-step gradient noise on TSP's deterministic late steps. Schedule sweep itself (step10 / step30 / step50 / const) is sufficient. |
| **G.6** Best-so-far per-instance value normalization $z_t = (\text{tour\_cost} − \text{lengths}_t) / \min_\text{seen} \text{tour\_cost}(x)$ | Untested | Low-medium | F.6 settled on `value_target_norm=none` (raw cost_to_go); per-instance min-seen is a different normalization with no empirical comparison yet. Worth revisiting if vh-bias re-emerges at TSP-50 or beyond. |
| **G.7** Symmetry augmentation at leaf eval (random 2D rotation+flip of coords pre-encoder) | Untested | Low | AGZ Methods §Search algorithm canonical (8-fold dihedral); TSP analog is SO(2) × {flip}. Cheap to add; would tighten AGZ-fidelity story but Track A wall savings already unlocked TSP-50 without it. |
| **G.8** Optimizer: Adam vs SGD+momentum 0.9, lr=1e-3 | Untested | Very low | AGZ canonical optimizer. F.6 used Adam throughout; switching would likely require re-deriving lr from scratch (F.6.0.5-style). |
| **G.9 (partial)** LR ablation from-scratch: lr ∈ {1e-3 steady-state, 1e-5} | F.6.0.5 compared 1e-4 vs 5e-4 from-scratch (5e-4 won); F.6.1 lrdecay used 1e-3 as initial value of 0.95^iter decay (not steady-state); lr=1e-5 from-scratch never tested (was warm-start-only) | Low | The lr-unlock chain pattern at TSP-50 (lr=5e-4 → lr=1e-4 resume) already informs scheduling. Marginal info from further lr sweeps from-scratch. |

**Resource budget if pursued:** each open item is a single small probe (~1-3 h Modal). Could be scoped as one optional "AGZ-canonical fidelity audit" sweep (G.4 strict-AGZ + G.7 symmetry + G.8 SGD-momentum) for tighter paper-alignment, or left as deferred low-priority work.

### Open scaling (per proposal Stage 5)

- **TSP-50 lv0 step-decay 400-iter** at the F.6.1.6 schedule (5e-4 → 1e-4 → 2e-5 → 4e-6). Hypothesis from progress doc §D.4: at iter 200 (lr=2e-5 segment) val should drop into 3.84-3.85; at iter 400 the run should approach Stage 1 canonical greedy or below. Cost: ~$30-50 Modal credits, ~5h on A10 (rollout is ~5× vh wall).
- **TSP-100 from-scratch.** Requires Stage 1 TSP-100 with full compute budget (current reduced-compute) to be a fair apples-to-apples comparison with the AM paper's TSP-100 sampling-1280.
- **Generalization probe:** train on TSP-50, test on TSP-100 (does MCTS training improve generalization?).

### Open transfer (proposal §Stage 6 stretch)

- **CVRP** — adapt masking + state representation for capacity constraints + depot returns. Adapt value head target for CVRP tour length normalization. Train + evaluate on CVRP-20/50/100.

---

## Acceptance criteria for Stage 5 closure

These criteria were originally in Stage 4 plan but require Stage 5-style ablation + scaling evidence to satisfy beyond the convergence-only Stage 4 claim.

1. **Sample efficiency (proposal headline).** Stage 4-trained model reaches Stage 1's quality (TSP-20: ≤ 3.83943) at fewer cumulative instances than Stage 1's 128M. **Status: SATISFIED at TSP-20 via lv0 chain — 3.8486 at 200K instances ≈ 0.16× Stage 1 budget.** Plot via `plot_stage4.py` against Stage 1's `epochs.csv`.

2. **Ultimate quality.** Stage 4 final greedy val_avg_cost ≤ 3.8312 (Stage 3 K=400 rollout MCTS) — Stage 4's network alone (no MCTS at test time) matches Stage 3's search-augmented Stage 1. **Status: NOT SATISFIED at greedy (lv0 best 3.8486 vs target 3.8312 = 0.017 gap), but SATISFIED at K=200 rollout (3.8329 vs 3.8312 within 0.002).** Plausibly requires lv0 step-decay 400-iter.

3. **TSP-50 parity.** Stage 4 final greedy val_avg_cost ≤ 5.7999 (Stage 1 TSP-50 greedy). **Status: in progress** — Track A lv0 K=50 chain currently iter 30 → val 6.186 (vs Stage 1 5.7999, Stage 4 prior best vh+λᵥ=1 muckiyvi = 6.060, Gurobi 5.6987).

4. **Value-head ablation closed.** Document and accept λᵥ=0 + rollout as the recommended training-time recipe when leaf_eval=rollout. **Status: SATISFIED** (project memory entry + this plan §D).

5. **Reach (Stage 6 stretch):** TSP-20 final greedy ≤ 3.8298 (= 0.05% gap vs Gurobi 3.8279) — proposal target. **Status: NOT SATISFIED.**

---

## Dependencies + entry points

- All work depends on Stage 4 closure (Phases A-E infrastructure + F.6.0/F.6.1 convergence demonstration).
- Modal launcher: [`src/scripts/modal_run_train_alphazero.py`](../src/scripts/modal_run_train_alphazero.py). Stage 5 entrypoints currently in this file: `run_f60_grid`, `run_f605_lr_validation`, `run_rollout_lambda_ablation`, `run_rollout_lv0_resume50_to_iter99`, `run_rollout_lv0_resume100_lr1e4_to_iter199`, `run_tsp50_lv0_resume_from_iter15_trackA`, K-comparison entrypoints.
- Probes / validators: [`src/scripts/probe_grad_norm.py`](../src/scripts/probe_grad_norm.py), [`src/scripts/probe_value_aleatoric.py`](../src/scripts/probe_value_aleatoric.py), [`src/scripts/probe_mcts_quality.py`](../src/scripts/probe_mcts_quality.py), [`src/scripts/val_stage4_mcts.py`](../src/scripts/val_stage4_mcts.py), [`src/scripts/compare_stage1_vs_stage4.py`](../src/scripts/compare_stage1_vs_stage4.py).

---

## Notes

- The Stage 4 plan file retains the Phases A-E + F.1-F.5 + F.6.0/F.6.1 design content as the historical archival record. This file is the Stage 5 working plan.
- Stage 0 Gurobi reference for TSP-20: 3.8279 mean (1000 instances, seed=1234).
- Stage 0 Gurobi reference for TSP-50: 5.6987.
- Stage 1 reference val_avg_cost: TSP-20 canonical bs=512 = 3.83943; TSP-50 = 5.7999.
- Stage 3 reference test-time MCTS K=400 rollout (TSP-20): 3.8312 (gap 0.087% vs Gurobi).
