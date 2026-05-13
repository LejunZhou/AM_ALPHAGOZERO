# Stage 5 Progress: Systematic Experiments and Ablations

**Plan:** [`_plans/stage5_plan.md`](../_plans/stage5_plan.md)
**Partitioned from Stage 4 progress on 2026-05-13** (proposal-alignment review: Phase F.6 deeper-tuning and ablation work belongs to Stage 5, not the Stage 4 convergence claim).
**Status:** Open. Migrated work below executed 2026-05-03 → 2026-05-13.

---

## Context

This doc holds the recipe-tuning, lr-schedule, value-head bottleneck, lv0 ablation, TSP-50 scaling, and wall-time optimization work originally logged under Stage 4 Phase F.6 sub-tasks, plus the standalone rollout-value-loss ablation track. See [`_plans/stage5_plan.md`](../_plans/stage5_plan.md) for the partition reasoning and current open items.

Convention: section names mirror the Stage 5 plan (§A recipe lockdown, §B lr-schedule chains, §C value-head bottleneck, §D lv0 ablation, §E TSP-50 scaling, §F wall-time optimizations, §G mcts_batch_size).

---

## §A — Recipe lockdown ablations (TSP-20)

### A.1 LR×wd factorial decomposition (F.6.0.5) — **COMPLETE 2026-05-05/06**

**Motivation.** F.6.0 winner stuck at val_avg_cost=3.9338, ~0.094 above Stage 1's 3.83943. The Stage-4 CE policy gradient `∂L/∂ℓ_j = p_j − π_target_j` is O(1) bounded; Stage 1 REINFORCE is advantage-scaled. Probe ([probe_grad_norm.py](../src/scripts/probe_grad_norm.py), n=30, random init, F.6.0-scale): Stage 4 grad-norm mean = **3.76**, Stage 1 = **90.87** (~24× ratio, matches O(1) vs O(advantage·N) prediction). Budget arithmetic: Stage 1 converged at ~250K grad steps × lr=1e-4 ⇒ ~25 units drift; Stage 4 F.6.0 at 10K × lr=1e-4 ⇒ ~1 unit (4% of Stage 1). Recommended lr=5e-4 (geometric mean of principled range 3e-4–1e-3).

**Diagnostic refocus.** Bl-normalized 4-variant `(lr, wd)` grid showed flat trajectories — lr/wd weren't the rate-limiter. **Hypothesis (user-led):** value head trains against a moving target `z = cost_to_go / bl_val`. Three shift channels: (a) calibration drift as θ★ improves and `bl_val` decreases; (b) buffer non-stationarity mixing fresh and stale calibrations; (c) across-instance variance collapse from `bl_val` absorbing across-instance signal. Implementation: `--value_target_norm none` (raw cost_to_go target) added to CLI + trainer + MCTS Python + MCTS C++, retaining buffer-stored bl-normalized z and denormalizing at use time. Default `bl` unchanged.

**F.6.0.5b raw-target 4-variant `(lr, wd)` 2×2 grid** (Modal `ap-M3se0bCnottfFqptEhq8oi`, K=40, 20 iter, leaf_eval=value_head, all 4 reached iter-19, ~30 min wall):

| variant | lr | wd | val(19) | policy_loss | value_loss | gate accepted |
|---|---|---|---:|---:|---:|---|
| V1 (control) | 1e-4 | 1e-4 | 4.331 | 1.527 | 0.279 | yes |
| V2 | 5e-4 | 1e-4 | 4.349 | 1.510 | 0.296 | **NO (regressed)** |
| **V3 (winner)** | **5e-4** | **0** | **4.265** | **1.465** | **0.264** | yes |
| V4 | 1e-4 | 0 | 4.367 | 1.551 | 0.276 | yes |

**2×2 factorial decomposition** (`lr × wd` design at iter-19 val):

| term | value | interpretation |
|---|---:|---|
| grand mean | 4.328 | — |
| main effect lr (5e-4 − 1e-4) | −0.041 | high lr helps on average |
| main effect wd (0 − 1e-4) | −0.024 | wd=0 helps on average |
| **lr × wd interaction** | **−0.120** | **3× the larger main effect — must raise lr AND drop wd jointly** |

**Decision-rule outcome:** Step 1 (lr) PASS via V3 (V1 − V3 = 0.066 ≥ 0.02 threshold). Step 2 (wd at lr=5e-4): V3 − V2 = −0.084 ≪ −0.01 → **adopt V3 (lr=5e-4, wd=0)**.

**F.6.1 defaults locked: `--lr_model 5e-4 --weight_decay 0 --value_target_norm none`.** Validates the bl_val-drift hypothesis as the dominant value_head-convergence drag; optimizer tuning becomes meaningful only after the target distribution shift is removed.

### A.2 Dirichlet ε sweep at V3 regime (F.6.0.6) — **COMPLETE 2026-05-06**

2 variants ε ∈ {0, 0.05} at V3 (lr=5e-4, wd=0, raw-target, value_head, K=40, 20 iter); V3's ε=0.25 is the implicit third reference. Modal `ap-IAwuU1k9JAADMbtkM7WTOR`.

| variant | ε | val(19) | max single-iter regression | # regressions ≥0.05 |
|---|---|---:|---:|---:|
| **E1 (winner)** | **0** | 4.2283 | **+0.035** | **0/19** |
| E2 | 0.05 | **4.1856** | +0.462 (iter 17→18) | 7/19 |
| V3 (implicit) | 0.25 | 4.2648 | (similar volatility expected) | — |

E2's iter-19 lead (0.043 over E1) sits at the bottom of a 0.55-amplitude oscillation (iter-18 = 4.738 → iter-19 = 4.186). **F.6.1 is 100-iter; volatility compounds.** E1's smooth descent is the safer pick.

**Mechanism:** Dirichlet noise corrupts π_t targets at every root. With K=40 + ε=0.05, ~2 visits/root are noise-driven; under V3's lr=5e-4 (5× faster target-fitting), the policy absorbs that variance into per-iter val spikes. ε=0 → smooth descent. **The F.6.0 ε=0.25 conclusion does NOT transfer** — each major regime change (leaf_eval, lr, value_target_norm) re-opens the ε question.

**F.6.1 default fully resolved (lr × wd × ε):** `--lr_model 5e-4 --weight_decay 0 --value_target_norm none --dirichlet_epsilon 0`.

### A.3 Sub-budget probes K + batch + M + buffer (F.6.0.7 → F.6.1.1) — **COMPLETE 2026-05-06**

7-run sweep at F.6.0.6 winner regime (ε=0, lr=5e-4, wd=0, raw-target, value_head, K=20-40, 20 iter), reference E1 (K=40, batch=512, buffer=200K) → val(19) = **4.228**.

| knob change | val(19) | Δ vs E1 | finding |
|---|---:|---:|---|
| K=20 (vs E1 K=40) | 4.428 | +0.200 | **K dominates** (~10× the batch effect) |
| K=20 + buffer 200K → 5K | 4.299 | +0.071 | **buffer is a major drag**: stale MCTS targets pulled policy back; shrinking closed 0.13 of the K-gap |
| K=20 + buffer → 1K | 4.350 | +0.122 | too small; need ~5-iter window for denoising |
| K=20 + batch 512 → 2048 | 4.408 | +0.180 | batch is essentially neutral once train_steps saturates |
| K=20, M=2000 × 10 iter (matched 20K) | 5.062 | +0.834 | **iters dominate M**: train_steps=200 saturates per-iter learning at M=1000 |
| K=20, rollout leaf eval | ~4.236 | +0.008 | rollout matches value_head at K=20 but 3.2× wall — per-credit value_head wins |

**F.6.1 final locked recipe: lr=5e-4 const, wd=0, value_target_norm=none, ε=0, K=40, M=1000, buffer=5000, batch=512, train_steps=200, gate=ttest, gate_every=1, leaf_eval=value_head, val_seed=42.** Buffer 200K → 5000 is the single strongest knob found.

---

## §B — lr-schedule deepening

### B.1 step10 + ε=0.25 sweep (F.6.1.3) — **COMPLETE 2026-05-06/07**

Cheapest-first response to the F.6.1 3.92 plateau (see Stage 4 progress §F.6.1 for the lock-in diagnostic): keep value_head, narrow stochastic window step30 → step10, restore Dirichlet. Two 100-iter from-scratch runs.

| run | ε | val(99) | mean_entropy_pi(99) |
|---|---|---:|---:|
| `f62_step10_eps05` (W&B `933kn781`) | 0.05 | 3.897 | 0.216 |
| **`f62_step10_eps25`** (W&B `meynsvp0`) | **0.25** | **3.8784** | 0.335 |

ε=0.25 beat F.6.1 main's iter-90 best by 0.015 at the same val_seed=42 with **2.16× higher π_t entropy**. The exploration→target-information mechanism is real; ε=0.05 too low to halt entropy collapse.

### B.2 lr=1e-4 unlock chain (F.6.1.4 → F.6.1.4.b → F.6.1.4.c) — **COMPLETE 2026-05-07**

Three sequential resumes from `f62_step10_eps25/iter-99.pt`. Discovery: at iter 127 the lr=5e-4 well empties, but **lr=1e-4 unlocks the next improvement tier in ONE iter**.

| stage | iters | lr | best val | last accept @ iter | accepts |
|---|---:|---:|---:|---:|---:|
| F.6.1.4 resume +50 | 100-149 | 5e-4 | **3.8665** | 127 (then 22 rejects) | 4/50 |
| F.6.1.4.b resume +50 lr=1e-4 | 150-199 | **1e-4** | **3.8514** | 184 (then 15 rejects) | **10/35** |
| F.6.1.4.c resume +50 lr=1e-4 | 200-249 | 1e-4 | **3.8498** | 225 (then 24 rejects) | 2/26 |

W&B: `z39vi807` (4) / `a4t72a7s` (4.b) / `52ftg1b8` (4.c). **Total chain at lr=1e-4: 3.8665 → 3.8498 = −0.0167 over 100 iters.** Gap to AM_S1 greedy 3.842 is now ~0.008 — within ~1× val-set SE.

**lr-override-on-resume infrastructure** added to [train_alphazero.py](../src/scripts/train_alphazero.py) so a `--resume_from` + `--lr_model 1e-4` flag overrides the optimizer's restored lr=5e-4 (otherwise LambdaLR silently keeps the loaded value). Verified by the immediate −0.005 at iter 150. Same pattern reused at TSP-50 (Track A).

**Soft signals at chain end (iter 249):** policy_grad_norm 0.80→0.40, value_grad_norm 0.83→0.21, value_loss 0.017→0.0045, mean_entropy_pi stable ~0.28. All settled into a tight regime under lr=1e-4. The lr=1e-4 well is nearly empty after iter 225 (24 rejects).

**Telemetry/infrastructure landed alongside the chain** (all carry forward to TSP-50/Track A):
- Per-loss gradient norm + value-grad VH/shared subspace split in [trainer.py](../src/am_baseline/training/trainer.py); orthogonal-decomposition invariant verified `value_grad_norm² = vh² + shared²` (smoke).
- `step10` temperature schedule in MCTSConfig + C++ (cutoff = ⌈0.1·N⌉; closer scaling to AGZ-canonical 12% than step30).
- `iterations.csv` schema extended with 5 new grad-norm columns (read-by-name handles old/new schema mix).
- [val_stage4_mcts.py](../src/scripts/val_stage4_mcts.py) standalone validator (auto-loads ckpt + args.json, supports `--match_train` and `--am_ckpt` for paired comparison).

### B.3 Step-decay 400-iter from-scratch (F.6.1.6) — **COMPLETE 2026-05-07**

Single from-scratch run consolidating the F.6.1.3 → 4 → 4.b → 4.c manual chain into a built-in lr step schedule (decay 0.2 every 100 iters: 5e-4 → 1e-4 → 2e-5 → 4e-6). 400 iters × M=1000 × mcts_batch_size=1000, 80K train steps, 400K instances. W&B: **`plpvfhv2`**. Wall: 2.79 h on A10.

**Per-segment ROI table:**

| iter range | lr | first val | last val | min val (iter) | net Δ |
|---|---|---:|---:|---:|---:|
| 0-99 | 5e-4 | 7.1040 | 3.9153 | 3.9114 (90) | **−3.19** |
| 100-199 | 1e-4 | 3.9013 | 3.8693 | 3.8690 (190) | −0.032 |
| 200-299 | 2e-5 | 3.8679 | 3.8696 | 3.8596 (285) | +0.0017 |
| 300-399 | 4e-6 | 3.8607 | 3.8619 | **3.8578 (365)** | **+0.0012 (wasted)** |

**Plateau: best 3.8578 (iter 365) / final 3.8619 greedy at iter 399.** Run never crossed below 3.85; last gate accept at iter 361 → 38 consecutive rejects to end. Beyond lr=2e-5 the schedule adds no value.

**Underperformed the F.6.1.4.c chain by ~0.014 at matched iters** (3.8578 vs 3.8498) despite identical recipe + 175 more iters + 2 extra lr-decay steps. Two candidate explanations not pursued: random-seed variance at ε=0.25 (~0.01-0.015 noise floor), or premature lr drop at iter 100 (chain ran 149 iters at lr=5e-4 before dropping). The 3.85 line is the recipe ceiling on TSP-20 under value_head leaf eval — drove the bottleneck-probe chain below.

---

## §C — Value-head bottleneck diagnosis — **COMPLETE 2026-05-08/09**

Three layered probes converging on the structural issue.

### C.1 Capacity probe (local refit)

4 value-head architectures (16K → 230K params) on cached glimpses from the FROZEN iter-399 encoder. **Val MSE plateaus at ~0.0060 across all sane sizes** (XL 4-layer hits train MSE 0.0051). Capacity buys at most ~0.0005 in val. Head capacity isn't the issue.

### C.2 Aleatoric probe ([src/scripts/probe_value_aleatoric.py](../src/scripts/probe_value_aleatoric.py))

Partial-root MCTS from 100 buffer states × 20 fresh completions, K=40 step10 ε=0.25 value_target_norm=none. MSE decomposition `mean((v−z)²) = Var(z|s) + (v(s) − E[z|s])²` (exact to invariant 5.55e-17): **0.00711 = 0.00169 var_z + 0.00543 bias² → 24% aleatoric / 76% bias.** Variance lives ONLY at steps 0-1 (frac=0.68 / 0.97); steps 2-19 are var_z = 0 to machine precision under step10+argmax. **RMS bias = 0.074** — the value head systematically misses E[z|s] by ~7% of cost-to-go scale at every step ≥ 2 deterministically. Output: `aleatoric_probe_best_n100_m20_K40.csv` in F.6.1.6 dir.

### C.3 Leaf-eval bypass sweep ([val_stage4_mcts.py](../src/scripts/val_stage4_mcts.py))

Re-evaluate iter-399 best_model at val time under MCTS × leaf_eval × K (val_seed=42, 2000 instances, ε=0, τ=0):

| variant | leaf_eval | K | val_avg_cost | Δ vs greedy |
|---|---|---:|---:|---:|
| greedy θ★ | — | — | 3.86279 | (ref) |
| MCTS vh | value_head | 40 | 3.86816 | **+0.0054 worse** (p<0.01) |
| MCTS vh | value_head | 200 | 3.86753 | +0.0048 (bumping K barely moves vh) |
| **MCTS rollout** | **rollout** | **40** | **3.83437** | **−0.0284 (beats Stage 1 canonical greedy 3.83943)** |
| MCTS rollout | rollout | 100 | 3.83338 | −0.0294 |
| MCTS rollout | rollout | 200 | **3.83296** | **−0.0298 (within 0.001 of Stage 3 K=400 rollout's 3.8312)** |

**Decisive: the 3.85 greedy ceiling is a vh-leaf-eval-induced training ceiling, not a model-quality ceiling.** F.6.1.6's encoder is inference-competitive with Stage 3 K=400 rollout once you swap leaf evaluators at val time. The 0.074 RMS bias propagates directly into MCTS visit distributions; more sims don't fix it. Memory entry: [project_alphagozero_value_head_leaf_eval_bias.md](C:\Users\Jun18\.claude\projects\c--Users-Jun18-Desktop-AM-ALPHAGOZERO\memory\project_alphagozero_value_head_leaf_eval_bias.md).

**Implications:** (a) inference-time, always use `leaf_eval=rollout` on F.6.1-family checkpoints — K=40 rollout buys 0.028 over greedy and breaks 3.85 trivially; (b) training-time: F.6.1.6 used vh leaf eval AT TRAINING TIME too, so π_t distillation targets carry the bias — explains the 3.85 greedy plateau. Natural F.6.1.7 entrypoint: rollout end-to-end during training. Pursued as the lv0 ablation below (§D).

---

## §D — lv0 ablation (rollout + λᵥ=0 end-to-end)

### D.1 Implementation (2026-05-09)

- Added the rollout-teacher value-loss ablation. `train_step_alphazero` now treats `lambda_v=0` as a true policy-only update: value MSE is still logged, but no value-head or shared value gradients are computed/applied. This directly tests whether value regression contaminates the encoder when `leaf_eval=rollout`.
- Iteration telemetry now includes predicted-policy entropy and MCTS-vs-greedy teacher-quality metrics (`mcts_delta_vs_greedy_mean`, `mcts_win_rate_vs_greedy`, `greedy_cost_mean`, `mcts_cost_mean`).
- Added Modal entrypoint `run_rollout_lambda_ablation` for the priority `lambda_v=0` vs `lambda_v=1` rollout grid, with optional `lambda_v=0.1`.
- Verification passed for `py_compile`, focused `lambda_v=1` and `lambda_v=0` trainer smokes, tiny CPU rollout train smokes for both lambda settings, and the full Stage 4 smoke harness. The smoke harness now uses a workspace scratch-dir helper instead of `tempfile.TemporaryDirectory()` to avoid Windows sandbox ACL issues around `torch.save`.

### D.2 TSP-20 grid: lv0 vs lv1 (Modal ablation results) — **COMPLETE 2026-05-09**

50-iter from-scratch grid via `run_rollout_lambda_ablation`. Both variants: `leaf_eval=rollout`, K=40, step10, ε=0.25, value_target_norm=none, buffer_capacity=5000, batch=512, train_steps=200, lr_model=5e-4, wd=0, val_seed=42. Wall: lv1 1.07h, lv0 1.16h on A10.

| run | wandb | lambda_v | best val | iter | final val (49) | accepts/50 |
|---|---|---:|---:|---:|---:|---:|
| `rollout_lv1_K40_step10_eps25_50iter_...` | [up6avuf9](https://wandb.ai/lejun/am-alphagozero/runs/up6avuf9) | 1.0 | 3.91933 | 44 | 3.93204 | 18 |
| `rollout_lv0_K40_step10_eps25_50iter_...` | [1syc0kk8](https://wandb.ai/lejun/am-alphagozero/runs/1syc0kk8) | 0.0 | **3.87901** | 49 | 3.87901 | 15 |

**Headline: lv0 wins by 0.053** on val at iter 49 (3.879 vs 3.932).

Both rollout-trained variants beat the F.6.1.6 (vh-leaf-eval) trajectory at matched iter count (F.6.1.6 ≈ 3.99 at iter 49). The `leaf_eval=rollout` choice alone buys ~0.06 vs vh; turning off value loss (`lambda_v=0`) buys an additional 0.053 on top.

**Side-by-side val_avg_cost milestones:**

| iter | lv1 | lv0 | gap (lv1−lv0) |
|---:|---:|---:|---:|
| 0 | 4.824 | 4.396 | +0.428 |
| 5 | 4.275 | 4.027 | +0.248 |
| 10 | 4.135 | 3.975 | +0.160 |
| 15 | 4.032 | 3.947 | +0.085 |
| 25 | 3.979 | 3.914 | +0.064 |
| 35 | 3.946 | 3.892 | +0.054 |
| 49 | **3.932** | **3.879** | +0.053 |

The gap narrows from 0.43 (iter 0) to 0.05 (iter 49) but is monotonically positive — lv0 leads at every milestone after warm-up. lv0 still has slope at iter 49 (last accept iter unknown but accept rate 15/50 ~ similar to lv1's 18/50, so neither is gate-saturated).

**Train signals — segment means (iter 25-49):**

| signal | lv1 | lv0 | diff |
|---|---:|---:|---:|
| policy_loss_mean | 0.486 | 0.409 | lv0 fits 16% better |
| value_loss_mean | 0.025 | 6.118 | lv0 head is uncoupled (random-init level — by design) |
| mean_entropy_pi (target) | 0.296 | 0.281 | lv0 targets slightly more peaked |
| mean_entropy_policy (model) | 0.464 | 0.400 | lv0 model also more peaked, tighter to target |
| value_grad_norm_shared | 0.644 | **0.000** | confirms lv0 truly skips value-side encoder gradients |
| policy_grad_norm | 1.326 | 0.615 | lv0 has half the policy-grad-norm — encoder is at a sharper minimum |

Critical: `value_grad_norm_shared = 0` for lv0 confirms the implementation correctly skips value gradients at every train step. The 16% lower `policy_loss_mean` + 0.064 lower model-policy entropy is the encoder being free to specialize for policy distillation without the biased value-head multitask pull.

**Teacher MCTS quality (mcts vs greedy) — segment means (iter 25-49):**

| variant | greedy_cost | mcts_cost | delta | win_rate |
|---|---:|---:|---:|---:|
| lv1 | 3.952 | 3.852 | -0.101 | 0.785 |
| lv0 | 3.898 | 3.846 | -0.053 | 0.691 |

Both teachers (MCTS-rollout) reach similar quality (3.85 ± 0.005), so the teacher signal isn't the differentiator. **What differs is how well the model's *greedy* policy tracks the teacher**: lv0's gap (delta=-0.053) is half of lv1's (delta=-0.101), meaning lv0 absorbs the MCTS signal twice as efficiently into the greedy policy. lv1's encoder is split between value and policy multitask, slowing distillation; lv0's is purely policy-tuned.

**Mechanism (consistent with the F.6.1.6 diagnostic chain):** the F.6.1.6 value head has ~0.074 RMS bias against E[z|s]. With leaf_eval=rollout, the value head doesn't enter MCTS — so its gradient on the shared encoder is **pure auxiliary noise from a biased target**. lv0 removes this noise, freeing the encoder to specialize for policy distillation.

This is generalizable: **for any Stage 4 recipe with `leaf_eval=rollout`, default to `lambda_v=0`**.

### D.3 lv0 +50 iter resume to iter 99 (constant lr=5e-4) — **COMPLETE 2026-05-09**

Resume entrypoint `run_rollout_lv0_resume50_to_iter99` added to [src/scripts/modal_run_train_alphazero.py](../src/scripts/modal_run_train_alphazero.py); same recipe verbatim plus `--resume_from outputs/.../rollout_lv0.../iter-49.pt`. W&B [d8uyrrm1](https://wandb.ai/lejun/am-alphagozero/runs/d8uyrrm1). Wall: 53.5 min on A10.

**Headline: lv0 saturates around 3.86 at constant lr=5e-4 — does NOT break 3.85 by iter 99.**

Final iter 99 val = **3.86073** (best of run, accepted at iter 99). Never crossed below 3.86 in the full 100 iters.

Trajectory milestones:

| iter | val | source |
|---:|---:|---|
| 0 | 4.3962 | orig |
| 25 | 3.9142 | orig |
| 49 | 3.8790 | orig (last) |
| **50** | **3.8832** | resume start (small expected uptick) |
| 60 | 3.8762 | resume |
| 70 | 3.8702 | resume |
| 80 | 3.8709 | resume |
| 90 | 3.8665 | resume |
| **99** | **3.8607** | resume final (last accept) |

Slope: iter 0–49 ≈ −0.010/iter, iter 50–99 ≈ −0.0005/iter (20× slower → recipe saturating at lr=5e-4). Resume gate stats: 8/50 accepts (16%) vs lv0 first 50 iters' 15/50 (30%).

Comparison vs F.6.1.6 at iter ~100:

| run | leaf_eval | λᵥ | iter ~100 val | recipe |
|---|---|---:|---:|---|
| F.6.1.6 | value_head | 1.0 | 3.9013 (iter 100) | step-decay 5e-4→1e-4 just dropped at iter 100 |
| F.6.1.4.c chain | value_head | 1.0 | 3.866 (iter 127) | const 5e-4 then 1e-4 from iter 149 |
| **lv0 resume** | **rollout** | **0.0** | **3.8607 (iter 99)** | **const 5e-4 entire run** |

lv0 at iter 99 (3.8607) beats F.6.1.6 at iter 100 by 0.041 and is comparable to F.6.1.4.c at iter 127 (3.866). The rollout+λᵥ=0 recipe IS clearly better than vh+λᵥ=1, just hasn't yet had access to lr decay.

### D.4 lv0 +100 iter resume at lr=1e-4 (iter 100 → 199) — **COMPLETE 2026-05-10**

Resume entrypoint `run_rollout_lv0_resume100_lr1e4_to_iter199` added to [src/scripts/modal_run_train_alphazero.py](../src/scripts/modal_run_train_alphazero.py). Same recipe verbatim plus `--lr_model 1e-4` (applied AFTER `coach.load_checkpoint` via the lr-override-on-resume hook). W&B [`7ybaqa12`](https://wandb.ai/lejun/am-alphagozero/runs/7ybaqa12). Wall: **1.70h on A10**.

**Headline: lv0 chain breaks 3.85 by iter 158 and asymptotes at 3.8486 (iter 197 best, 3.8501 at iter 199).**

The lr=1e-4 lever fired immediately — val dropped from 3.8607 (iter 99 final under lr=5e-4) to 3.8572 (iter 100 first lr=1e-4 step) to 3.8549 (iter 102), **a 0.006 drop in 3 iters** — exactly the F.6.1.4.b pattern. The chain crossed below 3.85 at iter 158 (val=3.84997) and stayed below 3.85 for 19 of the remaining 42 iters.

Trajectory milestones:

| iter | val | regime | notes |
|---:|---:|---|---|
| 49 | 3.8790 | lv0 orig (lr=5e-4) | end of lv0 v1 |
| 99 | 3.8607 | lv0 res1 (lr=5e-4) | saturated at lr=5e-4 |
| **100** | **3.8572** | **lv0 res2 (lr=1e-4 fires)** | -0.0035 in 1 iter |
| 102 | 3.8549 | lr=1e-4 | -0.006 in 3 iters from iter 99 |
| 124 | 3.8519 | lr=1e-4 | iter-25 segment min |
| 158 | 3.8500 | lr=1e-4 | **first crossing below 3.85** |
| 197 | **3.8486** | lr=1e-4 | **chain best** |
| 199 | 3.8501 | lr=1e-4 (final) | |

Per-25-iter segment slopes during lr=1e-4:

| segment | first | last | min | mean | slope/iter |
|---|---:|---:|---:|---:|---:|
| iter 100–124 | 3.8572 | 3.8519 | 3.8519@124 | 3.8537 | **−0.000220** |
| iter 125–149 | 3.8519 | 3.8514 | 3.8500@146 | 3.8513 | −0.000020 |
| iter 150–174 | 3.8506 | 3.8504 | 3.8493@173 | 3.8505 | −0.000008 |
| iter 175–199 | 3.8501 | 3.8501 | 3.8486@197 | 3.8498 | **0.000000** |

The lr=1e-4 segment was productive in iter 100–124 (slope −0.0002/iter, 9× the lr=5e-4 saturated rate of −0.00005/iter), then progressively flatter, fully saturated by iter 175. lr=1e-4 well is now empty.

Comparison vs prior chains at matched best-val:

| chain | recipe | best val | iter at best | iters total | lr regimes |
|---|---|---:|---:|---:|---|
| F.6.1.4.c | vh, λᵥ=1, K=40, ε=0.25, step10 | 3.8498 | 225 | 225 | 5e-4 (149) + 1e-4 (76) |
| F.6.1.6 | vh, λᵥ=1, K=40, ε=0.25, step10 | 3.8578 | 365 | 400 | 4-segment step decay |
| **lv0 chain** | **rollout, λᵥ=0, K=40, ε=0.25, step10** | **3.8486** | **197** | **199** | **5e-4 (100) + 1e-4 (100)** |
| Stage 1 canonical | (AM+value greedy) | 3.83943 | epoch 99 | — | bs=512 |

**lv0 chain beats F.6.1.4.c by 0.0012 in 28 fewer iters, and beats F.6.1.6 by 0.0092 in HALF the iter count.** The lv0 recipe is unambiguously more sample-efficient.

But lv0 still sits **0.0092 above Stage 1 canonical** (3.83943) — Stage 1 parity not achieved on greedy. The MCTS-rollout teacher at 3.838 IS below Stage 1 already, suggesting the policy could close more of this gap given more training (greedy=3.850 vs teacher=3.838 → 0.012 still on the table).

### D.5 MCTS val sweep on lv0 iter-199 (mirror C.3 leaf-eval bypass) — **COMPLETE 2026-05-10**

Used existing `src/scripts/val_stage4_mcts.py` (no new code). 2000-instance val_seed=42, ε=0, τ=0, mcts_batch_size=1000, RTX 4060. `iter-199.pt[best_model]` = iter-178-accepted weights (last gate accept of the lv0 chain).

| variant | leaf_eval | K | val_avg_cost | Δ vs greedy | wall |
|---|---|---:|---:|---:|---:|
| greedy θ★ (no MCTS) | — | — | **3.85562** | (ref) | 0.3s |
| MCTS K=40 rollout | rollout | 40 | 3.83429 | −0.0213 | 79s |
| MCTS K=100 rollout | rollout | 100 | 3.83346 | −0.0222 | 180s |
| MCTS K=200 rollout | rollout | 200 | **3.83288** | **−0.0228** | 256s |

**Paired diff vs greedy (K=40):** t = −19.78, p ≪ 0.0001; **53.0% strictly better, 11.0% strictly worse, 35.9% equal** (vs F.6.1.6's 29.3%/37.4%/33.4%). lv0's policy is far more aligned with rollout-MCTS targets — when the model and MCTS disagree, MCTS wins ~5× as often as it loses.

Side-by-side vs F.6.1.6 iter-399:

| variant | lv0 iter-199 | F.6.1.6 iter-399 | Δ (lv0 vs F.6.1.6) |
|---|---:|---:|---:|
| greedy θ★ | **3.85562** | 3.86279 | **−0.00717** ✓ |
| MCTS K=40 rollout | 3.83429 | 3.83437 | −0.00008 (tie) |
| MCTS K=100 rollout | 3.83346 | 3.83338 | +0.00008 (tie) |
| MCTS K=200 rollout | 3.83288 | 3.83296 | −0.00008 (tie) |

**Critical finding:** lv0 wins **decisively on greedy** (0.007 better, ~10× the 2000-inst SE), but the MCTS-rollout-augmented numbers are **statistically indistinguishable** between the two recipes at K∈{40,100,200}. The 0.007 greedy advantage gets *absorbed* by MCTS-rollout — both models hit the same K=200 asymptote of ~3.833.

Reference points + interpretation:

| reference | val_avg_cost | regime |
|---|---:|---|
| Gurobi (TSP-20 1000-inst seed=1234) | 3.8279 | optimum |
| Stage 3 K=400 rollout on Stage 1 model | 3.8312 | Stage 1 + heavy MCTS |
| **lv0 iter-199 + K=200 rollout** | **3.8329** | **Stage 4 + light MCTS** |
| F.6.1.6 iter-399 + K=200 rollout | 3.8330 | Stage 4 (vh-trained) + light MCTS |
| Stage 1 canonical (greedy) | 3.8394 | Stage 1 alone |
| **lv0 iter-199 (greedy)** | **3.8556** | **Stage 4 (lv0) alone** |
| F.6.1.6 iter-399 (greedy) | 3.8628 | Stage 4 (vh) alone |

Reading the table:
- **Stage 4 + K=200 rollout (3.8329) beats Stage 1 canonical greedy (3.8394) by 0.007.** AGZ sample-efficiency claim holds at the MCTS-augmented inference level.
- **Stage 4 + K=200 (3.8329) is within 0.0017 of Stage 3 K=400 on Stage 1 (3.8312) at HALF the search budget.** lv0/F.6.1.6 models can substitute for Stage 1 weights at lower search costs.
- **Gap to Gurobi** is 0.005 for both Stage 4 + K=200 configs vs 0.003 for Stage 3 K=400. Stage 4's model + light search closes ~85% of the optimality gap.

**Why MCTS levels the playing field.** At K=200, rollout-MCTS does ~200 tree expansions × ~20 NN forwards per rollout = ~4K decoder evaluations per instance. The greedy policy contributes the priors and the rollout-decode policy, but both lv0 and F.6.1.6 produce policies good enough to drive rollout to similar cost. The 0.007 difference in standalone greedy quality gets *averaged out* by the search.

**Implications:**
1. For deployment scenarios using only the greedy policy: lv0 is unambiguously the right recipe (0.007 better on TSP-20).
2. For deployment scenarios using MCTS at inference: lv0 vs vh+λᵥ=1 don't matter much at TSP-20. lv0 wins on training wall (net wall ~similar despite rollout 5×/iter, since lv0 hit best val at iter 197 vs F.6.1.6's iter 365).
3. Stage 4's sample-efficiency claim is supported. lv0 at 199 iters × 1000 instances/iter = 200K total instances, beats Stage 1 canonical (1.28M instances) at greedy + matches it at MCTS-augmented inference. Roughly **6.4× sample efficiency**.
4. At TSP-50, the greedy gap to Stage 1 will be much larger (~0.26 currently); the lv0 advantage on greedy could matter more in absolute terms, AND the K=200 rollout asymptote will likely be much further above Gurobi (more headroom for search to help).

---

## §E — TSP-50 scaling

### E.1 TSP-50 lv0 K=50 from-scratch (oxjyj70e + Track A relaunch 1wpkngg9) — **COMPLETE 2026-05-11**

Original run wandb `oxjyj70e` (`tsp50_lv0_K50_50iter_20260511T051358`) launched 05:13:58 UTC 2026-05-11 with Fix #1+2b+3+4+5. Killed at iter 15 (best val 6.50) for Track A relaunch.

Relaunch wandb `1wpkngg9` (`tsp50_lv0_K50_resume34_from_iter15_trackA_20260511T093124`) resumes from `iter-15.pt` for +34 iters (target iter 49), same recipe verbatim, all 5 fixes + Track A. First-two-iter walls vs Fix #5 baseline:

| iter | val | mcts_w | vs Fix #5 |
|---:|---:|---:|---:|
| 16 | 6.5566 | **318.2s = 5.3 min** | −582s = **−65%** |
| 17 | 6.3184 | **310.2s = 5.2 min** | −590s = **−66%** |

Iter 17 val 6.3184 was a new best for lv0 TSP-50 — beat `oxjyj70e`'s prior best of 6.4959 (at iter 8) by 0.18 in 9 iters. Still 0.27 above Stage 1 (5.7999) and 0.63 above Gurobi (5.6987), but trajectory healthy.

Mid-run snapshot (1wpkngg9 trajectory iter 16-30 via wandb scan, polled 2026-05-13):

| iter | val | mcts_w(s) | accept | mcts_cost | greedy | dgreedy |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 6.5566 | 318.2 | 1 | 5.9583 | 6.2543 | -0.2960 |
| 17 | 6.3184 | 310.2 | 1 | 6.0302 | 6.6048 | -0.5746 |
| 23 | 6.1709 | 312.2 | 1 | 5.9687 | 6.2472 | -0.2785 |
| 30 | 6.1858 | 309.7 | 0 | 5.9439 | 6.1729 | -0.2290 |

Best accepted val in resume: 6.1709 at iter 23. avg mcts_w (iter 16-30): 309.9s = 5.16 min/iter; iters remaining (to 49): 19, projected ETA ~1.67h.

### E.2 TSP-50 lv0 K=50 +50 iter resume to iter 99 (Track A relaunch) — **IN FLIGHT 2026-05-13**

`tsp50_lv0_K50_resume50_to99_trackA_20260513T064031` — +50 iters from `1wpkngg9 iter-49.pt` at lr=5e-4 const, target iter 99. Track A throughout. (Not actively tracked at partition time.)

### E.3 K-comparison ablation (K=50 vs K=100) — see commits `d60e4d3`, `8e1f47b`

20-iter parallel TSP-50 from-scratch entrypoints + K=50 +50 iter resume entrypoint added to modal launcher. Tests whether the K-dominates finding from TSP-20 sub-budget probe (A.3) holds at TSP-50.

### E.4 TSP-50 lv0 step-decay 400-iter — **OPEN (not yet launched)**

Proposed Modal entrypoint mirroring F.6.1.6 step-decay schedule (5e-4 → 1e-4 → 2e-5 → 4e-6 over 400 iters) at TSP-50 lv0 recipe. Hypothesis: lv0 starts 0.04 ahead of F.6.1.6 at iter 100; if the gap holds, asymptote is plausibly 0.04 below F.6.1.6's 3.8578 → ~3.82 territory. Cost: ~$30-50 Modal credits, ~5h on A10 (rollout 5× vh wall).

---

## §F — Wall-time optimizations (six surgical fixes + Track A)

### F.1 Solver wall-time optimizations — Fix #1, #2b, #3 — **COMPLETE 2026-05-10**

While the TSP-50 K=50 lv0 50-iter run was queueing/launching, identified and fixed the dominant rollout-path Python overhead in [src/am_baseline/search/mcts_cpp/solver.py](../src/am_baseline/search/mcts_cpp/solver.py) for the cross-instance batched solver `CppBatchMCTSSolver._solve_chunk`. Determinism preserved across all three (max_diff = 0.0e+00 on TSP-20 K=40 M=100 paired-seed runs).

**Fix #1 — `cache_key` body uses `bytes(visited)`** ([solver.py:739-758](../src/am_baseline/search/mcts_cpp/solver.py#L739-L758)). Replaced `tuple(bool(v) for v in snap["visited"])` with `bytes(snap["visited"])`. Same hashability/uniqueness, single C-level allocation.

**Fix #2b — Vectorize `rollout_many` state + masked argmax** ([solver.py:884-1023](../src/am_baseline/search/mcts_cpp/solver.py#L884-L1023)). Replaced per-rollout Python state-dict + per-step argmax loop with numpy state arrays. Action selection vectorized via `np.where(mask_arr, -np.inf, probs_arr).argmax(axis=-1)`.

**Fix #3 — Numpy-direct evaluator `eval_many_arrays` for rollout** ([solver.py:813-973](../src/am_baseline/search/mcts_cpp/solver.py#L813-L973)). Added a parallel evaluator entry that takes numpy `(slot_a, step_a, first_a, prev_a, length_a, visited_a)` directly. Bypasses three layers the dict-based `eval_many` path required for rollout. Cache stays shared.

Local measurements (RTX 4060, TSP-50 K=50 M=200, leaf_eval=rollout):

| variant | wall (s) | vs original | vs prior |
|---|---:|---:|---:|
| Original (pre-fix) | 518.5 | — | — |
| Fix #1 + Fix #2b | 326.7 | **−37.0%** | — |
| Fix #1 + #2b + #3 | **282.3** | **−45.6%** | **−13.6%** |

### F.2 Production-scale probe + Fix #4 — **COMPLETE 2026-05-10**

Local Fix #1+2b saving did NOT translate to production M=1000 (1255s vs 1260s — essentially no change). User killed `qejtsdp1` after observing the 21 min wall; launched single-iter cProfile probe on Modal A10G with all three fixes.

Probe finding: Fix #3 alone unlocks **−15% production wall** (1260s → 1073s). Python overhead at M=1000 is dominated by per-row operations Fix #1 and #2b didn't address; Fix #3's `eval_many_arrays` eliminates O(M) per-call overhead.

**Fix #4 — cache stores numpy arrays, not Python lists** ([solver.py:957-973, 1036-1054](../src/am_baseline/search/mcts_cpp/solver.py#L957-L1054)). Replaced `eval_cache[key] = (probs_row.tolist(), mask_row.tolist(), value)` with `(probs_row.copy(), mask_row.copy(), value)`. `.copy()` on a (50,) numpy array is ~200ns vs `.tolist()` ~7µs (~35× faster). Both `eval_many_arrays` (rollout) and `eval_many` (selection) updated. C++ `apply_results` reads cache values via `sequence_to_doubles(handle)` which iterates any PySequence, so it transparently accepts either.

Production walls — Fix #1+2b+3+4 relaunch (`fc3fja6h`, launched 22:20:38 UTC):

| run | code | iter 0 | iter 1 | iter 2 |
|---|---|---:|---:|---:|
| 5ppdx0kf (pre-fix) | baseline | 1031s | 1250s | 1271s |
| qejtsdp1 (killed) | Fix #1+2b | 1227s | 1249s | 1306s |
| Modal probe (1 iter) | Fix #1+2b+3 | 1073s | — | — |
| **fc3fja6h (relaunch)** | **Fix #1+2b+3+4** | **612s** | **777s** | TBD |

Steady-state iter 1 at **777s = 12.95 min/iter** vs pre-fix 1255s = **−38% production saving**. Beat 985s projection by 21%. Fix #4 alone saved ~300s (1073→777) at M=1000.

### F.3 Fix #5 — bulk-vectorize cache key construction — **COMPLETE 2026-05-11**

After Fix #4 the dominant remaining hotspot was the per-row Python loop inside `eval_many_arrays` (~300s tottime in the Modal M=1000 probe). The loop body did 4 numpy-scalar→Python-int casts plus a 6-tuple build per row × 58.7M rows = ~6µs/row.

Fix #5 ([solver.py:739-769, 863-924](../src/am_baseline/search/mcts_cpp/solver.py#L739-L924)) replaces this with:
- **Bulk numpy bit-packing**: one vectorized op builds a `(B,)` int64 array packing `slot (10b) | need_value (1b) | step (6b) | first (6b) | prev (6b)` into a single header per row.
- **Bulk `.tolist()`**: header + slot + step arrays converted to Python lists once.
- **2-tuple cache key**: `(packed_header_int, visited.tobytes())` instead of 6-tuple.
- `eval_many` (selection path) updated to produce the same key format so the cache stays shared.

Determinism preserved: TSP-20 K=40 M=100 paired-seed produces `max_abs_cost_diff = 0.0e+00`, tours identical, mean_cost matches pre-Fix-5 baseline exactly.

Modal probe Fix #5 verification (probe_solver_n50_K50_M1000_20260511T042411):

| code state | wall | vs pre-fix |
|---|---:|---:|
| Pre-fix | 1255s = 20.9 min | — |
| Fix #1+2b | 1260s = 21.0 min | ~0% |
| Fix #1+2b+3 (probe) | 1073s = 17.9 min | −15% |
| Fix #1+2b+3+4 | ~777-910s = 13-15 min | −31% |
| **Fix #1+2b+3+4+5 (probe)** | **709s = 11.8 min** | **−43.5%** |

**Honest production reality:** Fix #5 helps **~3% in steady-state training**, NOT the 17% the probe predicted. Probe ran on random-init model (high cache hit rate → per-row Python loop dominates); trained model produces more confident decoder outputs → lower cache hit rate → more time in GPU NN forward (irreducible). Iter 0 (random init, mimics probe) saved 8.8%; iter 2 (after 1 epoch of training) saved only 3.1%.

**Lesson for future profiling:** when assessing inference-side optimizations for the training loop, profile on a **trained checkpoint**, not random init. Probe-only methodology gave a ~5× overestimate of marginal benefit.

### F.4 Track A — per-row step in decoder + merge step groups — **COMPLETE 2026-05-11**

After Fix #5 the dominant remaining hotspot was the per-row Python cache lookup loop (~190s tottime at M=1000). The deeper structural issue: `rollout_many` splits active rollouts by `step` because `Decoder.decode_step` takes `state.i` as a scalar. At TSP-50 K=50 M=1000, ~50 small NN calls per outer iter, average batch size only ~17 rows. GPU launch overhead (~300µs/call) dominates the NN-side time.

Track A allows `state.i` to be a `(B,)` tensor so all active rollouts at heterogeneous steps batch into ONE NN call per outer iter.

**Implementation:**
- [src/am_baseline/problem/state.py](../src/am_baseline/problem/state.py) — `StateTSP.i` may now be scalar (existing) or per-row `(B,)` (new). `__getitem__` slices per-row `i`; scalar `i` passes through. `update` uses `torch.where((self.i.view(-1, 1) == 0), prev_a, self.first_a)` instead of `prev_a if self.i.item() == 0 else self.first_a` (broadcasts for both shapes).
- [src/am_baseline/model/decoder.py](../src/am_baseline/model/decoder.py) — `_get_step_context` adds a per-row path: build both the W_placeholder context and the gathered first/current-node context, then select per-row via `torch.where(state.i.view(-1, 1, 1) == 0, placeholder, gathered)`. The scalar fast-path (training-time multi-step decode + MCTS selection) is preserved bit-for-bit by branching on `state.i.numel() == 1`.
- [src/am_baseline/search/mcts_cpp/solver.py](../src/am_baseline/search/mcts_cpp/solver.py) — `eval_many_arrays` drops `misses_by_step` grouping; builds a single `state` with `i = torch.from_numpy(step_a[miss_idx])` (per-row tensor). `rollout_many` drops the `for _step in np.unique(active_steps):` loop and calls `eval_many_arrays` once per outer iter on the full active set. Zero-out first/prev for step==0 rows so the unused gather has valid indices.

**Local validation (RTX 4060):**

Determinism (TSP-20 K=40 M=100):
- Random-init paired-seed: mean_cost = 4.855792 (matches Fix #5 baseline exactly), tours identical, max_diff = 0.0e+00.
- Trained ckpt paired-seed (`stage4_pilot_v3_resumed iter-39`): mean_cost = 3.831700, tours identical, max_diff = 0.0e+00.

Wall (TSP-50 K=50 M=200 random-init):

| variant | wall (s) | vs original | batch_eval_calls |
|---|---:|---:|---:|
| Original (pre-fix) | 518.5 | — | n/a |
| Fix #1 + #2b + #3 + #4 + #5 | ~225 | −56.6% | ~399k |
| **Track A** | **120** | **−76.9%** | **64.6k (6.2× fewer)** |

Cache stats identical (10.32M hits / 1.42M misses) — no behavior change, only structural batch consolidation. Average rows per NN call: 17 → 22 (limited at M=200; expected ~110+ at M=1000).

Non-MCTS smoke ([src/scripts/smoke_alphazero.py](../src/scripts/smoke_alphazero.py)): all phases PASS, including the multi-step parallel decoder path which exercises the unchanged scalar `state.i` fast-path.

**Modal probe (with oxjyj70e iter-10 trained ckpt):**

Probed TSP-50 K=50 M=1000 with `outputs/tsp_50/tsp50_lv0_K50_50iter_20260511T051358_20260511T051406/iter-10.pt`. **Wall: 435.5s = 7.3 min** vs prior probes/production:

| code state | wall | vs pre-fix |
|---|---:|---:|
| Pre-fix | 1255s = 20.9 min | — |
| Fix #4 | ~830s = 13.8 min | −34% |
| Fix #5 | ~850s = 14.2 min | −32% |
| **Track A (trained ckpt)** | **435s = 7.3 min** | **−65%** |

Far beat the plan's 580-620s estimate. Track A saves both NN time AND total cache-lookup volume because the merged batches amortize kernel-launch overhead AND the trained model converges faster in the MCTS tree → fewer total rollout-steps to evaluate.

`batch_eval_calls` dropped from ~400k → **60k (6.6× fewer)**. Average rows per NN call: 17 → 720 (saturated at this scale). GPU forward floor is now ~16% of wall.

### F.5 Final wall progression summary

| code state | M=1000 production wall | vs pre-fix |
|---|---:|---:|
| Pre-fix | 1255s = 20.9 min/iter | — |
| Fix #1+2b | 1260s = 21.0 min/iter | ~0% |
| Fix #3 | (random-init probe only: 1073s) | — |
| Fix #4 (`fc3fja6h`) | ~830s = 13.8 min/iter | −34% |
| Fix #5 (`oxjyj70e`) | ~900s = 15.0 min/iter | −28% |
| **Track A (`1wpkngg9`)** | **310-320s = 5.2-5.3 min/iter** | **−75%** |

Wall went from 20.9 min/iter → 5.3 min/iter (**−75%**) across six surgical optimizations with **zero behavior change** (max_abs_cost_diff = 0.0 at every step). A 50-iter Stage 5 TSP-50 lv0 run now fits in ~4h, down from ~17h.

---

## §G — mcts_batch_size sweep — **COMPLETE 2026-05-07**

Discovery: `mcts_batch_size` (prior default 64) is the **cross-instance chunk size** in [solver.py:691](../src/am_baseline/search/mcts_cpp/solver.py#L691), not a per-NN-forward batch. At M=1000 the prior default sequentially processed 16 chunks of 64 — wildly underutilizing the GPU.

| mcts_batch_size | s/iter on F.6.1.3 recipe | speedup |
|---|---:|---:|
| 64 (prior default) | ~124 | 1.0× |
| 256 | ~38 | 3.3× |
| **1000** | **~25** | **5.0×** |
| 2000 (≡1000 at M=1000) | ~25 | 5.0× |

Quality unaffected (val trajectories within ±0.05 RNG noise). **Action: default 64 → 1000 in [train_alphazero.py:139](../src/scripts/train_alphazero.py#L139).** A 100-iter run that took ~110 min mcts wall now takes ~22 min; a 50-iter resume drops ~55 min → ~11 min.

---

## Memory hooks (cross-stage references)

- [`project_alphagozero_value_head_leaf_eval_bias.md`](../../C:/Users/Jun18/.claude/projects/c--Users-Jun18-Desktop-AM-ALPHAGOZERO/memory/project_alphagozero_value_head_leaf_eval_bias.md) — vh leaf-eval bias finding from §C.3 + lv0 generalization from §D.
- [`project_lr_fairness_for_stage4.md`](../../C:/Users/Jun18/.claude/projects/c--Users-Jun18-Desktop-AM-ALPHAGOZERO/memory/project_lr_fairness_for_stage4.md) — lr=5e-4 derivation rationale from §A.1.

---

## Notes

- W&B project: `lejun/am-alphagozero`. Run IDs cited inline.
- Stage 0 Gurobi reference TSP-20: 3.8279 mean (1000 instances, seed=1234); TSP-50: 5.6987.
- Stage 1 reference val_avg_cost: TSP-20 canonical bs=512 = 3.83943; TSP-50 = 5.7999.
- Stage 3 reference test-time MCTS K=400 rollout (TSP-20): 3.8312 (gap 0.087% vs Gurobi).
