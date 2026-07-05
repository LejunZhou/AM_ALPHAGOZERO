# Stage 5 §H — AlphaGo-Fan/Lee mixed leaf evaluation on TSP-20

Sibling of [stage5_plan.md](stage5_plan.md) §A–§G. Mirror progress file:
[`_progress/stage5_mix_leafeval_progress.md`](../_progress/stage5_mix_leafeval_progress.md).

> **STATUS: CLOSED 2026-07-02 — NEGATIVE, question answered.**
> §H.4 mix(λ=0.5) self-play lost to §B.4 rollout+lv0 (+0.0132; λᵥ-interference
> the best-supported dominant channel, mix-inference channel priced at
> +0.001 by Phase A; magnitude ~1σ of the ~0.01–0.015 seed-noise floor).
> §H.7 (separate ValueTrunk, Phase 0/0b) was also negative and produced the
> real answer: the value head fails **off-policy** — calibrated on trajectory
> states (|err| 0.08 raw) but ~0.6 raw optimistic on untaken siblings, so its
> sibling-ranking signal is ≈ 0 wherever MCTS reads it; richer representations
> make this *worse*, not better. Phase C (§7) was not run — evaluating a leaf
> eval that cannot rank actions adds nothing. The premise itself was flawed:
> the leaf rollout is deterministic argmax (zero sampling variance), so there
> was never rollout variance to reduce — see **§12** for the reframed,
> compute-based value proposition and the preconditions for reopening.
> Full evidence: progress §H.4 attribution review + §H.7.

## 1. Context

Two earlier Stage 5 findings drive this:

- **§C.3 leaf-eval bypass** ([`_progress/stage5_progress.md`](../_progress/stage5_progress.md) §C.3, memory `project_alphagozero_value_head_leaf_eval_bias.md`).
  F.6.1.6's trained value head, used as `leaf_eval='value_head'` at val time, is
  **statistically tied with greedy** at every K∈[40,200] (3.868 vs 3.863 at
  K=40, p<0.01 worse). The structural RMS bias of 0.074 against E[z|s] poisons
  visit distributions; more sims don't help. Switching the same checkpoint to
  `leaf_eval='rollout'` buys −0.028 over greedy at K=40 and breaks 3.85
  trivially.
- **§D.4 lv0 chain** (same progress doc). Training end-to-end with
  `leaf_eval='rollout', λᵥ=0` reaches **3.8486 at iter 199** — beats
  F.6.1.6 (3.8578 @ iter 365) at half the iter count. But the value head is
  *unused*, so we pay full rollout variance.

The current best recipe (lv0 chain) pays variance cost because rollout
estimators are noisy single greedy trajectories at each leaf. A
well-calibrated value head would reduce variance at no bias cost. AGZ
canonical (`leaf_eval='value_head'`) inherits the §C.3 bias. The classical
answer is the **AlphaGo Fan/Lee mixed evaluation** — already listed in
[`stage4_algorithm_spec.md`](stage4_algorithm_spec.md) §4.1.5 as the
intermediate point between AGZ-canonical and AlphaGo-Lee on the leaf-eval
spectrum, but never instantiated in this project:

    V(s_L) = λ · v_θ(s_L) + (1 − λ) · z_rollout / bl_val(x),   λ ∈ [0, 1]

with λ ≈ 0.5 in AGFan/AGLee. The hypothesis: an interior λ trades a small
dose of value-head bias for substantial rollout variance reduction, beating
both endpoints (λ=0 pure rollout = §B.4 3.8576 line; λ=1 pure value head =
§C.3 biased line). Used in **both** self-play (training) and inference,
following the AGFan/AGLee analog — the value-head **target** is unchanged
(z = realized cost_to_go / bl_val); only Q-values inside MCTS are blended.

## 2. Goal

Find λ\* ∈ (0,1) for TSP-20 that strictly improves on either endpoint at the
same train wall as §B.4 (K=10 step50 100-iter, 33 min on A10).

Success threshold (Phase C, 1000-instance seed=20260430):
greedy ≤ 3.85 AND MCTS K=40 ≤ 3.835 (clearly beats §B.4's 3.8576).
A null result (mix ties λ=0) is still informative — closes the AGFan/AGLee
question for the TSP-20 regime, complementing §C.3 / §D.

## 3. Approach

**Single search-code change.** Add `leaf_eval='mix'` with a
`mix_lambda ∈ [0,1]` field. Both existing branches in
[`mcts.py`](../src/am_baseline/search/mcts.py) `_expand` (lines 463–484)
return positive normalized cost-to-go in identical units, so the blend is
a one-line linear combination at the same dispatch point. The C++ batched
backend already supports computing both estimators at the same leaf (the
R²-probe path at [`mcts.cpp`](../src/am_baseline/search/mcts_cpp/mcts.cpp)
lines ~335–350), so mix is a small extension there too.

**No change to:**
- Backup / sign convention (Q = −total_norm at `mcts.py` lines 383–388).
- Value-head training target (z = realized cost_to_go / bl_val).
- Loss function (MSE on z, lambda_v-weighted).

**Two-phase experiment:**
1. **Phase A (Colab T4, cheap)** — λ-sweep at *inference only* on F.6.1.6
   to pick the 1–2 best λ for Phase B.
2. **Phase B (Modal A10, full)** — from-scratch training with mix at the
   chosen λ, recipe = §B.4 K=10 step50 100-iter with `lambda_v=1.0`
   (value head ON; §B.4 was lv0) and `leaf_eval='mix'`.
3. **Phase C** — canonical 1000-instance eval against F.6.1.6 best and
   the lv0 chain iter-199.

## 4. Critical files to modify

**Python backend (canonical reference):** [`src/am_baseline/search/mcts.py`](../src/am_baseline/search/mcts.py)
- `MCTSConfig` (line ~34): add `mix_lambda: float = 0.5`; extend the
  `leaf_eval` doc comment.
- `VALID_LEAF_EVAL` (line 125): add `'mix'`.
- `_validate_config` (line ~170): accept `'mix'`; require
  `0 ≤ mix_lambda ≤ 1`; require `model.value_head is not None` for `'mix'`
  (mirror the existing `value_head` check at line ~212); extend the
  `value_norm='sqrt_n'` incompatibility check at line ~224 to reject
  `'mix'` too.
- `_expand` (lines 463–484): add the `'mix'` branch — compute
  `v_head = _convert_value_head_output(model.value_head(glimpse), node, bl_val)`
  AND `v_rollout = _rollout_remaining_real(state, fixed) / bl_val`,
  return `λ·v_head + (1−λ)·v_rollout`.
- `_populate_priors` (lines 434–461): mirror the mix branch so the root's
  cached `node.v_estimate` is on the same scale as backed-up Q (required by
  `fpu_mode='node_value'`).

**C++ backend (production training path; `mcts_batch_size=1000` routes here):**
- [`mcts.hpp`](../src/am_baseline/search/mcts_cpp/mcts.hpp) /
  [`mcts.cpp`](../src/am_baseline/search/mcts_cpp/mcts.cpp): add
  `mix_lambda` to `Config` (parsed via `get_or<double>` at `mcts.cpp` line
  ~99); in the leaf-eval branch (around line ~482) set
  `need_value = (leaf_eval == "value_head" || leaf_eval == "mix")`; add
  `else if (cfg_.leaf_eval == "mix")` branch blending
  `convert_value_head_output(...)` with the rollout result from the
  existing `rollout_evaluator_` callback.
- [`solver.py`](../src/am_baseline/search/mcts_cpp/solver.py) (line ~169):
  pass `rollout_evaluator` when `leaf_eval ∈ {'rollout', 'mix'}`;
  serialize `mix_lambda` in `_cfg_dict()`. Mirror the existing R²-probe
  sanity check (lines ~120–124).
- Rebuild via `pip install -e .` (pyproject builds the pybind11 extension).

**Training-side plumbing:**
- [`train_alphazero.py`](../src/scripts/train_alphazero.py) (~line 128):
  add `--mix_lambda` float arg, default 0.5.
- [`coach.py`](../src/am_baseline/training/coach.py): add `mix_lambda` to
  the Coach config dataclass (line ~529) + forward into the MCTSConfig
  construction (line ~1089).

**Modal entrypoint (new):** [`modal_run_train_alphazero.py`](../src/scripts/modal_run_train_alphazero.py)
- New function `run_tsp20_k10_mix_step50(mix_lambda: str = "0.5", timestamp: str = "")`
  mirroring `run_tsp20_k10_lv0_step50` (line 2344). Reuse `_f61_args(...)`
  with `k=10, buffer_capacity=5000, lr_model="5e-4", lr_decay="0.2",
  lr_decay_step_size=50, temperature_schedule="step10",
  dirichlet_epsilon="0.25", leaf_eval="mix", lambda_v="1.0"`, then
  extend args with `["--mix_lambda", mix_lambda, "--mcts_batch_size", "1000"]`.
  Single change vs §B.4: **lv0→lv1 (lambda_v 0.0→1.0) +
  leaf_eval rollout→mix**.

**Probe / eval scripts:**
- [`probe_mcts_decomp.py`](../src/scripts/probe_mcts_decomp.py): already
  wired for `--leaf_eval`; add `--mix_lambda` and forward.
- **New** `src/scripts/eval_tsp20_mix_lambda_sweep.py`: skeleton copied from
  [`eval_tsp20_full_comparison.py`](../src/scripts/eval_tsp20_full_comparison.py);
  for each λ in the grid, run MCTS K=40 on `val_size=10000, val_seed=42`
  on a single fixed checkpoint; emit one CSV row per λ with
  `mean_cost, std, wall, fwd_count_{decode,value,rollout}`. Phase A
  entry point on Colab T4.

## 5. Phase A — Colab T4 inference probe

Checkpoint: F.6.1.6 winner — `outputs/tsp_20/f616_400iter_step_decay_20260507T101222_20260507T101229/iter-361_accepted.pt`.

Grid: **λ ∈ {0.00, 0.25, 0.50, 0.75, 1.00}**, K=40, ε=0, τ=0,
val_size=10000, val_seed=42, all other knobs via
`val_stage4_mcts.py --match_train`.

Anchors (from §C.3 / §D.5):
- λ=0 (pure rollout K=40) on F.6.1.6 → expect ≈ 3.834.
- λ=1 (pure value_head K=40) on F.6.1.6 → expect ≈ 3.868 (> greedy).

Plan-modifying outcomes:
- Monotone curve (λ=0 strictly wins) → mix offers no benefit *on
  F.6.1.6's biased value head*; Phase B still informative because
  from-scratch mix training may produce a less-biased head.
- Interior λ wins → take top 1–2 λ into Phase B.

Wall budget: ~5 × 15–25 min ≈ 1.5–2 h on Colab T4.

## 6. Phase B — Modal A10 training

Recipe (`run_tsp20_k10_mix_step50` at λ = λ\* from Phase A):
```
graph_size=20, n_iterations=100, M_instances=1000,
n_simulations_train=10, train_steps_per_iter=200,
buffer_capacity=5000, batch_size=512, gate_every=1, gate_mode=ttest,
temperature_schedule=step10, val_size=10000, val_seed=42,
leaf_eval=mix, mix_lambda=<λ*>,
lambda_v=1.0,                           # ← lv1; §B.4 was lv0
max_grad_norm=1.0, value_target_norm=none,
lr_model=5e-4, lr_decay=0.2, lr_decay_step_size=50,   # 5e-4 -> 1e-4 at iter 50
weight_decay=0.0, dirichlet_epsilon=0.25, dirichlet_alpha_factor=10.0,
mcts_batch_size=1000
```
Wall: ~1.5–2 h per λ on A10 (K=10 cheap; mix adds 1 value-head MLP call
per leaf on top of rollout — essentially free since the glimpse is already
computed for priors). Run 1–2 λ. Modal cost: ~$5–10 per run.

**Direct apples-to-apples baseline.** §B.4
`tsp20_k10_lv0_step50_100iter_…` reached 3.8576 at iter 100 / 33 min.
Phase B differs only by `leaf_eval` and `lambda_v` — any val_avg_cost
delta is attributable to "mix + value-head training" vs "pure rollout +
no value-head training".

Other reference points:
- F.6.1 K=40 step30 lv1 — 3.860 @ iter 99 (full vh-trained baseline)
- F.6.1.6 step-decay 400-iter — 3.8578 best (§B.3 ceiling)
- lv0 chain iter-199 — 3.8486 (§D.4 current TSP-20 frontier)
- Stage 1 canonical greedy — 3.83943

## 7. Phase C — Canonical eval

Run [`eval_tsp20_full_comparison.py`](../src/scripts/eval_tsp20_full_comparison.py)
(1000 instances, seed=20260430, Gurobi-optimal) on the trained mix
checkpoint. Per-checkpoint columns:
- Greedy.
- MCTS K=10, 40, 100 with `leaf_eval=mix, mix_lambda=λ\*` (matched to training).

Compare to F.6.1.6 best, lv0 iter-199, and Stage 1 canonical. Report
optimal-gap %, n_optimal/1000, paired-t vs lv0 iter-199, verdict.

## 8. Verification

1. **λ=0 parity test (rollout).** Smoke (extend
   [`smoke_mcts.py`](../src/scripts/smoke_mcts.py) or new `smoke_mix.py`):
   fix seed, run MCTS K=40 with `leaf_eval='rollout'` vs
   `leaf_eval='mix', mix_lambda=0.0` on the same instance → costs match
   to fp tolerance (the value-head contribution is multiplied by 0).
2. **λ=1 parity test (value_head).** Same idea,
   `leaf_eval='value_head'` vs `leaf_eval='mix', mix_lambda=1.0` →
   exact match (rollout contribution multiplied by 0).
3. **C++/Python bit-equivalence at mix=0.5.** Extend the A14 suite
   (smoke_mcts.py) to run Python vs `CppBatchMCTSSolver` at
   `leaf_eval='mix', mix_lambda=0.5` → costs identical (max_abs_diff = 0).
4. **Phase A produces a coherent λ-curve.** No NaNs; ordering of λ=0
   and λ=1 endpoints matches the §C.3 / §D.5 measurements within
   per-instance SE.
5. **Phase B 100-iter run completes** in ~2 h on A10; log per-iter
   `mcts_s`, `fwd_count_{decode,value,rollout}`, `val_avg_cost`,
   `value_loss_mean`, `policy_loss_mean`, `mean_entropy_policy`. Best
   val reported alongside §B.4 (3.8576) and §B.3 F.6.1.6 (3.8578).
6. **Phase C produces a per-λ comparison table** in
   [`_progress/stage5_mix_leafeval_progress.md`](../_progress/stage5_mix_leafeval_progress.md)
   §H.5, with optimal-gap %, paired-t vs lv0 iter-199, and verdict.

## 9. Out of scope (deferred)

- TSP-50 mix experiments — wait for TSP-20 verdict.
- Stochastic / multi-rollout leaf eval — orthogonal variance-reduction
  lever; can be revisited if mix shows promise.
- Mixing the value *target* (z = blend of realized return +
  bootstrapped value head) — that's λ-return / n-step TD, a different
  idea than AlphaGo Fan/Lee.
- Per-step adaptive λ (anneal high-λ early, low-λ late) — revisit only
  if static λ shows promise.
- AGZ-Master-style mixed schedule (binary switch when R²>0.99,
  `stage4_plan.md:614`) — orthogonal to continuous λ-blend and
  deferred separately.

## 10. Cross-references

- Algorithmic spec entry: [`stage4_algorithm_spec.md`](stage4_algorithm_spec.md) §4.1.5
  (AGFan/Lee row of the leaf-eval mapping table).
- Stage 5 antecedents: [`_progress/stage5_progress.md`](../_progress/stage5_progress.md)
  §B.4 (K=10 step50 recipe), §C.3 (vh leaf-eval bias diagnosis),
  §D (lv0 ablation).
- Mirror progress: [`_progress/stage5_mix_leafeval_progress.md`](../_progress/stage5_mix_leafeval_progress.md).
- Internal Claude plan: `~/.claude/plans/i-want-to-test-polymorphic-cookie.md` (this plan was approved there first, then transcribed here for the project record).

---

## 11. §H.7 — Separate value-trunk experiment (richer value input)

**Motivation.** The §H.4 mix value head and F.6.1.6 are equally biased
(~0.082 vs 0.080 raw RMS against E[z|s]; calibration probe 2026-06-28). The
bias is **state-dependent geometry error** (|signed| 0.016 ≪ RMS 0.082), and
the head's only input is the policy **glimpse** — a softmax-pooled, next-step
query readout that discards the remaining-sub-tour geometry that determines
cost-to-go. Hypothesis: a value head with its **own representation** (attention
over the *unvisited* nodes) reduces the bias. If so, mix may finally beat the
lv0/rollout frontier; if not, the bias is capacity/target-fundamental and the
AGZ value head is a scaling-only bet on TSP.

**Design — `ValueTrunk`.** Own MHA: query = proj([h_current ; h_first])
(placeholder at step 0); K/V = node embeddings with **visited nodes masked
out** → attends over the remaining sub-tour; → 2-layer MLP → scalar **raw**
cost-to-go (`value_target_norm='none'`, unchanged). Encoder shared but read
**detached** (stop-grad) so the value loss never perturbs the policy encoder —
net recipe = "lv0 policy + side value-trunk used only at mix leaf-eval", which
closes the §D λᵥ-interference channel by construction.

**Key backend fact (verified).** The C++ MCTS never computes the value head —
`mcts_cpp/solver.py:31-32` calls a Python evaluator (`solver.py:305-309`:
`decode_step → value_head(glimpse)`) that already holds `fixed`
(node_embeddings) + `state` (mask). So the trunk needs **zero C++ changes / no
`.pyd` rebuild**; only Python eval sites change.

### Phase 0 — decisive cheap test (supervised; no self-play, no MCTS, no C++)
Load §H.4 `iter-99` (encoder+policy frozen) + its `buffer.pt` (100k tuples,
local). Supervised-train **only** the trunk: `MSE(v_trunk(s), cost_to_go)` (raw,
= `z·bl_val`). Re-run `probe_value_aleatoric` (rollout-K=10 reference, raw
units — matched to the §H.4 Job-B 0.082 anchor).
**Gate:** RMS bias **< ~0.04** → representation bottleneck → Phase 1; **≈ 0.08**
→ fundamental → stop & report. Fallback 0b: unfreeze encoder for the trunk grad.
Files: new `model/value_trunk.py`, `model/attention_model.py`
(`value_head_type` flag + `value_from_state`), new
`scripts/train_value_trunk_supervised.py`, `scripts/probe_value_aleatoric.py`
(trunk branch).

### Phase 1 — self-play (only if Phase 0 passes)
Swap value forward → `value_from_state` at the 3 Python sites (`trainer.py`,
`mcts.py` mix/value branches, `solver.py` `eval_many` + BatchSearch evaluator).
Train from scratch on TSP-20, recipe = §H.4 (K=10 step10 ε=0.25 mix λ=0.5
vtn=none 100-iter) + stop-grad trunk. Compare best val vs §B.4 (3.8576) /
§H.4 (3.871); re-probe. Entrypoint `run_tsp20_k10_mix_vtrunk_step50`.

### Phase 2 — verdict
1000-instance canonical eval, paired-t vs lv0 iter-199, log in §H.5/§H.6.

### Verification
Phase-0 bias gate; smoke (trunk shapes, CPU, `mix(λ=0)==rollout` exact,
Py↔C++ value parity); stop-grad telemetry `value_grad_norm_shared==0`.

### Design defaults
stop-grad ON · query=[current,first] · 1 MHA layer + 2-layer MLP ·
Phase-0 encoder = §H.4 `iter-99`.

### §11 closure — **CLOSED 2026-07-02 (NEGATIVE for the representation hypothesis)**

Phase 0 + 0b ran (shared-trunk, own-encoder, own-encoder+holdout variants;
checkpoints next to §H.4 `iter-99.pt`). Outcome, in full in progress §H.7:

- The **Phase-0 gate as specified above was the wrong metric.** On-policy
  RMS bias is *not* what limits the head — the §H.4 glimpse head is already
  near-perfectly calibrated on-policy (signed −0.01, |err| 0.08 raw vs
  greedy-completion gt). The decision-relevant metric is **within-node
  sibling ranking** (`probe_action_ranking.py`), where every head variant
  scores Spearman ≈ 0 (−0.19…+0.19) vs the rollout's 0.89.
- Cause: **off-policy calibration failure**, not representation. The buffer
  contains only trajectory states; on untaken siblings the head is ~0.6 raw
  optimistic (own-encoder trunk: −0.84 — *lower* on-policy error buys *more*
  off-policy overconfidence, so the dissociation is causal).
- **Phases 1–2 cancelled.** Self-play with any current head as (part of)
  the leaf eval cannot beat rollout; a better trunk does not change that.
- Any future gate: `probe_action_ranking` decision regret vs the greedy
  rollout's 0.047 benchmark — not RMS.

---

## 12. Reframed value proposition — compute, not variance — **ADOPTED 2026-07-02**

**Premise correction.** §1 motivated the value head as a variance reducer
for "noisy" rollout leaf evals. That premise is false for this system: the
leaf rollout is a *deterministic* argmax completion
([`mcts.py:550-561`](../src/am_baseline/search/mcts.py)) — zero sampling
variance at a fixed leaf — and §C.2 measured var(z|s) ≈ 0 for steps ≥ 2
under step10+argmax. The AGFan/Lee mix averaged a *stochastic* fast-rollout
policy; there is no analogous noise dividend here. Mixing could only ever
trade nonexistent noise for value-head bias, which is what §H.3/§H.4 observed.

**What the value head is actually worth: forward passes.** A rollout from a
leaf at step t costs (N − t) extra decode forwards; the value head costs ~0
extra (an MLP on the glimpse the leaf already computed for priors).
Empirically (Phase A, λ=0 row): rollouts are **10.48M of 11.54M decode
forwards = 91%** of search compute at TSP-20 K=40. A head-only search
therefore buys **~11× more simulations at matched wall on TSP-20**, scaling
as ≈ 1 + N/2: **~26× at TSP-50, ~51× at TSP-100**. That dividend is real —
but currently unredeemable: §C.3 showed vh at K=200 (5× sims) still loses to
rollout at K=40, because a leaf eval that cannot rank siblings does not
convert extra sims into better visits.

**Consequences (standing decisions):**
1. **TSP-20 is closed for value-head experiments.** K=10 rollout self-play
   costs 33 min end-to-end; there is no compute problem to solve at N=20.
   The frontier recipe stays `leaf_eval=rollout, λᵥ=0` (§D verdict).
2. **The value head re-enters only as a compute play at TSP-100+**, where
   O(N) decodes/leaf is the binding cost, and only after its ranking is
   fixed. Preconditions for reopening (in order):
   a. **Off-policy value targets** — label counterfactual children /
      within-tree leaf states with their greedy-rollout cost-to-go (already
      computed for free during `leaf_eval=rollout` self-play) and train the
      head on that distribution. Cheap Phase-0-style test: supervised-train
      the *existing glimpse head* on augmented data from the §H.4 buffer +
      rollout labels, then re-run `probe_action_ranking`.
      → **Instantiated as Stage 5 §V0** —
      [`stage5_offpolicy_value_plan.md`](stage5_offpolicy_value_plan.md)
      (scaffold complete 2026-07-04; Colab T4 gate run pending).
   b. **Gate on ranking, not RMS**: decision regret approaching the
      rollout's 0.047 (TSP-20 anchor) before any self-play spend.
   c. Only then a self-play trial — at a graph size where the ~1+N/2×
      sim multiplier pays for the head's residual ranking deficit.
3. Optional attribution refinement (not required for closure): §B.4
   verbatim + λᵥ=1 + `leaf_eval=rollout` (~$5 / ~40 min A10) splits the
   H.4 deficit between the λᵥ and mix channels and starts bounding the
   seed-noise floor; run it only if the λᵥ-interference magnitude at K=10
   becomes decision-relevant elsewhere.
