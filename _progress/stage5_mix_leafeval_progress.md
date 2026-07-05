# Stage 5 §H — AlphaGo-Fan/Lee mixed leaf evaluation on TSP-20 — progress

Mirror of [`_plans/stage5_mix_leafeval_plan.md`](../_plans/stage5_mix_leafeval_plan.md).
Tracks implementation, smoke tests, and the Phase A/B/C experimental sweep
for `leaf_eval='mix', mix_lambda=λ` (`v_leaf = λ · v_head + (1−λ) · v_rollout`).

Status convention mirrors [`stage5_progress.md`](stage5_progress.md): each
sub-section header carries `**OPEN** / **IN FLIGHT** / **COMPLETE <date>**`.
"Iter N" refers to AGZ self-play iteration unless specified.

---

## Status overview

| Phase | Status | Date | Headline result |
|---|---|---|---|
| H.1 Implementation (Python + C++ + plumbing) | **COMPLETE 2026-05-21** | 2026-05-21 | mix mode wired across mcts.py, mcts.cpp, BatchSearch, Coach, train_alphazero CLI, val_stage4_mcts, probe_mcts_decomp, Modal entrypoint |
| H.2 Smoke tests (λ=0 parity, λ=1 parity, C++↔Py at λ=0.5) | **COMPLETE 2026-05-21** | 2026-05-21 | `smoke_mix.py` M1/M2/M3 all pass with \|Δ\|=0.000e+00 |
| H.3 Phase A — Colab T4 inference λ-sweep on F.6.1.6 | **COMPLETE 2026-05-21** | 2026-05-21 | monotone curve, λ=0 wins; total spread 0.004 (much smaller than §C.3 anchor of 0.034); proceed to Phase B per plan |
| H.4 Phase B — Colab T4 training at λ=0.5 (K=10 step50 100-iter) | **COMPLETE 2026-05-22, CLOSED as answered 2026-07-02** | 2026-07-02 | **NEGATIVE: 3.87083 vs §B.4 3.8576 (+0.0132)** — attribution review: no impl bug; λᵥ-interference dominant, mix channel ~+0.001; "4.4σ" overstates (n=1 seed vs ~0.01–0.015 replication floor) |
| H.5 Phase C — 1000-instance canonical eval | **SUPERSEDED by §H.7** | 2026-07-02 | not run — §H.7 closed mix as negative; evaluating a leaf-eval that cannot rank actions adds nothing |
| H.6 Verdict + open follow-ups | **SUPERSEDED by §H.7.3** | 2026-07-02 | verdict folded into §H.7.3 (mix dead; value head fails off-policy, not in representation) |
| H.7 Separate value-trunk (Phase 0/0b supervised + ranking probe) | **COMPLETE 2026-07-02 (NEGATIVE)** | 2026-07-02 | trunk does NOT fix the head; root cause re-diagnosed as **off-policy calibration failure**, not representation (§H.7 below) |

Reference points (existing, for comparison):

| Recipe | val_avg_cost | iter / wall | source |
|---|---|---|---|
| Stage 1 canonical greedy | 3.83943 | epoch 99 | §A header |
| F.6.1 K=40 step30 lv1, iter 99 | 3.860 | iter 99 / ~2 h | §A |
| F.6.1.6 step-decay 400-iter best | 3.8578 | iter 365 / ~8 h | §B.3 |
| **§B.4 K=10 lv0 step50 100-iter best** | **3.8576** | **iter 100 / 33 min** | **§B.4** |
| §D.4 lv0 chain best (iter 197) | 3.8486 | iter 197 / ~3 h | §D.4 |
| §C.3 F.6.1.6 + MCTS K=40 rollout (val) | 3.834 | val-only | §C.3 |
| §C.3 F.6.1.6 + MCTS K=40 vh (val) | 3.868 | val-only | §C.3 |

§B.4 is the **direct apples-to-apples baseline** for Phase B: same K=10
step50 100-iter, only `leaf_eval` and `lambda_v` differ.

---

## §H.1 Implementation — **COMPLETE 2026-05-21**

Landed in the same commit as this progress doc. Pybind11 extension rebuilt
(`pip install -e .`). Smoke verification under §H.2.

### H.1.1 Python backend (`src/am_baseline/search/mcts.py`)
- [x] `MCTSConfig`: added `mix_lambda: float = 0.5`; extended `leaf_eval` doc.
- [x] `VALID_LEAF_EVAL`: added `'mix'`.
- [x] `_validate_config`: accepts `'mix'`; `0 ≤ mix_lambda ≤ 1` enforced;
      requires `model.value_head is not None`; extends the
      `value_norm='sqrt_n'` incompatibility to reject `'mix'`.
- [x] `_expand`: `'mix'` branch returns `λ·v_head + (1−λ)·v_rollout`.
- [x] `_populate_priors`: mirrored — root `node.v_estimate` is on the same
      scale as backed-up Q.

### H.1.2 C++ backend (`src/am_baseline/search/mcts_cpp/`)
- [x] `mcts.hpp::Config`: added `mix_lambda` field.
- [x] `mcts.cpp::Config::from_python`: `get_or<double>("mix_lambda", ...)`
      with range check.
- [x] `Solver::populate_priors`, `Solver::expand`, `Solver::evaluate_pending`:
      `need_value` true for `value_head|mix`; `'mix'` branches blend
      `convert_value_head_output(...)` with `rollout_remaining_real/_many(...)`.
- [x] `BatchSearch::Impl::collect_request` (both call sites) emits requests
      with `need_value` / `need_rollout` both true under mix.
- [x] `BatchSearch::Impl::apply_result`: counters bumped for both branches;
      v_head / v_rollout blended via the new `convert_v_head` helper lambda.
- [x] `solver.py::CppMCTSSolver.solve_instance`: forwards `rollout_evaluator`
      when `leaf_eval ∈ {'rollout','mix'}`. `mix_lambda` flows through
      `_cfg_dict` via `asdict(self.cfg)` (no manual key).
- [x] Rebuild: `pip install -e .` (cp312 `.pyd` timestamp 2026-05-21 01:22).

### H.1.3 Training-side plumbing
- [x] `train_alphazero.py`: `--leaf_eval` choices extended to include
      `'mix'`; `--mix_lambda` (float, default 0.5) added.
- [x] `coach.py::make_self_play_config`: `mix_lambda` parameter forwarded
      into `MCTSConfig`. Coach call site at iteration loop passes
      `getattr(opts, 'mix_lambda', 0.5)`.
- [x] `val_stage4_mcts.py`: `--leaf_eval` choices extended;
      `--mix_lambda` added; `_build_mcts_config` honors `--match_train`
      by reading `mix_lambda` from sibling `args.json`.

### H.1.4 Modal entrypoint
- [x] `modal_run_train_alphazero.py::run_tsp20_k10_mix_step50(mix_lambda, timestamp)`
      added. Single recipe-delta from B.4 (`run_tsp20_k10_lv0_step50`):
      `leaf_eval=mix`, `lambda_v=1.0`, `--mix_lambda <λ>`. Run-name tag
      embeds λ (e.g. `tsp20_k10_mix0p5_step50_100iter_*`).

### H.1.5 Probe / eval scripts
- [x] `probe_mcts_decomp.py`: `--leaf_eval` choices extended; `--mix_lambda`
      added; forwarded into the constructed `MCTSConfig`.
- [x] New `src/scripts/eval_tsp20_mix_lambda_sweep.py`: Colab T4 entrypoint.
      Sweeps a CLI-supplied λ-grid (default `0.0,0.25,0.5,0.75,1.0`) at
      K=40 val_size=10000 val_seed=42 against a single checkpoint, writes
      one CSV row per λ + a `.npz` of per-instance costs.

---

## §H.2 Smoke tests — **COMPLETE 2026-05-21**

`src/scripts/smoke_mix.py` covers the three correctness invariants.
Random-init AttentionModel (`graph_size=10, embedding_dim=32,
n_encode_layers=2, n_heads=4, value_enabled=True`), one fixed TSP-10
instance, seeded MCTSConfig (`n_simulations=16, c_puct=1.0, ε=0, τ=0,
value_target_norm='none'`). Results from the live run on 2026-05-21:

| Test | a | b | \|Δ\| | Verdict |
|---|---|---|---|---|
| M1 rollout ↔ mix(λ=0)              | 3.7420294285 | 3.7420294285 | 0.000e+00 | OK |
| M2 value_head ↔ mix(λ=1)           | 4.2331485748 | 4.2331485748 | 0.000e+00 | OK |
| M3 Python MCTSSolver ↔ C++ Solver at mix(λ=0.5) | 3.5263400078 | 3.5263400078 | 0.000e+00 | OK |

M1 confirms the value-head contribution is zeroed when λ=0; M2 confirms
the rollout contribution is zeroed when λ=1; M3 confirms the C++ blend
matches Python bit-for-bit at λ=0.5 (the production training value).

### H.2.4 No-NaN / sign-convention sanity — deferred
Not required for correctness given M1–M3 are exact-match; the blend
formula `λ·v_head + (1−λ)·v_rollout` with both inputs ≥ 0 (cost-to-go)
trivially preserves non-negativity. Will re-open if Phase A surfaces
anomalies.

### Repro
```
PYTHONPATH=src python -m scripts.smoke_mix
```

---

## §H.3 Phase A — Colab T4 inference probe — **COMPLETE 2026-05-21**

Target checkpoint: F.6.1.6 winner —
`outputs/tsp_20/f616_400iter_step_decay_20260507T101222_20260507T101229/iter-361_accepted.pt`.

Grid: λ ∈ {0.00, 0.25, 0.50, 0.75, 1.00}, K=40, ε=0, τ=0,
`val_size=10000, val_seed=42, c_puct=0.05, mcts_batch_size=1000`.

### Results — Colab T4 (2026-05-21)

| λ    | val_avg_cost | SE      | wall (s) | fwd_decode | fwd_value | fwd_rollout |
|------|--------------|---------|----------|------------|-----------|-------------|
| 0.00 | 3.82798      | 0.00305 | 211.8    | 11,543,491 | 1,061,940 | 10,481,551  |
| 0.25 | 3.82814      | 0.00305 | 213.4    | 11,435,512 | 1,062,954 | 10,372,558  |
| 0.50 | 3.82893      | 0.00306 | 225.6    | 11,570,770 | 1,086,749 | 10,484,021  |
| 0.75 | 3.82992      | 0.00305 | 234.6    | 11,918,868 | 1,132,041 | 10,786,827  |
| 1.00 | 3.83222      | 0.00306 | 253.8    | 12,535,756 | 1,207,007 | 11,328,749  |

Artifacts:
- CSV: `_progress/eval_logs/tsp20_f616_mix_lambda_sweep_K40.csv`
- NPZ (per-instance costs, for paired-t): `_progress/eval_logs/tsp20_f616_mix_lambda_sweep_K40.npz`

### Interpretation

**Curve is monotone increasing in λ → λ=0 wins.** Matches the plan's
"monotone curve" branch of the decision rule: mix offers no inference
benefit on F.6.1.6's biased value head. Proceed to Phase B *per the plan*
because mix at training time may yield a less-biased head.

**The bias of F.6.1.6's value head at K=40 is much milder than §C.3
suggested.** Total inference spread λ=0→λ=1 is only **0.00424** here;
§C.3's same-checkpoint spread (3.868 − 3.834) was **0.034** — 8× larger.
Possible reasons:

- **Instance-set effect.** §C.3 used `val_size=2000, val_seed=42`,
  this used `val_size=10000, val_seed=42`. If the dataset generator's
  RNG consumption depends on `val_size`, the two subsets are different
  populations, and the value head may behave very differently on the
  2000-instance subset.
- **`mcts_batch_size` effect.** §C.3 used the script default
  `mcts_batch_size=64`; here we used `1000`. Larger batches may
  reduce per-leaf eval ordering effects that interact with the
  value-head's bias surface.
- **§C.3 was a noisy roll.** 2000-instance SE on F.6.1.6 is ~0.007;
  the 3.868 reading is ~5σ above this measurement's λ=1 value
  (3.832), which would be improbable as pure noise but possible
  given the population differences above.

This **weakens** the Phase B hypothesis: with so little vh-bias to
"undo," a mix-trained head needs to add even less noise reduction to win.
Phase B is still worth running — the plan called for it and self-play
dynamics under mix may differ qualitatively from inference — but expected
gains are smaller. **Single λ=0.5 run is the right scope (not a sweep).**

### Wall / compute cost

- Wall scales ~linearly with λ (211.8 s @ λ=0 → 253.8 s @ λ=1, +20%),
  consistent with one extra MLP call per leaf at λ>0. At λ=0.5 the
  overhead is ~7%.
- Total Phase A wall: ~19.5 min for all 5 λ values on Colab T4.
  Plan budgeted 60–90 min; came in 3-5× faster than expected.
  (The K=10 Phase B will be much shorter per leaf.)

### Decision

- **Phase B**: single Modal A10 run at **λ=0.5** (the canonical
  AGFan/Lee value, what `run_tsp20_k10_mix_step50` defaults to,
  and what the C++ smoke test validated bit-for-bit). Skip
  λ∈{0.25, 0.75} — flat inference curve doesn't justify the extra ~$10.

---

## §H.4 Phase B — Colab T4 training at λ=0.5 — **COMPLETE 2026-05-22**

Per §H.3 decision, ran a single λ=0.5 from-scratch training on Colab T4
(not Modal A10 as originally planned — Phase A was so fast on T4 that
the T4 path looked attractive for Phase B too, and verified ~1.5× of A10
wall, well inside the 12 h Colab Pro session envelope).

Recipe (matches `modal_run_train_alphazero.run_tsp20_k10_mix_step50` at λ=0.5):
```
graph_size=20, n_iterations=100, M_instances=1000,
n_simulations_train=10, train_steps_per_iter=200,
buffer_capacity=5000, batch_size=512, gate_every=1, gate_mode=ttest,
temperature_schedule=step10, val_size=10000, val_seed=42,
leaf_eval=mix, mix_lambda=0.5, lambda_v=1.0,
max_grad_norm=1.0, value_target_norm=none,
lr_model=5e-4, lr_decay=0.2, lr_decay_step_size=50,
weight_decay=0.0, dirichlet_epsilon=0.25, dirichlet_alpha_factor=10.0,
mcts_batch_size=1000
```

Notebook: [`notebooks/colab_phaseB_mix_train.ipynb`](../notebooks/colab_phaseB_mix_train.ipynb).
Output dir on Drive: `MyDrive/AM_AlphaGoZero/outputs/tsp_20/tsp20_k10_mix0p5_step50_100iter_20260522T055640_20260522T055644/`.

### Result

| λ | run_name | wandb | iter best | val best | gate accept rate | wall |
|---|---|---|---|---|---|---|
| 0.5 | tsp20_k10_mix0p5_step50_100iter_20260522T055640 | [h2ojy9qp](https://wandb.ai/lejun/am-alphagozero/runs/h2ojy9qp) | 99 | **3.87083** | 44/100 = 44 % | 50.3 min T4 (34.5 min MCTS + 15.8 min train) |

### Verdict — NEGATIVE (mix loses)

| Recipe | Best val @ iter 100 | Δ vs §B.4 | σ-dist (SE=0.003) |
|---|---|---|---|
| §B.4 lv0 K=10 step50 (rollout-only) | **3.8576** | — | — |
| **Phase B mix(λ=0.5)** | **3.87083** | **+0.01323** | **+4.4σ worse** |
| §B.3 F.6.1.6 ceiling (lv1 K=40 step-decay) | 3.8578 | +0.013 | — |

The trajectory plateaued at ~3.87 over the last 10 iters (iter 90→99 range:
3.876 → 3.871), with no signs of breaking through 3.86 even after the
lr step at iter 50 (5e-4 → 1e-4). Best val = final val at iter 99.

### What the result attributes — and what it doesn't

Phase B differs from §B.4 by **two simultaneous knob changes**:
1. `lambda_v`: 0.0 → 1.0 (value head goes from untrained-and-unused to
   trained-with-MSE-on-z each iter).
2. `leaf_eval`: rollout → mix (MCTS Q-backup uses
   `λ · v_head + (1−λ) · v_rollout` instead of just `v_rollout`).

So the clean apples-to-apples conclusion is: **"recipe ablation
{lv0+rollout} → {lv1+mix(0.5)} costs 0.013 on TSP-20 at this budget."**
We cannot yet attribute the loss to either knob in isolation. Two
plausible mechanisms:
- **vh-bias-poisoned visits** (mix path is the culprit): even a half-dose
  of a biased value head at each leaf shifts the visit distribution
  enough that the policy training target is degraded. §C.3 documented
  this bias on F.6.1.6; the question is whether it persists when the
  head is co-trained.
- **value-loss interference** (lambda_v path is the culprit): the value
  head's MSE-on-z loss may pull encoder gradients in a direction that
  hurts the policy head, especially in a small-K (K=10) regime where
  the policy loss already has high variance. F.6.1 sibling runs
  (lv1+value_head) sat at ~3.86 — close to Phase B's 3.87, suggesting
  lambda_v=1 alone may explain most of the deficit.

### Sample-efficiency check
- [x] Best val at iter 100 (3.87083) vs §B.4 (3.8576) — Phase B loses by 4.4σ.
- [x] Wall: 50.3 min T4 vs 33 min A10 (§B.4) — T4 is ~1.5× slower as expected;
      mix overhead of ~7% predicted in Phase A holds (mcts wall 34.5 min on T4
      vs estimated §B.4 ~22 min on T4 if scaled = ~57% of total wall, similar).

### Attribution review — 2026-07-02 (multi-agent code + record audit)

Full review (5 readers + 5 adversarial judges + critic) over mcts.py,
mcts.cpp/BatchSearch, coach/trainer plumbing, and the experimental record.

1. **No implementation bug.** Blend arithmetic, units, sign, and the
   `value_target_norm='none'` round-trip (trainer trains on raw
   `z·bl_val`; search divides `v_raw/bl_val` with the SAME frozen
   per-instance bl_val) verified line-for-line identical across Python
   and all five C++ mix sites. Config plumbing verified via H.4's
   `args.json`. The negative result is not an artifact of the mix code.
2. **The "4.4σ" overstates certainty.** σ there is the 10k-instance val
   SE (0.003) only; each arm is n=1 training seed against the record's
   own ~0.01–0.015 seed-noise floor (`stage5_progress.md` §B.1 note).
   Direction is corroborated (H.4's 3.871 sits inside the lv1 cluster
   3.858–3.878; every lv0 run is 3.848–3.861), but treat the magnitude
   as ~1σ of replication noise, not 4.4σ.
3. **λᵥ-interference is the best-supported dominant channel.** §D.2 is
   the only single-knob isolation in the record: lv1 vs lv0 at
   `leaf_eval=rollout` (K=40, 50 iter) costs +0.040 best / +0.053 final,
   with `value_grad_norm_shared` 0.644 vs 0.000. Phase A prices the mix
   *inference* channel at λ=0.5 at only **+0.00095**. The missing
   decisive control: **§B.4 verbatim + λᵥ=1 + leaf_eval=rollout**
   (K=10 step50 100-iter, ~$5 / ~40 min A10). ≈3.87 ⇒ mix exonerated,
   deficit is all λᵥ; ≈3.858 ⇒ the blend itself is convicted.
4. **The variance-reduction premise was false a priori.** The leaf
   rollout is a deterministic argmax completion (`mcts.py:558`) — zero
   sampling variance at a fixed leaf; §C.2 measured var(z|s)≈0 for
   steps ≥2. Mix can only trade *nonexistent noise* for value-head bias.
   See §H.7 for what the head's error actually looks like where MCTS
   queries it.
5. Residual caveats: C.3's 0.034 vh-vs-rollout spread vs Phase A's
   0.004 on the same checkpoint (8×) is still unreconciled (population
   2000@bs64 vs 10000@bs1000); Phase A CSV/NPZ artifacts were never
   committed; smoke M3 covered the sequential C++ solver, not
   BatchSearch (functionally exonerated by Phase A's smooth curve, not
   by a parity test). Also noted for a future fix: all C++
   BatchInstances share one `mt19937_64(seed)` stream, so root Dirichlet
   draws are not i.i.d. across the 1000 instances of an iteration
   (affects B.4 and H.4 equally — not a confound, but worth knowing).

**§H.4 CLOSED as answered (2026-07-02).** Verdict stands at the recipe
level ("{lv1 + mix(0.5)} loses to {lv0 + rollout} at this budget"); the
mechanism is understood via §H.7 (off-policy value-head failure + no
variance to reduce), so no Phase C and no further mix runs. The
λᵥ-vs-mix split (control run in item 3) is an optional refinement, kept
in plan §12.3 — not required for this closure.

---

## §H.5 Phase C — 1000-instance canonical eval — **SUPERSEDED by §H.7 (2026-07-02)**

Not run. §H.7's ranking + off-policy diagnosis closed mix as negative — a
1000-instance canonical eval of a mix leaf-eval that cannot rank actions
adds nothing. The scaffolding below is retained for reference only.

[`src/scripts/eval_tsp20_full_comparison.py`](../src/scripts/eval_tsp20_full_comparison.py)
(1000 instances, seed=20260430, Gurobi-optimal).

### Per-checkpoint comparison (to fill in)

| checkpoint | greedy | MCTS K=10 mix | MCTS K=40 mix | MCTS K=100 mix | gap-vs-Gurobi (K=40) | n_optimal/1000 |
|---|---|---|---|---|---|---|
| F.6.1.6 best | — | — | — | — | — | — |
| lv0 iter-199 | — | — | — | — | — | — |
| **mix λ\*** | — | — | — | — | — | — |

Paired t-test (1000 instances): mix vs lv0 iter-199 at K=40 rollout (lv0's
trained leaf eval) — report t, p, win/tie/loss split.

Success threshold: greedy ≤ 3.85 AND MCTS K=40 ≤ 3.835.

---

## §H.6 Verdict + follow-ups — **SUPERSEDED by §H.7.3 (2026-07-02)**

Verdict delivered in §H.7.3: mix loses; the value head is calibrated
on-policy but underestimates untaken siblings by ~0.6 raw → it cannot rank
actions where MCTS reads it → the bottleneck is the off-policy training
distribution, not representation. The pre-§H.7 decision tree below is kept
for the record.

Original expected decisions:

- **If mix wins**: log as the new TSP-20 frontier; consider TSP-50 port
  (§E sibling). Open question: per-step adaptive λ.
- **If mix ties λ=0 (pure rollout)**: closes the AGFan/Lee question for
  this regime; supports the §D verdict that the value head doesn't
  contribute useful signal at TSP-20 under the current value target.
- **If mix loses**: investigate whether F.6.1.6-style vh bias persists
  under mix self-play (value-head calibration probe at iter 50 / iter 100).

---

## §H.7 Separate value-trunk + ranking probes — **COMPLETE 2026-07-02 (NEGATIVE, reframed)**

Plan: [`_plans/stage5_mix_leafeval_plan.md`](../_plans/stage5_mix_leafeval_plan.md) §11.
Phase 0/0b artifacts (all next to the §H.4 checkpoint in
`outputs/tsp_20/tsp20_k10_mix0p5_step50_100iter_20260522T055640_20260522T055644/`):
`iter-99_vtrunk.pt` (shared frozen encoder), `iter-99_vtrunk_ownenc.pt`
(own value encoder), `iter-99_vtrunk_ownenc_ho.pt` (own encoder, trained
only on `inst%5!=0`). Supervised trainer:
`src/scripts/train_value_trunk_supervised.py`; probes:
`src/scripts/probe_value_aleatoric.py` (trunk branch),
`src/scripts/probe_action_ranking.py` (new).

### H.7.1 Action-ranking probe (2026-07-01/02, reconstructed + re-run 2026-07-02)

`probe_action_ranking.py`: 50 buffer decision nodes (seed=1234,
min_actions=4, mean 12.2 legal actions), ground truth g_a = E[cost-to-go |
child s′_a] from sampled-policy rollouts (SE 0.10 ≪ action separation 1.49);
`rank_gt_*.npz` caches in the H.4 output dir make re-runs model-only.
The **greedy rollout** row is computed cache-vs-cache (greedy-gt vs
sampled-gt, identical slots) — zero model calls.

| estimator | Spearman action-value (all / holdout) | Spearman v-vs-g intrinsic | top-1 | decision regret raw |
|---|---|---|---|---|
| **greedy rollout (leaf eval)** | **0.891 / 0.879** | — | **0.74 / 0.74** | **0.047 / 0.056** |
| §H.4 glimpse head (iter-99) | 0.674 / 0.669 | 0.185 / 0.081 | 0.66 / 0.60 | 0.168 / 0.199 |
| vtrunk (shared enc) | 0.652 / 0.621 | 0.007 / −0.046 | 0.48 / 0.56 | 0.311 / 0.307 |
| vtrunk (own enc) | 0.582 / 0.578 | −0.155 / −0.193 | 0.46 / 0.52 | 0.287 / 0.375 |
| vtrunk (own enc + holdout) | 0.530 / 0.549 | −0.175 / −0.172 | 0.54 / 0.64 | 0.227 / 0.173 |

(random-pick regret ≈ 0.75 for scale.)

- **Every value head has ≈ zero intrinsic sibling-ranking signal**
  (Spearman v-vs-g ∈ [−0.19, +0.19]); their decision-relevant Spearman
  (~0.53–0.67) comes entirely from the known edge-cost term.
- **The greedy rollout ranks siblings 3.5–7× better by regret** — and is
  identical on train vs held-out instances (deterministic function,
  nothing to memorize).
- **The ValueTrunk does NOT fix ranking — it is worse than the glimpse
  head.** Richer/own representation lowered on-policy RMS but bought no
  within-node discrimination.

### H.7.2 Off-policy calibration probe (2026-07-02) — the root cause

For the same 50 nodes, split children into the action self-play actually
TOOK (in the head's training distribution) vs the UNTAKEN siblings
(where MCTS needs discrimination). §H.4 glimpse head, raw units, vs
greedy-completion ground truth (the same estimator mix blends with):

| children | signed (v − g) | \|v − g\| |
|---|---|---|
| taken (on-policy), n=50 | **−0.012** | **0.0825** |
| untaken (off-policy), n=558 | **−0.598** | **0.620** |

(vs sampled-policy gt: −0.32/0.33 taken, −1.03/1.03 untaken; the
sampled-vs-greedy gt definition gap is +0.31/+0.44 — real but secondary.)

**The own-encoder trunk makes it worse — the dissociation is causal.** Same
probe, own-encoder vtrunk (the *lowest* on-policy error, |v−g|=0.040 on-policy):
off-policy signed bias **−0.838** vs the glimpse head's −0.598. Better
on-policy calibration buys *more* off-policy overconfidence, which is exactly
why its sibling-ranking Spearman is the worst of all heads (−0.12). So the
RMS-bias↔ranking dissociation is mechanistic, not incidental: minimizing MSE
on an on-policy buffer sharpens the head *toward* the trajectory and *away*
from the untaken siblings MCTS must discriminate. (Independently re-verified
2026-07-02.)

**Diagnosis.** The head is *beautifully calibrated on-policy* — the
documented "0.082 raw RMS" anchors (§C.2's 0.074, plan §11's 0.082) are
on-policy numbers. Off the trajectory it systematically **underestimates
cost-to-go by ~0.6 raw (~15% of tour length)** because buffer states are
only ever on-policy: the head effectively predicts "the cost an
on-policy state at this step would have." In PUCT this makes bad
branches look optimistic → visits stick to them → π targets degrade.
This one mechanism explains §C.3 (vh leaf eval ≤ greedy at any K), the
mild-but-real mix penalty (Phase A monotone, §H.4), and why the trunk
(same on-policy buffer) cannot help: **the bottleneck is the training
distribution + target, not the representation.** The plan §11 Phase-0
gate (on-policy RMS < 0.04) measures the wrong quantity — gate on
ranking regret / within-node Spearman instead.

### H.7.3 Verdict + implications — **§H CLOSED 2026-07-02**

- §H.7 CLOSED as negative for the representation hypothesis; positive
  as a diagnosis. The AGZ-style value head on TSP is not "biased" so
  much as **never trained where search reads it**.
- If a value head is pursued further, the promising lever is
  **off-policy value targets**: harvest counterfactual children (or
  within-tree leaf states) labeled with their greedy-rollout cost-to-go
  — the search already computes these for free under
  `leaf_eval=rollout` self-play — and gate on `probe_action_ranking`
  regret vs the rollout's 0.047, not on RMS.
- The rollout's real cost is **compute, not variance** (deterministic;
  §C.2 var(z|s)≈0 at steps ≥2). Quantified from Phase A (λ=0 row):
  rollouts are **10.48M / 11.54M = 91% of decode forwards** at TSP-20
  K=40, so a head-only search buys **~11× sims at matched wall**,
  scaling ≈ 1+N/2 → **~26× at TSP-50, ~51× at TSP-100**. Unredeemable
  today: §C.3's vh K=200 (5× sims) still lost to rollout K=40 — extra
  sims don't convert when the leaf eval can't rank siblings.
- **Value-head value proposition formally reframed as a compute play**
  — standing decisions in plan §12: (1) TSP-20 closed for value-head
  experiments, frontier stays `leaf_eval=rollout, λᵥ=0`; (2) the head
  re-enters only at TSP-100+ after off-policy targets fix ranking
  (gate: decision regret ≈ rollout's 0.047); (3) the λᵥ-vs-mix control
  run is optional, not a closure requirement.

Repro:
```
PYTHONPATH=src python -m scripts.probe_action_ranking \
  --ckpt outputs/tsp_20/tsp20_k10_mix0p5_step50_100iter_20260522T055640_20260522T055644/iter-99.pt \
  --buffer .../buffer.pt --which best --num_nodes 50 --rollouts_per_child 16 \
  --min_actions 4 --seed 1234 --gt_cache .../rank_gt_56345b29b6.npz   # all-split sampled gt
# holdout split: --holdout_k 5 --inst_split eval --gt_cache .../rank_gt_1b68dbd028.npz
# trunk variants: --value_head_type separate_trunk [--value_own_encoder] --ckpt .../iter-99_vtrunk*.pt
```

---

## Cross-references

- Plan: [`_plans/stage5_mix_leafeval_plan.md`](../_plans/stage5_mix_leafeval_plan.md).
- Algorithmic spec: [`_plans/stage4_algorithm_spec.md`](../_plans/stage4_algorithm_spec.md) §4.1.5 (AGFan/Lee row).
- Stage 5 antecedents in [`_progress/stage5_progress.md`](stage5_progress.md):
  §B.4 (K=10 step50 sibling recipe), §C.3 (vh leaf-eval bias),
  §D (lv0 ablation), §D.4 (current TSP-20 frontier).
- Memories: `project_alphagozero_value_head_leaf_eval_bias.md`,
  `feedback_iterate_on_results.md`.
