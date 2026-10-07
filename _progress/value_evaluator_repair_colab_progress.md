# Frozen-policy value evaluator repair — implementation progress

Date: 2026-09-26

Plan: [`value_evaluator_repair_colab_plan.md`](../_plans/value_evaluator_repair_colab_plan.md)

## Status

Notebook and reusable runner complete; main Colab/GPU experiment pending.
The experiment uses a separate evaluator adapter around the Python reference
search. This change does not modify the production model/decoder/search or
proposal.md. Other in-progress repository edits are preserved.

## Delivered

- [`colab_value_evaluator_repair.ipynb`](../notebooks/colab_value_evaluator_repair.ipynb):
  self-contained notebook with 20 embedded Python source files, source hash
  verification, Drive persistence, main/pilot/CPU presets, charts, and ZIP export.
  No GitHub push/clone, C++ extension, or W&B service is required.
- [`value_repair.py`](../src/am_baseline/experiments/value_repair.py): shared
  data generation, head fitting/resume, exact sibling probing, search adapter,
  paired comparisons, and provenance/invariant checks.
- [`run_value_repair.py`](../src/scripts/run_value_repair.py): JSON-config CLI.
- [`build_value_repair_notebook.py`](../src/scripts/build_value_repair_notebook.py):
  reproducibly refreshes the embedded source after runner changes.
- [`test_value_repair.py`](../src/scripts/test_value_repair.py): six CPU tests.

## Default main protocol

F.6.1.6 `iter-361_accepted.pt`, `best_model`, explicit original raw output units.
TSP-20: 1,024 train, 128 validation, 100 sibling-test, 8 timing-calibration,
100 search-test graphs; all instance splits use distinct RNG seeds. Three head
seeds (0/1/2), 30 epochs, batch 512, Adam lr 1e-3, weight decay 1e-4, gradient
norm cap 1. Train-only input standardization and parent-balanced MSE. The fixed
greedy-policy targets include the closing edge. Validation MSE selects the head
checkpoint; held-out ranking/search results never select weights.

Training parents cover selected depths from greedy and temperature-3 sampled
prefixes (first half of each tour). Every legal child is enumerated at these
parents, including off-policy actions. Probe states cover all nontrivial depths
under both prefix distributions. The empty root and forced final decision are
excluded from sibling metrics. A parent's known path and action edge are included
in child scores before computing optimal-completion regret.

The three refits share examples, targets, objective, optimization budget, and
seeded minibatch permutations:

| Variant | Features | Parameters (128-dimensional encoder) |
|---|---|---:|
| Original | Glimpse | 16,641 |
| Original wide | Glimpse; wider hidden layer | 66,821 |
| Repaired | Glimpse, first/current embeddings and coordinates, unvisited mean embedding/coordinates, remaining fraction, start flag | 66,817 |

The existing checkpoint head is an additional historical reference, not a
matched refit. Explicit coordinates remove the demonstrated endpoint alias;
the pooled representation is not claimed to be injective for arbitrary sets.
Original-head search computes only its needed glimpse features, so it is not
charged for the repaired head's additional feature construction.

Search uses the same prior/PUCT/reuse logic across all heads and rollout, MST,
and prior-only controls. The prior-only control gives constant total estimates
at nonterminal leaves; terminal exact costs still affect search. Equal-K uses
K=40. Time calibration chooses K on separate graphs, with a bounded candidate
budget up to 4K, to approximate rollout K=40 time. The report retains actual
held-out time ratios; it makes no strict equal-wall-time or production-throughput
claim. CPU/Python search is the default even when head fitting uses a GPU.

## Validation completed

Command: `PYTHONPATH=src .venv/bin/python -m unittest scripts.test_value_repair -v`.
All six tests pass, covering:

- alias and closing-edge invariants; exact DP versus brute force;
- reference-search parity for raw, baseline-normalized, and sqrt-N head units;
- batched child indexing/targets versus independently reconstructed states;
- distinct instance splits and parameter-count matching;
- resumed training after a simulated disconnect produces identical CPU weights;
- manifest mismatches rejected, unchanged policy/buffers, and full tiny-run resume.

The full notebook was executed in a clean Jupyter kernel using only its embedded
source plus the actual F.6.1.6 checkpoint. All **12 code cells completed without
errors**, including data, refits, exact DP, search, tables, plots, and ZIP export.
This was deliberately tiny: **TSP-20, 4 train / 2 validation / 2 probe / 2 search
graphs, 2 epochs, one head seed, CPU**. It is a functionality check, not evidence
for or against the research hypothesis. Both prefix sources gave 36 exact states
each. Prior regrets telescope to greedy minus optimal tour cost with maximum
absolute discrepancy **4.91e-7**. Frozen policy/buffer hashes matched after every
stage. Notebook schema and every Python cell parsed successfully; all embedded
source files matched the repository snapshot. Rendered plots were inspected;
search labels were replaced with a readable shared legend and seed SD bars.

Machine-readable check record:
[`value_repair_colab_validation_20260926.json`](eval_logs/value_repair_colab_validation_20260926.json).
Temporary smoke outputs are referenced there; the persistent record does not
claim those OS-temporary paths will remain available indefinitely.

## Outputs and resume

The run directory includes config and checkpoint/source hashes, runtime session
records, frozen-policy checks, cached training/validation/probe features,
head/optimizer checkpoints, training curves, per-child scores, sibling decisions,
rollout-target errors, per-instance search costs/tours/times, and paired contrasts.
Intervals cluster decisions by instance and are conditional on the head seed.
Notebook tables additionally report means and sample SD across all head seeds.

Data resumes per shard; training resumes per completed epoch using the same
order; probe/search reuse completed outputs. A different configuration or source
requires a new output directory. Avoid comparing publication timings spanning
different runtime sessions. The final ZIP excludes feature caches and weights.

## Environment

The requested conda environment is unavailable on this host (`conda` is not
installed). Local validation uses the repository's existing `.venv` and CPU
PyTorch. Notebook-validation tools were added to that environment; no conda
environment or GPU was created. Colab uses its supplied PyTorch without replacing
the CUDA build. NumPy/Numba, pandas, and matplotlib support the notebook; no
external optimal solver is needed for TSP-20.

## Interpretation boundary

The experiment fixes checkpoint-output units explicitly and tests a richer
value input. It does not establish that either repair cures the self-play
training plateau. Oracle-optimal decisions and frozen-policy rollout targets
are kept distinct. Publication-scale or GPU results remain pending execution.
For N>20, exact sibling metrics omit early horizons above the DP cap instead of
labeling a heuristic completion as exact. The first experiment should remain
TSP-20. Check convergence, all seeds, early horizons, the capacity control, and
the actual quality/time tradeoff before deciding whether to distill or pivot.

## Revision 2 — 2026-09-29 (Claude)

Plan revision: [`value_evaluator_repair_colab_plan.md` §Revision 2](../_plans/value_evaluator_repair_colab_plan.md).

### Changes
- `src/am_baseline/experiments/value_repair.py`: `VARIANTS` gains `repaired_geo`;
  new numba `_geometry_rows` / `geometry_features` (14 features, see docstring);
  `state_features` appends them; `RefitHead` slices columns per variant and adds the
  raw MST-bound column as a residual for `repaired_geo`; bias initialisation on the
  residual; `EvaluatorSolver` feeds full features to both repaired variants;
  `paired_report` emits contrasts for both variants (new `variant` column; extra
  `repaired_geo minus repaired` and `minus mst` rows); `check_invariants` verifies
  the geometry bound equals the `mst` control and the exact one-city tail cost.
- `src/scripts/test_value_repair.py`: 7 tests (new geometry test; capacity test
  extended). `PYTHONPATH=src .venv/bin/python -m unittest scripts.test_value_repair`
  → OK.
- `src/scripts/build_value_repair_notebook.py`: `EXPERIMENT` switch with per-experiment
  checkpoint paths, units, graph size and presets (`main` / `pilot`); TSP-50 guidance;
  contrast display for the new variant. Notebook rebuilt
  (`notebooks/colab_value_evaluator_repair.ipynb`, embedded-source sha
  `afdce0988a12…`).

### Local validation (Mac, CPU, `.venv`)
- Clean-kernel execution of the rebuilt notebook (nbclient) with `PROFILE=cpu_smoke`
  for **both** experiments, using the real F.6.1.6 and Stage 1 TSP-50 checkpoints:
  12/12 code cells, 0 errors each. Record:
  [`eval_logs/value_repair_colab_validation_20260929.json`](eval_logs/value_repair_colab_validation_20260929.json).
- Real-checkpoint runner pilots via `run_value_repair.py --phase all`:
  TSP-20 F.6.1.6 (96 train / 16 probe / 8 search instances, 12 epochs, 1 seed, K=20)
  and TSP-50 Stage 1 (4 / 2 / 2, 1 epoch, K=4; exercises the `bl` path and N=50
  geometry + DP oracle). Both completed all phases.

### Pilot signal (TSP-20, 288 held-out exact decisions per source; NOT a result)
Held-out sibling regret per decision / P(optimal child):

| evaluator | greedy states | τ=3 states | val MSE (raw) |
|---|---|---|---|
| prior | 0.0009 / 0.972 | 0.0280 / 0.767 | — |
| rollout | 0.0003 / 0.993 | 0.0208 / 0.865 | 0 |
| checkpoint head (corrected units) | 0.0199 / 0.837 | 0.0444 / 0.656 | — |
| original refit | 0.0270 / 0.799 | 0.0554 / 0.597 | 0.055 |
| original_wide refit | 0.0236 / 0.792 | 0.0488 / 0.618 | 0.051 |
| repaired refit | 0.0163 / 0.837 | 0.0474 / 0.663 | 0.047 |
| **repaired_geo refit** | **0.0097 / 0.885** | **0.0215 / 0.785** | **0.036** |

With 96 training graphs and one seed, `repaired_geo` already halves the repaired
head's regret and, on off-policy states, matches the rollout's regret and beats the
prior (fix rate 58% vs rollout 79%). TSP-50 tiny run: validation MSE 0.73
(`repaired_geo`) vs 2.27 (`repaired`) vs 4.5 (`original`) after one epoch on 4
graphs — direction only. The Colab `main` runs decide.

### How to run on Colab
1. Upload `notebooks/colab_value_evaluator_repair.ipynb` to Colab; GPU runtime (T4 is enough).
2. Put on Drive (`MyDrive/AM_AlphaGoZero/checkpoints/`):
   `f616_400iter_step_decay/iter-361_accepted.pt` (local:
   `outputs/tsp_20/f616_400iter_step_decay_20260507T101222_20260507T101229/`) and
   `stage1_tsp50_with_value/epoch-99.pt` **plus `args.json`** (local:
   `outputs/tsp_50/stage1_tsp50_with_value_20260424T032357/`).
3. Section 3: `EXPERIMENT="tsp20_f616"`, `PROFILE="main"` (~1 h, mostly CPU search);
   then `EXPERIMENT="tsp50_stage1"`, `PROFILE="main"` (~3 h; ~1 GB feature cache on
   Drive). Re-running unchanged cells resumes; a changed setting needs a new run name.
4. Read: section 6 sibling tables (`repaired_geo` vs `repaired` vs `original_wide`
   vs `rollout` vs `prior`, by horizon bucket, all seeds), section 7 search table
   (`calibrated_time` mode, `time_ratio_to_target`), section 8 contrasts. Download
   the results ZIP (section 9) and drop it under `_progress/eval_logs/` for the record.

## Results — Colab `main` runs, both experiments — **COMPLETE 2026-09-29/30**

Run on Colab (NVIDIA L4, torch 2.11, CPU search on 2 threads) by Lejun; bundles
pulled from Drive on 2026-09-29 into `_progress/eval_logs/value_repair/`
(`*_results.zip` plus the summary CSVs and plots). Manifest source digest
`b96a73f2…` equals the local revision-2 source; all freeze checks passed; invariant
checks (alias, geometry bound = `mst` control, one-city tail exact, DP = brute
force, search parity) passed in both runs.

| | TSP-20 (F.6.1.6 head, raw units) | TSP-50 (Stage 1 head, `bl` units) |
|---|---|---|
| train / val / probe / search graphs | 1024 / 128 / 100 / 100 | 512 / 64 / 64 / 64 |
| exact sibling states per source | 1,800 of 1,800 | 1,216 of 3,072 (≤ 20 cities left) |
| head seeds × epochs | 3 × 30 | 3 × 30 |

### Fitting (validation MSE on raw greedy-completion targets, mean over 3 seeds)

| head | TSP-20 | TSP-50 | train MSE TSP-50 |
|---|---:|---:|---:|
| original (glimpse) | 0.0493 | 0.261 | 0.192 |
| original_wide | 0.0474 | 0.251 | 0.176 |
| repaired (+ endpoints, pooled set) | 0.0357 | 0.170 | 0.104 |
| **repaired_geo** (+ geometry, residual over MST) | **0.0297** | **0.147** | 0.099 |

Curves are flat by epoch 15–20; seed SD ≤ 0.004. Capacity buys almost nothing
(wide vs original); each information addition buys a lot. The TSP-50 train/val gap
of `repaired_geo` (0.099 vs 0.147) says more training graphs would help.

### Held-out sibling decisions (exact oracle; mean over seeds)

TSP-50, greedy states (1,216 exact decisions per seed; prior wrong in 77):

| evaluator | P(optimal child) | regret / decision | harmful override | fix rate when prior wrong |
|---|---:|---:|---:|---:|
| prior | 0.937 | 0.0015 | 0 | 0 |
| rollout | 0.975 | 0.0009 | 0.023 | 0.96 |
| checkpoint head (corrected units) | 0.581 | 0.125 | 0.400 | 0.31 |
| original refit | 0.589 | 0.059 | 0.386 | 0.29 |
| original_wide refit | 0.605 | 0.057 | 0.373 | 0.29 |
| repaired refit | 0.812 | 0.0121 | 0.153 | 0.34 |
| **repaired_geo refit** | **0.899** | **0.0040** | **0.063** | 0.36 |

By horizon (regret, greedy states): `repaired_geo` 0.0015 / 0.0026 / 0.0058 at
1–5 / 6–10 / 11–20 cities left, vs rollout 0.0000 / 0.0002 / 0.0016 and prior
0.0004 / 0.0013 / 0.0020. τ=3 states: `repaired_geo` 0.0088 vs rollout 0.0045,
prior 0.0051, repaired 0.0169, original 0.066. TSP-20 mirrors this at smaller scale
(greedy: geo 0.0054, repaired 0.0079, original 0.0145, rollout 0.0011, prior 0.0016;
τ=3: geo 0.0216 ≈ prior 0.0218, rollout 0.0195).

### Search on held-out graphs (Python reference MCTS, CPU)

TSP-50, 64 graphs, greedy 5.7825:

| evaluator | equal K=40: cost (Δ greedy) | s/inst | calibrated K: cost (Δ greedy) | K | s/inst |
|---|---|---:|---|---:|---:|
| rollout | 5.7294 (−0.0530) | 3.13 | 5.7294 (−0.0530) | 40 | 3.13 |
| **repaired_geo** (3 seeds, SD 0.002/0.0015) | **5.7453 (−0.0371)** | 0.82 | **5.7345 (−0.0480)** | 160 | 2.33 |
| repaired | 5.7518 (−0.0307) | 0.84 | 5.7416 (−0.0408) | 160 | 2.42 |
| checkpoint head | 5.7474 (−0.0351) | 0.75 | 5.7399 (−0.0426) | 160 | 2.29 |
| original_wide | 5.7539 (−0.0286) | 0.74 | 5.7424 (−0.0400) | 160 | 2.22 |
| original | 5.7557 (−0.0268) | 0.74 | 5.7460 (−0.0365) | 160 | 2.25 |
| prior_only (no leaf signal) | 5.7571 (−0.0254) | 0.70 | 5.7481 (−0.0344) | 160 | 2.14 |
| mst (bound only) | 5.7609 (−0.0215) | 1.27 | 5.7526 (−0.0299) | 138 | 3.65 |

Paired contrasts for `repaired_geo` at calibrated time (mean over seeds, per-graph
SE): minus rollout **+0.0050 (0.0030)**; minus repaired −0.0072 (0.0041); minus
original_wide −0.0080 (0.0043); minus prior_only **−0.0136 (0.0041)**; minus mst
**−0.0181 (0.0062)**. The heads were under-budgeted: the calibration cap
(`max_time_K_multiplier=4` → K ≤ 160) left them at 2.1–2.4 s versus the rollout's
3.1 s per graph. Per simulation the geometry head is ~5× cheaper than a rollout here.

TSP-20 (100 graphs, greedy 3.8219): rollout 3.8003; at calibrated time every
evaluator including `prior_only` lands at 3.8013–3.8018 (geo − rollout +0.0010 ±
0.0009; geo − prior_only −0.0002). TSP-20 search is saturated by exact terminal
costs; it cannot discriminate evaluators, as anticipated.

### Verdict against the §Revision 2 decision rule (TSP-50)

- Regret within ~1.5× rollout on exact horizons: **not met** (≈4×: 0.0040 vs 0.0009;
  but 2.7× the prior's, and 3× better than `repaired`, 30× better than the
  checkpoint head).
- Calibrated-time search matches rollout: **not met, close**: +0.0050 ± 0.0030 at
  ~75% of the rollout's measured time, capped by K ≤ 160.
- `repaired_geo` minus `mst` clearly negative: **met** (−0.018, CI excludes 0);
  also minus `prior_only` −0.014 (CI excludes 0).
- Kill condition (geo ≈ repaired ≈ wide): **not triggered**; the three are cleanly
  separated on every metric.

**Reading.** Information, not capacity, was the value head's bottleneck.
Endpoint features halve the sibling regret, explicit geometry cuts it by another
3×, and the repaired head now carries real leaf signal in TSP-50 search: 70% of the
rollout's gain at equal K for a quarter of the time, 91% at three quarters of the
time. It is not yet a drop-in replacement for the rollout (4× its regret, +0.005
tour cost at less time).

### Next iteration proposed on 2026-09-30 — SUPERSEDED by the post-hoc check below

The 2026-09-30 proposal (heuristic-completion features + residual, 2048 training
graphs, `max_time_K_multiplier` 8, same splits, then Step 2 in parallel) was
re-audited on 2026-10-03 against the per-horizon data. Two of its premises do not
hold where it matters; see the corrected next step at the end of the next section.

### Post-hoc check (2026-10-03) — where the geometry gain lives

Re-analysis of `sibling_child_scores.npz` (every child of every probed parent,
all horizons) with
[`src/scripts/value_repair_horizon_analysis.py`](../src/scripts/value_repair_horizon_analysis.py);
CSVs `_progress/eval_logs/value_repair/tsp{20,50}_sibling_vs_rollout_by_horizon.csv`.
The reference is the frozen greedy rollout, i.e. the thing a learned evaluator must
replace, so every horizon is covered (the exact oracle stops at 20 cities left).
Head seeds averaged within a family.

TSP-50, greedy (on-policy) states:

| cities left | within-parent SD of rollout score | P(pick = rollout's pick) prior / repaired / geo | regret vs rollout's pick prior / repaired / geo |
|---|---:|---|---|
| 1–5 | 0.155 | 0.973 / 0.826 / 0.948 | 0.0005 / 0.0134 / 0.0016 |
| 6–10 | 0.262 | 0.922 / 0.807 / 0.900 | 0.0022 / 0.0113 / 0.0035 |
| 11–20 | 0.347 | 0.891 / 0.790 / 0.849 | 0.0045 / 0.0159 / 0.0087 |
| 21–30 | 0.383 | 0.875 / 0.735 / 0.812 | 0.0043 / 0.0200 / 0.0117 |
| 31–40 | 0.386 | 0.870 / 0.718 / 0.781 | 0.0078 / 0.0275 / 0.0155 |
| 41–49 | 0.420 | 0.845 / 0.628 / 0.706 | 0.0098 / 0.0405 / 0.0271 |

Off-policy (τ=3) states, regret vs the rollout's pick at 21–30 / 31–40 / 41–49:
prior 0.044 / 0.071 / 0.117, geo 0.049 / 0.074 / 0.100, repaired 0.057 / 0.081 /
0.111 — head and prior comparable, the head ahead only beyond 30 cities left.
Sibling-centred RMSE of the predicted rollout target (`rollout_target_errors.csv`,
greedy states), geo vs repaired: 0.031 vs 0.059 at 1–5, 0.046 vs 0.062 at 6–10,
0.088 vs 0.103 at 11–20, **0.138 vs 0.147 at 21+**.

Reading:
1. **The geometry gain is confined to short horizons**: −48% within-parent error at
   ≤5 cities left, −6% at 21+. At 21+ cities left, where a TSP-50/100 evaluator has
   to earn its keep, the geometry head is still 2–3× the prior's regret against the
   rollout's pick on on-policy states and only equal to the prior off-policy. "More
   geometry keeps paying" was an extrapolation from the aggregate; the long-horizon
   within-parent error (0.14 for both repaired heads) is a feature limitation.
2. **The train/val gap is not evidence of data limitation.** Every head shows a
   similar gap (original 0.19/0.26, wide 0.18/0.25, repaired 0.10/0.17, geo
   0.10/0.15) and the geometry head's validation curve is flat from epoch 9: it is
   memorisation of the 512 training graphs. More graphs may still help — the Stage 1
   glimpse head trained on 128M instances beats the same architecture refit on 512
   graphs in search (5.7474 vs 5.7557 at K=40) — but that is a different argument.
3. **Standalone sibling regret under-predicts search usefulness**, consistent with
   the §I.3 retraction: the checkpoint head agrees with the rollout's pick on 3.5% of
   parents at 41+ cities left, yet beats `prior_only` in search by 0.010 at K=40 and
   0.008 at calibrated time (about 2 per-graph SE), and is within 0.001–0.004 of the
   geometry head at equal K. Level calibration across subtrees matters to PUCT.
   Search cost at matched wall time stays the only decision metric for a head.
4. **The calibration cap bound.** Per simulation on 2 CPU threads: 0.098 s rollout,
   0.0145 s for every head and for `prior_only` (the head's own cost is invisible),
   so matching the 3.93 s target needs K ≈ 250, above the cap of 160. Any
   time-matched claim needs `max_time_K_multiplier ≥ 8`. The `mst` control costs
   0.029 s per simulation because `mst_remaining` is numpy, not numba (hence K=138);
   cosmetic.
5. **Splits.** Each split has its own seed offset (`split_coordinates`), so a run
   with more training graphs keeps val/probe/calibration/search identical; a run
   with more search graphs gets a new search set (pairing with this run is then
   internal only). The manifest signature includes every protocol field, so any
   changed knob needs a new `output_dir`; data + training + probe cost ~16 min on
   the L4, search ~1 h per 64 graphs on 2 CPU threads.

**Corrected next step (2026-10-03), replacing the 2026-09-30 proposal.**
- Step 2 (student isolation at TSP-50 with the rollout teacher; see
  `_progress/stage5_progress.md` §I.4) is the critical path. The evaluator line is
  only worth more effort if Step 2 passes (cheaper teacher, TSP-100). Step 2 runs
  as a diagnostic under the current proposal: warm-start runs were classified as
  diagnostics on 2026-05-02 (`stage4_progress.md` §F reframing). No TSP-50
  warm-start has ever been run. A claim decision (narrow proposal.md to "improves
  a converged REINFORCE policy" vs port the winning target to the from-scratch
  loop) arises only if Step 2 passes, and needs Lejun's approval then.
- Before any further Colab spend on the evaluator: an offline feature check on the
  Mac (CPU, local Stage 1 TSP-50 checkpoint). Do nearest-neighbour / 2-opt
  completions and the 1-tree bound explain within-parent rollout variation at 21–49
  cities left? Go rule: ≥25% lower long-horizon sibling-centred RMSE than the
  geometry set in a per-bucket linear fit, or regret vs the rollout's pick at 21+ at
  or below the prior's. No-go: park the evaluator line, keep the rollout teacher.
- Optional, config-only: rerun `tsp50_stage1/main` in a new output dir with
  `max_time_K_multiplier: 8` (and 128 search graphs) to settle the time-matched
  comparison for the record. It does not change the next action.
