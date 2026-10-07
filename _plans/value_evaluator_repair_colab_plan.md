# Frozen-policy value evaluator repair — Colab experiment

Date: 2026-09-26

## Objective and scope

Determine whether supplying missing state information improves the value head's
held-out sibling decisions and the cost/time tradeoff of search. Freeze the
checkpoint policy, encoder, decoder, and normalization buffers throughout.
This is an evaluator experiment, not another AlphaZero training run. The
one-remaining-city alias is established; its contribution at longer horizons is
an empirical question. Do not revise proposal.md.

## Matched protocol

1. Load the existing F.6.1.6 checkpoint with explicit original output units.
   Record checkpoint/source hashes, architecture, seeds, and runtime versions.
2. Generate disjoint train, validation, sibling-test, timing-calibration, and
   search-test instances. Generate greedy and temperature-3 prefix states;
   enumerate all children of selected parents. Cache frozen greedy completion
   targets, including the closing edge, in raw cost units.
3. Refit a glimpse-only MLP and an enriched-state MLP on identical examples,
   MSE objective, optimizer settings, minibatch orders, and budgets. Include a
   parameter-matched wider glimpse-only MLP. Train three seeds by default.
   Retain the existing checkpoint head as a reference, not a matched control.
4. On held-out instances, report sibling regret, optimal-child picks, harmful
   overrides and repairs of the prior, and within-parent ranking by horizon.
   Use exact Held–Karp completions only as evaluation oracles, never labels.
   Explicitly exclude uncertified horizons when the exact-oracle cap is exceeded.
5. Evaluate equal-K search, plus K calibrated on separate timing instances to
   approximate a common wall budget. Include greedy, original/refitted/repaired
   heads, a geometric MST leaf evaluator, and greedy-rollout leaves. Report
   actual timings and forward counts; no GPU/C++ performance generalization.
6. Make the notebook self-contained with an embedded, hash-checked source
   snapshot. Persist inputs, head/optimizer checkpoints, raw metrics, and plots
   to Drive. Resume completed stages and interrupted training without mixing
   different configurations. Provide small CPU settings and correctness checks.

## Validation and deliverables

- CPU unit/integration checks: exact oracle vs brute force, target accounting,
  endpoint alias removed from inputs, policy parameters/buffers unchanged,
  checkpoint units, and custom-search parity with the existing Python solver.
- End-to-end tiny run through data, training, exact sibling probing, and search;
  notebook schema/code-cell validation and embedded-source consistency check.
- Notebook, reusable experiment runner, README instructions, and matching
  progress record. Report local checks separately from unrun Colab/GPU results.

## Completion (2026-09-26)

Notebook and runner delivered. Six CPU tests passed; all 12 notebook code cells
executed from the embedded snapshot with the real F.6.1.6 checkpoint on a tiny
TSP-20 run. Full Colab/GPU experiment remains for the user to launch. See the
[progress record](../_progress/value_evaluator_repair_colab_progress.md) for exact
settings, evidence, output meanings, and limitations.

## Revision 2 (2026-09-29) — geometry-residual head and the TSP-50 decision run

Adopted as Stage 5 §I Step 1 (see `_plans/stage5_plan.md` §I and
`_progress/stage5_progress.md` §I.4). Two changes to the delivered protocol:

1. **Fourth matched head, `repaired_geo`.** The `repaired` head still has to infer
   tour geometry from a centroid and two endpoints. `repaired_geo` adds 14 explicit
   geometry features of the remaining sub-problem (MST of the unvisited set plus the
   nearest links from the current and start cities, the three nearest distances from
   each endpoint, current-to-start distance, coordinate spread and bounding box, mean
   distances) and predicts a residual over the MST bound, so the existing `mst`
   control is its zero-MLP baseline. Same data, targets, optimizer, minibatch order
   and epoch budget as the other refits; about 1.8K more parameters than
   `original_wide`. Features are computed identically at training and search time
   by one numba kernel, verified against the `mst` control and against the exact
   one-city tail cost in the pre-fit invariant checks.
2. **Experiment switch.** `EXPERIMENT="tsp20_f616"` (Stage 4 checkpoint, raw-cost
   head; every decision has an exact oracle; pipeline check) or
   `EXPERIMENT="tsp50_stage1"` (Stage 1 TSP-50 checkpoint, `bl` head; exact oracles
   for the last 20 cities, search comparison for everything). TSP-50 is the decision
   run because a value head only pays off where rollouts are expensive, and that is
   where the current head was measured near random at early steps.

**Decision rule for Step 1.** On `tsp50_stage1`/`main`, all three head seeds:
- PASS if `repaired_geo` sibling regret on the exact horizons is within ~1.5× the
  rollout's and its calibrated-time search matches or beats `rollout`, with
  `repaired_geo minus mst` clearly negative (the learned residual adds to the bound).
- KILL if `repaired_geo` ≈ `repaired` ≈ `original_wide`: the input is not the
  bottleneck at TSP-50 and the head cannot be repaired cheaply; the teacher stays
  rollout-only and the value-head line on TSP stops.
- Anything in between: the head is usable only at short horizons; report it as such.
