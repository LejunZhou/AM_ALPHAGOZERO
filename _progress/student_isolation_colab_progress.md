# Step 2 — Student isolation at TSP-50 (Colab) — progress

Plan: [`_plans/student_isolation_colab_plan.md`](../_plans/student_isolation_colab_plan.md).
Stage 5 §I Step 2. Approved by Lejun 2026-10-03 (staged: screen first; Colab).

## Status

**COMPLETE 2026-10-06. Screen outcome: STOP.** Lejun ran `main` on Colab in one
L4 runtime (2026-10-04, 05:38–08:50 UTC, 3 h 12 min). All three targets beat
continued REINFORCE at matched wall time. None kept 25% of the teacher's gain.
Under the rule fixed before the run, Confirm is not run and the TSP claim ends.
Results are archived in `_progress/eval_logs/student_isolation/`: the run's ZIP,
the key result files, and `tsp50_review_20261006.json` (the independent checks
below).

## Results (Colab `main`, NVIDIA L4, 2026-10-04)

**Run integrity.**
- One runtime, with the L4 in every phase (`sessions.jsonl`).
- The frozen Stage 1 was unchanged at every phase boundary.
- C++/Python root-Q parity 9e-8; teacher forcing reproduces the model's
  log-likelihood exactly; Stage 1's test costs recomputed identically.
- The control ran 4,090 s against a 3,967 s budget, 3% in its favour. The budget
  is the teacher on the training split (2,317 s) plus the mean student grid
  (1,650 s).

**Test set: 2,048 graphs, paired per graph.** The teacher (K=40 rollout MCTS)
scores 5.7389 against Stage 1 greedy 5.8023, a gain of 0.0635 (SE 0.0015).

| policy | test cost | Δ vs Stage 1 | Δ vs control [95% CI] | retention [95% CI] | fix / keep | canonical 10K Δ |
|---|---:|---:|---:|---:|---|---:|
| Stage 1 greedy | 5.8023 | 0 | — | 0 | — | 0 (5.8013) |
| teacher | 5.7389 | −0.0635 | — | 1 | — | — |
| visits (lr 3e-5) | 5.7930 | −0.0093 | −0.0060 [−0.0098, −0.0021] | 0.147 [0.114, 0.181] | 0.27 / 0.994 | −0.0102 |
| **gumbel_q** (lr 3e-5) | **5.7892** | **−0.0131** | −0.0097 [−0.0137, −0.0058] | **0.207** [0.169, 0.242] | 0.37 / 0.988 | −0.0145 |
| best_tour (lr 3e-5) | 5.7915 | −0.0108 | −0.0074 [−0.0113, −0.0035] | 0.170 [0.132, 0.205] | 0.32 / 0.991 | −0.0125 |
| control (REINFORCE, 17,501 steps, 9.0M graphs) | 5.7989 | −0.0034 [−0.0075, +0.0007] | 0 | 0.05 | 0.45 / 0.963 | −0.0056 |

**Decision: STOP.**
- Every target passes the first criterion (it beats the control); none reaches
  the floor.
- The best target, `gumbel_q`, misses by 0.0028 of tour cost (the floor needs
  −0.0159).
- 1.2% of 20,000 bootstrap resamples reach 0.25.
- A recomputation from `test_costs.npz` reproduces `decision.json`.

**What the numbers say.**
1. **Distillation is the better use of GPU time at the Stage 1 plateau.** In the
   same ~66 min, `gumbel_q` gains 0.0131. Continued REINFORCE gains 0.0034, and
   its interval includes zero.
2. **The student keeps a fifth of what search finds.**
   - The teacher overrides the Stage 1 argmax at 2.5% of steps (2,544 of the test
     trajectories' steps).
   - Along the teacher's own paths, the best student now makes 37% of those
     overrides. It also changes 1.2% of the other decisions.
   - On its own greedy paths, that nets 21% of the teacher's gain.
3. **The target matters, modestly (one seed).**
   - `gumbel_q` beats `visits` by 0.0038 [−0.0061, −0.0015].
   - `gumbel_q` beats `best_tour` by 0.0023 [−0.0045, −0.0002].
   - `best_tour` vs `visits` is not significant.
4. **The students were not under-trained.**
   - At the chosen learning rate, validation is flat over the second half of
     training: slope within ±0.0001 per epoch, against noise SD 0.0012.
   - Every target chose the middle learning rate, 3e-5.
   - More epochs on these graphs would not close the gap. More teacher data and
     on-policy relabelling were not tested.
5. **`fix_rate` alone is not absorption.**
   - The control has the highest fix rate (0.45) and the lowest keep rate
     (0.963). It changes more decisions of every kind, and its test gain is not
     significant.
   - Splitting graphs by "teacher improved / teacher tied" is dominated by the
     same effect. The control gains 0.018 where the teacher improved and loses
     0.043 where it tied. Those splits are not used as evidence.
6. **Timing.**
   - The teacher took 0.071 s per graph in the Colab runtime, 3.4× the Mac CPU
     (0.021). This lengthened the control's budget, which works against the
     students; they won anyway.
   - Students took 9 × ~550 s, and the whole run took 3 h 12 min (estimate:
     2–3 h).

**Corrections made while reviewing.**
- **Mislabelled key.** `decision.json`'s `teacher_minus_stage1` held Stage 1 minus
  teacher, i.e. the teacher's gain (+0.0635). The key is now `teacher_gain`. The
  notebook already printed it correctly as "Teacher gain".
- **Misleading STOP text.** The generated INTERPRETATION.md said STOP means "did
  not beat more REINFORCE". In this run, STOP came from the floor.
  - The report now opens with the outcome and which criterion each target met.
  - Re-running the fixed report on the downloaded files reproduces every decision
    value.
  - A test now covers the "beats the control, below the floor" case; 11 tests
    pass.
  - The notebook was rebuilt (snapshot `d7c0b34e2801`) and re-executed clean with
    `cpu_smoke`.
  - The Drive copy of INTERPRETATION.md predates the fix.
- **Canonical-set description.** The plan called the canonical set the one behind
  Stage 1's 5.7999. In fact:
  - It is the self-play loop's validation draw (`train_alphazero.py --val_seed 42`).
  - Stage 1's 5.7999 came from its own training draw: seed 1234, drawn after model
    and baseline init.
  - Stage 1 scores 5.8013 on the seed-42 set. The 0.0014 difference is within
    set-to-set noise (SE 0.0041).
  - Earlier TSP-50 comparisons of the loop against Stage 1 mixed the two sets.
    Their gaps were ≥ 0.13, so no conclusion changes.

**Decided 2026-10-07:** Lejun approved closing TSP and moving to stochastic PCTSP (proposal revised; see `_plans/stage6_spctsp_plan.md`).

**Next step: Lejun's decision (asked 2026-10-06).**
- Per the rule, no Confirm run. The TSP claim ends for both the from-scratch and
  the warm-start framing.
- Recommended: close the TSP chapter as a negative/partial result. Then move to
  stochastic PCTSP, the problem already agreed as next. Start with:
  - the sibling-ranking probe;
  - an early one-round absorption check of this design (new from this run),
    because the student side may limit any problem.
- That changes the proposal's scope, so it needs Lejun's approval before
  proposal.md is edited.

**Repository note (2026-10-06, at push time).** This Mac's checkout was 3 commits behind
`origin/main`: Stage 5 §H.7 / §V0 / §V1 (2026-07-04, made on another machine), recorded in
[`stage5_offpolicy_value_progress.md`](stage5_offpolicy_value_progress.md) and
[`stage5_mix_leafeval_progress.md`](stage5_mix_leafeval_progress.md). §V0 found the value
head fails off-policy and reached rollout-level sibling ranking on TSP-20 by distilling
rollouts on counterfactual children. The September audit, Step 1 and Step 2 were done
without that work. The Step 2 result does not depend on it (rollout teacher, no value head),
but the evaluator-side narrative (§I, Step 1) should be reconciled with §V0/§V1 before the
proposal revision.

## Delivered

| File | Role |
|---|---|
| `notebooks/colab_student_isolation.ipynb` | Self-contained notebook (embedded hash-checked source snapshot incl. C++; compiles the search in the runtime). The `main` run used snapshot `26188b0f9bfc`; rebuilt 2026-10-06 as `d7c0b34e2801` (report labels only) |
| `src/scripts/build_student_isolation_notebook.py` | Regenerates the notebook from the sources |
| `src/am_baseline/experiments/student_isolation.py` | Runner: phases check, teacher, students, control, evaluate, report |
| `src/scripts/run_student_isolation.py` | CLI (profiles or JSON config) |
| `src/scripts/watch_student_isolation.py` | Polls a Drive folder shared by link, reports the running phase, downloads results when `decision.json` appears (stdlib only; tested on the Step 1 folder and a simulated Step 2 timeline) |
| `src/scripts/test_student_isolation.py` | 11 CPU unit/integration tests (incl. a Screen + Confirm report in one folder) |
| C++ `return_root_q` (mcts.cpp/hpp, solver.py, mcts.py) | Opt-in export of root child Q values and the root estimate, both C++ paths and the Python reference; default off, wire format unchanged |

## Decisions made while building (with evidence)

1. **One round of distillation from a fixed teacher**, not a 40-iteration loop, so
   all targets train on identical data and the comparison isolates the target. The
   decision rule changed accordingly from "keep half the teacher's gain" to "beat
   continued REINFORCE at matched wall time AND keep ≥ 25% of the teacher's gain".
2. **Gumbel completed-Q target adapted.** Measured on 128 real Stage 1 TSP-50 graphs
   (K=40, c_puct 0.05): the search visits exactly one root child at 78% of steps;
   that child's averaged Q differs from the root's greedy-rollout value by ~1e-3 in
   either direction (root higher 51%, lower 37%, equal 13%). With the faithful mctx
   mixed value, the min-max rescale turned that into a 5–9 nat swing toward every
   unvisited move: target entropy 0.69 vs prior 0.06, argmax on the teacher's move
   only 66% of steps. Training variant: unvisited moves valued at the visited
   children's prior-weighted mean Q; spreads ≤ 1e-6 are ties (float32 Q); computed in
   float64. Result: equal to the prior (L1 < 5e-7) where one child was visited; at the
   21% of steps with ≥ 2 visited children (all correction steps are among them) it
   puts 98% of its mass on the teacher's move vs 77% for `visits`. Both forms are
   unit-tested against a row-by-row transcription.
3. **Teacher is cheap.** 0.03 s per graph at K=40 on this Mac's CPU (narrow search on
   a sharp prior, NN cache hits), against ~0.25 s/graph for the exploratory from-scratch
   searches of May. Training set raised from 16,384 to 32,768 graphs.
4. **REINFORCE resume bug avoided (not fixed in train.py).** `train.py --resume` wraps
   the rollout baseline in `WarmupBaseline`, whose `epoch_callback` only sets α while
   `epoch < n_epochs`; resuming at epoch 99 leaves α = 0 and trains with the
   exponential baseline. The control builds `RolloutBaseline` directly and restores its
   saved state (the checkpoint's baseline model is from epoch 94).
5. **Selection on validation only.** Each student run starts its best-checkpoint
   tracking at the Stage 1 weights, so a target that never improves validation
   reports Stage 1 itself (retention 0) rather than a degraded model.

## Target statistics on real teacher data (64 Stage 1 TSP-50 graphs, K=40)

| | value |
|---|---|
| Teacher gain over Stage 1 greedy | +0.063 (teacher better on 75% of graphs) |
| Steps where the teacher overrides the Stage 1 argmax | 2.4% |
| Root children visited per step | mean 1.25, median 1, max 7 |
| Entropy prior / visits / gumbel_q, 1-child steps | 0.0008 / 0.0000 / 0.0008 |
| Entropy prior / visits / gumbel_q, ≥2-child steps | 0.29 / 0.19 / 0.008 |
| Mass on the teacher's move at correction steps: prior / visits / gumbel_q | 0.23 / 0.77 / 0.98 |
| `best_tour` uses the MCTS tour | 75% of graphs |

## Local validation (Mac, CPU, `.venv`)

- C++ batched and sequential vs Python reference on Stage 1 TSP-50 weights (12-city
  graphs, K=16): identical tours and visit counts, max |ΔQ| 1.7e-7.
- Existing `smoke_mix.py` and `smoke_alphazero.py` pass against the rebuilt extension.
- `python -m unittest scripts.test_student_isolation`: 11 tests pass (Gumbel vs
  transcription, masks, best tour, end-to-end with matched budget, exact student
  resume after a simulated disconnect, teacher shard regeneration bit-identical,
  manifest rules, decision rule, retention CI).
- Notebook executed in a clean kernel with `cpu_smoke` (real checkpoint, 10 cities):
  all 9 code cells, including compiling the C++ search from the embedded snapshot.
- Notebook rebuilt from the final sources (snapshot `26188b0f9bfc`) and re-executed
  clean with `cpu_smoke`. The local checkpoint's SHA-256 equals the Drive copy used in
  Step 1 (`04150b55…`). Record: `_progress/eval_logs/student_isolation_validation_20261003.json`.

## CPU preview at one-eighth scale (2026-10-03) — NOT a result

Same runner on this Mac: 4,096 training graphs, 512 validation and 512 test graphs
(the first 512 of the main test set), 4 epochs, learning rates {1e-5, 3e-5, 1e-4},
REINFORCE control with 128K-graph epochs. Purpose: check that students learn at all
and exercise every phase on real TSP-50 data.

| policy | test cost (512) | Δ vs Stage 1 | selected lr | fix / keep rate |
|---|---:|---:|---:|---|
| Stage 1 greedy | 5.7980 | 0 | — | 0 / 1 |
| teacher (K=40) | 5.7368 | −0.0612 | — | — |
| visits | 5.7896 | −0.0084 | 3e-5 | 0.24 / 0.995 |
| gumbel_q | 5.7891 | −0.0089 | 3e-5 | 0.30 / 0.990 |
| best_tour | 5.7894 | −0.0086 | 3e-5 | 0.25 / 0.993 |
| control (CPU, 101 steps) | 5.7997 | +0.0017 | — | 0.41 / 0.964 |

- Every target learned; retention ≈ 0.14 for all three (bootstrap 95% ≈ 0.08–0.21).
  The learning-rate grid is well placed: 3e-5 won for every target.
- Validation gains (−0.011 to −0.016) exceed test gains: selection over many
  checkpoints on 512 graphs is optimistic, as expected. Test is the honest number.
- The control is meaningless on CPU (REINFORCE ran 101 steps). On a GPU it gets
  roughly 10–20× more updates per minute, so "beats the control" is not settled here.
- The pre-registered floor (retention ≥ 0.25) would STOP at this scale. The main run
  has 8× the training graphs and 2× the epochs; whether retention rises is the open
  question. The rule is unchanged; Lejun was told it may be the deciding criterion.

## Notes for other machines

The C++ change needs a rebuild before `return_root_q` works: `pip install -e .`
(or `python setup.py build_ext --inplace`). The checked-in Windows `.pyd` binaries
are older; without a rebuild everything else behaves as before and a request for
root Q raises a clear error. The Colab notebook always compiles its own copy.

## How to run (Lejun)

Upload `notebooks/colab_student_isolation.ipynb` to Colab, pick an L4 GPU, run all
cells with `PROFILE = "main"`. The checkpoint and `args.json` from Step 1 are already
in `MyDrive/AM_AlphaGoZero/checkpoints/stage1_tsp50_with_value/`. Output and the
results ZIP land in `MyDrive/AM_AlphaGoZero/outputs/student_isolation/`. After a
disconnect, rerun from the top; finished work is skipped.

GPU choice (answered 2026-10-03): keep ONE GPU type for every phase. The control's
budget is the teacher and student wall time measured earlier in the same run, so a
reconnect onto a different GPU breaks the matching; `sessions.jsonl` records the GPU
per phase and must be checked when reviewing. An A100 is not expected to help much:
on an A10G the batched search spent ~20% of its wall in NN forward (Stage 5 §F
profile), and T4 vs A10 differed only 1.5× on the §H self-play notebook. A faster GPU
also speeds the GPU-bound REINFORCE control more than the CPU-bound teacher, which
makes the matched-time test harder for the students. L4 stays the recommendation.
