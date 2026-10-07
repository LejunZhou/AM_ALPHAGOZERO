# Step 2 — Student isolation at TSP-50 (Colab)

Date: 2026-10-03. Stage 5 §I Step 2. Approved by Lejun on 2026-10-03: staged
version (screen first, add seeds only for a target that passes), run on Colab.
Runs as a diagnostic under the current proposal; proposal.md is not changed.

**Outcome (2026-10-06): STOP.** Every target beat continued REINFORCE at matched wall time
(`gumbel_q` −0.0097 [−0.0137, −0.0058] vs the control). None kept 25% of the teacher's gain
(best: `gumbel_q` 0.207 [0.169, 0.242]). STOP therefore came from the retention floor, not
from the control. Confirm is not run. Results and review are in
[`student_isolation_colab_progress.md`](../_progress/student_isolation_colab_progress.md).

## Question

Can a REINFORCE-converged AM absorb the improvements that a rollout-MCTS teacher
finds, and does the choice of training target decide it? The from-scratch loop
stalls once the policy is good and the teacher's gains are small; this recreates
that endgame on TSP-50, where the teacher still has room (≈0.05 over greedy).
TSP-50 has never been warm-started in this project.

## Design: one round of distillation from a fixed teacher

- **Teacher (frozen).** Stage 1 TSP-50 `epoch-99.pt`; batched C++ MCTS, rollout
  leaves, K=40, c_puct 0.05, tree reuse, no noise, τ=0 (argmax visits). Per
  instance it stores the Stage 1 greedy tour/cost, the MCTS tour/cost, and per
  tour-step the root visit counts, root child Q values and the root's own
  estimate (new opt-in `return_root_q`, C++ both paths + Python reference).
- **Identical data for every target.** One teacher dataset per seed; all
  targets train on it.
  - `visits`: N(s,a)/ΣN along the teacher's trajectory (current AGZ target).
  - `gumbel_q`: completed-Q improved policy, softmax(log π + σ(q̂)) with the
    mctx `qtransform_completed_by_mix_value` defaults (c_visit 50, c_scale 0.1,
    mixed value for unvisited children, min-max rescale over legal children),
    computed from the PUCT tree's statistics and the frozen prior.
  - `best_tour`: imitate the better of {MCTS tour, Stage 1 greedy tour} per
    instance (the greedy incumbent is always a candidate).
- **Students.** Initialised from Stage 1; teacher-forced cross-entropy over all
  non-forced steps; Adam, grad clip 1.0, 8 epochs, 64 instances per step, the
  same minibatch order for every target. Learning rate grid {1e-5, 3e-5, 1e-4};
  each target's checkpoint and learning rate are chosen on a 1,024-graph
  validation set by greedy cost. The test set never selects anything.
- **Control: more REINFORCE at matched wall time.** Continue Stage 1 from
  `epoch-99.pt` with its optimizer and rollout-baseline state and the Stage 1
  recipe (lr 1e-4, 1.28M-instance epochs, batch 512, value loss λ 1). Uses the
  rollout baseline directly: the repo's `train.py --resume` would leave the
  warm-up wrapper at α=0 and silently train with the exponential baseline.
  Budget = teacher wall on the training split + one target's student training
  wall (its full learning-rate grid). Same validation-based selection.

## Splits (fixed seeds; val/test identical across seeds)

| split | graphs | use |
|---|---:|---|
| train (per seed) | 32,768 | teacher data for students |
| val | 1,024 | checkpoint + learning-rate selection (greedy only) |
| test | 2,048 | reporting; teacher also runs here for the retention denominator |
| canonical val | 10,000 | `torch.manual_seed(42)` set: the self-play loop's validation draw (`--val_seed 42`). Stage 1's 5.7999 was on its own training draw; corrected 2026-10-06. Greedy only |

## Metrics and decision rule

On the test set, with G0 = Stage 1 greedy, T = teacher, S = student greedy,
R = control greedy, all paired per graph:
- Δ_T = G0 − T, Δ_S = G0 − S, Δ_R = G0 − R; retention = Δ_S / Δ_T (bootstrap CI).
- Absorption diagnostic along the teacher's test trajectories: the share of
  "correction" steps (teacher action ≠ Stage 1 argmax) where the student now
  picks the teacher's action, and the regression rate elsewhere.

**Screen (seed 0, all targets).** A target PASSES if S − R has a 95% CI below
zero (it beats more REINFORCE at matched wall time) **and** retention ≥ 0.25.
CONTINUE to Confirm with the passing target that has the best validation cost,
plus `visits` as reference. STOP if no target passes: one round of search
distillation does not beat more REINFORCE, and the TSP claim ends for both the
from-scratch and the warm-start framing. (Note added 2026-10-06: STOP also follows
when targets beat the control but stay below the floor. That is what happened, so
the gloss "does not beat more REINFORCE" does not describe this run.)

**Confirm (seeds 1 and 2: new training graphs, same val/test).** The target holds
if both new seeds beat their own controls and mean retention ≥ 0.25.

*Change from the 2026-10-03 chat description, stated to Lejun:* the earlier
wording said "keep at least half of the teacher's gain". That bar was set for a
multi-iteration loop. With one round of distillation the claim-relevant test is
"beats more REINFORCE at the same wall time"; retention 0.25 is the floor that
keeps a trivial win from passing.

## Engineering

- `src/am_baseline/experiments/student_isolation.py` (phases: check, teacher,
  students, control, evaluate, report); CLI `src/scripts/run_student_isolation.py`;
  tests `src/scripts/test_student_isolation.py`; notebook built by
  `src/scripts/build_student_isolation_notebook.py` →
  `notebooks/colab_student_isolation.ipynb`.
- Self-contained notebook: embedded, hash-checked source snapshot including the
  C++ sources; the extension is compiled in the Colab runtime. No GitHub push.
- Resumable on Drive: teacher shards of 1,024 graphs, per-epoch student
  checkpoints, per-epoch control checkpoints, manifest signature that refuses a
  changed protocol in the same output directory (seeds/targets are work units,
  so Screen and Confirm share one directory and one report).
- Profiles: `main` (Screen), `confirm`, `pilot` (small real run, measures
  throughput), `cpu_smoke` (10-city end-to-end check; cannot answer anything).

## Local validation before handing over

1. C++/Python parity of tours, visits, root Q and root values (done 2026-10-03:
   identical tours and visits, max |ΔQ| 1.7e-7), existing smoke tests still pass.
2. Unit tests: target construction (visits, Gumbel formula against a direct
   implementation, best-tour choice), teacher forcing reproduces the model's own
   greedy log-likelihood, control budget accounting, decision logic.
3. `cpu_smoke` end to end through every phase from the notebook in a clean kernel.
4. A small real-checkpoint TSP-50 run on CPU to measure per-phase cost.

## Expected cost (measured locally; the first Colab shard prints the real ETA)

The teacher is far cheaper than the May loop searches: 0.02–0.03 s per graph at K=40
on this Mac's CPU, because the converged prior keeps PUCT narrow (median one visited
root child) and the evaluator cache absorbs repeated rollouts. A student step (64
graphs, TSP-50, teacher-forced) takes 0.16 s on the Mac CPU. Estimated `main` on an
L4: teacher 15–30 min, students 35–55 min for nine runs, control the same wall time
as teacher + one target, evaluation a few minutes; about 2–3 h in total. Confirm
roughly doubles the teacher and control time for two seeds and two targets.
