# Stage 5 §V — off-policy rollout distillation into the value head

Successor to [stage5_mix_leafeval_plan.md](stage5_mix_leafeval_plan.md) §12
(the compute-not-variance reframing). Mirror progress file:
[`_progress/stage5_offpolicy_value_progress.md`](../_progress/stage5_offpolicy_value_progress.md).

## 1. Context

§H closed with the diagnosis (progress §H.7): the AGZ-style value head fails
**off-policy** — calibrated on trajectory states (|err| 0.08 raw) but ~0.6 raw
optimistic on untaken sibling children, so its within-node ranking Spearman is
≈ 0 where MCTS reads it. Representation changes don't help (ValueTrunk worse).
Plan §12 reframed the head's value proposition as **compute**: rollouts are
91% of decode forwards at TSP-20 K=40; a trustworthy head buys ≈ 1+N/2× sims
at matched wall (11×@N=20, 26×@N=50, 51×@N=100), which is the lever for the
stalled TSP-50 program (§E: +0.132 above Stage-1 greedy at 270 s/iter).

The fix candidate: **distill the greedy rollout into the head on the
counterfactual children themselves** — labels the search already computes for
free under `leaf_eval=rollout`. The head stops predicting "game outcome z" and
becomes an *amortized rollout*: the exact function `leaf_eval='rollout'`
evaluates (mcts.py::_rollout_remaining_real), at ~0 marginal forwards per leaf.

## 2. §V0 — the decisive cheap test (Colab T4, ~30-40 min)

All on the frozen §H.4 policy (`iter-99.pt` + `buffer.pt`). No self-play, no
C++ changes, no encoder training — only the 16.6k-param glimpse-head MLP.

1. **Dataset** (`build_offpolicy_value_dataset.py`): sample ~20k buffer states
   (train split `inst%5!=0`, ~19/step, parent steps 0..N-2), enumerate ALL
   legal children, label each with greedy-rollout cost-to-go (raw, excludes
   the edge into the child — probe convention). ≈200k pairs, stored as
   (slot, action, label) triples. Plus a 2k-state eval-split dataset for
   held-out MSE monitoring.
2. **Distill** (`train_value_head_offpolicy.py`): MSE on raw labels, policy
   frozen, glimpse computed no-grad, warm-start from the §H.4 head.
   20 epochs × 400 steps × bs 512, lr 1e-3 → 2e-4 @ 10 → 4e-5 @ 20.
3. **Gates** (pre-registered, held-out instances `inst%5==0`, 50 nodes,
   sampled-E[z|s'] gt, same slots as §H.7.1 via committed caches in
   `_progress/eval_logs/rank_gt/`):
   - **G1 depth-1**: decision regret ≤ **0.08** raw AND intrinsic
     Spearman(v,g) ≥ **0.5**. Anchors: rollout 0.056 / 0.88; pre-distill head
     0.199 / 0.081.
   - **G2 depth-2** (`probe_action_ranking --root_hop 1`, new): regret ≤
     **2× the rollout anchor on the same probe** AND Spearman(v,g) ≥ **0.4**.
     Tests generalization beyond the depth-1 training distribution — the
     training children are 1 step off-trajectory; MCTS leaves go deeper.

Decision rule: **G1∧G2 PASS → V1. G1 pass only → training-data lever (add
hop-augmented children to the dataset), re-gate. Both fail after the data
lever → the amortization claim fails on TSP-20; record and stop.**

## 3. §V1 — matched-wall inference validation (TSP-20, val-only, ~15 min T4)

**V0 PASSED 2026-07-04** (depth-1 rollout parity: regret 0.0554 vs anchor
0.0563; see progress). V1 cashes the dividend end-to-end via
`notebooks/colab_V1_matched_wall_vh.ipynb` — four paired arms on
val_size 10000 seed 42 (`val_stage4_mcts.py --save_costs` for cross-arm
paired t-tests; ε=0, τ=const, c_puct=0.05, mcts_batch_size=1000):

| arm | ckpt | leaf_eval | K |
|---|---|---|---|
| R (wall budget) | distilled | rollout | 40 |
| V40 | distilled | value_head | 40 |
| **VM (primary)** | distilled | value_head | K_matched = 40·W_R/W_V40, rounded to 20, clamp [60, 800] |
| O40 (contrast) | ORIGINAL | value_head | 40 |

Pre-registered criteria:
- **S1 (mechanism):** V40 < greedy, paired one-sided p < 0.01. The original
  head fails this at every K (§C.3); the distilled head must pass.
- **S2 (primary, non-inferiority):** paired Δ(VM − R) ≤ +0.002. Strictly
  better ⇒ headline; parity ⇒ mechanism validated and the prize moves to
  N≥50 (multiplier ~26× vs ~11× here).

Known residual risk: V0 trained on depth-1 children only; deeper MCTS trees
query farther off-distribution (depth-2 regret 0.161 vs rollout's 0.082).
If S1 passes but S2 fails, the first lever is hop-augmented training data
(labeler on hopped states), not architecture. This stays within the §12
standing decision — a *gate*, not a TSP-20 training experiment.

## 4. §V2 — deploy at TSP-50 (the actual prize)

Only on V1 PASS. Two modes, in order:
(a) inference: distilled head at matched wall against §E's rollout numbers;
(b) training: "value head as rollout cache" — self-play with cheap head
leaf-eval, re-distilled from rollouts on a small state sample every k
iterations, λᵥ=0 on the shared encoder throughout (§D). Requires a TSP-50
labeling pass on an §E buffer; budget ~1 Colab session or ~$10 Modal.

## 5. Critical files

- **New** `src/scripts/build_offpolicy_value_dataset.py` — batched
  rollout-labeler (uniform-step batches, rectangular children layout,
  fp32-parity with the sequential probe verified 2.4e-7).
- **New** `src/scripts/train_value_head_offpolicy.py` — frozen-policy
  distillation trainer (glimpse head; trunk variants supported via flags).
- **New** `src/scripts/rank_rollout_benchmark.py` — rollout ranking anchor
  from paired (greedy, sampled) gt caches; zero model calls.
- **Edited** `src/scripts/probe_action_ranking.py` — `--root_hop h` depth
  probe (seeded hop stream; hop=0 hashes unchanged, committed caches valid).
- **New** `notebooks/colab_V0_offpolicy_value_distill.ipynb` — T4 pipeline:
  datasets → distill → G1/G2 gates → verdict cell.
- Committed gt caches: `_progress/eval_logs/rank_gt/rank_gt_{1b68dbd028,
  323b72d066,56345b29b6,bdec4956cc}.npz` (provenance: §H.7.1 probes on the
  §H.4 buffer, seed 1234).

## 6. Verification (local CPU, done 2026-07-04)

1. `rank_rollout_benchmark` reproduces the §H.7.1 anchors exactly
   (0.04676 all / 0.05635 eval).
2. Batched labeler ↔ sequential probe greedy-gt parity: 101 labels across 8
   nodes, max |Δ| = 2.4e-07 (fp32).
3. Tiny end-to-end (219 pairs, 3 epochs): held-out child RMSE 1.06 → 0.29;
   held-out Spearman(v,g) 0.081 → 0.489 — directional signal at 0.1% of the
   full data budget.
4. `--root_hop 1` path: gt gen + cache reuse + summary run clean; original
   head at depth-2: regret 0.70 ≈ random-pick 0.76 (6-node smoke).
5. Terminal-children exclusion (parent step N-1): decode_step has no glimpse
   at all-visited states and MCTS never queries value there.

## 7. Out of scope

- Mixing on-policy z targets into the distillation loss (the taken child's
  rollout label ≈ realized z under argmax play; redundant at TSP-20).
- Trunk-head distillation (flags exist; run only if the glimpse head fails
  G1/G2 with the data lever exhausted).
- Any TSP-20 self-play training with the head (§12 standing decision).
