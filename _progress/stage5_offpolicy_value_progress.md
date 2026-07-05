# Stage 5 §V — off-policy rollout distillation — progress

Mirror of [`_plans/stage5_offpolicy_value_plan.md`](../_plans/stage5_offpolicy_value_plan.md).

## Status overview

| Phase | Status | Date | Headline |
|---|---|---|---|
| V0 scaffold (labeler + trainer + probes + Colab notebook + smokes) | **COMPLETE 2026-07-04** | 2026-07-04 | batched labeler fp32-parity 2.4e-7 with sequential probe; tiny 219-pair distill already moves held-out Spearman(v,g) 0.081→0.489 |
| V0 Colab T4 run (~200k pairs, gates G1/G2) | **COMPLETE 2026-07-04 — PASS (both gates)** | 2026-07-04 | **distilled head reaches ROLLOUT PARITY at depth-1 (regret 0.0554 vs anchor 0.0563); Spearman(v,g) 0.081→0.788; depth-2 0.161 ≤ 2× anchor 0.082** |
| V1 matched-wall vh-only val (TSP-20) | **OPEN — scaffold ready** | 2026-07-04 | `notebooks/colab_V1_matched_wall_vh.ipynb` (4 arms, paired, ~15 min T4) |
| V2 TSP-50 deployment | BLOCKED on V1 PASS | — | — |

Reference anchors (held-out instance split, 50 nodes, sampled-E[z|s'] gt —
§H.7.1 [`stage5_mix_leafeval_progress.md`](stage5_mix_leafeval_progress.md)):

| estimator | decision regret raw | Spearman(v,g) | top-1 |
|---|---|---|---|
| greedy rollout (target function) | **0.056** | — | 0.74 |
| §H.4 glimpse head, pre-distill | 0.199 | 0.081 | 0.60 |
| gate G1 threshold | ≤ 0.08 | ≥ 0.5 | (report) |

## §V0 scaffold — **COMPLETE 2026-07-04**

- [x] `src/scripts/build_offpolicy_value_dataset.py` — greedy-rollout labels
      for ALL legal children of sampled buffer states, batched per parent
      step (rectangular children layout: at step t exactly t nodes visited).
      Parent step N−1 excluded (terminal children have no glimpse; MCTS uses
      exact cost there). Stores compact (slot, action, step, label) + meta.
- [x] `src/scripts/train_value_head_offpolicy.py` — frozen-policy MSE
      distillation; trainable = `value_head` MLP only (16,641 params);
      glimpse computed under no_grad; warm-start default, `--reinit_head`
      optional; `separate_trunk`/`--value_own_encoder` variants supported.
- [x] `src/scripts/probe_action_ranking.py` + `--root_hop h` — depth-(h+1)
      off-policy generalization probe; hop actions from a seeded side stream
      (`seed+4242`) so slot sampling and hop-0 cache hashes are unchanged.
- [x] `src/scripts/rank_rollout_benchmark.py` — rollout ranking anchor from
      paired greedy/sampled gt caches (zero model calls).
- [x] gt caches committed: `_progress/eval_logs/rank_gt/rank_gt_*.npz`
      (4 files, 26 KB each — §H.7.1 provenance, previously only in the local
      outputs dir).
- [x] `notebooks/colab_V0_offpolicy_value_distill.ipynb` — full T4 pipeline
      with a programmatic G1/G2 verdict cell.

### Local CPU smokes (2026-07-04)

| check | result |
|---|---|
| benchmark reproduces §H.7.1 anchors | 0.04676 (all) / 0.05635 (eval) — exact |
| batched labeler vs sequential probe (101 labels, 8 nodes) | max \|Δ\| = 2.4e-07 |
| tiny distill (219 pairs, 3 epochs) held-out child RMSE | 1.06 → 0.29 |
| tiny distill held-out Spearman(v,g) / regret | 0.081 → **0.489** / 0.199 → 0.201 |
| `--root_hop 1` smoke (orig head, 6 nodes) | regret 0.70 ≈ random-pick 0.76 — head useless at depth-2, consistent with §H.7.2 |
| terminal-children NaN (parent step 19) | fixed — excluded from dataset |

The starting held-out child RMSE of the warm-started head (≈1.06 raw)
independently reproduces the §H.7.2 off-policy error magnitude (−1.03 vs
sampled gt) on a fresh state sample — the diagnosis holds out of sample.

## §V0 Colab T4 run — **COMPLETE 2026-07-04 — PASS (G1 ∧ G2)**

Run: `notebooks/colab_V0_offpolicy_value_distill.ipynb` on T4, ~200k
rollout-labeled child pairs (train split `inst%5!=0`), 20-epoch distill into
the 16.6k-param glimpse head. Verdict table (held-out instances, Section 6):

| probe | regret | sp(v,g) | sp(act) | top1 |
|---|---|---|---|---|
| rollout anchor depth-1 | 0.0563 | — | 0.879 | 0.74 |
| head ORIG depth-1 | 0.1994 | 0.081 | 0.669 | 0.60 |
| **head DISTILLED depth-1** | **0.0554** | **0.788** | **0.867** | **0.76** |
| rollout anchor depth-2 (hop1) | 0.0819 | — | — | — |
| head ORIG depth-2 | 0.3435 | 0.120 | 0.334 | 0.30 |
| **head DISTILLED depth-2** | **0.1610** | **0.765** | **0.710** | **0.52** |

- **G1 PASS** (regret ≤ 0.08 ∧ sp_ctg ≥ 0.5): 0.0554 / 0.788.
- **G2 PASS** (regret ≤ 2×0.0819 ∧ sp_ctg ≥ 0.4): 0.1610 / 0.765.

### Interpretation

1. **The off-policy diagnosis is confirmed BY INTERVENTION.** Same
   architecture, same 16.6k params, same frozen policy — only the training
   distribution changed — and held-out sibling ranking went from ≈0 signal
   to **statistical parity with the greedy rollout itself** (0.0554 vs
   0.0563 on identical slots). §H.7's "not representation, distribution"
   verdict is now causal, not correlational.
2. **Depth-2 generalizes but degrades gracefully.** The rollout anchor
   itself degrades off-trajectory (0.056 → 0.082 — harder nodes); the
   distilled head holds 0.161 with sp(v,g) 0.765 (vs ORIG's useless
   0.344/0.120 at depth-2). Training data was depth-1 children only, so
   deeper MCTS trees see states farther from the training distribution —
   the residual risk V1 measures end-to-end.
3. The amortized-rollout framing holds: the head now IS a ~free
   approximation of the function `leaf_eval='rollout'` computes.

Artifacts on Drive (§H.4 run dir): `iter-99_vh_offpolicy.pt`, both datasets,
4 probe CSVs, 2 hop-1 gt caches.

## §V1 matched-wall vh-only val (TSP-20) — **OPEN — scaffold ready 2026-07-04**

`notebooks/colab_V1_matched_wall_vh.ipynb` (~15 min T4). Four paired arms on
val_size 10000 seed 42 (per-instance costs via the new
`val_stage4_mcts.py --save_costs`, smoke-verified paired across invocations):
R (rollout K=40, the wall budget), V40 (vh-distilled K=40), VM (vh-distilled
at K_matched = 40·W_R/W_V40, computed in-notebook, clamp [60,800]), O40
(vh-ORIGINAL K=40 contrast).

Pre-registered: **S1** V40 < greedy (paired p<0.01; ORIG head fails this per
§C.3); **S2 (primary)** Δ(VM − R) ≤ +0.002 non-inferiority at matched wall.
PASS → V2 (TSP-50, where the multiplier is ~26× and the +0.132 parity gap is
the target).

## Cross-references

- Plan: [`_plans/stage5_offpolicy_value_plan.md`](../_plans/stage5_offpolicy_value_plan.md).
- Diagnosis: [`stage5_mix_leafeval_progress.md`](stage5_mix_leafeval_progress.md) §H.7.
- Standing decisions: [`_plans/stage5_mix_leafeval_plan.md`](../_plans/stage5_mix_leafeval_plan.md) §12.
- Memory: `project_alphagozero_value_head_leaf_eval_bias.md`.
