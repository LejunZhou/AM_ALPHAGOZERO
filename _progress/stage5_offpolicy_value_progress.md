# Stage 5 §V — off-policy rollout distillation — progress

Mirror of [`_plans/stage5_offpolicy_value_plan.md`](../_plans/stage5_offpolicy_value_plan.md).

## Status overview

| Phase | Status | Date | Headline |
|---|---|---|---|
| V0 scaffold (labeler + trainer + probes + Colab notebook + smokes) | **COMPLETE 2026-07-04** | 2026-07-04 | batched labeler fp32-parity 2.4e-7 with sequential probe; tiny 219-pair distill already moves held-out Spearman(v,g) 0.081→0.489 |
| V0 Colab T4 run (~200k pairs, gates G1/G2) | **OPEN — ready to run** | — | `notebooks/colab_V0_offpolicy_value_distill.ipynb` |
| V1 matched-wall vh-only val (TSP-20) | BLOCKED on V0 PASS | — | — |
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

## §V0 Colab T4 run — **OPEN**

Run `notebooks/colab_V0_offpolicy_value_distill.ipynb` top to bottom
(~30–40 min). Paste the Section 6 verdict table back. Artifacts persist to
Drive in the §H.4 run dir: `iter-99_vh_offpolicy.pt`, both datasets, probe
CSVs, hop-1 gt caches.

Gates (pre-registered in plan §2): G1 regret ≤ 0.08 ∧ Spearman(v,g) ≥ 0.5;
G2 (hop-1) regret ≤ 2× rollout anchor ∧ Spearman(v,g) ≥ 0.4.
PASS → V1 (matched-wall vh-only val). G1-only → hop-augmented training data,
re-gate. Both fail after data lever → record negative, stop per plan §12.

## Cross-references

- Plan: [`_plans/stage5_offpolicy_value_plan.md`](../_plans/stage5_offpolicy_value_plan.md).
- Diagnosis: [`stage5_mix_leafeval_progress.md`](stage5_mix_leafeval_progress.md) §H.7.
- Standing decisions: [`_plans/stage5_mix_leafeval_plan.md`](../_plans/stage5_mix_leafeval_plan.md) §12.
- Memory: `project_alphagozero_value_head_leaf_eval_bias.md`.
