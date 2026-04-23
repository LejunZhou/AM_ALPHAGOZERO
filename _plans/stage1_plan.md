# Stage 1 Plan: Extend AM with an Auxiliary Value Head

**Created:** 2026-04-23
**Predecessor:** `_plans/stage0_plan.md` (completed, TSP-20 trained to 3.8426 — matches published)
**Status:** Approved 2026-04-23 — implementation pending

---

## Context

Stage 0 delivered a clean AM reimplementation that reproduces the paper's TSP-20 numbers. Stage 1 is the first real extension toward the AlphaGo-Zero-style training loop: we add a **value head** that predicts cost-to-go from partial tour states, and we train it jointly with the existing REINFORCE policy via a combined loss. The value head is purely auxiliary in this stage — it does **not** enter the policy-gradient estimator.

**Why this scoping (confirmed in design discussion):**
- If we kept trajectory-level REINFORCE (every action shares one scalar advantage), the only place `v` could plug into the policy gradient is as `v(s_0)` replacing `bl_val` — which is strictly worse than the existing greedy-rollout baseline (an actual rollout beats a learned estimate).
- Using `v(s_t)` as a per-step baseline requires pairing with reward-to-go; that's a Stage 5 ablation ("AlphaGo Zero loss vs. REINFORCE + value auxiliary loss", proposal line 154), not a Stage 1 change.
- Stage 1's explicit success criteria (proposal lines 67-69) are "value R² > 0.7" and "policy gap unchanged or improved." Mixing two changes (value head + advantage formulation) would destroy attribution.

The value head's real payoff comes in Stages 3-4, where it serves as the MCTS leaf evaluator — not as a PG baseline. In Stage 1 the only way `v` influences the policy is via **shared-trunk representation pressure**, and whether that helps, hurts, or is neutral is exactly the measurement we want.

---

## Design Summary

### What changes

1. **Model:** add `ValueHead(nn.Module)` — a small MLP that maps the decoder's per-step *glimpse* vector (128-dim, already returned by `decoder.decode_step(return_glimpse=True)`) to a scalar value estimate.
2. **Decoder loop:** `decode()` collects per-step glimpses alongside log-probs, so the training forward returns `(cost, log_likelihood, per_step_values)`. Inference (greedy eval / rollout baseline) skips value computation to keep eval cheap.
3. **Problem helper:** add `TSP.get_edge_costs(dataset, pi) -> (B, N)` returning per-edge costs, so we can compute realized cost-to-go targets `R_t`.
4. **Trainer:** extend `train_batch` to compute `L_value` (per-step MSE against normalized cost-to-go) and optimize `L = L_reinforce + lambda_v * L_value`. Policy-gradient expression unchanged.
5. **Config:** add `lambda_v`, `value_hidden_dim`, `value_target_norm` ('bl' or 'sqrt_n'), `value_enabled` (bool, default True).
6. **Logger:** add per-epoch diagnostics — value R² (overall + by tour position bucket), value loss magnitude, calibration plot (W&B only), vs. policy loss magnitude.

### What does NOT change

- REINFORCE loss expression (`trainer.py:115`) — bit-identical.
- Greedy rollout baseline (`RolloutBaseline` in `baselines.py`) — bit-identical; `bl_val` still enters the PG the same way.
- Challenger/champion gating — unchanged; it evaluates policy performance only (greedy cost), not value quality.
- Encoder — fully shared between policy and value (this is the "shared-trunk regularization" we're measuring).

### Value head specification

- **Input:** per-step glimpse vector from `decoder._one_to_many_logits` (already returned when `return_glimpse=True` — see `decoder.py:111`, `decoder.py:162`). Shape `(batch, embed_dim)` per step.
- **Architecture:** `Linear(embed_dim → value_hidden_dim) → ReLU → Linear(value_hidden_dim → 1)`. Default `value_hidden_dim = embed_dim = 128`.
- **Target:** realized cost-to-go from the sampled trajectory, normalized:
  - `R_t = sum of edge costs from step t to close-of-tour` (over the *sampled* `pi`, not greedy)
  - `R_t_normalized = R_t / Z` where `Z = bl_val` (per-instance; default) or `sqrt(N)` (fallback).
- **Loss:** `L_value = mean_{i, t} (R_{t,i}/Z_i - v_theta(s_{t,i}))^2`, averaged over batch and over all N decoding steps.
- **Weighting:** `L = L_reinforce + lambda_v * L_value`. Default `lambda_v = 1.0` for TSP-20; small sweep on {0.1, 0.5, 1.0, 2.0} if the policy gap regresses.
- **Gradient flow:** value MSE backprops through value head **and** shared trunk (encoder + decoder up to the query/glimpse). This is intentional — it's the mechanism for representation transfer. `bl_val` stays detached (never touched).
- **Source policy:** on-policy — values are computed during the sampling-mode forward used for REINFORCE. Same trajectory used for both losses.
- **When skipped:** during `rollout()` (greedy eval + baseline eval) and `validate()`. Gated by `model.training` check or an explicit `compute_values` flag in `decode()`.

### Why glimpse (not query, not graph-embedding)

The glimpse is the decoder's attention-aggregated view of remaining nodes, *after* mask-based feasibility filtering is applied in `_one_to_many_logits` (`decoder.py:147`). It carries: graph context (via fixed context projection), current state (first + last node), *and* remaining-node structure (via attention). Query alone omits the remaining-node integration; graph embedding alone omits the partial-tour state. Glimpse integrates both and is already plumbed through `return_glimpse=True`.

---

## Architecture: `src/am_baseline/` Changes

```
src/am_baseline/
  model/
    value_head.py               # NEW: ValueHead(nn.Module)
    decoder.py                  # MODIFY: decode() collects glimpses when compute_values=True
    attention_model.py          # MODIFY: forward returns (cost, ll, values); construct ValueHead
  problem/
    tsp.py                      # MODIFY: add get_edge_costs(dataset, pi) -> (B, N)
  training/
    trainer.py                  # MODIFY: compute L_value in train_batch; handle detached bl_val normalizer
    logging.py                  # MODIFY: log_step / log_epoch accept value metrics
  baseline/
    baselines.py                # UNCHANGED (deep-copy still copies value head harmlessly)
  config.py                     # MODIFY: add lambda_v, value_hidden_dim, value_target_norm, value_enabled
  evaluation/
    evaluate.py                 # MODIFY: accept --eval_value flag to report R² on eval set
scripts/
  train.py                      # UNCHANGED (config passes value args automatically)
  eval_value.py                 # NEW: standalone value-head diagnostic (R², calibration by tour position)
```

---

## Implementation Sequence

### Phase A: Model changes (value head wired, forward returns values)

1. **`model/value_head.py`** — new file. Minimal MLP as specified above. Defensive: accept `(B, D)` or `(B, 1, D)` and squeeze consistently.
2. **`model/decoder.py`** — modify `decode()` (lines 77-92) to accept `compute_values=False` kwarg. When True, it must:
   - Call `decode_step(fixed, state, return_glimpse=True)` (already supported — see `decoder.py:94-112`).
   - Accumulate glimpses into a list alongside `outputs`.
   - Return `(log_p_stack, sequences, glimpses_stack)` — glimpses shape `(B, N, embed_dim)`.
   - When `compute_values=False`, preserve existing signature so Stage 0 callers (baseline rollout, validate) don't break.
3. **`model/attention_model.py`** — modify `__init__` to construct `ValueHead` when `config.value_enabled`. Modify `forward()` (lines 50-63):
   - In training mode (or explicit `compute_values=True`), request glimpses from decoder, run value head over them, return `(cost, ll, values)` where `values` is `(B, N)`.
   - In eval mode (default), return `(cost, ll)` — unchanged. Preserves `RolloutBaseline.eval` call path at `baselines.py:101-104`.
4. **Milestone A:** smoke test — forward pass in training mode returns three tensors with correct shapes; eval mode still returns two.

### Phase B: Target and loss computation

5. **`problem/tsp.py`** — add `get_edge_costs(dataset, pi)`:
   - Gather coordinates in tour order, compute `norms = (d[:, 1:] - d[:, :-1]).norm(dim=2)` (shape `(B, N-1)`).
   - Append the closing edge `(d[:, 0] - d[:, -1]).norm(dim=1).unsqueeze(-1)` (shape `(B, 1)`).
   - Return concatenated `(B, N)` — element `i` is the edge cost incurred at decoding step `i+1` (0-indexed).
   - Verify: `get_edge_costs(...).sum(dim=1) == get_costs(...)[0]` (within fp tolerance).
6. **Cost-to-go tensor:** add `utils/tensor_ops.py::cost_to_go(edge_costs)` → `(B, N)` where entry `[b, t]` = `edge_costs[b, t:].sum()`. Implementable as `torch.flip(torch.cumsum(torch.flip(edge_costs, [1]), 1), [1])`.
7. **`training/trainer.py`** — in `train_batch`:
   - After `cost, log_likelihood, values = model(x)`, compute `edge_costs = TSP.get_edge_costs(x, pi)` — note: `pi` is currently not returned by `model(x)`; either extend forward to also return `pi` (via `return_pi=True`, which is already supported at `attention_model.py:50-63`), or recompute via `return_pi=True`. Prefer: pass `return_pi=True` always in training and destructure.
   - Compute `Z = bl_val.detach()` (per-instance) or `sqrt(graph_size)` based on `config.value_target_norm`.
   - `targets = cost_to_go(edge_costs) / Z[:, None]` — shape `(B, N)`, detached.
   - `value_loss = F.mse_loss(values, targets)` — auto-averages over batch and steps.
   - `loss = reinforce_loss + config.lambda_v * value_loss + bl_loss`.
   - Log `value_loss.item()` and, periodically, per-bucket R² on the current batch.
8. **Milestone B:** smoke test — 2-epoch run on TSP-20 with `lambda_v=1.0` completes without NaN, value_loss decreases, policy_loss behavior matches Stage 0 reference within noise.

### Phase C: Diagnostics and configuration

9. **`config.py`** — add fields `lambda_v: float = 1.0`, `value_hidden_dim: int = 128`, `value_target_norm: str = 'bl'` (choices: 'bl', 'sqrt_n'), `value_enabled: bool = True`. Add matching CLI args in `from_args`.
10. **`training/logging.py`** — extend `log_step` to accept `value_loss`, `value_r2_overall` (optional). Extend `log_epoch` to accept `val_value_r2_overall`, `val_value_r2_early`, `val_value_r2_mid`, `val_value_r2_late`. Add new CSV columns, W&B `define_metric` entries, TensorBoard scalars.
11. **Per-epoch validation value metrics:**
    - In `trainer.validate()` (or a sibling `validate_value()`), run one forward pass on `val_dataset` with `compute_values=True` under `torch.no_grad()`.
    - Collect `(values, targets)` across batches. Compute overall R² (`1 - var(residuals)/var(targets)`) and bucketed R² for step ranges `[0, N/4)`, `[N/4, 3N/4)`, `[3N/4, N]`.
    - Log to W&B.
12. **`scripts/eval_value.py`** — standalone script that loads a trained checkpoint, runs value-head forward on a held-out dataset, emits: overall R², per-bucket R², calibration plot (binned mean `v` vs. binned mean `R_t`), scatter.

### Phase D: Experiments

13. **Smoke (CPU/local GPU, TSP-20, ~3 epochs, small epoch_size):** confirm no NaN, shapes correct, losses plausible.
14. **Canonical TSP-20 run (Modal A10G, 100 epochs, full epoch_size, `lambda_v=1.0`, `value_target_norm='bl'`, seed=1234):** match Stage 0 settings so the policy-gap comparison is apples-to-apples. Expected: val_avg_cost within noise of Stage 0's 3.8426 (target: the gap must not regress more than ~1%).
15. **Lambda sweep (TSP-20, truncated 30-epoch runs):** `lambda_v in {0.1, 0.5, 1.0, 2.0}`. Select the largest `lambda_v` for which policy gap is statistically indistinguishable from Stage 0. Record value R² at that setting.
16. **Target-norm ablation (TSP-20, 30 epochs):** `bl` vs `sqrt_n`. Expect `bl` to give better R² (per-instance normalization cancels instance difficulty). If `sqrt_n` wins, investigate — likely bug.
17. **TSP-50 verification (optional, only if TSP-20 succeeds):** single 100-epoch run at best-lambda with `bl` normalization, confirm value head generalizes to larger graphs without re-tuning.

---

## Key Files to Modify (with anchors)

| File | Current line(s) | Change |
|---|---|---|
| `src/am_baseline/model/value_head.py` | NEW | `ValueHead(nn.Module)` — 2-layer MLP |
| `src/am_baseline/model/decoder.py` | 77-92 (`decode()`) | Add `compute_values` kwarg; collect glimpses via existing `return_glimpse` path (94-112) |
| `src/am_baseline/model/attention_model.py` | 17-92 | Construct `ValueHead` in `__init__`; branch `forward()` on training/compute_values |
| `src/am_baseline/problem/tsp.py` | 12-23 (`get_costs`) | Add `get_edge_costs` (reuse tour-order gathering) |
| `src/am_baseline/utils/tensor_ops.py` | append | Add `cost_to_go()` (reverse cumsum) |
| `src/am_baseline/training/trainer.py` | 103-127 (`train_batch`) | Extend to request pi + values, compute `L_value`, combine losses |
| `src/am_baseline/training/trainer.py` | 12-18 (`validate`) | Add value-diagnostic pass (new sibling fn) |
| `src/am_baseline/training/logging.py` | 80-118 / 120-141 | Extend step/epoch logging with value metrics |
| `src/am_baseline/config.py` | 7-60 + 79-134 | Add fields + CLI args |
| `src/am_baseline/evaluation/evaluate.py` | — | Add value-R² reporting flag |
| `src/scripts/eval_value.py` | NEW | Standalone value diagnostic CLI |

---

## Verification Criteria (Stage 1 done when)

- [ ] Forward pass in training mode returns `(cost, ll, values)` with `values.shape == (B, N)`; eval mode still returns `(cost, ll)` — Stage 0 code paths unaffected.
- [ ] `get_edge_costs(x, pi).sum(dim=1) == get_costs(x, pi)[0]` within 1e-5 fp tolerance.
- [ ] `cost_to_go(edge_costs)[:, 0] == edge_costs.sum(dim=1)` and `cost_to_go(edge_costs)[:, -1] == edge_costs[:, -1]` (spot checks).
- [ ] TSP-20 full run with `lambda_v=1.0`: final val_avg_cost within 1% of Stage 0 (3.8426) — policy not degraded. (One-sided paired t-test on held-out eval, `p > 0.05` for "worse than Stage 0"; or raw delta within ±0.04.)
- [ ] **Value head R² ≥ 0.7 overall** on held-out TSP-20 eval set (proposal's explicit target, line 68).
- [ ] Bucketed R² shape matches intuition: **late-tour R² > early-tour R²** (uncertainty collapses as tour completes). If flat or reversed, investigate the input features.
- [ ] Calibration plot (W&B) shows `E[v | bin] ≈ E[R_t | bin]` across bins — unbiased predictions.
- [ ] Loss curves recorded to CSV + W&B; `value_loss`, `actor_loss`, `val_value_r2_overall` visible as separate traces.
- [ ] `_progress/stage1_progress.md` updated with results.

---

## Known Risks / Decisions

1. **Deep-copy of baseline model will now also copy the value head.** This is fine — the baseline never calls the value head (it goes through `rollout()` which uses greedy decoding with `compute_values=False`). No correctness issue, minor memory overhead (~few MB for a 128-hidden MLP).

2. **Training forward is slower** — glimpse collection + value MLP per step. Expected overhead: ~5-15% wall-clock per epoch on TSP-20. Acceptable; will measure.

3. **Normalization choice `bl` depends on `bl_val` being available per-instance.** This is true after `RolloutBaseline.wrap_dataset` runs at epoch start, and `bl_val` is already plumbed into `train_batch` via `baseline.unwrap_batch`. No new data flow needed.

4. **During warmup epochs,** `ExponentialBaseline` is used and its `bl_val` is a scalar (batch-level EMA), not per-instance. Handle: if `bl_val` is scalar or 0-dim, fall back to `value_target_norm='sqrt_n'` for that epoch. Add an assertion that per-instance norm requires a `RolloutBaseline`-phase.

5. **Value head needs a fair chance to warm up before R² is meaningful.** Don't trigger alarms on low R² in the first ~5 epochs. Hold reporting thresholds until ~epoch 10.

6. **If `lambda_v=1.0` regresses the policy gap**, the loss scales are mismatched. Symptom: `actor_loss` magnitude shrinks relative to `value_loss`. Fix by reducing `lambda_v`; do not rescale inputs. The sweep in Phase D step 15 diagnoses this.

7. **Ablation hook for Stage 5.** This plan intentionally does *not* insert `v` into the policy gradient. If we later want the Stage 5 ablation ("AlphaGo Zero loss vs. REINFORCE + value auxiliary loss"), we'll add a separate config flag `use_value_as_baseline` and an alternate loss path — out of scope for Stage 1.

---

## Verification Commands (end-to-end)

```bash
# Conda env (per CLAUDE.md)
conda activate AM_AlphaGoZero

# Smoke test: CPU, 2 epochs, small settings
python -m scripts.train --graph_size 20 --batch_size 32 --epoch_size 640 \
  --n_epochs 2 --no_cuda --run_name stage1_smoke --lambda_v 1.0

# Canonical TSP-20 (Modal A10G)
python -m scripts.modal_run_train --graph_size 20 --n_epochs 100 \
  --batch_size 512 --epoch_size 1280000 --baseline rollout \
  --bl_warmup_epochs 1 --seed 1234 --lambda_v 1.0 \
  --value_target_norm bl --run_name stage1_tsp20_canonical

# Value diagnostic on checkpoint
python -m scripts.eval_value --model outputs/tsp_20/stage1_tsp20_canonical/epoch-99.pt \
  --val_size 10000 --seed 1234
```

Expected first-read of results: `actor_loss` trace looks identical to Stage 0's `wpqp1dpp`; `val_avg_cost` lands ≈ 3.84; `val_value_r2_overall` rises from ≈ 0 to ≥ 0.7 over the first ~30 epochs; bucketed R² climbs from early to late.
