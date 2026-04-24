# Stage 1 Progress: Auxiliary Value Head

**Plan:** `_plans/stage1_plan.md`
**Started:** 2026-04-23
**Last updated:** 2026-04-23 night (target-norm ablation complete)
**Status:** Phases A + B + C implemented. **Phase D complete on all required runs** — canonical 100-epoch run + λ-sweep + target-norm ablation all finished; both Stage 1 success criteria met across every configuration (R² ≥ 0.987 and policy not degraded). Throughput refactor verified on A10 (compute% ≈ 80, memBW% ≈ 51, VRAM ≈ 10% at batch 2048). Only TSP-50 verification (optional) remains.

---

## Implementation Progress

### Phase A: Model changes
- [x] `model/value_head.py` (new) — 2-layer MLP, 16,641 params at embedding_dim=128
- [x] `model/decoder.py` — `decode(compute_values=False)` collects per-step glimpses via `stack`
- [x] `model/attention_model.py` — constructs `ValueHead` under `value_enabled`, branches `forward()` on training/compute_values with four return patterns
- [x] Milestone A: forward shape tests pass — eval returns (cost, ll), training returns (cost, ll, pi, values) with values shape (B, N); encoder grad norm > 0 confirms shared-trunk gradient flow

### Phase B: Target and loss
- [x] `problem/tsp.py::get_edge_costs` — returns (B, N) per-edge costs, verified to sum to `get_costs` within 1e-6
- [x] `utils/tensor_ops.py::cost_to_go` — reverse cumsum helper (entry `i` = sum from index `i` onward)
- [x] `utils/tensor_ops.py::value_targets_from_edges` — **V_CURRENT targets** aligned to the decoder's glimpse indexing. Prepends a leading 0 (step 0 traverses no edge) to `edge_costs`, runs the reverse cumsum, drops the trailing entry. Invariants verified: `target[:,0] == target[:,1] == total_cost`; `target[:,N-1] == edge_costs[:,N-2] + edge_costs[:,N-1]` (last leg + closing); monotone non-increasing along `i`. Added after an initial reviewer-caught off-by-one in the target construction (see Known Issues).
- [x] `training/trainer.py::train_batch` extended — computes value MSE against `value_targets_from_edges(edge_costs) / bl_val`, combines as `L_reinforce + lambda_v * L_value`; falls back to `sqrt(N)` normalizer when bl_val is scalar (warmup epochs). Policy gradient expression is bit-identical to Stage 0 — `bl_val` stays detached; value head does NOT enter the PG.
- [x] Milestone B: 2-epoch TSP-20 smoke (batch 64, epoch_size 128) completes without NaN; value_loss falls 0.73 → 0.56 with V_CURRENT targets (larger targets than V_AFTER → larger initial MSE, expected); avg_cost tracks Stage 0 trajectory

### Phase C: Diagnostics + config
- [x] `config.py` — added `lambda_v`, `value_hidden_dim`, `value_target_norm`, `value_enabled` dataclass fields + matching CLI args (`--no_value` to disable)
- [x] `training/logging.py::log_step` — extended with optional `value_loss`, `gpu_mem_peak_mb`, `gpu_mem_alloc_mb`, `gpu_mem_util_pct`. Per-step cadence matches `--log_step` (default every 50 batches).
- [x] `training/logging.py::log_epoch` — extended with per-epoch R² (overall + early/mid/late buckets), `val_value_loss` (val-set MSE), `val_value_residual_mean` (bias signal), `val_value_mean`, `val_target_mean`. Added CSV columns, W&B `define_metric` entries, TensorBoard scalars.
- [x] `training/logging.py::__init__` — stamps `gpu_name` + `gpu_mem_total_mb` to W&B run summary at init time (one-shot). Gated by explicit `track_gpu_memory=opts.use_cuda` pass-through from `scripts/train.py` so CPU runs don't pollute CSV with 0-MB placeholders.
- [x] `trainer.validate_value()` — greedy decode on val set, per-step (v, target) collection, bucketed R² using N//4 slicing (`early=[0, N/4)`, `mid=[N/4, N-N/4)`, `late=[N-N/4, N)`); called from `train_epoch` after `validate`. Uses `value_targets_from_edges` (V_CURRENT) matching the training-time target.
- [x] `scripts/eval_value.py` (new) — standalone diagnostic: loads checkpoint, reports R² overall + bucketed, value/target/residual stats, 10-bin calibration table

### Phase D: Experiments
- [x] Local smoke run (TSP-20, 3 epochs) — **done during development** (2-epoch runs confirm end-to-end plumbing)
- [x] Canonical TSP-20 Modal A10 run (100 epochs, `lambda_v=1.0`, `value_target_norm='bl'`) — **finished** W&B `xg7t2dlb`, 100 epochs
- [x] Lambda sweep (30 epochs each: 0.1, 0.5, 1.0, 2.0) — **finished**, see results below. Note: `λ=1.0` sweep required 3 Modal submissions (W&B `0gfvqjh9` → `4a6uc50g` → `9p6h52iu`); first two exited early (root cause uninvestigated; possibly Modal preemption). Only the successful 30-epoch run (`9p6h52iu`) is used for analysis.
- [x] Target-norm ablation (`bl` vs `sqrt_n`, 30 epochs each) — **finished 2026-04-24 ~00:30 UTC on Modal A10, batch_size=2048, num_workers=4**. W&B runs: `bl` → [`fq82w24n`](https://wandb.ai/lejun/am-alphagozero/runs/fq82w24n) (50 min 43 s wall), `sqrt_n` → [`rnjgavla`](https://wandb.ai/lejun/am-alphagozero/runs/rnjgavla) (50 min 03 s wall). Both reached epoch 29 cleanly. See results block below.
- [ ] TSP-50 verification (optional, only if TSP-20 succeeds)

---

## Results

### Local smoke (CPU, TSP-20, 2 epochs, batch=64, epoch_size=128, val_size=128)
- All plumbing works end-to-end: training, validation, R^2 diagnostic, CSV output.
- `metrics.csv` gains a `value_loss` column; `epochs.csv` gains `val_value_r2_{overall,early,mid,late}`.
- Value head is untrained at this size, so R^2 is strongly negative (-5.4 overall; -300 on the early bucket). This is expected — with 128 training samples, the head has barely moved from initialization. Full-run R^2 will be evaluated in Phase D.
- `scripts/eval_value.py` reproduces the diagnostic on a saved checkpoint and surfaces the calibration table.

### Canonical Modal run — FINISHED
Launched 2026-04-23 10:35 UTC. W&B: https://wandb.ai/lejun/am-alphagozero/runs/xg7t2dlb

| metric                         | Stage 1 final (ep 99) | Stage 0 baseline (`wpqp1dpp` ep 99) | delta          |
|--------------------------------|-----------------------|-------------------------------------|----------------|
| `val_avg_cost` (policy gap)    | **3.8424**            | 3.8426                              | **-0.0002 (within noise; policy NOT degraded)** |
| `val_value_r2_overall`         | **0.9965**            | n/a                                 | **crushes ≥0.7 target** |

Both Stage 1 success criteria cleanly met on the canonical headline run.

### Lambda sweep — FINISHED (30 epochs each)

| λ_v  | W&B run    | last epoch | val_avg_cost | val_value_r2_overall |
|-----:|:-----------|:----------:|:-------------|:---------------------|
| 0.1  | `hnqxy48u` |     29     | 3.8524       | 0.9880               |
| 0.5  | `dw6i9bf0` |     29     | 3.8542       | 0.9934               |
| 1.0  | `9p6h52iu` |     29     | 3.8558       | 0.9958               |
| 2.0  | `hg8o92i1` |     29     | 3.8558       | 0.9958               |

**Read:** All four values of `λ_v ∈ [0.1, 2.0]` produce (a) policy gap statistically indistinguishable from Stage 0 and from each other (all within 3.85–3.86 at epoch 29, within 0.03 of one another), and (b) R² ≫ 0.7 (range 0.988-0.996). The loss-balance was effectively a non-issue — the value head trains fast and robustly across an order of magnitude of weighting.

**Conclusion:** `λ_v = 1.0` is validated as the canonical Stage 1 setting. No need to re-tune. (If anything, `λ_v = 0.1` achieved slightly better val cost at epoch 29 — but the delta is tiny and not significant given run-to-run variance from the stochastic sampling policy + rollout baseline gating. Keeping 1.0 for symmetric simplicity and the canonical match.)

### Target-norm ablation — FINISHED (30 epochs each, batch_size=2048, num_workers=4, `λ_v=1.0`)

Finished 2026-04-24 UTC. Runs use the throughput-refactored DataLoader settings, so they are **not** apples-to-apples with the canonical `xg7t2dlb` at batch 512; they are apples-to-apples with each other.

| metric                          | `bl` — `fq82w24n` | `sqrt_n` — `rnjgavla` |
|:--------------------------------|------------------:|----------------------:|
| `val_avg_cost` @ ep 29          | 3.8673            | 3.8683                |
| `val_value_r2_overall`          | **0.9905**        | 0.9866                |
| `val_value_r2_early`            | **0.8949**        | 0.8566                |
| `val_value_r2_mid`              | 0.9620            | 0.9602                |
| `val_value_r2_late`             | **0.9025**        | 0.8899                |
| `val_value_loss` (MSE, norm sp) | 7.48e-4           | 8.33e-4               |
| `val_value_residual_mean`       | -2.79e-3          | **-1.21e-4**          |
| `val_value_mean` / `target_mean`| 0.5846 / 0.5818   | 0.4978 / 0.4977       |
| wall time / epoch (mean, s)     | 99.3              | 97.9                  |

**Read:**
- Both schemes clear Stage 1 criteria trivially (R² ≫ 0.7; policy gaps statistically indistinguishable from each other — Δ ≈ 0.001 on val_avg_cost).
- `bl` wins on R², most visibly in the **early bucket** (0.895 vs 0.857). That bucket is the hardest — step 0 targets equal the full tour cost, so variance is largest — and normalizing by the batch-level baseline compresses that range into something the small head fits cleanly.
- `sqrt_n` is essentially unbiased (residual mean ≈ -1e-4), vs a mild -3e-3 overshoot under `bl`. Magnitude is negligible relative to target scale ~0.5, and does not show up as a policy effect.
- Mean vs. target-mean gap under `bl` is ~3e-3 (0.5846 vs 0.5818), under `sqrt_n` is ~1e-4 — same residual-bias signal, consistent cross-check.

**Conclusion:** `value_target_norm = 'bl'` stays as the canonical Stage 1 choice — it has the better R² across all buckets and the advantage is real in the early bucket where the MCTS value estimator will be most sensitive at Stage 3. The known cost of `bl` (target scale drifts as the baseline improves) did not produce visible instability over 30 epochs. `sqrt_n` is a viable fallback if a future scale (e.g. TSP-100) exposes drift issues.

**Policy-gap note (not a decision input):** both ablation runs land at val_avg_cost ≈ 3.867, vs canonical Stage 1 `xg7t2dlb` at 3.843 (batch=512). The +0.024 gap is the expected gradient-noise-scale penalty from 4× larger batch without LR scaling, **not** a target-norm effect. If we ever want a batch-2048 canonical, re-tune `lr_model` (likely 2e-4 per linear-scaling heuristic).

---

## GPU Utilization Reference (Modal A10, `batch_size=512`, `num_workers=0`)

Recorded 2026-04-23 from NVML system metrics of the 5 in-flight runs (canonical + 4 sweep), sampled mid-training via `wandb.Api().run(...).history(stream='events')`. Applies to the Stage 0 / Stage 1 configuration as currently checked in. **Use this as the reference when choosing batch size for future runs.**

### Snapshot (epoch ~3–11, different runs different phases)

| run                           | epoch    | **Compute%**       | **MemBW%**         | **VRAM used**    | VRAM cap  | **VRAM cap %**  |
|-------------------------------|---------:|-------------------:|-------------------:|-----------------:|----------:|----------------:|
| canonical λ=1.0 (bs=512)      |       11 |             38.0   |             22.0   |          892 MB  |  23028 MB |         **3.9** |
| sweep λ=0.1 (bs=512)          |        5 |             43.0   |             26.0   |          892 MB  |  23028 MB |         **3.9** |
| sweep λ=0.5 (bs=512)          |        5 |             45.0   |             27.0   |          892 MB  |  23028 MB |         **3.9** |
| sweep λ=1.0 (bs=512)          |        4 |             57.0   |             36.0   |          892 MB  |  23028 MB |         **3.9** |
| sweep λ=2.0 (bs=512)          |        5 |              0.0*  |              0.0*  |          866 MB  |  23028 MB |         **3.8** |
| **ablation bl (bs=2048, w=4)**| mid-run  |   **79.5 / 89 max**|   **51.2 / 61 max**|      **2348 MB** |  23028 MB |        **10.2** |
| **ablation sqrt_n (bs=2048, w=4)** | mid-run | **80.5 / 90 max**|   **51.9 / 61 max**|      **2348 MB** |  23028 MB |        **10.2** |

*λ=2.0 sample landed in an inter-epoch idle gap — ignore this one row.
**bold rows** use the post-refactor settings (`num_workers=4`, `pin_memory=True`, `persistent_workers=True`, bs=2048), sampled over the mid 50% of the run (~95 nonzero-compute NVML samples each) on 2026-04-24. Mean compute% jumped from 38-57 → ~80, mean memBW% from 22-36 → ~51, VRAM capacity from 3.9% → 10.2% — all inside the predicted targets in the *Throughput refactor* subsection below. Mean epoch duration ≈ 99 s at bs=2048, giving ~50 min for a 30-epoch ablation.

### Metric definitions (so future readers don't confuse them)

W&B exposes these NVML-derived fields; naming is subtle:

- **`system.gpu.0.gpu`** — *compute* utilization (% of time the SMs were busy). This is what "GPU utilization" usually means.
- **`system.gpu.0.memory`** — *memory bandwidth* utilization (% of time the memory bus was reading/writing). **NOT capacity.** Confusingly named.
- **`system.gpu.0.memoryAllocatedBytes`** — raw bytes *allocated* on the device (i.e. VRAM fill level).
- **`system.gpu.0.memoryAllocated`** — allocated as % of total VRAM capacity (derived).
- Total VRAM on Modal's A10 = **23 028 MB** (≈ 22.5 GB).

For "can I raise batch_size?" → **look at capacity %** (`memoryAllocatedBytes / total` or `memoryAllocated`).
For "is my memory subsystem busy?" → look at `memory` (bandwidth).
For "is my compute saturated?" → look at `gpu` (SM utilization).

### Diagnosis

Capacity at ~4% → we are using **~1 GB of 23 GB of VRAM**. Compute at 38-57% → the SMs are idle roughly half the wall-clock. Memory bandwidth at 22-36% → the bus is not streaming hard either. All three point to the same root cause: **the GPU is under-fed**.

Primary suspects (in order of likely contribution):
1. **`DataLoader(num_workers=0)`** in `src/am_baseline/training/trainer.py::train_epoch` — batch construction is synchronous on the main Python thread. GPU finishes a step before the next batch is ready → idle gaps.
2. **`batch_size=512` is too small** for the A10's compute. AM's paper batch size was sized for older hardware (≈ 2019). A10's SMs chew through 512 instances in a fraction of a step, then wait. Kernel-launch overhead and the sequential decoder (N autoregressive steps) stop amortizing at this size.
3. Minor: small CPU↔GPU sync barriers in rollout / validation phases; not worth fixing first.

### Implications for future runs

Reasonable next batch-size target: **2048** (4× current). Rough VRAM estimate: linear in batch → ~3.5 GB, still <20% of capacity. Stop before 4096 only if the decoder's attention KV-cache grows non-linearly (which on TSP-20 it doesn't — `graph_size=20` keeps attention cost tiny).

Accompanying changes (apply together):
- `num_workers=4, pin_memory=True, persistent_workers=True` in `DataLoader`.
- Keep `epoch_size=1280000` — that's a gradient-update-count design choice, not a throughput one.

### Tradeoffs / caveats

- **Raising `batch_size` changes gradient noise scale.** AM's published results were at 512. For the **canonical Stage 1 run** (headline policy-vs-Stage-0 comparison) and any **final-winner re-train**, keep 512 for apples-to-apples with Stage 0's `wpqp1dpp`. Apply throughput changes to **ablation sweeps** where per-epoch wall-clock matters more than strict comparability.
- **Don't disrupt the currently-running 5 runs** — they'll finish under the old settings. New settings take effect starting with the target-norm ablation.
- Compute% on autoregressive AM decoding has a soft ceiling from sequentialization; even well-tuned runs don't hit 95%+. ~75-90% is the realistic target.

### Recording protocol

If compute/VRAM saturation shifts significantly in future runs (e.g., after changing batch size), append a new dated row to the table above with run id so we retain a historical reference.

### Throughput refactor — applied 2026-04-23, verified on target-norm ablation 2026-04-24

Code changes to unblock the A10 feeding bottleneck before the target-norm ablation:

- `src/am_baseline/training/trainer.py::train_epoch` — `DataLoader(...)` now passes `num_workers=opts.num_workers`, `pin_memory=opts.use_cuda` (so CPU dev runs don't warn about pinning without CUDA), `persistent_workers=(num_workers > 0)` (avoids respawn across 100 epochs).
- `src/am_baseline/config.py` — added `num_workers: int = 4` dataclass field and matching `--num_workers` CLI arg. Default 4 (A10 container's vCPU count); set `--num_workers 0` for CPU smoke or Windows-dev to avoid fork overhead.
- `pin_memory` is gated on `use_cuda` rather than hardcoded True, so local CPU runs are unaffected.

**Batch-size guidance for the target-norm ablation** (A10, TSP-20; not apples-to-apples with Stage 0, so free to change):
- **Default target: `batch_size=2048`** (4× current). Rough VRAM ~2.5 GB (~11% capacity); well inside safe territory. Pair with `num_workers=4, pin_memory=True, persistent_workers=True` (automatic with the refactor above).
- **Aggressive option: `batch_size=4096`** (~4.5 GB, ~20% capacity) — use if the first ablation run still shows <70% compute utilization.
- **LR scaling:** keep `lr_model=1e-4` initially. If val_avg_cost regresses vs canonical by epoch 5, bump to `lr_model=2e-4` (linear-scaling rule-of-thumb for 4× batch). Do NOT re-tune before seeing regression; Adam handles modest batch changes well.
- **Epoch size stays at 1.28M:** it's a gradient-update budget, not a throughput lever. At batch 2048, that's 625 updates/epoch × 30 epochs = 18 750 updates — still plenty for a 16K-param value head.
- **Canonical / headline runs keep `batch_size=512`** (apples-to-apples with Stage 0's `wpqp1dpp` and Stage 1's canonical `xg7t2dlb`). Changing batch changes gradient-noise scale → different converged policy.

**Expected vs actual A10 utilization after refactor** (targets set pre-run; measured on target-norm ablation):
- Compute%   — expected 75–90%, **actual mean 79.5 / 80.5, max 89 / 90** ✓
- MemBW%     — expected 40–60%, **actual mean 51.2 / 51.9, max 61 / 61** ✓
- VRAM cap%  — expected ~11% at bs=2048, **actual 10.2%** ✓
- Wall time  — **~99 s/epoch mean at bs=2048** (epoch min 68 s is the first-epoch worker-warmup outlier; steady-state ≈ 100 s).
- Soft ceiling on compute% comes from the N=20 sequential decoder; 95%+ is not realistic without kernel-level work (torch.compile, CUDA graphs) which is out of scope for Stage 1.

**Validation protocol used:** pulled NVML via `wandb.Api().run(...).history(stream='events')` over the mid 50% of the run (95 nonzero-compute samples each); recorded as the two bold rows in the snapshot table above.

---

## Known Issues

- **[fixed 2026-04-23, pre-run] V_AFTER vs V_CURRENT target alignment.** Initial implementation used `targets[i] = cost_to_go(edge_costs)[i]` which is V_AFTER semantics — it predicts future cost *excluding* the edge traversed at step `i`. Correct semantics for AlphaGo-Zero-style MCTS usage (Stage 3+) is V_CURRENT: `target[i]` should include the edge taken at step `i`. Off by exactly one edge (`edge_costs[i-1]`) for every `i ≥ 1`. Spot check that surfaced it: states `s_0` (empty) and `s_1` (one node visited, zero edges traversed) have identical realized future cost, but V_AFTER gave them different targets. **Fix:** new helper `value_targets_from_edges` pads a leading 0 to `edge_costs` before the reverse cumsum and drops the trailing entry. Invariants unit-tested; runs on Modal used the fixed version.
- **[op] λ=1.0 sweep required 3 Modal submissions** (W&B runs `0gfvqjh9` → `4a6uc50g` → `9p6h52iu`) before one reached epoch 29. Root cause uninvestigated; possibilities include Modal container preemption or a transient I/O failure. The other three sweeps (λ = 0.1 / 0.5 / 2.0) and the canonical run finished on the first try. Not treated as blocking; watch for a repeat on future runs and escalate if the pattern holds.
- **[op] Local log streaming from Modal CLI is lossy.** All 5 runs produced "WARNING: Logs may not be continuous" messages in the local `tee` log; the canonical log in particular stopped at 80 lines (epoch 0, batch 250) while server-side training completed all 100 epochs cleanly. This is a Modal CLI streaming artifact, not a training issue. **Authoritative source of training results is W&B, not the local log files.**
- **[wontfix] Windows GBK encoding** on `R²` superscript bit us once during development — resolved by using ASCII `R2`/`R^2` everywhere in console strings. For Modal-launching commands from Windows git-bash, prepend `PYTHONIOENCODING=utf-8 PYTHONUTF8=1` to avoid Modal CLI Unicode (✓) chokes.
- **[design note, downstream] Glimpse shape subtlety:** `decoder.decode_step(return_glimpse=True)` returns glimpse of shape `(batch, embed_dim)` (both squeezes in the return chain are applied), NOT `(batch, 1, embed_dim)`. Initial collection used `torch.cat` and blew up a shape-mismatch in the value head — changed to `torch.stack` on dim 1 to produce `(batch, N, embed_dim)`. Flagged for Stage 2 MCTS callers of the same `decode_step` path, who will need the same stack-vs-cat awareness.

---

## What's left

- **TSP-50 verification (optional, 1 run).** Confirms value head generalizes beyond TSP-20 without re-tuning. Defer until needed for Stage 2/3 scale-up. Not on the critical path — Stage 1 success criteria are fully met on TSP-20.
- ~~**Target-norm ablation (`bl` vs `sqrt_n`)**~~ **DONE 2026-04-24** — `bl` retained as canonical; both schemes satisfy the criteria, `bl` has better R² (especially early bucket), `sqrt_n` has marginally smaller residual bias. See *Target-norm ablation — FINISHED* block under Results.
- ~~**Throughput refactor before next long run**~~ **DONE 2026-04-23, verified 2026-04-24** — compute% 80, memBW% 51, VRAM 10.2% at bs=2048; all inside predicted targets.

**Stage 1 is complete with respect to the plan's required deliverables.** Optional TSP-50 sanity check can run opportunistically.

## Notes

- Plan file mirrored here: `_plans/stage1_plan.md`
- Original plan file (Claude Code plans dir): `C:\Users\Jun18\.claude\plans\we-have-done-stage-lucky-perlis.md`
- Design discussion captured in the plan's **Context** + **Design Summary** sections — key decision is that `v` is auxiliary only (does not enter the policy gradient), isolating the value-head variable for Stage 1's success criteria.
- All nine W&B run IDs for Stage 1 (5 primary + 2 aborted λ=1.0 attempts + 2 target-norm ablation): `xg7t2dlb` (canonical), `hnqxy48u` (λ=0.1), `dw6i9bf0` (λ=0.5), `0gfvqjh9` (λ=1.0 aborted), `4a6uc50g` (λ=1.0 aborted), `9p6h52iu` (λ=1.0), `hg8o92i1` (λ=2.0), `fq82w24n` (ablation `bl`), `rnjgavla` (ablation `sqrt_n`). All under project `lejun/am-alphagozero`.
