# Stage 0 Progress: Reproduce AM Baseline on TSP

**Plan:** `_plans/stage0_plan.md`  
**Started:** 2026-04-22  
**Status:** Completed 2026-04-23 — TSP-20 full training converged to 3.8426 (matches AM paper), Stage 0 done

---

## Implementation Progress

### Phase A: Foundation
- [x] `config.py`
- [x] `utils/tensor_ops.py`
- [x] `utils/misc.py`
- [x] `problem/state.py`
- [x] `problem/tsp.py`

### Phase B: Model
- [x] `model/encoder.py`
- [x] `model/decoder.py`
- [x] `model/attention_model.py`
- [x] Smoke test: forward pass works
- [x] Weight loading: pretrained ref weights produce same output

### Phase C: Training
- [x] `training/logging.py`
- [x] `baseline/baselines.py`
- [x] `training/trainer.py`

### Phase D: Entry Points
- [x] `scripts/train.py`
- [x] `scripts/evaluate.py`
- [x] `scripts/modal_run_train.py` — Modal cloud GPU entry point
- [ ] `scripts/generate_data.py` (not needed — datasets generated on-the-fly)

### Phase E: Validation
- [x] Smoke test: end-to-end training (2 epochs, cost 9.57 -> 5.11), no crashes
- [x] Pretrained model evaluation matches published numbers (see below)
- [x] Baseline solver comparison (Gurobi optimal, LKH, insertion heuristics)
- [x] `scripts/eval_baselines.py` — unified evaluation with `--lkh`, `--gurobi` flags
- [x] TSP-20 full training (100 epochs) convergence verified
- [x] Training curves recorded

---

## Results

### Pretrained Model Evaluation (10,000 instances, seed=1234)
| Model | Greedy | Expected Greedy | Match? |
|-------|--------|-----------------|--------|
| TSP-20 | 3.8416 | ~3.85 | Yes |
| TSP-50 | 5.7955 | ~5.80 | Yes |
| TSP-100 | 8.0820 | ~8.12 | Yes |

### Baseline Solver Comparison (1,000 instances, TSP-20, seed=1234)

| Method | Avg Cost | Gap to Optimal | Time |
|--------|----------|----------------|------|
| Gurobi (optimal) | 3.8279 | 0% | 8.1s |
| LKH (elkai) | 3.8279 | 0.00% | 15.5s |
| AM (greedy, pretrained) | 3.8383 | 0.27% | 0.4s |
| Farthest Insertion | 3.9218 | 2.45% | 0.6s |
| Random Insertion | 3.9845 | 4.09% | 0.4s |
| Nearest Insertion | 4.3203 | 12.86% | 0.6s |
| Nearest Neighbour | 4.4680 | 16.72% | 0.02s |

Solver setup: `elkai` (LKH wrapper), `gurobipy` (restricted license, expires 2027-11-29).
Concorde not available on Windows (no pre-built wheel); LKH matches optimal anyway.
Script: `src/scripts/eval_baselines.py --lkh --gurobi`

### Training from Scratch (run `am_tsp20`, W&B id `wpqp1dpp`, seed=1234, Modal A10G)
| Size | Epochs | Final Val | Best Val | Published | Wall-clock | Avg Epoch |
|------|--------|-----------|----------|-----------|------------|-----------|
| TSP-20 | 100/100 | **3.8426** | **3.8418** (ep 94) | 3.84 | 4.52 h | 161 s |

Settings: `--graph_size 20 --batch_size 512 --epoch_size 1280000 --baseline rollout --bl_warmup_epochs 1 --n_epochs 100 --lr_model 1e-4 --seed 1234` (all AM defaults).

Training curve (val_avg_cost, every 10th epoch):
- Epoch  0: 3.9632
- Epoch 10: 3.8690
- Epoch 20: 3.8561
- Epoch 30: 3.8541
- Epoch 40: 3.8498
- Epoch 50: 3.8493
- Epoch 60: 3.8452
- Epoch 70: 3.8453
- Epoch 80: 3.8445
- Epoch 90: 3.8424
- Epoch 99: 3.8426

Smooth monotone descent, plateau ≈ epoch 60. Total improvement 0.1206 (3.04%) from init. Matches AM paper TSP-20 greedy (~3.85) / sampling (~3.84).

W&B: https://wandb.ai/lejun/am-alphagozero/runs/wpqp1dpp

---

## Infrastructure

### Modal + W&B Integration (completed)
- [x] `pyproject.toml` — added `wandb`, `modal`, `cloud` optional deps
- [x] `training/logging.py` — W&B logging alongside CSV (always-on) and TensorBoard (optional)
- [x] `config.py` — added `wandb_project`, `wandb_entity`, `wandb_mode` fields + CLI args
- [x] `scripts/train.py` — passes W&B config to MetricsLogger
- [x] `scripts/modal_run_train.py` — Modal cloud GPU entry point

### Environment
- GPU (training): Modal A10G cloud; local RTX 4060 Laptop (8 GB VRAM) for dev/eval
- Conda env: `AM_AlphaGoZero` (torch 2.11.0 cloud, 2.7.0+cu118 local)
- Baseline solvers: `elkai` (LKH), `gurobipy` (Gurobi MIP, restricted license)
- Modal: workspace `lejunzhou`, volume `am-alphagozero-volume`
- W&B: project `am-alphagozero`, entity `lejun`

---

## Known Issues

- BatchNorm `num_batches_tracked` keys are not in pretrained checkpoints (expected — not learnable parameters)
- Reference key remapping needed for loading pretrained ref models (decoder params moved under `decoder.*`)
- **[fixed 2026-04-23]** `WarmupBaseline.epoch_callback` at `src/am_baseline/baseline/baselines.py:167` dropped the return value from the inner `RolloutBaseline.epoch_callback`, so `baseline_updated` was always logged as `False`/`None`. Did **not** affect training correctness (final val matches AM paper) — only the W&B metric. Fix: capture inner return and propagate.

---

## Notes

- `decode_step()`, `encode()`, `precompute_decoder()` all exposed and tested for Stage 1/2 use
- `return_glimpse=True` in decode_step returns (128,)-dim glimpse vector — value head attachment point
