# Stage 0 Plan: Reproduce AM Baseline on TSP

**Created:** 2026-04-22  
**Status:** Not started

## Context

Stage 0 is the foundation for the entire AM+AlphaGoZero project. We need a working, verified Attention Model that matches the published TSP results from Kool et al. (ICLR 2019), structured so we can extend it with value heads (Stage 1) and MCTS (Stage 2). The reference code at `ref/attention-learn-to-route-master/` is a working implementation but has compatibility issues (Python 3.10+, CPU-only runs, deprecated APIs) and mixes 6 problem types. We'll port the TSP-relevant subset into a clean `src/` structure.

Pretrained checkpoints exist at `ref/attention-learn-to-route-master/pretrained/tsp_{20,50,100}/epoch-99.pt` — we can use these for immediate verification of our ported code.

---

## Target Architecture: `src/am_baseline/`

```
src/
  am_baseline/
    __init__.py
    config.py                  # Dataclass config (replaces argparse options.py)
    model/
      __init__.py
      encoder.py               # GraphAttentionEncoder, MultiHeadAttention
      decoder.py               # Decode loop, decode_step(), get_log_p()
      attention_model.py       # Top-level: composes encoder + decoder
    problem/
      __init__.py
      tsp.py                   # TSP cost, dataset, make_state()
      state.py                 # StateTSP NamedTuple
    baseline/
      __init__.py
      baselines.py             # NoBaseline, ExponentialBaseline, RolloutBaseline, WarmupBaseline
    training/
      __init__.py
      trainer.py               # train_epoch(), train_batch(), validate(), rollout()
      logging.py               # CSV metrics logger (+ optional TensorBoard)
    evaluation/
      __init__.py
      evaluate.py              # Greedy eval, sampling-K eval
    utils/
      __init__.py
      tensor_ops.py            # compute_in_batches, do_batch_rep, move_to
      misc.py                  # torch_load_cpu, checkpoint helpers
  scripts/
    train.py                   # CLI: python -m scripts.train --graph_size 20
    evaluate.py                # CLI: python -m scripts.evaluate --model ...
    generate_data.py           # Generate fixed test datasets
```

---

## Key Design Decisions (for Stage 1/2 extensibility)

1. **Expose `decode_step()` as public** — MCTS needs to call single decoding steps, not the full autoregressive loop. Split the reference `_inner()` into `decode_step(fixed, state) -> (log_p, mask)` + a loop wrapper.

2. **Expose `encode()` separately** — MCTS encodes once, decodes many times per tree search. `model.encode(input) -> embeddings` and `model.precompute_decoder(embeddings) -> fixed` must be callable independently.

3. **Return glimpse vector from decode_step** — The decoder's attention glimpse is the natural attachment point for a value head MLP in Stage 1. Return it as optional output.

4. **Config dataclass over constructor args** — Adding `value_head_hidden_dim` etc. in Stage 1 should not require changing every call site.

---

## Compatibility Fixes (ref -> src)

| Issue | Fix |
|-------|-----|
| `collections.Iterable` (Python 3.10+) | Delete `monkey_patch.py` entirely |
| `tensorboard_logger` (unmaintained) | Replace with CSV + `torch.utils.tensorboard` |
| `torch.cuda.get_rng_state_all()` on CPU | Guard with `if torch.cuda.is_available()` |
| `torch.uint8` masks (deprecated) | Use `torch.bool` |
| `torch.arange(..., out=tensor.new())` | Use `torch.arange(..., device=tensor.device)` |
| `torch.load()` missing `weights_only` | Add explicit `weights_only=False` |
| `DataLoader(num_workers=1)` on Windows | Default to `num_workers=0` |
| Hardcoded CUDA device selection | Config-based device, auto-detect (prefer CUDA) |
| `-np.inf` in forward pass | Use `float('-inf')` |
| Non-TSP code (VRP, OP, PCTSP) | Strip entirely — TSP only for Stage 0 |

---

## Training Strategy

**Hardware:** NVIDIA RTX 4060 Laptop GPU (8 GB VRAM). Auto-detect device (prefer CUDA, fallback CPU).

**We only train TSP-20 from scratch.** TSP-50 and TSP-100 are NOT retrained — we use the pretrained reference checkpoints (`ref/.../pretrained/tsp_{50,100}/epoch-99.pt`) for evaluation only. This saves significant compute while still validating that our code loads and evaluates all sizes correctly.

| Task | batch | epoch_size | epochs | Estimated time |
|------|-------|------------|--------|----------------|
| TSP-20 training | 512 | 1,280,000 | 100 | ~2-3 hours |
| TSP-50 eval only | — | — | — | pretrained checkpoint |
| TSP-100 eval only | — | — | — | pretrained checkpoint |

A smoke-test tier (batch=32, epoch_size=640, epochs=3) remains useful for fast debugging on either CPU or GPU.

Code must still run on CPU (for quick debugging and future portability), but GPU is the primary training target.

---

## Implementation Sequence

### Phase A: Foundation (utilities, problem, config)
1. `config.py` — dataclass with all hyperparameters + CPU presets
2. `utils/tensor_ops.py`, `utils/misc.py` — port utility functions
3. `problem/state.py` — StateTSP with `torch.bool` masks
4. `problem/tsp.py` — TSP class, dataset, costs

### Phase B: Model (forward pass works)
5. `model/encoder.py` — GraphAttentionEncoder (port near-verbatim from `ref/nets/graph_encoder.py`)
6. `model/decoder.py` — decoder with `decode_step()` exposed (refactored from `ref/nets/attention_model.py` `_inner`, `_get_log_p`, `_one_to_many_logits`)
7. `model/attention_model.py` — compose encoder + decoder, expose `encode()` (thin shell over encoder + decoder)
8. **Milestone**: smoke-test forward pass:
   ```python
   model = AttentionModel(config)
   cost, ll = model(torch.rand(4, 20, 2))
   ```
9. **Milestone**: load pretrained ref weights into our model, verify same outputs on same input

### Phase C: Training infrastructure
10. `training/logging.py` — CSV metrics writer
11. `baseline/baselines.py` — all baselines in one file (resolve circular import by passing `rollout` fn as argument)
12. `training/trainer.py` — REINFORCE training loop (ported from `ref/train.py`)

### Phase D: Entry points
13. `scripts/train.py` — CLI training (replaces `ref/run.py`)
14. `scripts/evaluate.py` — CLI evaluation (simplified from `ref/eval.py`)
15. `scripts/generate_data.py` — test data generation

### Phase E: Validation
16. Smoke test — verify no crashes, loss decreases (CPU or GPU, small settings)
17. Load pretrained ref models (TSP-20/50/100), evaluate — verify matches published numbers
18. Train TSP-20 from scratch on GPU — verify convergence matches published ~3.85 greedy
19. Record TSP-20 training curves (no TSP-50/100 training — use pretrained checkpoints only)

---

## Key Reference Files

| Our file | Ported from | Key classes/functions |
|----------|------------|----------------------|
| `model/encoder.py` | `ref/.../nets/graph_encoder.py` | `SkipConnection`, `MultiHeadAttention`, `Normalization`, `MultiHeadAttentionLayer`, `GraphAttentionEncoder` |
| `model/decoder.py` | `ref/.../nets/attention_model.py` | `AttentionModelFixed`, `_inner()` -> `decode()`/`decode_step()`, `_get_log_p()`, `_one_to_many_logits()`, `_select_node()`, `_precompute()`, `_get_parallel_step_context()`, `_make_heads()` |
| `model/attention_model.py` | `ref/.../nets/attention_model.py` | `AttentionModel` (thin shell), `set_decode_type()`, `forward()`, `sample_many()`, `_calc_log_likelihood()`, `_init_embed()` |
| `problem/state.py` | `ref/.../problems/tsp/state_tsp.py` | `StateTSP` NamedTuple |
| `problem/tsp.py` | `ref/.../problems/tsp/problem_tsp.py` | `TSP`, `TSPDataset` |
| `baseline/baselines.py` | `ref/.../reinforce_baselines.py` | `Baseline`, `NoBaseline`, `ExponentialBaseline`, `RolloutBaseline`, `WarmupBaseline`, `BaselineDataset` |
| `training/trainer.py` | `ref/.../train.py` | `train_epoch()`, `train_batch()`, `validate()`, `rollout()`, `clip_grad_norms()` |
| `config.py` | `ref/.../options.py` | Dataclass replacing argparse |
| `utils/tensor_ops.py` | `ref/.../utils/tensor_functions.py` + `ref/.../utils/functions.py` | `compute_in_batches()`, `do_batch_rep()`, `sample_many()`, `move_to()` |
| `utils/misc.py` | `ref/.../utils/functions.py` | `torch_load_cpu()`, `load_model()`, `load_args()` |
| `training/logging.py` | `ref/.../utils/log_utils.py` | `log_values()` -> CSV-based logger |
| `scripts/train.py` | `ref/.../run.py` | `run()` entry point |

---

## Published Baseline Numbers (targets)

| Problem | n | Optimal (Concorde) | Greedy (gap%) | Sampling 1280 (gap%) |
|---------|---|-------------------|--------------|----------------------|
| TSP | 20 | ~3.84 | 3.85 (0.34%) | 3.84 (0.08%) |
| TSP | 50 | ~5.69 | 5.80 (1.76%) | 5.72 (0.52%) |
| TSP | 100 | ~7.76 | 8.12 (4.53%) | 7.94 (2.26%) |

Pretrained model hyperparameters (from `ref/.../pretrained/tsp_20/args.json`):
- seed=1235, baseline=rollout, batch_size=512, epoch_size=1280000
- n_epochs=100, embedding_dim=128, hidden_dim=128, n_encode_layers=3
- lr_model=1e-4, lr_decay=1.0, max_grad_norm=1.0
- bl_alpha=0.05, bl_warmup_epochs=1, tanh_clipping=10.0, normalization=batch

---

## Verification Criteria (Stage 0 done when)

- [ ] Forward pass matches reference (same weights -> same outputs)
- [ ] Pretrained TSP-20 model evaluates to ~3.85 (greedy) and ~3.84 (sampling-1280)
- [ ] TSP-20 trained from scratch converges to ~3.85 greedy (only size we train)
- [ ] Metrics CSV generated with training curves
- [ ] `encode()`, `decode_step()`, `precompute_decoder()` exposed for Stage 1/2
- [ ] Code runs on both GPU (primary) and CPU (fallback)
- [ ] `_progress/stage0_progress.md` updated with results
