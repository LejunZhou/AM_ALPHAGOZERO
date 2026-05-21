# Stage 5 §H — AlphaGo-Fan/Lee mixed leaf evaluation on TSP-20 — progress

Mirror of [`_plans/stage5_mix_leafeval_plan.md`](../_plans/stage5_mix_leafeval_plan.md).
Tracks implementation, smoke tests, and the Phase A/B/C experimental sweep
for `leaf_eval='mix', mix_lambda=λ` (`v_leaf = λ · v_head + (1−λ) · v_rollout`).

Status convention mirrors [`stage5_progress.md`](stage5_progress.md): each
sub-section header carries `**OPEN** / **IN FLIGHT** / **COMPLETE <date>**`.
"Iter N" refers to AGZ self-play iteration unless specified.

---

## Status overview

| Phase | Status | Date | Headline result |
|---|---|---|---|
| H.1 Implementation (Python + C++ + plumbing) | **COMPLETE 2026-05-21** | 2026-05-21 | mix mode wired across mcts.py, mcts.cpp, BatchSearch, Coach, train_alphazero CLI, val_stage4_mcts, probe_mcts_decomp, Modal entrypoint |
| H.2 Smoke tests (λ=0 parity, λ=1 parity, C++↔Py at λ=0.5) | **COMPLETE 2026-05-21** | 2026-05-21 | `smoke_mix.py` M1/M2/M3 all pass with \|Δ\|=0.000e+00 |
| H.3 Phase A — Colab T4 inference λ-sweep on F.6.1.6 | OPEN | — | — |
| H.4 Phase B — Modal A10 training at λ\* (K=10 step50 100-iter) | OPEN | — | — |
| H.5 Phase C — 1000-instance canonical eval | OPEN | — | — |
| H.6 Verdict + open follow-ups | OPEN | — | — |

Reference points (existing, for comparison):

| Recipe | val_avg_cost | iter / wall | source |
|---|---|---|---|
| Stage 1 canonical greedy | 3.83943 | epoch 99 | §A header |
| F.6.1 K=40 step30 lv1, iter 99 | 3.860 | iter 99 / ~2 h | §A |
| F.6.1.6 step-decay 400-iter best | 3.8578 | iter 365 / ~8 h | §B.3 |
| **§B.4 K=10 lv0 step50 100-iter best** | **3.8576** | **iter 100 / 33 min** | **§B.4** |
| §D.4 lv0 chain best (iter 197) | 3.8486 | iter 197 / ~3 h | §D.4 |
| §C.3 F.6.1.6 + MCTS K=40 rollout (val) | 3.834 | val-only | §C.3 |
| §C.3 F.6.1.6 + MCTS K=40 vh (val) | 3.868 | val-only | §C.3 |

§B.4 is the **direct apples-to-apples baseline** for Phase B: same K=10
step50 100-iter, only `leaf_eval` and `lambda_v` differ.

---

## §H.1 Implementation — **COMPLETE 2026-05-21**

Landed in the same commit as this progress doc. Pybind11 extension rebuilt
(`pip install -e .`). Smoke verification under §H.2.

### H.1.1 Python backend (`src/am_baseline/search/mcts.py`)
- [x] `MCTSConfig`: added `mix_lambda: float = 0.5`; extended `leaf_eval` doc.
- [x] `VALID_LEAF_EVAL`: added `'mix'`.
- [x] `_validate_config`: accepts `'mix'`; `0 ≤ mix_lambda ≤ 1` enforced;
      requires `model.value_head is not None`; extends the
      `value_norm='sqrt_n'` incompatibility to reject `'mix'`.
- [x] `_expand`: `'mix'` branch returns `λ·v_head + (1−λ)·v_rollout`.
- [x] `_populate_priors`: mirrored — root `node.v_estimate` is on the same
      scale as backed-up Q.

### H.1.2 C++ backend (`src/am_baseline/search/mcts_cpp/`)
- [x] `mcts.hpp::Config`: added `mix_lambda` field.
- [x] `mcts.cpp::Config::from_python`: `get_or<double>("mix_lambda", ...)`
      with range check.
- [x] `Solver::populate_priors`, `Solver::expand`, `Solver::evaluate_pending`:
      `need_value` true for `value_head|mix`; `'mix'` branches blend
      `convert_value_head_output(...)` with `rollout_remaining_real/_many(...)`.
- [x] `BatchSearch::Impl::collect_request` (both call sites) emits requests
      with `need_value` / `need_rollout` both true under mix.
- [x] `BatchSearch::Impl::apply_result`: counters bumped for both branches;
      v_head / v_rollout blended via the new `convert_v_head` helper lambda.
- [x] `solver.py::CppMCTSSolver.solve_instance`: forwards `rollout_evaluator`
      when `leaf_eval ∈ {'rollout','mix'}`. `mix_lambda` flows through
      `_cfg_dict` via `asdict(self.cfg)` (no manual key).
- [x] Rebuild: `pip install -e .` (cp312 `.pyd` timestamp 2026-05-21 01:22).

### H.1.3 Training-side plumbing
- [x] `train_alphazero.py`: `--leaf_eval` choices extended to include
      `'mix'`; `--mix_lambda` (float, default 0.5) added.
- [x] `coach.py::make_self_play_config`: `mix_lambda` parameter forwarded
      into `MCTSConfig`. Coach call site at iteration loop passes
      `getattr(opts, 'mix_lambda', 0.5)`.
- [x] `val_stage4_mcts.py`: `--leaf_eval` choices extended;
      `--mix_lambda` added; `_build_mcts_config` honors `--match_train`
      by reading `mix_lambda` from sibling `args.json`.

### H.1.4 Modal entrypoint
- [x] `modal_run_train_alphazero.py::run_tsp20_k10_mix_step50(mix_lambda, timestamp)`
      added. Single recipe-delta from B.4 (`run_tsp20_k10_lv0_step50`):
      `leaf_eval=mix`, `lambda_v=1.0`, `--mix_lambda <λ>`. Run-name tag
      embeds λ (e.g. `tsp20_k10_mix0p5_step50_100iter_*`).

### H.1.5 Probe / eval scripts
- [x] `probe_mcts_decomp.py`: `--leaf_eval` choices extended; `--mix_lambda`
      added; forwarded into the constructed `MCTSConfig`.
- [x] New `src/scripts/eval_tsp20_mix_lambda_sweep.py`: Colab T4 entrypoint.
      Sweeps a CLI-supplied λ-grid (default `0.0,0.25,0.5,0.75,1.0`) at
      K=40 val_size=10000 val_seed=42 against a single checkpoint, writes
      one CSV row per λ + a `.npz` of per-instance costs.

---

## §H.2 Smoke tests — **COMPLETE 2026-05-21**

`src/scripts/smoke_mix.py` covers the three correctness invariants.
Random-init AttentionModel (`graph_size=10, embedding_dim=32,
n_encode_layers=2, n_heads=4, value_enabled=True`), one fixed TSP-10
instance, seeded MCTSConfig (`n_simulations=16, c_puct=1.0, ε=0, τ=0,
value_target_norm='none'`). Results from the live run on 2026-05-21:

| Test | a | b | \|Δ\| | Verdict |
|---|---|---|---|---|
| M1 rollout ↔ mix(λ=0)              | 3.7420294285 | 3.7420294285 | 0.000e+00 | OK |
| M2 value_head ↔ mix(λ=1)           | 4.2331485748 | 4.2331485748 | 0.000e+00 | OK |
| M3 Python MCTSSolver ↔ C++ Solver at mix(λ=0.5) | 3.5263400078 | 3.5263400078 | 0.000e+00 | OK |

M1 confirms the value-head contribution is zeroed when λ=0; M2 confirms
the rollout contribution is zeroed when λ=1; M3 confirms the C++ blend
matches Python bit-for-bit at λ=0.5 (the production training value).

### H.2.4 No-NaN / sign-convention sanity — deferred
Not required for correctness given M1–M3 are exact-match; the blend
formula `λ·v_head + (1−λ)·v_rollout` with both inputs ≥ 0 (cost-to-go)
trivially preserves non-negativity. Will re-open if Phase A surfaces
anomalies.

### Repro
```
PYTHONPATH=src python -m scripts.smoke_mix
```

---

## §H.3 Phase A — Colab T4 inference probe — **OPEN**

Target checkpoint: F.6.1.6 winner —
`outputs/tsp_20/f616_400iter_step_decay_20260507T101222_20260507T101229/iter-361_accepted.pt`.

Grid: λ ∈ {0.00, 0.25, 0.50, 0.75, 1.00}, K=40, ε=0, τ=0,
`val_size=10000, val_seed=42`.

Reference anchors (must match within per-instance SE):
- λ=0 ≈ 3.834 (§C.3 K=40 rollout)
- λ=1 ≈ 3.868 (§C.3 K=40 vh)

### Results table (to fill in)

| λ | val_avg_cost | std | wall (min) | fwd_decode | fwd_value | fwd_rollout |
|---|---|---|---|---|---|---|
| 0.00 | — | — | — | — | — | — |
| 0.25 | — | — | — | — | — | — |
| 0.50 | — | — | — | — | — | — |
| 0.75 | — | — | — | — | — | — |
| 1.00 | — | — | — | — | — | — |

Decision rule:
- Monotone curve (λ=0 strictly wins) → mix offers no inference benefit on
  F.6.1.6's biased head; proceed to Phase B *anyway* because mix at
  training time may yield a less-biased head.
- Interior λ\* wins → pick top 1–2 λ into Phase B.

W&B / Colab notebook link: (to fill in)

---

## §H.4 Phase B — Modal A10 training — **OPEN**

Recipe (`run_tsp20_k10_mix_step50` at λ = λ\* from §H.3):
```
graph_size=20, n_iterations=100, M_instances=1000,
n_simulations_train=10, train_steps_per_iter=200,
buffer_capacity=5000, batch_size=512, gate_every=1, gate_mode=ttest,
temperature_schedule=step10, val_size=10000, val_seed=42,
leaf_eval=mix, mix_lambda=<λ*>, lambda_v=1.0,
max_grad_norm=1.0, value_target_norm=none,
lr_model=5e-4, lr_decay=0.2, lr_decay_step_size=50,
weight_decay=0.0, dirichlet_epsilon=0.25, dirichlet_alpha_factor=10.0,
mcts_batch_size=1000
```

Wall budget: ~1.5–2 h per λ on A10. Modal cost: ~$5–10 per run.

### Results table (to fill in)

| λ\* | run_name | wandb | iter best | val best | gate accept rate | wall |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |

Per-iter signals to log: `mcts_s`, `fwd_count_{decode,value,rollout}`,
`val_avg_cost`, `value_loss_mean`, `policy_loss_mean`,
`mean_entropy_policy`, `mcts_delta_vs_greedy_mean`.

### Sample-efficiency check
- [ ] Best val at iter 100 vs §B.4 (3.8576).
- [ ] Per-iter wall vs §B.4 (33 min total ≈ 20 s/iter). Mix adds 1 vh MLP
      call per leaf; should be ≤ 5 % wall overhead.

---

## §H.5 Phase C — 1000-instance canonical eval — **OPEN**

[`src/scripts/eval_tsp20_full_comparison.py`](../src/scripts/eval_tsp20_full_comparison.py)
(1000 instances, seed=20260430, Gurobi-optimal).

### Per-checkpoint comparison (to fill in)

| checkpoint | greedy | MCTS K=10 mix | MCTS K=40 mix | MCTS K=100 mix | gap-vs-Gurobi (K=40) | n_optimal/1000 |
|---|---|---|---|---|---|---|
| F.6.1.6 best | — | — | — | — | — | — |
| lv0 iter-199 | — | — | — | — | — | — |
| **mix λ\*** | — | — | — | — | — | — |

Paired t-test (1000 instances): mix vs lv0 iter-199 at K=40 rollout (lv0's
trained leaf eval) — report t, p, win/tie/loss split.

Success threshold: greedy ≤ 3.85 AND MCTS K=40 ≤ 3.835.

---

## §H.6 Verdict + follow-ups — **OPEN**

To fill in after §H.5. Expected decisions:

- **If mix wins**: log as the new TSP-20 frontier; consider TSP-50 port
  (§E sibling). Open question: per-step adaptive λ.
- **If mix ties λ=0 (pure rollout)**: closes the AGFan/Lee question for
  this regime; supports the §D verdict that the value head doesn't
  contribute useful signal at TSP-20 under the current value target.
- **If mix loses**: investigate whether F.6.1.6-style vh bias persists
  under mix self-play (value-head calibration probe at iter 50 / iter 100).

---

## Cross-references

- Plan: [`_plans/stage5_mix_leafeval_plan.md`](../_plans/stage5_mix_leafeval_plan.md).
- Algorithmic spec: [`_plans/stage4_algorithm_spec.md`](../_plans/stage4_algorithm_spec.md) §4.1.5 (AGFan/Lee row).
- Stage 5 antecedents in [`_progress/stage5_progress.md`](stage5_progress.md):
  §B.4 (K=10 step50 sibling recipe), §C.3 (vh leaf-eval bias),
  §D (lv0 ablation), §D.4 (current TSP-20 frontier).
- Memories: `project_alphagozero_value_head_leaf_eval_bias.md`,
  `feedback_iterate_on_results.md`.
