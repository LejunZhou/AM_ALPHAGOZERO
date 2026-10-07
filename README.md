# AM AlphaGoZero

An Attention Model policy for the Traveling Salesman Problem, refined by Monte
Carlo Tree Search at test time and improved further by an AlphaGo Zero–style
self-play training loop.

The Attention Model proposes likely next cities in a tour. MCTS searches
around those proposals to find lower-cost tours than greedy decoding. The same
search is then recycled into training data: the model learns from its own
search statistics, the way AlphaGo Zero learned from MCTS-guided self-play.

**Direction (2026-10-07).** The TSP chapter is closed.
- On deterministic TSP, the AlphaGo Zero–style loop did not beat REINFORCE.
- One round of search distillation beat continued REINFORCE at equal GPU time, but
  kept only about a fifth of the search's gain.

The project now tests the thesis on the **stochastic Prize-Collecting TSP**. There,
a single rollout is a noisy value estimate, so a learned value function can add
information rather than only save compute. See the revised
[proposal](proposal.md) (TSP chapter outcome and Stage 6) and the
[Stage 6 plan](_plans/stage6_spctsp_plan.md).

Earlier status (2026-09-25): from-scratch search-supervised greedy
policies still trail the REINFORCE baselines on the matched TSP-20/50 checks.
The latest [research audit](_progress/research_diagnosis_20260925_progress.md)
identifies an evaluation-normalization omission and a value-input limitation.
Correctly scaled value-guided search improves on greedy in the audit, although
rollouts remain stronger. Consult that report before using older value-failure
or sample-efficiency claims. The proposed fixes and training ablations are pending.

The [frozen-policy value repair experiment](notebooks/colab_value_evaluator_repair.ipynb)
ran on Colab (2026-09-30). Missing inputs, not capacity, limited the value head:
explicit endpoint and geometry features cut its TSP-50 decision regret 30-fold and
recovered most of the rollout's search gain at lower cost. The gain is concentrated
in the last ten cities, so rollouts remain the stronger teacher; see the
[results record](_progress/value_evaluator_repair_colab_progress.md).

The [student isolation experiment](notebooks/colab_student_isolation.ipynb) ran on
Colab (2026-10-04). It asked whether a REINFORCE-trained TSP-50 model can absorb the
improvements a rollout-MCTS teacher finds.
- One round of distillation from the search beat continued REINFORCE at the same wall
  time for all three training targets. The best target cost 0.0131 less than Stage 1
  greedy; REINFORCE improved by 0.0034.
- The student kept only about a fifth of the teacher's 0.0635 gain, below the 25%
  floor set before the run.
- The outcome is therefore STOP for the TSP claim. See the
  [results record](_progress/student_isolation_colab_progress.md).

## How MCTS works

TSP construction is treated as a sequence of small decisions: each state is a
partial tour, each action picks one unvisited city, and the search builds a
tree of partial tours rooted at the current state. Every simulation cycles
through four phases.

```
                       (root: current partial tour)
                                  │
       ┌─── 1. Selection ─────────┤
       │     PUCT walk down       │
       │     expanded children    │
       │                          ▼
       │                     (leaf)
       │                          │
       │     2. Expansion ────────┤   AM decoder → priors P
       │                          │
       │     3. Evaluation ───────┤   value head OR greedy rollout
       │                          │
       └─── 4. Backup ◄───────────┘   propagate value up,
                                       update N, W, Q on each edge
```

**1. Selection (PUCT).** Starting from the root, descend by picking the action
that maximizes the AlphaGo Zero PUCT score
(`src/am_baseline/search/puct.py:8-11`):

```
a* = argmax_a [ Q(s, a) + c_puct · P(s, a) · sqrt(Σ_b N(s, b)) / (1 + N(s, a)) ]
```

The first term exploits actions with good measured value `Q`; the second
explores actions with high prior `P` and low visit count `N`. The exploration
constant is `c_puct = 0.05` (`mcts.py:41`) — much smaller than AlphaGo's 1.0,
because routing has subtler cost differences between near-optimal actions.

**2. Expansion.** When selection reaches an unexpanded leaf, run one decoder
forward pass on its partial tour. The softmax over legal next cities becomes
the prior `P` (`mcts.py:387-407`, renormalized over legal actions only —
correctness invariant 4 in `mcts.py:1-19`).

**3. Evaluation.** Score the leaf one of two ways, controlled by
`cfg.leaf_eval`:

- `value_head`: a small MLP (`src/am_baseline/model/value_head.py:5-26`)
  predicts the normalized cost-to-go directly from the current encoder state.
- `rollout`: finish the partial tour greedily under the AM policy and use the
  realized cost.

Both evaluators are normalized by the per-instance greedy baseline
(`value_norm='bl'`, `mcts.py:60`) so different TSP instances share a
comparable value scale at the leaves. The `rollout` evaluator is the test-time
default; the `value_head` evaluator is what makes the training loop possible
(see below).

**4. Backup.** Convert cost to value by negation —
`value = − total_normalized_cost` (`mcts.py:329-334`, invariant 3) — so that
"larger is better" selection logic still solves a minimization problem. Walk
the simulation path back to the root, incrementing `N`, accumulating `W`, and
recomputing `Q = W / N` on each edge.

After `K` simulations, MCTS commits to the most-visited root action and
*reuses the chosen subtree* as the next root (`tree_reuse=True`,
`mcts.py:248-256`). Search statistics compound across tour steps instead of
being thrown away.

## From search to training: the AlphaGo Zero loop

The same MCTS that improves a single tour can also improve the model itself.
Each tour step produces a search-refined action distribution and, once the
tour is finished, a realized cost. Both are training signal.

```
       ┌────────────────────────────────────────────────────────┐
       │                                                        │
       ▼                                                        │
   best model θ★                                                │
       │                                                        │
       │   self-play on M fresh TSP instances                   │
       │   with MCTS guided by θ★                               │
       ▼                                                        │
   per state s_t:  π_t = N(s_t, ·) / Σ N(s_t, ·)                │
                   z_t = (tour_cost − lengths_t) / bl_val       │
       │                                                        │
       ▼                                                        │
   replay buffer (stratified by tour step)                      │
       │                                                        │
       ▼                                                        │
   SGD on  L = − Σ_a π_t(a) log p_θ(a|s_t)  +  λ · ‖v_θ − z_t‖² │
       │                                                        │
       ▼                                                        │
   gate: paired t-test vs θ★ on a fixed validation set ─────────┘
        accept → θ★ ← θ
```

**Self-play.** Each iteration, the coach generates `M` fresh TSP instances
and plays each one under MCTS guided by the current best model `θ★`
(`src/am_baseline/training/coach.py:641-767`). At every tour step the solver
snapshots the root visit counts (`return_root_visits=True`) and stores the
realized per-state cost-to-go.

**Targets.** For each visited state `s_t`:

- Policy target `π_t = N(s_t, ·) / Σ N(s_t, ·)` — the raw MCTS visit
  distribution at that step. Action selection during self-play uses a
  separate temperature schedule (`step30`); the *training target* stays at
  τ=1 by design, so policy distillation always learns from a well-defined
  distribution.
- Value target `z_t = (tour_cost − lengths_t) / bl_val` — the per-state
  cost-to-go, normalized by the same per-instance baseline used inside MCTS.
  Reuses Stage 1's `value_targets_from_edges`
  (`src/am_baseline/utils/tensor_ops.py:57-78`) so the value head is trained
  on the exact quantity it predicts at MCTS leaves.

**Replay buffer.** A stratified-by-step ring buffer of pre-allocated tensors
(`coach.py:40-436`) — fixed memory footprint, deterministic eviction.

**Distillation.** A combined cross-entropy + MSE step
(`src/am_baseline/training/trainer.py:240-356`):

```
L = − Σ_a π_t(a) · log p_θ(a | s_t)   +   λ_v · ‖v_θ(s_t) − z_t‖²
```

with L2 weight decay 1e-4 and Adam at lr=1e-4 (matches the warm-start
checkpoint, so the optimizer does not re-destabilize a converged model).

**Validation and gating.** Every few iterations, the working model is
evaluated by greedy rollout on a fixed validation set and compared to `θ★`
with a paired t-test at α=0.05 (`coach.py:1079-1097`). Only on accept does
the working model become the new `θ★`; subsequent self-play uses the
promoted model.

The loop then repeats: better `θ★` produces better self-play, which produces
better targets, which produces a better `θ★`.

## Repository layout

```
src/am_baseline/
  model/        # AttentionModel encoder/decoder + value head
  problem/      # TSP state, edge-cost helpers
  search/       # MCTS — Python reference + C++ backend (mcts_cpp/)
  training/     # Stage 1 trainer; Stage 4 coach + replay buffer
  utils/        # tensor ops, value-target construction
src/scripts/    # CLI entry points (train.py, run_mcts.py, train_alphazero.py)
ref/            # reference implementations (Kool et al. AM, alpha-zero-general, KataGo)
```

## Running the experiments

```bash
conda activate AM_AlphaGoZero
pip install -e .
```

**Student isolation — Colab (Stage 5 §I Step 2):**

1. Upload [`notebooks/colab_student_isolation.ipynb`](notebooks/colab_student_isolation.ipynb)
   to Colab and select a GPU runtime (L4 recommended).
2. Place the Stage 1 TSP-50 checkpoint and its `args.json` at
   `MyDrive/AM_AlphaGoZero/checkpoints/stage1_tsp50_with_value/`.
3. Run all cells with `PROFILE = "main"` (the screen). If the report says
   `CONTINUE`, rerun with `PROFILE = "confirm"`; it reuses the same Drive folder.

The notebook embeds a hash-checked source snapshot and compiles the C++ search in
the runtime; no repository push is needed. A frozen Stage 1 policy plus batched
rollout MCTS labels 32,768 training graphs once. Students start from Stage 1 and
learn from visit counts, a completed-Q improved policy, or the best tour found;
validation picks checkpoints and learning rates. Continued REINFORCE with the
checkpoint's optimizer and rollout baseline gets the same wall time. Runs resume
from Drive under `outputs/student_isolation/<run_name>/`. CLI:
`PYTHONPATH=src python src/scripts/run_student_isolation.py --profile cpu_smoke
--checkpoint <epoch-99.pt> --output_dir <dir>`. Regenerate the notebook with
`python src/scripts/build_student_isolation_notebook.py`.

**Frozen-policy value evaluator repair — Colab diagnostic:**

1. Upload [`notebooks/colab_value_evaluator_repair.ipynb`](notebooks/colab_value_evaluator_repair.ipynb)
   to Colab and select a GPU runtime (CPU is supported).
2. Place the F.6.1.6 checkpoint at
   `MyDrive/AM_AlphaGoZero/checkpoints/f616_400iter_step_decay/iter-361_accepted.pt`.
   The notebook explicitly uses its raw-output convention, `original_value_norm='none'`.
3. Run the cells in order. The default `main` preset uses 1,024 training graphs,
   128 validation graphs, 100 sibling-test graphs, and 100 separate search-test
   graphs; each of three architectures is fitted with seeds 0, 1, and 2 for 30
   epochs. `pilot` and `cpu_smoke` presets provide smaller runs.

The notebook embeds a hash-checked source snapshot, so no repository push,
clone, C++ build, or W&B login is required. Its `EXPERIMENT` switch also runs the
Stage 1 TSP-50 checkpoint (the decision run). It freezes all policy parameters and
normalization buffers, caches raw greedy-completion targets for all children of
selected greedy/sampled prefix states, and compares the original head, a wider
original head, and a head with explicit endpoint/remaining-set features.
Held-out decisions use exact DP at up to 20 remaining nodes. Search comparisons
include prior-only, geometric, and rollout evaluators, both equal-K and with K
calibrated on separate timing instances. Actual held-out times are reported;
calibration is not a strict wall-clock deadline.

Runs resume from Drive under `outputs/value_repair/<run_name>/`. Change the run
name when changing settings. The final cell exports raw results and plots to a
compact ZIP; feature caches and resumable heads stay on Drive. The source runner
is [`src/am_baseline/experiments/value_repair.py`](src/am_baseline/experiments/value_repair.py).
For CLI use, pass the notebook's saved `config.json` to
`PYTHONPATH=src python -m scripts.run_value_repair --config <path> --phase all`.
After editing the experiment sources, regenerate the embedded notebook with
`python src/scripts/build_value_repair_notebook.py`.

**Stage 1 — REINFORCE + auxiliary value-head training** (produces the warm-start
checkpoint that everything else builds on):

```bash
PYTHONPATH=src python src/scripts/train.py \
  --graph_size 20 \
  --n_epochs 100 \
  --run_name stage1_tsp20
```

**Stage 3 — test-time MCTS inference** (no training; loads a checkpoint and
benchmarks tour quality):

```bash
PYTHONPATH=src python -m scripts.run_mcts \
  --model outputs/tsp_20/stage1_tsp20/epoch-99.pt \
  --graph_size 20 \
  --val_size 1000 \
  --n_simulations 200 \
  --c_puct 0.05 \
  --leaf_eval rollout \
  --tree_reuse \
  --backend cpp_batch \
  --output_csv outputs/stage3/tsp20_mcts.csv
```

The Python backend is the readable reference; `--backend cpp_batch` runs the
C++ implementation in `src/am_baseline/search/mcts_cpp/` and is roughly 20×
faster.

**Stage 4 — AlphaGo Zero–style self-play training** (the loop described
above):

```bash
PYTHONPATH=src python src/scripts/train_alphazero.py \
  --load_path outputs/tsp_20/stage1_tsp20/epoch-99.pt \
  --graph_size 20 \
  --n_iterations 100 \
  --M_instances 1000 \
  --n_simulations_train 50 \
  --gate_mode ttest \
  --temperature_schedule step30 \
  --dirichlet_epsilon 0.25
```

This example warm-starts from Stage 1. Omitting `--load_path` starts from random
initialization; the main Stage 5 experiments use that setting. Warm-start and
from-scratch results answer different training-efficiency questions.

## Further reading

- `proposal.md` — research questions, methods, and expected outcomes.
- `_plans/stage{0,1,2,3,4,5}_plan.md` — design documents per stage.
- `_progress/stage{0,1,2,3,4,5}_progress.md` — running results, decisions, and
  open questions.
- `ref/` — reference codebases the project draws on.
