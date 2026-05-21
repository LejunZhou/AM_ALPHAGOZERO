# Stage 4 Algorithm & Formulas — Reference Spec

**Companion to:** `_plans/stage4_plan.md` (Stage 4 engineering phases, closed), `_plans/stage5_plan.md` (Stage 5 ablations + scaling, active), `_progress/stage4_progress.md`, `_progress/stage5_progress.md`.
**Purpose:** mathematical / algorithmic description of the AlphaGo-Zero-style training loop Stage 4 implements, grounded in actual code paths and mapped 1-to-1 to AGZ paper equations.
**Created:** 2026-04-30. **Refined 2026-05-13** to reflect the **F.6.1.3+lv0 locked recipe** for production runs (lr=5e-4 / wd=0 / value_target_norm='none' / ε=0.25 / step10 / leaf_eval='rollout' / λᵥ=0 / mcts_batch_size=1000 / gate_every=1 / buffer=5000) plus Track A per-row decoder, lr-override-on-resume, per-loss grad-norm split, and the MCTS wall-time optimization stack.

---

## Context

The Stage 4 plan (`_plans/stage4_plan.md`) describes the original engineering work in phases (A–G). Stage 5 (`_plans/stage5_plan.md`) extends with ablations and scaling. This document presents the loop in algorithm/equation form, grounded in actual code paths it composes:

- AM model API in `src/am_baseline/model/attention_model.py` (`encode`, `precompute_decoder`, `decode_step`, `value_head`)
- MCTS solvers in `src/am_baseline/search/mcts.py` (Python reference) + `mcts_cpp/` (production C++ backend; `CppBatchMCTSSolver` is the cross-instance batched scheduler)
- PUCT selection in `src/am_baseline/search/puct.py:7-33` and `src/am_baseline/search/mcts_cpp/mcts.cpp::Solver::select_action`
- Gating in `src/am_baseline/baseline/baselines.py:106-123`
- Value normalization + per-loss grad-norm split in `src/am_baseline/training/trainer.py::train_step_alphazero`
- Coach orchestration in `src/am_baseline/training/coach.py::MCTSCoach`

It maps onto the AlphaGo Zero paper (Silver et al., *Nature* 550, 354–359, 2017) eq. (1) + Methods §Search algorithm, with **six documented deviations from AGZ canonical** at the current F.6.1+lv0 production recipe:
1. Adam (lr=5e-4) vs SGD-momentum (canonical step-anneal)
2. wd=0 vs 1e-4
3. raw cost-to-go value target vs ±1 win/loss bits
4. ε=0.25 retained but with `step10` temperature (was ε=0/step30 in earlier F.6.1)
5. `leaf_eval='rollout'` (AlphaGo-Lee-style) vs AGZ-canonical value-net-only — this is the lv0 recipe
6. `λᵥ=0` (policy-only training, value head receives no gradient) — the lv0 ablation winner

All six validated by the F.6.0.5 → F.6.1.6 → lv0 ablation chain (see [`_progress/stage5_progress.md`](../_progress/stage5_progress.md) §A–§D). Earlier F.6.1 deviation set (4 items) is superseded.

---

## §1 Notation

| Symbol | Meaning |
|---|---|
| $N$ | TSP graph size (= 20 in Stage 4 main runs; F.6.1 TSP-50 probe extends to $N=50$) |
| $s$ | partial-tour state (StateTSP NamedTuple — `loc, dist, first_a, prev_a, visited_, lengths, i`) |
| $a$ | action = next-city index, $a \in \{0,\dots,N{-}1\}$, masked by `visited_` |
| $\theta$ | network parameters (encoder + decoder + value head); trainer's working copy |
| $\theta^\star$ | best-player parameters (used for self-play data generation) |
| $f_\theta(s) \to (\mathbf{p}, v)$ | dual-head AM: $\mathbf{p} \in \Delta^{N-1}$ via softmax over decoder logits, $v \in \mathbb{R}$ from value head |
| $\pi_t \in \Delta^{N-1}$ | **Training target** — temperature-1 normalized visit distribution at root of tour-step $t$, $\pi_t(a) = N(s_t,a)/\sum_b N(s_t,b)$ (raw normalized; richer than action-selection $\tau$-schedule, see §4.2) |
| $\sigma_t \in \Delta^{N-1}$ | **Action-selection distribution** at tour-step $t$, $\sigma_t(a) \propto N(s_t,a)^{1/\tau_t}$ with `step30` $\tau$-schedule. The played action $a_t \sim \sigma_t$ |
| $z_t \in \mathbb{R}_+$ | **Per-state value target** — V_CURRENT cost-to-go from partial state $s_t$. **Three normalization modes** via `--value_target_norm` ∈ {`bl`, `none`, `sqrt_n`}; F.6.1 default is `none` (raw cost-to-go, see §3.7). |
| $\alpha_\theta$ | MCTS solver wrapping current $f_\theta$; given $s_0$, returns $(\text{tour}, z, \{\pi_t\}_{t=0}^{N-1})$ |
| $\mathcal{D}$ | replay buffer: ring-buffer of instance records $\{(s_t, \pi_t, z_t)\}_{t=0}^{N-1}$, fixed capacity in *instances* (default 5000 = ~5-iter window at M=1000) |
| $K$ | MCTS simulations per move (F.6.1 K=20 default; K=40 / K=50 also tested) |
| $M$ | self-play instances per iteration (= 1000; F.6.1.0 confirmed this saturates per-iter train budget) |
| $\tau$ | sampling temperature for MCTS root action selection |
| $c_\text{puct}$ | PUCT exploration constant (= 0.05, Stage 2-locked) |
| $\varepsilon, \alpha$ | Dirichlet noise weight and concentration. **F.6.1.3+lv0 default $\varepsilon=0.25$** with `step10` τ-schedule (paired choice — see §4.3). $\alpha = 10/N$ retained ($\alpha N = 10$ matches AGZ effective concentration). |
| $\eta, \mathrm{wd}$ | optimizer lr / weight decay. **Default $\eta=5\times 10^{-4}$ Adam, $\mathrm{wd}=0$** (F.6.0.5b V3 winner). lr_decay schedule supported via `--lr_decay` + `--lr_decay_step_size` (LambdaLR; default 1.0 = constant). |
| $\lambda_v$ | value-loss weight in joint MSE+CE objective. **lv0 default $\lambda_v=0$** (policy-only, value head receives no gradient). $\lambda_v=1$ is the AGZ-canonical "lv1" mode, retained for ablation. |
| $G$ | gate cadence in iters. **Default $G=1$** (revised from earlier $G=5$ for best_model freshness — see §5). |
| $K_\text{chunk}$ | `mcts_batch_size` — cross-instance chunk size in `CppBatchMCTSSolver`. **Default 1000** (was 64; 5× wall reduction via GPU saturation, see §F.G). NOT the per-NN-forward batch. |

---

## §2 Outer loop (one Stage 4 iteration)

For iteration $i = 0, 1, \dots, I{-}1$:

$$
\boxed{\quad
\begin{aligned}
&\textbf{(1) Self-play.}\quad \text{Sample } M \text{ random TSP-}N \text{ instances } \{x^{(m)}\}_{m=1}^M,\ x^{(m)} \in [0,1]^{N\times 2}.\\
&\quad\text{For each } m,\ \text{run } \alpha_{\theta^\star}(x^{(m)}) \to (\text{tour}^{(m)}, z^{(m)}, \{\pi_t^{(m)}\}_{t=0}^{N-1}). \\
&\quad\text{Push records } \{(s_t^{(m)}, \pi_t^{(m)}, z_t^{(m)})\}_{m,t} \text{ into } \mathcal{D};\ \mathcal{D} \text{ evicts oldest instance on overflow}.\\[4pt]
&\textbf{(2) Train.}\quad \text{For } j = 1, \dots, J=\texttt{train\_steps\_per\_iter}: \\
&\quad B \sim \text{Stratified-by-step}(\mathcal{D},\ |B|=\texttt{batch\_size})\quad \text{(see §3.6)}\\
&\quad \theta \leftarrow \theta - \eta_i\, \nabla_\theta \mathcal{L}(\theta; B)\quad \text{(Adam, } \eta_i=\eta_0\cdot \texttt{lr\_decay}^i, \text{wd}=0\text{)}\\[4pt]
&\textbf{(3) Gate.}\quad \text{If } (i+1) \bmod G = 0\ \text{(F.6.1 default } G=1\text{)}:\\
&\quad \text{accept} \leftarrow \text{Gate}(\theta, \theta^\star,\ \text{val\_size}=10000,\ \alpha=0.05)\\
&\quad \text{If accept: } \theta^\star \leftarrow \theta\\[2pt]
&\textbf{(4) Log + checkpoint.}\quad \text{Record } \mathrm{val\_avg\_cost}(\theta),\ \mathrm{lr}=\eta_i,\ \text{totals, gating outcome.}\\[2pt]
&\textbf{(5) Step lr scheduler.}\quad \eta_{i+1} \leftarrow \eta_0\cdot\texttt{lr\_decay}^{i+1}\ \text{via LambdaLR (no-op if lr\_decay=1)}.
\end{aligned}
\quad}
$$

**Key invariant** (matches AGZ Methods §Self-play training pipeline, p.357-358): $\theta^\star$ is updated only on gate accept; $\theta$ is updated every train step regardless. No rollback on reject.

**Current production (F.6.1.3+lv0) defaults:**
- $\eta_0 = 5\times 10^{-4}$ (F.6.0.5b V3). `lr_decay` ∈ {`1.0` (const, default), `0.2` with `lr_decay_step_size=100` (F.6.1.6 step-decay: 5e-4 → 1e-4 → 2e-5 → 4e-6 across 400 iters)}.
- **lr-override-on-resume**: `--resume_from PATH --lr_model 1e-4` overrides the optimizer's restored lr (LambdaLR would otherwise silently keep the loaded value). Implemented in `train_alphazero.py:331-342`. Pattern used by the F.6.1.4 chain and the TSP-50 lv0 +100 iter resume.
- $\mathrm{wd} = 0$ (F.6.0.5b drops AGZ-canonical wd=1e-4 in favor of AM-paper Adam wd=0).
- $G = 1$ (per-iter best_model refresh propagates each accepted improvement to the next iter's self-play).
- Buffer capacity = 5000 instances (≈5-iter window at M=1000). F.6.1.1 winner — 200K stale targets actively drag the policy back.
- `mcts_batch_size = 1000` (= M; one chunk per iter; 5× faster than the prior default 64 — see §F.G in this doc / §G in stage5_progress).
- $\lambda_v = 0$ (lv0): value head still evaluated for telemetry but receives no gradient; encoder is freed from value-multitask pull. See §3.4.
- `leaf_eval = 'rollout'`: AlphaGo-Lee-style rollout to terminal from leaf. Paired with lv0 (see §4.1.5).

---

## §3 Loss function

Per-tuple loss for one record $(s_t, \pi_t, z_t)$, equivalent to AGZ eq. (1) extended with the lv0 value-weight switch:

$$
\boxed{\quad
\mathcal{L}(\theta;\, s_t, \pi_t, z_t) \;=\; \lambda_v\cdot\underbrace{\big(z_t - v_\theta(s_t)\big)^2}_{\text{value MSE (per-state cost-to-go)}}
\;-\; \underbrace{\sum_{a \in \mathcal{A}(s_t)} \pi_t(a) \cdot \log p_\theta(a \mid s_t)}_{\text{policy distillation (cross-entropy)}}
\;+\; \underbrace{c\,\|\theta\|_2^2}_{\text{L2, via } \texttt{weight\_decay}}
\quad}
$$

with $\lambda_v \in \{0, 1\}$ controlled by `--lambda_v` (see §3.4 for the **lv0** semantics when $\lambda_v=0$).

with the components computed as:

$$
\begin{aligned}
\mathbf{e} &= \mathrm{Encoder}_\theta(\mathrm{coords}(s)) \in \mathbb{R}^{N\times d}\\
\mathcal{F} &= \mathrm{precompute\_decoder}_\theta(\mathbf{e})\quad \text{(AttentionModelFixed NamedTuple)}\\
(\log \mathbf{p}_\theta(\cdot\mid s),\ \texttt{mask},\ \mathbf{g}) &= \mathrm{decode\_step}_\theta(\mathcal{F},\, s,\, \texttt{return\_glimpse=True})\\
v_\theta(s) &= \mathrm{value\_head}_\theta(\mathbf{g}) \in \mathbb{R}
\end{aligned}
$$

**Mask handling** (numerical stability): $\log p_\theta(a\mid s) = -\infty$ on visited cities, so we replace those entries with $0$ before the dot product (their $\pi_a = 0$ makes the contribution 0):

$$
\sum_a \pi_a \log p_\theta(a\mid s) \;\equiv\; \sum_a \pi_a \cdot \big[\![\mathrm{mask}(a)\!=\!0]\!\big] \cdot \log p_\theta(a\mid s)
$$

**Value target — per-state cost-to-go** (matches Stage 1's V_CURRENT target shape; reuses `value_targets_from_edges` in `src/am_baseline/utils/tensor_ops.py:57-78`). **Three normalization modes** controlled by `--value_target_norm` ∈ {`bl`, `none`, `sqrt_n`} (added 2026-05-06 in F.6.0.5b's Option B):

$$
\boxed{\quad
z_t \;=\;
\begin{cases}
\dfrac{\mathrm{tour\_cost} - \mathrm{lengths}_t}{\mathrm{bl\_val}(x)} & \text{if } \texttt{value\_target\_norm} = \texttt{bl}\ \text{(legacy default; AGZ-canonical-style)}\\[8pt]
\mathrm{tour\_cost} - \mathrm{lengths}_t & \text{if } \texttt{value\_target\_norm} = \texttt{none}\quad \textbf{(F.6.1 default)}\\[6pt]
\dfrac{\mathrm{tour\_cost} - \mathrm{lengths}_t}{\sqrt{N}} & \text{if } \texttt{value\_target\_norm} = \texttt{sqrt\_n}\ \text{(theoretical scaling; ablation only)}
\end{cases}
\quad}
$$

where $\mathrm{lengths}_t$ is the cumulative cost of edges already traversed (= `state.lengths` at step $t$) and $\mathrm{bl\_val}(x) = \mathrm{cost}(\mathrm{greedy\_rollout}_{\theta^\star}(x))$ when used.

**MCTS leaf-eval inverse mapping** (matches the buffer→trainer pipeline; in [`mcts.py::_convert_value_head_output`](src/am_baseline/search/mcts.py) and [`mcts.cpp::convert_value_head_output`](src/am_baseline/search/mcts_cpp/mcts.cpp)): the value head's raw output $\hat v$ is converted at MCTS time to a unit-comparable scale before backup:

$$
V(s_L) - \mathrm{state.lengths}/\mathrm{bl\_val} \;=\;
\begin{cases}
\hat v(s_L) & \text{if } \texttt{norm} = \texttt{bl}\\[4pt]
\hat v(s_L) / \mathrm{bl\_val}(x) & \text{if } \texttt{norm} = \texttt{none}\\[4pt]
\hat v(s_L) \cdot \sqrt{N} / \mathrm{bl\_val}(x) & \text{if } \texttt{norm} = \texttt{sqrt\_n}
\end{cases}
$$

This keeps the MCTS leaf-evaluator invariant ($V(s_L) = \mathrm{state.lengths}/\mathrm{bl\_val} + V_\text{cost-to-go-fraction}$) intact across all three target conventions.

**Why per-state, not broadcast `z`:** the existing MCTS leaf evaluator (`src/am_baseline/search/mcts.py:1-15`, invariant 1) computes $V(s_L) = \mathrm{state.lengths}/\mathrm{bl\_val} + v_\theta(s_L)$, which assumes $v_\theta$ predicts **remaining cost-to-go from a partial state**, not the full tour. Training $v_\theta$ on broadcast full-tour `z` would double-count the path cost at MCTS time. Per-state $z_t$ matches the V_CURRENT shape Stage 1 trained against and makes Phase A's leaf evaluator a no-op compared to Stage 3.

**`bl_val` recomputation cadence:** once per training epoch under $\theta^\star$ (the model that produced the tour). Frozen at buffer-push time (`bl_val_frozen`) for the record's full lifetime in $\mathcal{D}$. Under `value_target_norm=none`, this number is no longer in the training target — only used at MCTS time for scale-conversion.

**Total objective** over a batch of size $B$, with **stratified-by-step sampling** (see §3.6 below):

$$
\mathcal{L}_\text{batch}(\theta) \;=\; \frac{1}{B}\sum_{(s_{t^\star},\pi_{t^\star},z_{t^\star})\in \mathcal{B}_{t^\star}}\Big[(z_{t^\star} - v_\theta(s_{t^\star}))^2 - \pi_{t^\star}^\top \log \mathbf{p}_\theta(\cdot\mid s_{t^\star})\Big] \;+\; c\,\|\theta\|_2^2
$$

where $t^\star \sim \text{Uniform}\{0,\dots,N-1\}$ is drawn once per minibatch and $\mathcal{B}_{t^\star}$ is a uniform sample of $B$ records from $\mathcal{D}$ restricted to step $t^\star$.

The L2 term is implemented via `torch.optim.Adam(..., weight_decay=c)` with current default $c = 0$ (F.6.0.5b winner — wd=1e-4 actively hurt at the F.6.1 lr regime).

### §3.4 lv0 mode — $\lambda_v = 0$ semantics

**Implementation** (`train_step_alphazero` in `src/am_baseline/training/trainer.py:345-394`): when `lambda_v == 0`, the trainer:

1. Computes $v_\theta(s_t)$ under `torch.no_grad()` for telemetry only — `value_loss` is still logged.
2. Calls `torch.autograd.grad(policy_loss, params)` with `retain_graph=False` and skips the value-gradient pass entirely.
3. Writes `pg` (policy grad) directly into each parameter's `.grad`. No value-head or shared-encoder value gradient touches the model.
4. Verification check: `value_grad_norm_shared` logged as exactly 0.0 every train step under lv0.

**Why lv0 wins under `leaf_eval='rollout'`** (Stage 5 §D.2 ablation, 2026-05-09): when the value head doesn't enter MCTS at leaf eval (rollout takes over), the value loss is **pure auxiliary noise from a biased target** — the value head's structural ~0.074 RMS bias against $E[z\mid s]$ (Stage 5 §C aleatoric probe) propagates through the shared encoder as biased gradient. Setting $\lambda_v=0$ removes the gradient and frees the encoder to specialize for policy distillation.

Empirical TSP-20 result (50-iter from-scratch, K=40, step10+ε=0.25, leaf_eval=rollout): **lv0 (3.879) beats lv1 (3.932) at iter 49 by 0.053**. lv0 endpoint at iter 197 = **3.8486 greedy** (beats Stage 1 canonical 3.83943 at ~6.4× sample efficiency).

**When NOT to use lv0**: when `leaf_eval='value_head'` (AGZ-canonical), the value head IS in the search loop — turning off its training would break MCTS quality. Use $\lambda_v=1$ in that regime.

### §3.5 Per-loss gradient-norm split telemetry

**Implementation** (`train_step_alphazero` in `src/am_baseline/training/trainer.py:382-432`): rather than calling `total_loss.backward()`, we run two separate `torch.autograd.grad` traversals (`policy_loss` first with `retain_graph=True`, then `value_loss`), measure their norms separately, then write the linear combination $\partial(\text{policy} + \lambda_v\cdot\text{value})/\partial\theta$ into each parameter's `.grad` slot. Cost: ~2× backward traversal time (negligible vs MCTS self-play wall).

**Logged metrics** (per train step):

| metric | meaning |
|---|---|
| `policy_grad_norm` | $\lVert \nabla_\theta \mathcal{L}_\pi \rVert$ over all trainable params |
| `value_grad_norm` | $\lVert \nabla_\theta \mathcal{L}_v \rVert$ over all trainable params (== 0 under lv0) |
| `value_grad_norm_vh` | restriction of $\nabla_\theta \mathcal{L}_v$ to `model.value_head` params only |
| `value_grad_norm_shared` | restriction of $\nabla_\theta \mathcal{L}_v$ to the encoder+decoder "shared" subspace |
| `gradient_norm` | $\lVert$ combined gradient $\rVert$ post-clip |

The split enables cosine-of-conflict analysis between policy and value losses on shared parameters and direct verification that lv0 truly zeros out the shared value gradient. The invariant $\lVert\nabla\mathcal{L}_v\rVert^2 = \lVert\nabla\mathcal{L}_{v,\text{vh}}\rVert^2 + \lVert\nabla\mathcal{L}_{v,\text{shared}}\rVert^2$ holds to machine precision (smoke A1).

### §3.6 Stratified-by-step sampling — why and how

**Decoder constraint:** AM's `model.decode_step(fixed, state)` takes a single `StateTSP` whose `i` field is a *scalar* (`src/am_baseline/problem/state.py:5-19`); the decoder branches on `state.i == 0` vs `> 0` to swap the placeholder graph-context for the (first_a, prev_a) embedding. A minibatch that mixes multiple step values under one scalar `state.i` would silently produce wrong $\log p_\theta$ on all rows whose step disagrees with the scalar — gradients would be biased without raising NaN.

**Resolution:** the buffer's `sample()` draws $t^\star \sim \text{Uniform}\{0..N-1\}$ first, then samples $B$ records uniformly from $\mathcal{D}$ restricted to records with step $= t^\star$. Every row's step matches the decoder's scalar `state.i`. The marginal distribution of (instance, step) over many train steps is uniform — gradients are unbiased; only per-batch variance is slightly higher (negligible at $J=200$ train steps × $N=20$ steps → ~10× per-step coverage per iter).

**Implementation:** the buffer maintains `_step_index: list[np.ndarray]` of length $N$, with `_step_index[t]` indexing tuple slots currently filled at step $t$. Updated atomically with each push/eviction. O(1) sampling. **Persistence:** `_step_index` is not saved; it is deterministically rebuilt on `load()` from the locked invariant `tuple_slot = inst_idx \cdot N + step \implies step = \text{tuple\_slot} \bmod N`, scanning filled slots in O(`capacity_tuples`) (microseconds). This makes `_step_index` a pure runtime-cached projection of the data and eliminates a "saved index out of sync with saved data" failure class.

**All $N$ per-step records are stored.** No skip-rule for forced last-step actions (`legal_actions(t) = 1`). Trivial CE at those steps is finite (the AM decoder's legality mask makes $\log p_\theta(\cdot \mid s_t)$ sharp on the only legal action, matching $\pi_t$'s one-hot mass), the value-loss MSE remains informative, and the dense `capacity_instances × N` layout stays consistent. Matches AGZ Methods §Self-play (tuples stored for every step up to termination).

**Three alternatives considered and rejected** (recorded for future reference):
- *Group-by-step within a minibatch* (~20 decoder dispatches per train step at TSP-20) — defeats batching; dispatch overhead dominates.
- *Vectorize the decoder over per-row step* — non-trivial change to AM decoder API; risks regressing Stage 1/2/3 callers.
- *Per-instance batches* (whole-tour rollouts) — Stage 1's bottleneck; sequential within instance, no within-instance parallelism.

---

## §3.7 Value normalization — caveats and design notes

Three points worth tracking; informed both default choices in the plan and the open Stage 5 ablation.

### Why we can't simply copy AGZ's reward shape

AGZ's targets are $z \in \{-1, +1\}$ — literal win/loss bits (Methods §Self-play, p.358). The paper notes (Methods §Optimization, p.358):

> "The cross-entropy and MSE losses are weighted equally (this is reasonable because rewards are unit scaled, $r \in \{-1, +1\}$)."

This unit-scaled target gives AGZ four free properties we don't have:
- **Stationary**: a win is a win across all training time.
- **Instance-independent**: every game returns ±1 regardless of board configuration.
- **Model-independent**: the winner of a played game is fixed once the game is played.
- **Loss balance trivial**: $(z-v)^2 \in [0,4]$ regardless of any other detail.

TSP is a continuous-cost minimization problem; quantizing $z$ to ±1 ("did MCTS beat greedy?") would discard the magnitude signal we're trying to learn. The continuous ratio $z = \text{tour\_cost}/\text{bl\_val}$ preserves it but introduces three concerns below.

### Concern 1: `bl_val` *target-side* drift — DOMINANT problem under `value_target_norm='bl'`; RESOLVED by `'none'` (F.6.0.5b Option B)

The frozen-at-generation `bl_val` makes any single record's target stationary, but **across-record** drift is large and hurts the value head's training. Three concrete drift channels in the bl-normalized regime, surfaced by F.6.0.5b:

1. **Calibration drift across records.** As $\theta^\star$ improves, newer records have smaller `bl_val` than older records. The value head is trained on a *moving distribution* of $z = \text{cost\_to\_go}/\text{bl\_val\_frozen}$ — even though each record is individually stationary, the population the value head sees over training is a noisy mixture of "older / weaker θ★" and "newer / stronger θ★" calibrations.
2. **Buffer non-stationarity.** Even within one training step, the minibatch can mix records from very different θ★ snapshots, blurring the per-state target.
3. **Across-instance variance collapse at random init.** Early in training, tour_cost and bl_val are highly correlated (both produced by ~uniform-random policies on the same instance), so $z = \text{cost\_to\_go}/\text{bl\_val} \approx \mathrm{const}$ across instances. The value head trivially fits the per-state mean and **provides no leaf-discrimination at MCTS time** (F.6.0 root-cause: see [progress F.6.0 headline 4](_progress/stage4_progress.md)).

**Option B fix (F.6.0.5b, now F.6.1 default): `value_target_norm='none'`.** Train on raw $z_t = \text{cost\_to\_go}_t$ (no division). Removes all three drift channels at once. Empirical impact ([F.6.0.5b results](_progress/stage4_progress.md)): the value head finally learns non-trivial across-instance variance (value_loss climbs from ~0.013-0.024 in the broken-bl regime to ~0.06-0.30 in the working-raw regime — an order of magnitude *more* loss, but the loss is now meaningful signal rather than constant-mean trivial fitting). The cost-to-go magnitude scale is rougher (~3-5× for TSP-20 random init) but bounded; gradients stay O(1).

The cost: the policy-iteration "newer self-play is closer to optimal" signal is encoded only via better MCTS visit distributions, not the value scale. Acceptable trade.

### Concern 2: `bl_val` model asymmetry — N/A under `'none'`

Under `value_target_norm='none'`, `bl_val` is no longer in the training target — only used at MCTS leaf-eval time for scale conversion. The tour-vs-baseline-model asymmetry that mattered under `'bl'` (resolved earlier by computing `bl_val` from $\theta^\star$ at push time) is absent here.

### Concern 3: Stage 5 alternative — `best-so-far` normalization

Originally, G.6 ablation was "per-step cost-to-go target instead of broadcast z". With per-state cost-to-go being the F.4 default (and `value_target_norm` now the active knob via F.6.1), G.6 is repurposed to **"best-so-far per-instance normalization"**:

$$
z_t^\text{best-so-far} \;=\; \frac{\mathrm{tour\_cost} - \mathrm{lengths}_t}{\min_{\text{seen}}\mathrm{tour\_cost}(x)}
$$

where the min is over all self-play attempts on instance $x$ across all iterations. Doesn't require an oracle, doesn't drift with the trainer, range $\geq 0$ with monotone optimum-tracking. Worth re-evaluating now that `'none'` is validated — could outperform raw if value-loss magnitude becomes a stability concern at TSP-50+.

---

## §4 MCTS inner loop ($\alpha_\theta$)

For one instance $x \in [0,1]^{N\times 2}$, MCTS produces $(\text{tour}, z, \{\pi_t\})$ via $N$ tour-steps, each running $K$ simulations.

### §4.1 Per-simulation traversal — PUCT

Starting at root $s_0$, descend the tree by **PUCT** (exact code in `src/am_baseline/search/puct.py:7-33`):

$$
\boxed{\quad
a^\star \;=\; \arg\max_{a \in \mathcal{A}(s)}\Big[\, Q(s,a) \;+\; c_\text{puct}\,P(s,a)\,\frac{\sqrt{\sum_b N(s,b)}}{1 + N(s,a)} \,\Big]
\quad}
$$

where for unvisited edges $Q(s,a)$ takes the **FPU value** (= running mean of expanded children's Q at this node when `fpu_mode='running_q'`, with fallback $-1.0$ if no children expanded yet — matches `mcts.py:_fpu_value_for`).

On reaching a leaf $s_L$:

- **Expand:** evaluate $f_\theta(s_L) \to (\mathbf{p}_L, v_L)$; for each legal $a$, set $P(s_L, a) = p_L[a]$, $N(s_L,a)=0$, $W(s_L,a)=0$.
- **Evaluate** (see §4.1.5): $V(s_L) = v_L$.
- **Backup:** for each edge $(s,a)$ on the simulation path,

$$
N(s,a) \mathrel{+}= 1,\qquad W(s,a) \mathrel{+}= V(s_L),\qquad Q(s,a) = \frac{W(s,a)}{N(s,a)}.
$$

### §4.1.5 Leaf evaluation choice — **rollout** (lv0 default), value-net alternative

| Version | Leaf evaluation |
|---|---|
| AlphaGo Fan / Lee | $V(s_L) = \lambda \cdot v_\theta(s_L) + (1-\lambda) \cdot z_\text{rollout}$ — value mixed with fast-rollout-policy game completion |
| **AlphaGo Zero (canonical)** | $V(s_L) = v_\theta(s_L)$ — value net only |
| AlphaGo Master | same as Zero (value-net only), but with handcrafted features and SL initialization |
| **Stage 4 lv0 (current default)** | $V(s_L) = z_\text{rollout}/\mathrm{bl\_val}(x)$ — pure greedy rollout to terminal, **no value head in MCTS path** |

**Current Stage 4 default: `leaf_eval='rollout'`.** Implemented in `mcts_cpp/solver.py::rollout_many` + `mcts.cpp::Solver::rollout_remaining_real`. From a leaf $s_L$, perform a greedy rollout (argmax of decoder priors at every step) to a terminal state, return $z_\text{rollout}$ (the realized remaining cost), normalize by $\mathrm{bl\_val}(x)$, back up.

**Why we deviated from AGZ-canonical** (Stage 5 §C leaf-eval bypass probe, 2026-05-09): F.6.1.6's trained value head is **statistically tied with greedy** at MCTS val time (vh K=40 → 3.868 vs greedy 3.863, p<0.01 WORSE). The structural RMS bias of 0.074 against $E[z\mid s]$ kills MCTS visit distributions; more sims don't fix it (vh K=200 still at 3.868). Switching to `leaf_eval='rollout'` on the SAME checkpoint: K=40 → 3.834 (beats Stage 1 canonical greedy 3.83943), K=200 → 3.833 (within 0.005 of Stage 3 K=400 rollout's 3.8312).

**The decisive finding**: the 3.85 greedy ceiling on F.6.1.6 is a **vh-leaf-eval-induced TRAINING ceiling, not a model-quality ceiling**. Training with rollout-as-leaf-eval (and pairing with $\lambda_v=0$ to remove the biased value gradient on the shared encoder) lifts the ceiling. Memory landmark: [`project_alphagozero_value_head_leaf_eval_bias.md`](C:\Users\Jun18\.claude\projects\C--Users-Jun18-Desktop-AM-ALPHAGOZERO\memory\project_alphagozero_value_head_leaf_eval_bias.md).

**Inference-time recommendation** (from Stage 5 §D.5 sweep): on ANY F.6.1-family checkpoint at val/test time, default to `--leaf_eval rollout`. K=40 rollout buys ~0.028 over greedy on TSP-20 and breaks 3.85 trivially.

**`leaf_eval='value_head'` is retained** for AGZ-canonical fidelity audits + diagnostic probes. Stage 4 originally launched with it as default; superseded.

### §4.2 Root action sampling and training target — *two distinct distributions*

After $K$ simulations from root $s_t$, we extract **two** distributions from the visit counts $\{N(s_t,a)\}_{a \in \mathcal{A}(s_t)}$:

**(a) Action-selection distribution $\sigma_t$** (used to sample the played action $a_t \sim \sigma_t$):

$$
\boxed{\quad
\sigma_t(a) \;=\; \frac{N(s_t,a)^{1/\tau_t}}{\sum_b N(s_t,b)^{1/\tau_t}},\qquad
\tau_t \;=\; \begin{cases} 1.0 & \text{if } t < \lceil p \cdot N \rceil \\ 0^+ & \text{otherwise} \end{cases}
\quad}
$$

where $\tau \to 0^+$ means deterministic argmax and $p \in \{0.1, 0.3, 0.5\}$ is the schedule fraction. **`step10` is the F.6.1.3+lv0 default** ($p=0.1$, exploration in the first ⌈0.1·N⌉ steps only); `step30` ($p=0.3$, AGZ-proportional) and `step50` are G.4 alternatives. `const` keeps $\tau = $ `cfg.temperature` uniformly. The chosen action becomes the new root via tree reuse.

**(b) Training target $\pi_t$** (the cross-entropy target in §3, stored in the buffer):

$$
\boxed{\quad
\pi_t(a) \;=\; \frac{N(s_t,a)}{\sum_b N(s_t,b)}\quad\text{(temperature 1, *always*)}
\quad}
$$

**Why decouple $\sigma_t$ and $\pi_t$:** AGZ uses $\tau_t$ for both action selection *and* the training target (Methods §Self-play, p.358), which means late-game training targets in AGZ are one-hot. That works for Go (250-ply games with rich late-game tactical structure even at $\tau \to 0$). For TSP-20, the action space shrinks deterministically with $t$ (legal_actions(18) = 2, legal_actions(19) = 1), so one-hot late targets carry no information beyond "the chosen action was legal" — a wasted distillation signal. Stage 4 default (choice B) keeps the AGZ-faithful exploration *behavior* (sampled action via `step30`) but trains against the *richer* raw normalized visit distribution, preserving multimodal information when MCTS visited multiple actions roughly equally.

**G.4 ablation** compares this default against strict-AGZ (where $\pi_t = \sigma_t$, one-hot late targets) on the same TSP-20 recipe.

### §4.3 Root exploration noise — Dirichlet (training only)

At each root $s_t$, the priors are perturbed exactly once before the $K$ simulations (skipped entirely when ε=0):

$$
\boxed{\quad
P(s_t, a) \;\leftarrow\; (1-\varepsilon)\, p_\theta(a\mid s_t) \;+\; \varepsilon\, \eta_a,\qquad \boldsymbol{\eta} \sim \mathrm{Dir}(\alpha\,\mathbf{1}_{|\mathcal{A}(s_t)|})
\quad}
$$

with $\alpha = 10/N$ retained ($\alpha N = 10$ matches AGZ's $\alpha = 0.03$ at $|\mathcal{A}|=362$).

**ε history and current default ($\varepsilon=0.25$ paired with `step10` τ-schedule, F.6.1.3+lv0 winner):**

- F.6.0 originally used $\varepsilon=0.25$ (AGZ inheritance) + `step30` τ-schedule.
- F.6.0.6 (under F.6.1 V3 regime, leaf_eval=value_head, `step30`): tested ε∈{0, 0.05, 0.25} — **ε=0 won on stability** for the 100-iter horizon. Spec briefly defaulted to ε=0.
- F.6.1.3 (2026-05-07, `step10` introduced): paired ε∈{0.05, 0.25} × `step10`. **ε=0.25 + step10 won with val=3.8784 at iter 99**, breaking F.6.1's 3.92 plateau. Mechanism: `step10` collapses to argmax after just 2/20 steps, dropping target entropy; **ε=0.25 restores multimodality in the π_t targets** by perturbing root priors at every step. Validated via `mean_entropy_pi` 2.16× higher at ε=0.25.
- **Current production: $\varepsilon=0.25$ paired with `step10`**, used by all F.6.1.4+ runs and the lv0 recipe.

**Coupled choice:** ε and the τ-schedule must be tuned **together**. ε=0.25 with `step30` (the original F.6.0 default) over-explores; ε=0 with `step10` (the F.6.1 pre-F.6.1.3 setting) under-explores. ε=0.25 + step10 is the local optimum at lr=5e-4.

**TSP-50 also uses $\varepsilon=0.25$ + `step10`** in production (F.6.1.x TSP-50 K=50 lv0 chain — `0d48yqys` etc.).

### §4.3.5 Train-only vs eval-only — when does noise apply?

**Dirichlet noise is added during self-play training only, not during evaluation/inference.** AGZ Methods §Self-play (p.358) is the only place noise is described:

> "Additional exploration is achieved by adding Dirichlet noise to the prior probabilities in the root node $s_0$ ... this noise ensures that all moves may be tried, but the search may still overrule bad moves."

Methods §Evaluator (p.358) describes candidate-vs-best evaluation with no mention of noise:

> "Each evaluation consists of 400 games, using an MCTS with 1,600 simulations to select each move, **using an infinitesimal temperature $\tau \to 0$** (that is, we deterministically select the move with maximum visit count, to give the strongest possible play)."

The asymmetry: noise is an *exploration mechanism for data generation* (ensures the visit distribution being distilled is informative even on actions the prior dismisses), not a search-quality lever (it can only hurt by pulling priors toward random actions). Three call sites in Stage 4:

| Call site | Phase | $\tau$-schedule | Dirichlet | Rationale |
|---|---|---|---|---|
| `generate_self_play_batch` | training data gen | **`step10`** | **ε=0.25** | F.6.1.3 winner: step10's low target entropy needs ε to restore π_t multimodality |
| Val-time MCTS (`val_stage4_mcts.py`) | inference | τ→0 throughout | **none** | AGZ §Evaluator — strongest play |
| Greedy val (`coach.validate`) | evaluation | n/a (greedy decoding, no MCTS) | **none** | Network's standalone quality |

Implementation note: under tree reuse, the Dirichlet draw is *resampled freshly* each time a child becomes the new root — never propagated. `mcts.py:226-228, 444-460` and the C++ mirror already handle this correctly.

### §4.4 Final outputs

After $N$ tour-steps, MCTS returns:

- $\text{tour} = (a_0, a_1, \dots, a_{N-1})$
- $\text{tour\_cost} = \sum_{t=0}^{N-1} \|x_{a_t} - x_{a_{t+1 \bmod N}}\|_2$
- $\{\mathrm{lengths}_t\}_{t=0}^{N-1}$ — cumulative cost of edges traversed before $s_t$ (= `state.lengths`)
- $\{\pi_t\}_{t=0}^{N-1}$ — temperature-1 normalized visit distributions (training target)
- $\{z_t\}_{t=0}^{N-1}$ where $z_t = (\text{tour\_cost} - \mathrm{lengths}_t)/\mathrm{bl\_val}(x)$ — per-state V_CURRENT targets

Phase A's `MCTSConfig.return_root_visits=True` exposes the raw visit dicts $\{N(s_t, \cdot)\}$; the buffer-push step computes $\pi_t$ and $z_t$ from these.

### §4.5 Production MCTS backend — `CppBatchMCTSSolver` cross-instance scheduling

The production self-play backend is `src/am_baseline/search/mcts_cpp/solver.py::CppBatchMCTSSolver` (NOT the Python reference `MCTSSolver`). Key structural differences from a per-instance Python loop:

1. **C++ owns tree state + PUCT + backup** (`mcts_cpp::Solver` for the single-instance API; `mcts_cpp::BatchSearch` for the cross-instance scheduler). Python only provides the NN evaluator callback.

2. **Cross-instance batching via `BatchSearch.collect_requests` / `apply_results`** (mcts.hpp:176-189):
   - `engine.collect_requests()` → C++ runs PUCT selection on every active tree in parallel, returns a Python list of leaf-evaluation requests (one per tree with a pending simulation slot).
   - Python's `evaluate_requests` batches the requests by `need_value` flag and `need_rollout` flag, calls `model.decoder.decode_step(...)` ONCE per group on the merged batch, returns numpy `(probs, mask, value, rollout_remaining)` tuples.
   - `engine.apply_results(results)` → C++ does backup on every tree, advances state machine.
   - Loop until `engine.is_done()`.

3. **Chunked over `mcts_batch_size`**: `solve_batch` partitions $M$ instances into $\lceil M / K_\text{chunk}\rceil$ sequential chunks. Each chunk runs its own `BatchSearch`. **Production default $K_\text{chunk}=1000$** means M=1000 instances run as ONE chunk per iter (full GPU saturation).

4. **Eval cache**: `_solve_chunk` maintains a Python dict `eval_cache: {(packed_header_int, visited_bytes) → (probs, mask, value)}` shared across selection-path and rollout-path NN evaluations within a chunk. Hit rate at production scale ~82% (trained ckpt). The cache survives across all simulations within one chunk, then drops at chunk end.

5. **Per-row `state.i` in the decoder (Track A, 2026-05-11)** — `solver.py::rollout_many` + `eval_many_arrays` allow `StateTSP.i` to be a per-row `(B,)` tensor rather than the scalar the original AM decoder required. All active rollouts at heterogeneous tour-steps merge into **one** NN call per outer iter, replacing the prior `for _step in np.unique(active_steps):` loop that produced ~50 small NN calls per outer iter at TSP-50 K=50.

   - Implementation: `Decoder._get_step_context` branches per-row via `torch.where(state.i.view(-1,1,1) == 0, placeholder_ctx, gathered_ctx)`. The scalar fast-path is preserved bit-for-bit via `state.i.numel() == 1` guard, so the **training-time `train_step_alphazero` is unchanged** — only the MCTS rollout path uses the per-row form.

   - **Determinism preserved**: paired-seed runs produce `max_abs_cost_diff = 0.0e+00` on TSP-20 K=40 M=100 and TSP-50 K=50 M=200 — Track A is a structural batch-consolidation, not a numerics change.

   - **Stratified-by-step sampling for training (§3.6) is still load-bearing** — the trainer uses scalar `state.i` for one decoder forward per minibatch.

See `_progress/stage5_progress.md` §F for the full wall-time optimization stack (Fix #1-5 + Track A → 75% wall reduction).

---

## §5 Gating (paired t-test)

**Procedure** (matches `RolloutBaseline.epoch_callback`, `baselines.py:106-123`):

1. Fix a held-out validation set $\mathcal{V}$ of size $|\mathcal{V}| = 10{,}000$ TSP-$N$ instances.
2. Compute candidate costs $\{c^{\text{cand}}_v\}_{v\in\mathcal{V}}$ using $\theta$ (greedy rollout, no MCTS).
3. Compute baseline costs $\{c^{\text{base}}_v\}_{v\in\mathcal{V}}$ using $\theta^\star$ (cached as `self.bl_vals`).
4. **Reject early** if $\bar c^\text{cand} \ge \bar c^\text{base}$ → return False.
5. Otherwise run paired t-test:

$$
t = \frac{\bar d}{s_d / \sqrt{|\mathcal{V}|}},\qquad d_v = c^{\text{cand}}_v - c^{\text{base}}_v
$$

6. **Accept** iff one-sided $p\text{-value} = \tfrac{1}{2}\Pr(T \le t) < \alpha = 0.05$.

On accept: $\theta^\star \leftarrow \theta$; the cache `self.bl_vals` is recomputed under the new baseline.

### §5.5 Why the gate exists

The gate is taken directly from AGZ Methods §Self-play training pipeline + §Evaluator:

> "AlphaGo Zero's self-play training pipeline consists of three main components ... the best performing player so far, $\alpha_{\theta^\star}$, is used to generate new self-play data."
>
> "**To ensure we always generate the best quality data**, we evaluate each new neural network checkpoint against the current best network ... If the new player wins by a margin of >55% (to avoid selecting on noise alone) then it becomes the best player."

**Primary purpose: prevent regressions from poisoning the replay buffer.** Self-play data quality is bounded by the model that produced it. Without a gate, a noisy training step that temporarily worsens the model would immediately start polluting $\mathcal{D}$, which would push the next training step further off — a feedback loop that can spiral. The gate breaks it: $\theta^\star$ only advances when there's *statistical evidence* the new model is better, so the data-generating distribution is **monotonically improving** by construction.

**Secondary effects:**
1. **Filter optimizer noise.** Adam's per-step updates can momentarily worsen the policy on val even if the long-run direction is correct. Gating at every iter ($G=1$ default) checks per-iter; the t-test against the cached $\theta^\star$ baseline rejects if the candidate doesn't beat by a statistically significant margin.
2. **Snapshot for evaluation.** Headline plots use $\theta^\star$, not whatever $\theta$ happened to be at the last gradient step. Stabilizes reported numbers.
3. **Enable safe optimizer continuity.** Per scope decision 3 (matching AGZ), the trainer's optimizer state is *not* reset on gate reject. That's only safe because the gate decoupled "what generates data" from "what's training" — the trainer can explore noisy directions in parameter space without immediately corrupting $\mathcal{D}$.

**Gate cadence revised 2026-05-06: $G=5$ → $G=1$.** Initial $G=5$ inherited AGZ's "evaluate every 1000 batches" pattern proportionally to our setup (200 train_steps × 5 iters = 1000 batches). Under the F.6.1 regime, this caused up to a 5-iter staleness in $\theta^\star$ — at lr=5e-4 with ~0.05/iter improvement, that's ~0.25 quality units of self-play data drag for free. $G=1$ refreshes immediately; the t-test on 10K val has plenty of power to distinguish per-iter improvements; cost is +5-10s/iter (≪ ~25-115s/iter mcts_s). F.6.1 K=20 trajectories (lrdecay variant: gate accepts at iter 0, 20, 30, 98) confirm the t-test is appropriately conservative and doesn't accept on noise.

**Gating is not strictly necessary at scale.** KataGo's `ref/KataGo-master/SelfplayTraining.md` notes:

> "Note that not using gating ... will be faster and will save compute power, and the whole loop works perfectly fine without it, but having it at first can be nice to help debugging and make sure that things are working and that the net is actually getting stronger."

KataGo can drop the gate because at scale (millions of games) individual bad checkpoints' contributions to the replay buffer get diluted by many other good checkpoints' data — the buffer-level distribution is robust even if individual snapshots aren't. Stage 4 doesn't have that dilution (1K instances/iter, ~1% of buffer per iter), so the gate is load-bearing.

**G.5.c ablation tests this directly**: drop the gate, always set $\theta^\star = \theta$, see if the loop still converges. Either outcome (gating-faster vs gating-load-bearing) is publishable evidence at our scale.

---

## §6 Algorithm in one block — full pseudocode

**Current production recipe (F.6.1.3+lv0)** for TSP-20; values in `[brackets]` are the locked defaults. TSP-50 differs only in `N=50`, `K=50`, and `K_chunk=1000` (otherwise identical).

```
Input:  No checkpoint (from-scratch random init; F.6 supersedes F.4 warm-start)
        I = 100..400              # outer iterations (typical chain: 50 → 99 → 199 → 399)
        M = 1000                  # self-play instances per iter         [F.6.1.0 saturates train budget]
        K = 40                    # MCTS simulations per move (TSP-20); 50 at TSP-50
        K_chunk = 1000            # mcts_batch_size — cross-instance chunk; one chunk/iter at M=1000
        J = 200                   # train steps per iter                 [F.6.1.0]
        B = 512                   # mini-batch size
        G = 1                     # gate every G iterations              [G=5 → 1, 2026-05-06]
        η₀ = 5e-4                 # Adam initial lr                      [F.6.0.5b V3 winner]
        lr_decay = 1.0            # constant lr (or 0.2 + step_size=100 for F.6.1.6 4-segment step decay)
        lr_decay_step_size = 1    # >1 enables StepLR-like step decay
        c = 0.0                   # weight_decay                         [F.6.0.5b]
        λᵥ = 0.0                  # lv0 — policy-only training, value head untouched   [Stage 5 §D winner]
        ε = 0.25                  # Dirichlet root noise                 [F.6.1.3, paired with step10]
        α = 10/N = 0.5            # Dirichlet concentration
        c_puct = 0.05             # PUCT exploration constant            [Stage 2-locked]
        leaf_eval = 'rollout'     # AlphaGo-Lee-style rollout in MCTS    [Stage 5 §C diagnostic, §D ablation]
        value_target_norm = 'none'# raw cost-to-go target                [F.6.0.5b Option B]
        gate_mode = 'ttest'       # paired-t α=0.05
        temperature_schedule = 'step10'   # τ=1 for first ⌈0.1N⌉ steps, τ=0 otherwise   [F.6.1.3]
        buffer_capacity = 5000    # ≈5-iter window at M=1000             [F.6.1.1 winner]
        val_seed = 42, val_size = 10000

Initialize:
        θ ← random init                  # trainer's working copy
        θ★ ← deepcopy(θ)                 # best-player snapshot
        D ← MCTSReplayBuffer(capacity_instances=5000)
        Opt ← Adam(θ.parameters(), lr=η₀, weight_decay=c=0)
        Sched ← LambdaLR(Opt, lambda k: lr_decay ** (k // lr_decay_step_size))
        V ← fixed val set of 10K instances (val_seed=42); bl_vals ← rollout(θ★, V)
        # ---- Optional: lr-override-on-resume ----
        # If resuming via `--resume_from PATH --lr_model 1e-4`, override the
        # optimizer's restored lr AND scheduler.base_lrs to the new value.
        # Required for the F.6.1.4 chain pattern: lr=5e-4 plateau → lr=1e-4 unlock.

For i = 0, …, I−1:

    # ---- (1) Self-play with θ★ via CppBatchMCTSSolver ----
    {x⁽ᵐ⁾} ← sample M random TSP-N instances                  # one batch
    bl_val[1..M] ← greedy_cost(θ★, {x⁽ᵐ⁾})                     # one forward pass
    # CppBatchMCTSSolver runs all M instances concurrently per chunk of K_chunk:
    (tours, mcts_costs, root_visit_dists) ← CppBatchMCTSSolver(θ★, cfg, K_chunk).solve_batch({x⁽ᵐ⁾}, bl_val)
        # Per outer iter inside MCTS:
        #   - C++ engine.collect_requests() returns pending leaf evals across all trees
        #   - Python evaluates: model.decoder.decode_step(...) ONCE on the merged batch
        #     (Track A: per-row state.i so all rollout steps batch together)
        #   - For each leaf needing rollout: rollout_many({x⁽ᵐ⁾}) does greedy-argmax to terminal
        #   - C++ engine.apply_results(...) does backup
        # Per tour-step: πₜ ← N/ΣN (raw temperature-1), σₜ ← step10 schedule (argmax after step 2 at N=20)
    For each m, t:
        cost_to_go_t ← mcts_cost_m − lengthsₜ                  # remaining tour cost from sₜ
        zₜ ← cost_to_go_t                                       # value_target_norm='none'
        D.push((s_t^(m), π_t^(m), z_t^(m), bl_val_m))

    # ---- (1.5) MCTS-quality telemetry ----
    mcts_delta_vs_greedy_mean ← mean(mcts_costs - bl_val)       # negative when MCTS wins
    mcts_win_rate_vs_greedy ← mean(mcts_costs < bl_val)
    {greedy_cost_mean, mcts_cost_mean} logged for trajectory diagnostics

    # ---- (2) Distillation training ----
    For j = 1, …, J:
        Batch ← D.sample(B)                                    # stratified by step (§3.6); scalar state.i
        # Forward (lv0 mode):
        log_p, mask, glimpse ← model.decode_step(...)
        if λᵥ == 0:
            v ← model.value_head(glimpse.detach())              # logged but no gradient flows back
        else:
            v ← model.value_head(glimpse)
        policy_loss ← -(π_target * log_p_safe).sum(-1).mean()  # mask-safe CE
        value_loss ← MSE(v, z_target)                           # always computed for telemetry
        # Per-loss grad-norm split (§3.5):
        policy_grads ← autograd.grad(policy_loss, params, retain_graph=(λᵥ>0))
        value_grads  ← autograd.grad(value_loss,  params)  if λᵥ > 0  else None
        for p, pg, vg in zip(params, policy_grads, value_grads or [None]·len(params)):
            p.grad = pg + λᵥ·vg   (or just pg under lv0)
        clip_grad_norm_(θ.parameters(), max_norm=1.0)
        Opt.step()
        log {policy_grad_norm, value_grad_norm, value_grad_norm_vh, value_grad_norm_shared}

    # ---- (3) Validation + Gating ----
    val_cost ← greedy_rollout(θ, V)
    If (i+1) mod G == 0:                                       # G=1 → every iter
        candidate_vals ← rollout(θ, V)
        if mean(candidate_vals) < mean(bl_vals):
            t, p = scipy.stats.ttest_rel(candidate_vals, bl_vals)
            if (p/2) < 0.05 and t < 0:
                θ★ ← deepcopy(θ)
                bl_vals ← candidate_vals
                save_checkpoint(tag=f'{i}_accepted')

    # ---- (4) Log + checkpoint ----
    log(iter=i, total_instances=(i+1)·M, val_avg_cost=val_cost,
        mcts_wall_s, train_wall_s, buffer_size, lr,
        mcts_delta_vs_greedy_mean, mcts_win_rate_vs_greedy,
        greedy_cost_mean, mcts_cost_mean, gated, accepted)
    save_checkpoint(tag=f'{i}')
    save buffer.pt                                              # rolling save (large)

    # ---- (5) Step lr scheduler ----
    Sched.step()                                                # advances iter counter

Return θ★, θ
```

**Headline results (current):**
- **TSP-20 lv0 chain**: 199 iters × 1000 instances → best val 3.8486 greedy (lv0 iter-197 best_model) / 3.8329 K=200 rollout val-MCTS. Beats Stage 1 canonical greedy (3.83943) at **~6.4× sample efficiency** (200K vs 1.28M instances).
- **TSP-50 lv0 chain** (`oxjyj70e → 1wpkngg9 → 0d48yqys`): 99 iters in flight at 2026-05-13; current best val **6.058** at iter 68 (matches/beats prior Stage 4 best `muckiyvi` 6.060). Stage 1 TSP-50 = 5.7999; Gurobi = 5.6987.

---

## §7 Mapping to AGZ paper equations

| Stage 4 (F.6.1.3+lv0 locked) | AGZ paper | Identity / deviation |
|---|---|---|
| §3 loss | eq. (1), p.355 | $\ell = \lambda_v(z-v)^2 - \pi^\top\log\mathbf{p} + c\lVert\theta\rVert^2$ — structurally similar. **Reward shape**: $z \in \mathbb{R}_+$ raw cost-to-go vs $\{-1,+1\}$. **$c=0$** vs AGZ $c=10^{-4}$. **$\lambda_v=0$** under lv0 (value head receives no gradient — see §3.4); AGZ-canonical is $\lambda_v=1$. |
| §4.1 PUCT | Methods §Search algorithm, p.358 | $U(s,a) = c_\text{puct}\,P(s,a)\,\sqrt{\sum_b N(s,b)}/(1+N(s,a))$ — **literal match**. |
| §4.1.5 leaf eval | Methods §Search algorithm + §AlphaGo versions, p.357-358 | AGZ: $V(s_L) = v_\theta(s_L)$ (value net only). **lv0 default: `leaf_eval='rollout'`** (AlphaGo-Lee-style greedy rollout to terminal) — Stage 5 §C bypass probe showed vh leaf eval is statistically worse than greedy on F.6.1.6 due to a 0.074 RMS structural bias; rollout breaks 3.85 trivially. Documented deviation. |
| §4.2 root sampling | Methods §Self-play, p.358 | $\sigma_a \propto N(s_0,a)^{1/\tau}$ — **literal match for σ**; $\pi$ (training target) decoupled (raw τ=1 always), Stage 4 choice B. $\tau$-schedule scaled (Go: 30 of ~250 plies = 12%; **Stage 4 TSP-20: step10 = ⌈0.1·N⌉ = 2 of 20 = 10%**, F.6.1.3 winner). |
| §4.3 Dirichlet | Methods §Self-play, p.358 | $P \leftarrow (1-\varepsilon)p + \varepsilon\eta$ — formula identical. **$\varepsilon=0.25$ in F.6.1.3+lv0** (matches AGZ canonical 0.25; F.6.0.6's earlier ε=0 default was paired with step30 — under step10 the entropy drop requires ε=0.25 to restore π_t multimodality). $\alpha = 10/N$ retained. Train-only (val/test use no noise). |
| §5 gating | Methods §Evaluator, p.358 | AGZ: 400-game match, 55% win threshold. Stage 4: paired t-test on 10K val, $\alpha=0.05$ — comparable confidence, same spirit. **$G=1$** vs AGZ's ~G=5 proportional. |
| §2 loop structure | Methods §Self-play training pipeline, p.357-8 | best-player $\theta^\star$ generates data; trainer $\theta$ continues regardless of gate — **literal match**. |
| Optimizer | Methods §Optimization | AGZ: SGD+momentum 0.9, lr step-anneal {1e-2 → 1e-3 → 1e-4}. **Stage 4: Adam, lr=5e-4 const or lr=1e-4 unlock chain or F.6.1.6 4-segment step decay**. Documented deviation; SGD-momentum is plan G.8 ablation. |
| Replay buffer | Methods §Self-play | AGZ: last 500K games (~20-iter window @ 25K games/iter). **Stage 4: 5000 instances (~5-iter window @ 1K instances/iter)**. Proportionally tighter — F.6.1.1 found longer windows hurt by retaining stale MCTS targets. |

**Net deviation count: 6 documented** (optimizer, ε+τ-schedule coupling, wd, value-target-norm, **leaf_eval='rollout'**, **λᵥ=0**); all empirically validated by F.6.0.5+ ablations and Stage 5 §C/§D bottleneck-probe chain. Structural algorithm (PUCT + visit-distillation + gated-best-player) is AGZ-faithful.

---

## §F MCTS wall-time optimization stack (engineering enabler — see stage5_progress §F)

All optimizations preserve **bit-exact determinism** (`max_abs_cost_diff = 0.0e+00` on paired-seed runs). They are pure engineering wins, not algorithm changes.

| optimization | scope | wall payoff (TSP-50 K=50 M=1000) | code location |
|---|---|---|---|
| Fix #1 — `bytes(visited)` cache key body | `_solve_chunk.cache_key` | (part of cumulative −34%) | `solver.py:739-758` |
| Fix #2b — vectorize `rollout_many` state + masked argmax (numpy) | rollout path | (part of cumulative −34%) | `solver.py:1098-1210` |
| Fix #3 — numpy-direct evaluator `eval_many_arrays` for rollout | rollout path | (part of cumulative −34%) | `solver.py:823-1011` |
| Fix #4 — cache stores numpy arrays (`.copy()` 35× faster than `.tolist()`) | eval cache | 1255 → 830 s/iter | `solver.py:957-1005, 1075-1093` |
| Fix #5 — bulk-vectorize cache-key construction (2-tuple `(packed_header_int, visited_bytes)`) | eval cache | 830 → ~900 s/iter (probe-overestimated; ~3% in steady-state) | `solver.py:739-769, 863-924` |
| **Track A** — per-row `state.i` in decoder; merge rollout step groups into one NN call | decoder + rollout | **900 → 310 s/iter (−65%)** | `state.py:5-19`, `decoder.py:_get_step_context`, `solver.py:911-1011, 1138-1210` |
| §G — `mcts_batch_size` default 64 → 1000 | cross-instance chunking | 124 → 25 s/iter on F.6.1.3 recipe (5×) | `train_alphazero.py:150-158` |

**Combined: 1255 s/iter (pre-fix) → 310-320 s/iter (Track A), −75%.** A 50-iter TSP-50 lv0 run now fits in ~4h, down from ~17h. See `_progress/stage5_progress.md` §F for the full trajectory.

**Production decomposition at trained ckpt** (TSP-50 K=50 M=1000, `iter-68_accepted.pt`, A10G, 2026-05-13 probe — `src/scripts/probe_mcts_decomp.py`):

| phase | wall (s) | share |
|---|---:|---:|
| Decoder NN forward | 74.7 | 19.5% |
| C++ `collect_requests` (PUCT walk) | 39.7 | 10.3% |
| C++ `apply_results` (backup) | 8.9 | 2.3% |
| Python remainder (cache loop + tensor bridge + numpy state ops) | 260.7 | **67.9%** |
| Total | 383.9 | 100% |

Per-row Python cache lookup loop (`eval_many_arrays` body, ~100s) is the dominant remaining slab. T2.1 `torch.compile` probe (2026-05-13) bought ~5% but introduces dynamo dispatch tax — parked. Identified next-tier candidate: T2.2 = move rollout state machine into C++ (estimated −40% wall, ~2-3 days dev; not scheduled).

---

## §8 Critical files referenced

**All implementation work (Phases A–F) is complete.** Below maps each algorithm component to its current file.

**Consumed as-is from earlier stages:**
- `src/am_baseline/model/attention_model.py` — `model.encode`, `model.precompute_decoder`, `model.decode_step(return_glimpse=True)`, `model.value_head`.
- `src/am_baseline/search/puct.py:7-33` — exact PUCT formula in §4.1.
- `src/am_baseline/utils/tensor_ops.py:57-78` — `value_targets_from_edges` produces V_CURRENT shape (used at smoke tests; F.6.1 trainer recomputes $z_t$ inline per the §3.7 normalization mode).
- `src/am_baseline/baseline/baselines.py:106-123` — `RolloutBaseline.epoch_callback` — exact gating procedure in §5.
- `src/am_baseline/search/mcts_cpp/solver.py` — `CppBatchMCTSSolver` (the self-play backend $\alpha_\theta$).

**Edited / extended through F.6.1.3+lv0 + Stage 5 ablations:**
- `src/am_baseline/problem/state.py` — `StateTSP.i` may now be scalar (existing fast-path) or per-row `(B,)` tensor (Track A, for MCTS rollout path).
- `src/am_baseline/model/decoder.py` — `_get_step_context` has per-row branch path via `torch.where(state.i.view(-1,1,1) == 0, placeholder, gathered)`; scalar fast-path preserved bit-for-bit via `state.i.numel() == 1` guard. Training-time `decode_step` uses scalar path; MCTS rollout uses per-row path.
- `src/am_baseline/search/mcts.py` — `MCTSConfig.{return_root_visits, temperature_schedule, value_target_norm, n_simulations_per_step}`; `_apply_dirichlet`, `_resolve_tau`, `_convert_value_head_output` helpers.
- `src/am_baseline/search/mcts_cpp/{mcts.hpp, mcts.cpp, bindings.cpp, solver.py}` — Python-side mirror; `BatchSearch` cross-instance scheduler; `Solver::convert_value_head_output` for value_target_norm scale conversion; six wall-time optimizations (Fix #1-5 + Track A) — see §F.
- `src/am_baseline/training/trainer.py::train_step_alphazero` — §3 loss with lv0 mode + per-loss grad-norm split (`policy_grad_norm`, `value_grad_norm`, `value_grad_norm_vh`, `value_grad_norm_shared`).
- `src/am_baseline/training/coach.py` — `MCTSCoach` (§2 outer loop with mcts-quality telemetry), `MCTSReplayBuffer` (§3.6 stratified-by-step sampler), `make_self_play_config` plumbing all knobs.
- `src/am_baseline/training/logging.py` — `iterations.csv` schema + W&B aliases; new fields: `mcts_delta_vs_greedy_mean`, `mcts_win_rate_vs_greedy`, `greedy_cost_mean`, `mcts_cost_mean`.
- `src/scripts/train_alphazero.py` — CLI flags: `--lr_model`, `--lr_decay`, `--lr_decay_step_size`, `--weight_decay`, `--lambda_v`, `--dirichlet_epsilon`, `--dirichlet_alpha_factor`, `--value_target_norm`, `--leaf_eval`, `--gate_mode`, `--gate_every`, `--temperature_schedule`, `--buffer_capacity`, `--mcts_batch_size`, `--n_simulations_schedule`, `--n_simulations_first/late/last`, `--val_seed`, `--freeze_encoder`, `--resume_from`. **lr-override-on-resume**: when `--resume_from` is paired with `--lr_model`, the optimizer's restored lr is overwritten in both `optimizer.param_groups[0]['lr']` and `lr_scheduler.base_lrs` (train_alphazero.py:331-342) — required for the F.6.1.4 chain unlock pattern.
- `src/scripts/modal_run_train_alphazero.py` — Modal entrypoints for the F.6.0.5+ ablation series, F.6.1 main + lrdecay variants, F.6.1.3 step10+ε sweep, F.6.1.4 lr-unlock chain, F.6.1.6 step-decay 400-iter, rollout-λᵥ ablation (lv0 from-scratch + resumes), TSP-50 K-comparison + K=50 lv0 chain + Track A relaunches, K-bracket, mcts_batch_size sweep. Plus probe entrypoints: `run_probe_mcts_decomp`, `run_probe_triton_diag`.
- `src/scripts/probe_*.py` — diagnostic probes (Stage 5 §C):
  - `probe_grad_norm.py` — Stage 1 vs Stage 4 raw-gradient-norm (motivates lr=5e-4).
  - `probe_value_aleatoric.py` — value-head MSE decomposition into Var(z|s) + bias² (Stage 5 §C.2).
  - `probe_mcts_quality.py` — MCTS-vs-greedy buffer-quality probe.
  - `probe_mcts_decomp.py` — wall decomposition + `torch.compile` A/B (Stage 5 F.6/F.7).
- `src/scripts/val_stage4_mcts.py` — val-time MCTS sweep across leaf_eval × K (used for §C.3 vh-bypass result and §D.5 lv0 sweep).
- `src/scripts/smoke_alphazero.py` — A1..A6 smokes (gradient flow, buffer invariants, save/load, decoder graph correctness, gating mock, full coach round-trip).

---

## §9 Verification

This document is descriptive (algorithm spec, not new code). It is verified by:

1. **Smoke A1** (`smoke_alphazero.py`): construct a 5-instance buffer with random $\pi_t, z_t$; one call to `train_step_alphazero` produces a finite scalar matching the §3 formula. Also verifies $\lVert\nabla\mathcal{L}_v\rVert^2 = \lVert\nabla\mathcal{L}_{v,\text{vh}}\rVert^2 + \lVert\nabla\mathcal{L}_{v,\text{shared}}\rVert^2$ (per-loss split invariant from §3.5).
2. **Smoke A1.5**: cost-to-go consistency on a known tour: buffer-push pipeline produces identical $z_t$ as `value_targets_from_edges` from Stage 1.
3. **Smoke A3**: in self-play with `step10`, $\sigma_t$ collapses to one-hot at step $\lceil 0.1N \rceil$, but $\pi_t$ stays bounded above zero throughout — entropy decay is on $\sigma_t$, not $\pi_t$.
4. **Smoke A4**: visit-count consistency $\sum_t \sum_a N(s_t, a) \le K \cdot N$ for one instance.
5. **Smoke A6 (full coach round-trip)**: 2-iter end-to-end on TSP-8 random init produces a save/load-roundtrippable state. Determines whether new flags (lv0, n_simulations_per_step, lr_decay_step_size) break the full loop.
6. **Production trajectory descent**: monotone-ish descent of `val_avg_cost` along the F.6.1.3+lv0 chain matches §2 outer-loop expectation (well-documented via wandb runs `1syc0kk8`, `d8uyrrm1`, `7ybaqa12`, `1wpkngg9`, `0d48yqys`).
7. **MCTS backend bit-equivalence (deterministic settings)**: Python and C++ backends produce identical $\pi_t$ at fixed seed **with `dirichlet_epsilon=0` and `temperature=0`**. Under production config (ε=0.25, step10), the two backends consume RNG state independently — distributional equivalence only, no bit-match invariant.
8. **MCTS $\pi_t$ legality** (tree_reuse=True, production config): for every tour-step, $\pi_t$ sums to 1, $\text{support}(\pi_t) \subseteq \text{unvisited}(s_t)$ (subset, not equality), $\arg\max \pi_t$ is unvisited, $\pi_t \ge 0$ everywhere.
9. **Track A determinism**: TSP-20 K=40 M=100 and TSP-50 K=50 M=200 paired-seed produce `max_abs_cost_diff = 0.0e+00` between pre-Track-A and post-Track-A code paths. Tours identical.
10. **Wall-time fix determinism**: Each of Fix #1-5 verified the same paired-seed identity individually (max_diff = 0.0).
11. **lv0 value-grad invariant**: `value_grad_norm_shared == 0` exactly on every train step when `lambda_v=0` (mechanical check via the per-loss split telemetry from §3.5).

---

**Last updated:** 2026-05-13. Cross-links: `_plans/stage4_plan.md` (Stage 4 engineering phases, closed), `_plans/stage5_plan.md` (Stage 5 ablations + scaling, active), `_progress/stage4_progress.md`, `_progress/stage5_progress.md`.

**Refinement log:**
- **2026-05-13**: Major update to reflect F.6.1.3+lv0 locked recipe and Stage 5 §C–§D findings.
  - **Recipe drift**: ε=0 → **ε=0.25 (paired with step10 τ-schedule)**; `step30` → **`step10`**; `leaf_eval='value_head'` → **`leaf_eval='rollout'`**; new $\lambda_v=0$ (**lv0**) default; `mcts_batch_size` 64 → **1000**.
  - **New §3.4** documenting lv0 mode (policy-only training; value head receives no gradient).
  - **New §3.5** documenting per-loss gradient-norm split telemetry (`policy_grad_norm`, `value_grad_norm`, `value_grad_norm_vh`, `value_grad_norm_shared`).
  - **§4.1.5 rewrite**: rollout is the new default; AGZ-canonical value-net leaf eval is the alternative; cite Stage 5 §C vh-leaf-eval bias finding (0.074 RMS bias, structural).
  - **§4.3 rewrite**: ε=0.25 + step10 coupling rationale (F.6.1.3 winner); old ε=0 default superseded.
  - **New §4.5**: `CppBatchMCTSSolver` cross-instance scheduling + Track A per-row `state.i` (MCTS rollout uses per-row; training-time decode_step preserves scalar fast-path).
  - **New §F**: MCTS wall-time optimization stack (Fix #1-5 + Track A → 75% wall reduction at TSP-50 K=50 M=1000) with production decomposition.
  - **§6 pseudocode rewrite** with lv0 defaults, CppBatchMCTSSolver invocation, per-loss grad-norm split flow, lr-override-on-resume note.
  - **§7 AGZ mapping**: deviation count 4 → **6** (added leaf_eval='rollout' and λᵥ=0).
  - **§8 file inventory** updated: new probes (`probe_value_aleatoric.py`, `probe_mcts_decomp.py`), `val_stage4_mcts.py`, lr-override infrastructure, lv0 implementation in trainer.
  - **§9 verification** updated: smokes reflect current flag surface; Track A determinism check added.
- **2026-05-06**: F.6.0.5→F.6.1 lockdown (lr=5e-4, wd=0, value_target_norm='none', ε=0, gate_every=1, buffer=5000). 4 deviations from AGZ canonical documented (optimizer, ε, wd, value-target-norm). LambdaLR infrastructure. **Superseded by 2026-05-13 refinement above** — ε=0 default reversed when paired with step10; leaf_eval, λᵥ defaults migrated to lv0.
- **2026-04-30**: Initial creation. Reflected Phase F.4 design: $\eta=10^{-4}$, wd=1e-4, ε=0.25, value_target_norm='bl' (single mode), G=5, buffer=200K. Implementation phases A-D planned but not complete.
