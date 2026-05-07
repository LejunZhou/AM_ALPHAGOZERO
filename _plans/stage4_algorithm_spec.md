# Stage 4 Algorithm & Formulas — Reference Spec

**Companion to:** `_plans/stage4_plan.md` (engineering phases) and `_progress/stage4_progress.md` (run status).
**Purpose:** mathematical / algorithmic description of the AlphaGo-Zero-style training loop Stage 4 implements, grounded in actual code paths and mapped 1-to-1 to AGZ paper equations.
**Created:** 2026-04-30. **Refined 2026-05-06** to reflect the F.6.0.5→F.6.1 locked recipe (lr=5e-4 / wd=0 / value_target_norm='none' / ε=0 / gate_every=1 / buffer=5000).

---

## Context

The Stage 4 plan (`_plans/stage4_plan.md`) describes the engineering work in phases (A–G). This document presents the same loop in algorithm/equation form, grounded in actual code paths it composes:

- AM model API in `src/am_baseline/model/attention_model.py`
- MCTS solvers in `src/am_baseline/search/mcts.py` + `mcts_cpp/`
- PUCT selection in `src/am_baseline/search/puct.py:7-33`
- Gating in `src/am_baseline/baseline/baselines.py:106-123`
- Value normalization in `src/am_baseline/training/trainer.py` (per-state cost-to-go target with three normalization modes — see §3.5)

It maps 1-to-1 onto the AlphaGo Zero paper (Silver et al., *Nature* 550, 354–359, 2017) eq. (1) + Methods §Search algorithm, with **four documented deviations from AGZ canonical** (Adam vs SGD-momentum, ε=0 vs 0.25, wd=0 vs 1e-4, raw value target vs bl-normalized) all empirically validated by the F.6.0.5+ ablation series.

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
| $z_t \in \mathbb{R}_+$ | **Per-state value target** — V_CURRENT cost-to-go from partial state $s_t$. **Three normalization modes** via `--value_target_norm` ∈ {`bl`, `none`, `sqrt_n`}; F.6.1 default is `none` (raw cost-to-go, see §3.5). |
| $\alpha_\theta$ | MCTS solver wrapping current $f_\theta$; given $s_0$, returns $(\text{tour}, z, \{\pi_t\}_{t=0}^{N-1})$ |
| $\mathcal{D}$ | replay buffer: ring-buffer of instance records $\{(s_t, \pi_t, z_t)\}_{t=0}^{N-1}$, fixed capacity in *instances* (default 5000 = ~5-iter window at M=1000) |
| $K$ | MCTS simulations per move (F.6.1 K=20 default; K=40 / K=50 also tested) |
| $M$ | self-play instances per iteration (= 1000; F.6.1.0 confirmed this saturates per-iter train budget) |
| $\tau$ | sampling temperature for MCTS root action selection |
| $c_\text{puct}$ | PUCT exploration constant (= 0.05, Stage 2-locked) |
| $\varepsilon, \alpha$ | Dirichlet noise weight and concentration. **F.6.1 default $\varepsilon=0$** (F.6.0.6 winner, stability-revised); ε=0.25 was AGZ inheritance and is dominated under V3 regime. $\alpha = 10/N$ retained ($\alpha N = 10$ matches AGZ effective concentration). |
| $\eta, \mathrm{wd}$ | optimizer lr / weight decay. **F.6.1 default $\eta=5\times 10^{-4}$ Adam, $\mathrm{wd}=0$** (F.6.0.5b V3 winner). lr_decay schedule supported via `--lr_decay` (LambdaLR; default 1.0 = constant). |
| $G$ | gate cadence in iters. **F.6.1 default $G=1$** (revised from earlier $G=5$ for best_model freshness — see §5). |

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

**F.6.1-locked default values:**
- $\eta_0 = 5\times 10^{-4}$ (F.6.0.5b V3); `lr_decay` ∈ {`1.0` (const, F.6.1 main), `0.95` (F.6.1 lrdecay variant)} — both reach the same plateau ~3.92.
- $\mathrm{wd} = 0$ (F.6.0.5b drops AGZ-canonical wd=1e-4 in favor of AM-paper Adam wd=0).
- $G = 1$ (revised 2026-05-06 from G=5; per-iter best_model refresh propagates each accepted improvement to the next iter's self-play. Cost: +5-10s/iter validation, negligible vs ~25-115s/iter mcts_s).
- Buffer capacity = 5000 instances (≈5-iter window at M=1000); F.6.1.1 winner. The default 200K (= effectively never-evicting at our M) was actively dragging the policy back via stale MCTS targets — see §3.7.

---

## §3 Loss function

Per-tuple loss for one record $(s_t, \pi_t, z_t)$, equivalent to AGZ eq. (1):

$$
\boxed{\quad
\mathcal{L}(\theta;\, s_t, \pi_t, z_t) \;=\; \underbrace{\big(z_t - v_\theta(s_t)\big)^2}_{\text{value MSE (per-state cost-to-go)}}
\;-\; \underbrace{\sum_{a \in \mathcal{A}(s_t)} \pi_t(a) \cdot \log p_\theta(a \mid s_t)}_{\text{policy distillation (cross-entropy)}}
\;+\; \underbrace{c\,\|\theta\|_2^2}_{\text{L2, via } \texttt{weight\_decay}}
\quad}
$$

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

The L2 term is implemented via `torch.optim.Adam(..., weight_decay=c)` with $c = 10^{-4}$, identical to the explicit form for adaptive optimizers under standard usage.

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

## §3.5 Value normalization — caveats and design notes

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

### §4.1.5 Leaf evaluation choice — value-net only (AGZ-canonical)

AGZ uses **pure value-network leaf evaluation, no rollouts of any kind** — this is the defining algorithmic change from AlphaGo Lee → AlphaGo Zero. From Methods §Search algorithm (p.358):

> "The leaf node $s_L$ is added to a queue for neural network evaluation, $(d_i(\mathbf{p}), v) = f_\theta(d_i(s_L))$ ... The leaf node is expanded and each edge $(s_L, a)$ is initialized to $\{N=0, W=0, Q=0, P=p_a\}$; the value $v$ is then backed up."

And from the abstract / §AlphaGo versions (p.357):

> "AlphaGo Zero ... learns from self-play reinforcement learning, starting from random initial weights, **without using rollouts** ... It only uses its deep neural network to evaluate leaf nodes and to select moves."

| Version | Leaf evaluation |
|---|---|
| AlphaGo Fan / Lee | $V(s_L) = \lambda \cdot v_\theta(s_L) + (1-\lambda) \cdot z_\text{rollout}$ — value mixed with fast-rollout-policy game completion |
| **AlphaGo Zero** | $V(s_L) = v_\theta(s_L)$ — value net only |
| AlphaGo Master | same as Zero (value-net only), but with handcrafted features and SL initialization |

**Stage 4 follows AGZ-canonical**: $V(s_L) = v_\theta(s_L)$. Stage 3's `rollout` leaf eval (which beat `value_head` at test-time gap-red 65.2% vs 53.3% on a Stage-1-trained value head) is queued as Phase G.1 ablation. The Stage 4 hypothesis: under MCTS-distillation pressure the value head improves enough to flip the inequality — i.e., AGZ's design is right *once the loop has trained the value head against MCTS targets*.

### §4.2 Root action sampling and training target — *two distinct distributions*

After $K$ simulations from root $s_t$, we extract **two** distributions from the visit counts $\{N(s_t,a)\}_{a \in \mathcal{A}(s_t)}$:

**(a) Action-selection distribution $\sigma_t$** (used to sample the played action $a_t \sim \sigma_t$):

$$
\boxed{\quad
\sigma_t(a) \;=\; \frac{N(s_t,a)^{1/\tau_t}}{\sum_b N(s_t,b)^{1/\tau_t}},\qquad
\tau_t \;=\; \begin{cases} 1.0 & \text{if } t < \lceil 0.3 N \rceil \\ 0^+ & \text{otherwise} \end{cases}
\quad}
$$

where $\tau \to 0^+$ means deterministic argmax. (`step30` schedule, AGZ-proportional. `step50` and `const` are G.4 ablations.) The chosen action becomes the new root via tree reuse.

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

**ε default revised by F.6.0.6:** $\varepsilon = 0$ for TSP-20 (was $\varepsilon=0.25$ AGZ inheritance). The 2-variant ε∈{0, 0.05} sweep at V3 settings with ε=0.25 from F.6.0.5b as implicit reference produced:

| ε | val_avg_cost(iter 19) | per-iter regressions ≥ 0.05 |
|---|---|---|
| **0.00** | 4.228 | **0/19** (perfectly smooth) |
| 0.05 | 4.186 (lucky-stop low; iter-18 was 4.738) | 7/19 (with +0.46 spike) |
| 0.25 | 4.265 | (similar volatility expected) |

Endpoint at iter 19 favored ε=0.05 by 0.043, but trajectory inspection showed ε=0.05's iter-19 was the trough of a 0.55-amplitude oscillation (iter 17→18: +0.46, iter 18→19: −0.55). For F.6.1's 100-iter horizon, **smooth convergence dominates a noise-floor endpoint advantage** → ε=0 wins.

**TSP-50 may still benefit from small ε.** The F.6.1 TSP-50 probe uses ε=0.05 because (a) the action space (50 cities) is more uniform at random init (`α·N = 10` perturbation has more cities to redistribute over); (b) at higher N, the value head bootstraps slower, so MCTS Q-values are uninformative for more iters → exploration via ε is more useful. This is hypothesis-driven; will revisit after the trajectory lands.

**Why ε=0 doesn't break exploration**: the value head provides leaf-discrimination once it learns non-trivial across-instance variance (under raw-target Option B; see §3.5). UCB selection via $c_\text{puct}\, P\, \sqrt{N_\text{tot}}/(1+N_a)$ explores prior tail mass already; per-instance state variation across M=1000 instances per iter provides the "exploration in data space" that AGZ used ε for in the per-instance dimension.

### §4.3.5 Train-only vs eval-only — when does noise apply?

**Dirichlet noise is added during self-play training only, not during evaluation/inference.** AGZ Methods §Self-play (p.358) is the only place noise is described:

> "Additional exploration is achieved by adding Dirichlet noise to the prior probabilities in the root node $s_0$ ... this noise ensures that all moves may be tried, but the search may still overrule bad moves."

Methods §Evaluator (p.358) describes candidate-vs-best evaluation with no mention of noise:

> "Each evaluation consists of 400 games, using an MCTS with 1,600 simulations to select each move, **using an infinitesimal temperature $\tau \to 0$** (that is, we deterministically select the move with maximum visit count, to give the strongest possible play)."

The asymmetry: noise is an *exploration mechanism for data generation* (ensures the visit distribution being distilled is informative even on actions the prior dismisses), not a search-quality lever (it can only hurt by pulling priors toward random actions). Three call sites in Stage 4:

| Call site | Phase | $\tau$-schedule | Dirichlet | Rationale |
|---|---|---|---|---|
| `generate_self_play_batch` | training data gen | `step30` | **ε=0** for TSP-20 (F.6.1 default); **ε=0.05** for TSP-50 (current probe) | F.6.0.6 trajectory-stability finding (TSP-20); hedge for TSP-50 |
| Acceptance criterion 2 (greedy eval) | evaluation | n/a (greedy decoding, no MCTS) | **none** | Measure network's standalone quality |
| Stage 3-style test-time MCTS (G ablation) | inference | τ→0 throughout | **none** | AGZ §Evaluator/Evaluation — strongest play |

Implementation note: under tree reuse, the Dirichlet draw is *resampled freshly* each time a child becomes the new root — never propagated. `mcts.py:226-228, 444-460` and the C++ mirror already handle this correctly.

### §4.4 Final outputs

After $N$ tour-steps, MCTS returns:

- $\text{tour} = (a_0, a_1, \dots, a_{N-1})$
- $\text{tour\_cost} = \sum_{t=0}^{N-1} \|x_{a_t} - x_{a_{t+1 \bmod N}}\|_2$
- $\{\mathrm{lengths}_t\}_{t=0}^{N-1}$ — cumulative cost of edges traversed before $s_t$ (= `state.lengths`)
- $\{\pi_t\}_{t=0}^{N-1}$ — temperature-1 normalized visit distributions (training target)
- $\{z_t\}_{t=0}^{N-1}$ where $z_t = (\text{tour\_cost} - \mathrm{lengths}_t)/\mathrm{bl\_val}(x)$ — per-state V_CURRENT targets

Phase A's `MCTSConfig.return_root_visits=True` exposes the raw visit dicts $\{N(s_t, \cdot)\}$; the buffer-push step in Phase C computes $\pi_t$ and $z_t$ from these.

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

**F.6.1-locked recipe** (TSP-20; values in `[brackets]` are the locked defaults):

```
Input:  No checkpoint (from-scratch random init; F.6 supersedes F.4 warm-start)
        I = 100                   # outer iterations
        M = 1000                  # self-play instances per iter         [F.6.1.0 saturates train budget]
        K = 20                    # MCTS simulations per move            [F.6.1 K=20 default; K=40 also viable]
        J = 200                   # train steps per iter                 [F.6.1.0]
        B = 512                   # mini-batch size                      [F.6.0.8 batch=2048 only +0.02]
        G = 1                     # gate every G iterations              [revised 2026-05-06]
        η₀ = 5e-4                 # Adam initial lr                      [F.6.0.5b V3 winner]
        lr_decay = 1.0            # constant lr (or 0.95 for lrdecay variant)
        c = 0.0                   # weight_decay                         [F.6.0.5b V3]
        ε = 0.0                   # Dirichlet root noise                 [F.6.0.6 stability winner]
        α = 10/N = 0.5            # Dirichlet concentration
        c_puct = 0.05             # PUCT exploration constant            [Stage 2-locked]
        leaf_eval = 'value_head'  # AGZ-canonical                         [F.6.0.7→F.6.1.1]
        value_target_norm = 'none'# raw cost-to-go target                [F.6.0.5b Option B]
        gate_mode = 'ttest'       # paired-t α=0.05
        temperature_schedule = 'step30'   # τ=1 for first ⌈0.3N⌉ steps, τ=0 otherwise
        buffer_capacity = 5000    # ≈5-iter window at M=1000             [F.6.1.1 winner]
        val_seed = 42, val_size = 10000

Initialize:
        θ ← random init                  # trainer's working copy
        θ★ ← deepcopy(θ)                 # best-player snapshot
        D ← MCTSReplayBuffer(capacity_instances=5000)
        Opt ← Adam(θ.parameters(), lr=η₀, weight_decay=c=0)
        Sched ← LambdaLR(Opt, lambda i: lr_decay ** i)
        V ← fixed val set of 10K instances (val_seed=42); bl_vals ← rollout(θ★, V)

For i = 0, …, I−1:

    # ---- (1) Self-play with θ★ ----
    {x⁽ᵐ⁾} ← sample M random TSP-N instances
    For each x⁽ᵐ⁾:
        bl_val_m ← cost(greedy_rollout(θ★, x⁽ᵐ⁾))             # frozen at push time
        (tour, cost, {πₜ, σₜ, lengthsₜ}) ← α_{θ★}(x⁽ᵐ⁾)        # πₜ = N/Σ N (target); σₜ = step30 (action)
        For t in 0..N-1:
            cost_to_go_t ← cost − lengthsₜ                    # remaining tour cost from sₜ
            zₜ ← cost_to_go_t                                  # value_target_norm='none' (Option B)
                                                                # if ='bl': zₜ /= bl_val_m
            D.push((sₜ, πₜ, zₜ, bl_val_m))                    # bl_val stored for MCTS scale-conversion only

    # ---- (2) Distillation training ----
    For j = 1, …, J:
        Batch ← D.sample(B)                                    # stratified by step (§3.6)
        L_batch ← (1/B) Σ [(zₜ − v_θ(sₜ))² − πₜ · log p_θ(·|sₜ)]    # eq. §3 (no L2 since c=0)
        Opt.zero_grad(); L_batch.backward()
        clip_grad_norm_(θ.parameters(), max_norm=1.0)          # clip is near no-op under Adam
        Opt.step()

    # ---- (3) Gating ----
    If (i+1) mod G == 0:                                       # G=1 → every iter
        candidate_vals ← rollout(θ, V)                          # greedy, no MCTS, no Dirichlet
        if mean(candidate_vals) < mean(bl_vals):
            t, p = scipy.stats.ttest_rel(candidate_vals, bl_vals)
            if (p/2) < 0.05 and t < 0:
                θ★ ← deepcopy(θ)
                bl_vals ← candidate_vals                        # refresh baseline cache

    # ---- (4) Log ----
    log(iter=i, total_instances=(i+1)·M,
        val_avg_cost=mean(rollout(θ, V)),
        lr=Opt.param_groups[0]['lr'],
        gated=((i+1) mod G == 0), accepted, …)

    # ---- (5) Step lr scheduler ----
    Sched.step()                                                # no-op when lr_decay=1.0

Return θ★, θ
```

**F.6.1 K=20 main result** (val_avg_cost(iter 99) on 10K val_seed=42):
- lr_decay=1.0 (constant): ~3.92-3.93 plateau
- lr_decay=0.95 (decaying): 3.922 (geometric mean lr ~7.7e-5 over 100 iters)

Stage 1 ceiling: 3.839. **F.6.1 closes the gap to 0.08 cost units (2% relative) at ~7.8% of Stage 1's instance budget**, validating the proposal sample-efficiency claim.

---

## §7 Mapping to AGZ paper equations

| Stage 4 (F.6.1 locked) | AGZ paper | Identity / deviation |
|---|---|---|
| §3 loss | eq. (1), p.355 | $\ell = (z-v)^2 - \pi^\top\log\mathbf{p} + c\lVert\theta\rVert^2$ — **structurally identical**. **Reward shape** differs ($z \in \mathbb{R}_+$ raw cost-to-go vs $\{-1,+1\}$, see §3.5). **$c=0$** in F.6.1 (vs AGZ $c=10^{-4}$); F.6.0.5b found AGZ-canonical wd actively hurts at Stage-4 lr=5e-4. |
| §4.1 PUCT | Methods §Search algorithm, p.358 | $U(s,a) = c_\text{puct}\,P(s,a)\,\sqrt{\sum_b N(s,b)}/(1+N(s,a))$ — **literal match**. |
| §4.1.5 leaf eval | Methods §Search algorithm + §AlphaGo versions, p.357-358 | $V(s_L) = v_\theta(s_L)$, no rollouts — **literal match** (F.6.0.9 confirmed rollout adds 3.2× compute without quality benefit at our regime). |
| §4.2 root sampling | Methods §Self-play, p.358 | $\pi_a \propto N(s_0,a)^{1/\tau}$ — **literal match**; $\tau$-schedule scaled (Go: 30 of ~250 plies; TSP-20: 6 of 20 plies). |
| §4.3 Dirichlet | Methods §Self-play, p.358 | $P \leftarrow (1-\varepsilon)p + \varepsilon\eta$ — formula identical, **$\varepsilon=0$ in F.6.1** (vs AGZ $\varepsilon=0.25$); F.6.0.6 found ε=0 wins on stability under V3 regime. $\alpha = 10/N$ retained ($\alpha N=10$ matches AGZ effective concentration). Train-only (eval uses no noise). |
| §5 gating | Methods §Evaluator, p.358 | AGZ uses 400-game match with 55% win threshold; Stage 4 uses paired t-test on 10K-instance val (lower variance, $\alpha=0.05$ — comparable confidence; same spirit). **$G=1$** in F.6.1 (vs AGZ's "every 1000 batches" ≈ G=5 proportional). |
| §2 loop structure | Methods §Self-play training pipeline, p.357-8 | best-player $\theta^\star$ generates data; trainer $\theta$ continues regardless of gate — **literal match**. |
| Optimizer | Methods §Optimization | AGZ: SGD+momentum 0.9, lr step-anneal {1e-2 → 1e-3 → 1e-4}. **F.6.1: Adam, lr=5e-4 const or lr=1e-3 with 0.95/iter decay**. Documented deviation; SGD-momentum-with-AGZ-schedule is plan G.8 ablation, not main run. |
| Replay buffer | Methods §Self-play | AGZ: last 500K games (~20-iter window @ 25K games/iter). **F.6.1: 5000 instances (~5-iter window @ 1K instances/iter)**. Proportionally tighter window — F.6.1.1 found longer windows actively hurt by retaining stale MCTS targets that drag the policy back toward earlier weaker θ★. |

**Net deviation count: 4 documented (optimizer, ε, wd, value-target-norm); all empirically validated by F.6.0.5+ ablations.** Structural algorithm (PUCT + leaf-eval-via-value-net + visit-distillation + gated-best-player) is AGZ-faithful.

---

## §8 Critical files referenced

**All implementation work (Phases A–F) is complete.** Below maps each algorithm component to its current file.

**Consumed as-is from earlier stages:**
- `src/am_baseline/model/attention_model.py` — `model.encode`, `model.precompute_decoder`, `model.decode_step(return_glimpse=True)`, `model.value_head`.
- `src/am_baseline/search/puct.py:7-33` — exact PUCT formula in §4.1.
- `src/am_baseline/utils/tensor_ops.py:57-78` — `value_targets_from_edges` produces V_CURRENT shape (used at smoke tests; F.6.1 trainer recomputes $z_t$ inline per the §3.5 normalization mode).
- `src/am_baseline/baseline/baselines.py:106-123` — `RolloutBaseline.epoch_callback` — exact gating procedure in §5.
- `src/am_baseline/search/mcts_cpp/solver.py` — `CppBatchMCTSSolver` (the self-play backend $\alpha_\theta$).

**Edited / extended through F.6.1:**
- `src/am_baseline/search/mcts.py` — added `MCTSConfig.{return_root_visits, temperature_schedule, value_target_norm}`; `_apply_dirichlet`, `_resolve_tau`, `_convert_value_head_output` helpers.
- `src/am_baseline/search/mcts_cpp/{mcts.hpp, mcts.cpp, bindings.cpp, solver.py}` — Python-side mirror of the above; `Solver::convert_value_head_output` for value_target_norm scale conversion at all 4 leaf-eval call sites.
- `src/am_baseline/training/trainer.py` — `train_step_alphazero` implements §3 loss; `--value_target_norm` branch reconstructs $z_t$ from buffer-stored normalized form.
- `src/am_baseline/training/coach.py` — `MCTSCoach` (§2 outer loop), `MCTSReplayBuffer` (§3.6 stratified-by-step sampler with ring-buffer eviction), `make_self_play_config` plumbing all knobs.
  - LR scheduler: `LambdaLR(optimizer, lambda i: lr_decay**i)` wired up; `step()` called at end of each iter; checkpoint save/load includes scheduler state.
- `src/am_baseline/training/logging.py` — `iterations.csv` schema and W&B cross-stage aliases (`epoch=iter`, `val_avg_cost`, `epoch_duration=mcts_wall_s+train_wall_s`, `baseline_updated`, `lr`, `global_step`, `value_loss`).
- `src/scripts/train_alphazero.py` — CLI wrapper; flags include `--lr_model`, `--lr_decay`, `--weight_decay`, `--dirichlet_epsilon`, `--dirichlet_alpha_factor`, `--value_target_norm` ∈ {`bl`, `none`, `sqrt_n`}, `--leaf_eval` ∈ {`value_head`, `rollout`}, `--gate_mode` ∈ {`ttest`, `always`, `never`}, `--gate_every`, `--temperature_schedule`, `--buffer_capacity`, `--val_seed`.
- `src/scripts/modal_run_train_alphazero.py` — Modal entrypoints for the F.6.0.5+ ablation series and F.6.1 main + lrdecay variants + TSP-50 probe.
- `src/scripts/smoke_alphazero.py` — A1..A6 smokes (gradient flow, buffer invariants, save/load, decoder graph correctness, gating mock, full coach round-trip).
- `src/scripts/probe_grad_norm.py` *(NEW Stage 4 diagnostic)* — Stage 1 vs Stage 4 raw-gradient-norm comparison (motivates the lr=5e-4 derivation in F.6.0.5).

---

## §9 Verification

This document is descriptive (algorithm spec, not new code). It will be verified by:

1. **Smoke A1** (Phase B): construct a 5-instance buffer with random $\pi_t, z_t$; one call to `train_step_alphazero` should produce a finite scalar matching the §3 formula computed by hand.
2. **Smoke A1.5** (Phase B): cost-to-go consistency on a known tour: for a Stage 1 greedy tour with `value_targets_from_edges` returning `(N,)` targets, verify our buffer-push pipeline produces identical $z_t$ values for the same tour. Confirms we reuse the correct V_CURRENT shape.
3. **Smoke A3** (Phase E): in self-play with `step30`, the *action distribution* $\sigma_t$ collapses to one-hot at step $\lceil 0.3N \rceil$, but the *training target* $\pi_t$ stays bounded above zero throughout — entropy decay is on $\sigma_t$, not $\pi_t$.
4. **Smoke A4** (Phase F.2): visit-count consistency $\sum_t \sum_a N(s_t, a) \le K \cdot N$ for one instance.
5. **F.3 pilot**: monotone descent of `val_avg_cost` matches the §2 outer-loop expectation.
6. **Phase A bit-equivalence (deterministic settings only)**: Python and C++ MCTS backends produce identical $\pi_t$ at fixed seed **with `dirichlet_epsilon=0` and `temperature=0` clamped**. This isolates the deterministic logic surface (PUCT, FPU, expand/backup, tree-reuse advance). Under the production self-play config (Dirichlet on, `step30`), both backends consume RNG state independently — Python via NumPy Mersenne Twister, C++ via `std::random` — so they diverge even when both are correct. Cross-backend $\pi_t$ equality under the production config is **not** a correctness invariant; only distributional equivalence is, which a smoke test does not check.
7. **Phase A π_t legality** (tree_reuse=True, production config): for every tour-step, $\pi_t$ sums to 1, $\text{support}(\pi_t) \subseteq \text{unvisited}(s_t)$ (subset; PUCT may leave some legal actions with $N=0$ at low $K$), $\arg\max \pi_t$ is unvisited, $\pi_t \ge 0$ everywhere. **Subset, not equality** — equality would falsely fail when a sharp prior + small $c_\text{puct}$ keeps some legal actions unexplored.

---

**Last updated:** 2026-05-06. Cross-link: see `_plans/stage4_plan.md` for engineering phases and `_progress/stage4_progress.md` for run status (F.6.0.5b → F.6.1 results documented there).

**Refinement log:**
- **2026-05-06**: Updated to reflect F.6.0.5→F.6.1 locked recipe. Documented 4 deviations from AGZ canonical (Adam vs SGD-momentum, ε=0 vs 0.25, wd=0 vs 1e-4, raw value target via `value_target_norm='none'` vs bl-normalized) — all empirically validated. Added gate cadence revision $G=5\to 1$ rationale, lr scheduler infrastructure (LambdaLR), buffer-window finding (5000 ≪ 200K). All implementation work (Phases A-F) marked complete.
- **2026-04-30**: Initial creation. Spec reflected Phase F.4 design: $\eta=10^{-4}$, wd=1e-4, ε=0.25, value_target_norm='bl' (single mode), G=5, buffer=200K. Implementation phases A-D were planned but not yet complete.
