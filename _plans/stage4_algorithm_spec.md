# Stage 4 Algorithm & Formulas — Reference Spec

**Companion to:** `_plans/stage4_plan.md` (engineering phases) and `_progress/stage4_progress.md` (run status).
**Purpose:** mathematical / algorithmic description of the AlphaGo-Zero-style training loop Stage 4 implements, grounded in actual code paths and mapped 1-to-1 to AGZ paper equations.
**Created:** 2026-04-30.

---

## Context

The Stage 4 plan (`_plans/stage4_plan.md`) describes the engineering work in phases (A–G). This document presents the same loop in algorithm/equation form, grounded in actual code paths it composes:

- AM model API in `src/am_baseline/model/attention_model.py`
- MCTS solvers in `src/am_baseline/search/mcts.py` + `mcts_cpp/`
- PUCT selection in `src/am_baseline/search/puct.py:7-33`
- Gating in `src/am_baseline/baseline/baselines.py:106-123`
- Value normalization in `src/am_baseline/training/trainer.py:208-219`

It maps 1-to-1 onto the AlphaGo Zero paper (Silver et al., *Nature* 550, 354–359, 2017) eq. (1) + Methods §Search algorithm.

---

## §1 Notation

| Symbol | Meaning |
|---|---|
| $N$ | TSP graph size (= 20 in Stage 4) |
| $s$ | partial-tour state (StateTSP NamedTuple — `loc, dist, first_a, prev_a, visited_, lengths, i`) |
| $a$ | action = next-city index, $a \in \{0,\dots,N{-}1\}$, masked by `visited_` |
| $\theta$ | network parameters (encoder + decoder + value head); trainer's working copy |
| $\theta^\star$ | best-player parameters (used for self-play data generation) |
| $f_\theta(s) \to (\mathbf{p}, v)$ | dual-head AM: $\mathbf{p} \in \Delta^{N-1}$ via softmax over decoder logits, $v \in \mathbb{R}$ from value head |
| $\pi_t \in \Delta^{N-1}$ | **Training target** — temperature-1 normalized visit distribution at root of tour-step $t$, $\pi_t(a) = N(s_t,a)/\sum_b N(s_t,b)$ (raw normalized; richer than action-selection $\tau$-schedule, see §4.2) |
| $\sigma_t \in \Delta^{N-1}$ | **Action-selection distribution** at tour-step $t$, $\sigma_t(a) \propto N(s_t,a)^{1/\tau_t}$ with `step30` $\tau$-schedule. The played action $a_t \sim \sigma_t$ |
| $z_t \in \mathbb{R}_+$ | **Per-state value target** — V_CURRENT cost-to-go from partial state $s_t$, normalized by `bl_val`. Matches Stage 1's training target shape (see §3) |
| $\alpha_\theta$ | MCTS solver wrapping current $f_\theta$; given $s_0$, returns $(\text{tour}, z, \{\pi_t\}_{t=0}^{N-1})$ |
| $\mathcal{D}$ | replay buffer: deque of instance records $\{(s_t, \pi_t, z)\}_{t=0}^{N-1}$ |
| $K$ | MCTS simulations per move (50 pilot, 100 main) |
| $M$ | self-play instances per iteration (= 1000) |
| $\tau$ | sampling temperature for MCTS root action selection |
| $c_\text{puct}$ | PUCT exploration constant (= 0.05, Stage 2-locked) |
| $\varepsilon, \alpha$ | Dirichlet noise weight and concentration ($\varepsilon=0.25$, $\alpha=10/N$) |

---

## §2 Outer loop (one Stage 4 iteration)

For iteration $i = 0, 1, \dots, I{-}1$:

$$
\boxed{\quad
\begin{aligned}
&\textbf{(1) Self-play.}\quad \text{Sample } M \text{ random TSP-}N \text{ instances } \{x^{(m)}\}_{m=1}^M,\ x^{(m)} \in [0,1]^{N\times 2}.\\
&\quad\text{For each } m,\ \text{run } \alpha_{\theta^\star}(x^{(m)}) \to (\text{tour}^{(m)}, z^{(m)}, \{\pi_t^{(m)}\}_{t=0}^{N-1}). \\
&\quad\text{Push records } \{(s_t^{(m)}, \pi_t^{(m)}, z^{(m)})\}_{m,t} \text{ into } \mathcal{D}.\\[4pt]
&\textbf{(2) Train.}\quad \text{For } j = 1, \dots, J=\texttt{train\_steps\_per\_iter}: \\
&\quad B \sim \text{Uniform}(\mathcal{D},\ |B|=\texttt{batch\_size})\\
&\quad \theta \leftarrow \theta - \eta\, \nabla_\theta \mathcal{L}(\theta; B)\quad \text{(Adam, } \eta=10^{-4}, \text{wd}=10^{-4}\text{)}\\[4pt]
&\textbf{(3) Gate.}\quad \text{If } (i+1) \bmod G = 0:\\
&\quad \text{accept} \leftarrow \text{Gate}(\theta, \theta^\star,\ \text{val\_size}=10000,\ \alpha=0.05)\\
&\quad \text{If accept: } \theta^\star \leftarrow \theta\\[2pt]
&\textbf{(4) Log + checkpoint.}\quad \text{Record } \mathrm{val\_avg\_cost}(\theta), \text{ totals, gating outcome.}
\end{aligned}
\quad}
$$

**Key invariant** (matches AGZ Methods §Self-play training pipeline, p.357-358): $\theta^\star$ is updated only on gate accept; $\theta$ is updated every train step regardless. No rollback on reject.

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

**Value target — per-state cost-to-go** (matches Stage 1's V_CURRENT target shape; reuses `value_targets_from_edges` in `src/am_baseline/utils/tensor_ops.py:57-78`):

$$
\boxed{\quad
z_t \;=\; \frac{\mathrm{tour\_cost} - \mathrm{lengths}_t}{\mathrm{bl\_val}(x)} \;=\; \frac{\sum_{u=t}^{N-1}\|x_{a_u} - x_{a_{u+1 \bmod N}}\|_2}{\mathrm{bl\_val}(x)}
\quad}
$$

where $\mathrm{lengths}_t$ is the cumulative cost of edges already traversed (= `state.lengths` at step $t$) and $\mathrm{bl\_val}(x) = \mathrm{cost}(\mathrm{greedy\_rollout}_{\theta^\star}(x))$.

**Why per-state, not broadcast `z`:** the existing MCTS leaf evaluator (`src/am_baseline/search/mcts.py:1-15`, invariant 1) computes $V(s_L) = \mathrm{state.lengths}/\mathrm{bl\_val} + v_\theta(s_L)$, which assumes $v_\theta$ predicts **remaining cost-to-go from a partial state**, not the full tour. Training $v_\theta$ on broadcast full-tour `z` would double-count the path cost at MCTS time. Per-state $z_t$ matches the V_CURRENT shape Stage 1 trained against and makes Phase A's leaf evaluator a no-op compared to Stage 3.

**`bl_val` recomputation cadence:** once per training epoch under $\theta^\star$ (the model that produced the tour, not the trainer's evolving $\theta$ — see §3.5 Concern 2).

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

### Concern 1: `bl_val` drift — RESOLVED by frozen-at-generation design

In a hypothetical broadcast-z design where `bl_val` is recomputed under the trainer's evolving θ, the same tour's $z$ would grow across iterations as θ improves. **Stage 4 sidesteps this entirely:** `bl_val` is computed once at instance-push time (under $\theta^\star$, the model that produced the tour) and frozen for the record's lifetime in the buffer. Combined with per-state cost-to-go $z_t = (\text{tour\_cost} - \text{lengths}_t)/\text{bl\_val}_\text{frozen}$, the training target for any given buffer record is **stationary across all training steps that draw it**.

The only buffer-level "drift" is across records: newer instances were generated under stronger $\theta^\star$ with smaller `bl_val`, so their $z_t$ is on a slightly different scale than older records. This is the *correct* policy-iteration signal — the loop should learn that recent self-play is closer to optimal — not a moving-target pathology.

### Concern 2: `bl_val` model asymmetry

Tours come from $\theta^\star$ but `bl_val` could be computed from $\theta$. In between gate accepts these can diverge.

**Resolution (now baked into §3 default):** `bl_val(x) = cost(greedy_rollout_{θ★}(x))` — the same model that produced the tour. Computed inside `generate_self_play_batch` at self-play time, frozen until next gate accept. Cleaner semantics, zero compute cost.

### Concern 3: Stage 5 alternative

Originally, Stage 4's G.6 ablation was "per-step cost-to-go target instead of broadcast z". Per-state cost-to-go is **now the F.4 default** (this section), so G.6 is repurposed to **"best-so-far per-instance normalization"**:

$$
z_t^\text{best-so-far} \;=\; \frac{\mathrm{tour\_cost} - \mathrm{lengths}_t}{\min_{\text{seen}}\mathrm{tour\_cost}(x)}
$$

where the min is over all self-play attempts on instance $x$ across all iterations. Doesn't require an oracle, doesn't drift with the trainer, range $\geq 0$ with monotone optimum-tracking. **Strongest candidate for Stage 5 G.6 ablation**.

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

At each root $s_t$, the priors are perturbed exactly once before the $K$ simulations:

$$
\boxed{\quad
P(s_t, a) \;\leftarrow\; (1-\varepsilon)\, p_\theta(a\mid s_t) \;+\; \varepsilon\, \eta_a,\qquad \boldsymbol{\eta} \sim \mathrm{Dir}(\alpha\,\mathbf{1}_{|\mathcal{A}(s_t)|})
\quad}
$$

with $\varepsilon = 0.25$, $\alpha = 10/N = 0.5$ for TSP-20 (AGZ uses $\alpha = 0.03$ at $|\mathcal{A}|=362$; the heuristic $\alpha \approx 10/|\mathcal{A}|$ scales it).

### §4.3.5 Train-only vs eval-only — when does noise apply?

**Dirichlet noise is added during self-play training only, not during evaluation/inference.** AGZ Methods §Self-play (p.358) is the only place noise is described:

> "Additional exploration is achieved by adding Dirichlet noise to the prior probabilities in the root node $s_0$ ... this noise ensures that all moves may be tried, but the search may still overrule bad moves."

Methods §Evaluator (p.358) describes candidate-vs-best evaluation with no mention of noise:

> "Each evaluation consists of 400 games, using an MCTS with 1,600 simulations to select each move, **using an infinitesimal temperature $\tau \to 0$** (that is, we deterministically select the move with maximum visit count, to give the strongest possible play)."

The asymmetry: noise is an *exploration mechanism for data generation* (ensures the visit distribution being distilled is informative even on actions the prior dismisses), not a search-quality lever (it can only hurt by pulling priors toward random actions). Three call sites in Stage 4:

| Call site | Phase | $\tau$-schedule | Dirichlet | Rationale |
|---|---|---|---|---|
| `generate_self_play_batch` | training data gen | `step30` | **ε=0.25, α=0.5** | AGZ §Self-play — exploration for data diversity |
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
1. **Filter optimizer noise.** Adam's per-step updates can momentarily worsen the policy on val even if the long-run direction is correct. Gating at every $G$ iters checks at a coarser cadence than per-step.
2. **Snapshot for evaluation.** Headline plots use $\theta^\star$, not whatever $\theta$ happened to be at the last gradient step. Stabilizes reported numbers.
3. **Enable safe optimizer continuity.** Per scope decision 3 (matching AGZ), the trainer's optimizer state is *not* reset on gate reject. That's only safe because the gate decoupled "what generates data" from "what's training" — the trainer can explore noisy directions in parameter space without immediately corrupting $\mathcal{D}$.

**Gating is not strictly necessary at scale.** KataGo's `ref/KataGo-master/SelfplayTraining.md` notes:

> "Note that not using gating ... will be faster and will save compute power, and the whole loop works perfectly fine without it, but having it at first can be nice to help debugging and make sure that things are working and that the net is actually getting stronger."

KataGo can drop the gate because at scale (millions of games) individual bad checkpoints' contributions to the replay buffer get diluted by many other good checkpoints' data — the buffer-level distribution is robust even if individual snapshots aren't. Stage 4 doesn't have that dilution (1K instances/iter, ~1% of buffer per iter), so the gate is load-bearing.

**G.5.c ablation tests this directly**: drop the gate, always set $\theta^\star = \theta$, see if the loop still converges. Either outcome (gating-faster vs gating-load-bearing) is publishable evidence at our scale.

---

## §6 Algorithm in one block — full pseudocode

```
Input:  θ₀  (Stage 1 canonical TSP-20 checkpoint)
        I = 100   # outer iterations (main run)
        M = 1000  # self-play instances per iter
        K = 100   # MCTS simulations per move
        J = 200   # train steps per iter
        B = 512   # mini-batch size
        G = 5     # gate every G iterations
        η = 1e-4, c = 1e-4 (Adam lr / weight_decay)

Initialize:
        θ ← θ₀                           # trainer's working copy
        θ★ ← deepcopy(θ₀)                # best-player snapshot
        D ← MCTSReplayBuffer(capacity=200_000 instances)
        Opt ← Adam(θ.parameters(), lr=η, weight_decay=c)
        V ← fixed val set of 10K instances; bl_vals ← rollout(θ★, V)

For i = 0, …, I−1:

    # ---- (1) Self-play with θ★ ----
    {x⁽ᵐ⁾} ← sample M random TSP-N instances
    For each x⁽ᵐ⁾:
        bl_val_m ← cost(greedy_rollout(θ★, x⁽ᵐ⁾))             # bl_val from θ★ (§3.5 Concern 2)
        (tour, cost, {πₜ, σₜ, lengthsₜ}) ← α_{θ★}(x⁽ᵐ⁾)        # πₜ = N/Σ N (target); σₜ = step30-tempered (action)
        For t in 0..N-1:
            cost_to_go_t ← cost − lengthsₜ                    # remaining tour cost from sₜ (§3 boxed)
            zₜ ← cost_to_go_t / bl_val_m                      # per-state V_CURRENT target
            D.push((sₜ, πₜ, zₜ))                              # store πₜ (raw, τ=1), not σₜ

    # ---- (2) Distillation training ----
    For j = 1, …, J:
        Batch ← D.sample(B)                                    # uniform over per-step records
        L_batch ← (1/B) Σ [(zₜ − v_θ(sₜ))² − πₜ · log p_θ(·|sₜ)]    # eq. §3
        Opt.zero_grad(); L_batch.backward(); Opt.step()         # +c·‖θ‖² via weight_decay

    # ---- (3) Gating ----
    If (i+1) mod G == 0:
        candidate_vals ← rollout(θ, V)                          # greedy, no MCTS, no Dirichlet
        if mean(candidate_vals) < mean(bl_vals):
            t, p = scipy.stats.ttest_rel(candidate_vals, bl_vals)
            if (p/2) < 0.05 and t < 0:
                θ★ ← deepcopy(θ)
                bl_vals ← candidate_vals                        # refresh baseline cache

    # ---- (4) Log ----
    log(iter=i, total_instances=(i+1)·M,
        val_avg_cost=mean(rollout(θ, V)),
        gated=((i+1) mod G == 0), accepted, …)

Return θ★, θ
```

---

## §7 Mapping to AGZ paper equations

| Stage 4 | AGZ paper | Identity |
|---|---|---|
| §3 loss | eq. (1), p.355 | $\ell = (z-v)^2 - \pi^\top\log\mathbf{p} + c\lVert\theta\rVert^2$ — **literal match**; reward shape differs ($z \in \mathbb{R}_+$ vs $\{-1,+1\}$, see §3.5) |
| §4.1 PUCT | Methods §Search algorithm, p.358 | $U(s,a) = c_\text{puct}\,P(s,a)\,\sqrt{\sum_b N(s,b)}/(1+N(s,a))$ — **literal match** |
| §4.1.5 leaf eval | Methods §Search algorithm + §AlphaGo versions, p.357-358 | $V(s_L) = v_\theta(s_L)$, no rollouts — **literal match** |
| §4.2 root sampling | Methods §Self-play, p.358 | $\pi_a \propto N(s_0,a)^{1/\tau}$ — **literal match**; $\tau$-schedule scaled (Go: 30 of ~250 plies; TSP-20: 6 of 20 plies) |
| §4.3 Dirichlet | Methods §Self-play, p.358 | $P \leftarrow (1-\varepsilon)p + \varepsilon\eta$ — **literal match**; $\alpha$ scaled by $10/|\mathcal{A}|$ heuristic; train-only |
| §5 gating | Methods §Evaluator, p.358 | AGZ uses 400-game match with 55% win threshold; Stage 4 uses paired t-test on 10K-instance val (lower variance; same spirit) |
| §2 loop structure | Methods §Self-play training pipeline, p.357-8 | best-player $\theta^\star$ generates data; trainer $\theta$ continues regardless of gate — **literal match** |

---

## §8 Critical files referenced

**Consumed as-is (no edits):**
- `src/am_baseline/model/attention_model.py` — `model.encode`, `model.precompute_decoder`, `model.decode_step(return_glimpse=True)`, `model.value_head`.
- `src/am_baseline/search/puct.py:7-33` — exact PUCT formula in §4.1.
- `src/am_baseline/utils/tensor_ops.py:57-78` — `value_targets_from_edges` produces V_CURRENT shape; Stage 4's $z_t$ computation reuses this exact target shape (§3).
- `src/am_baseline/search/mcts.py:1-15` — leaf-evaluator invariant ($V(s_L) = \text{state.lengths}/\text{bl\_val} + v_\theta(s_L)$); confirms why per-state $z_t$ (not broadcast `z`) is the correct training target.
- `src/am_baseline/training/trainer.py:208-219` — value normalization scaling pattern.
- `src/am_baseline/baseline/baselines.py:106-123` — `RolloutBaseline.epoch_callback` — exact gating procedure in §5.
- `src/am_baseline/search/mcts_cpp/solver.py` — `CppBatchMCTSSolver` — the self-play backend $\alpha_\theta$.

**To be edited (Phases A + E):**
- `src/am_baseline/search/mcts.py:34-72` — `MCTSConfig` defaults; Stage 4 adds `return_root_visits` (Phase A) + `temperature_schedule` (Phase E).
- `src/am_baseline/search/mcts_cpp/{mcts.hpp, mcts.cpp, bindings.cpp, solver.py}` — same Phase-A and -E plumbing in C++.

**To be created:**
- `src/am_baseline/training/coach.py` *(Phase D)* — implements §2 outer loop, holds $\theta$ and $\theta^\star$ + replay buffer + gating baseline.
- `src/am_baseline/training/trainer.py::train_step_alphazero` *(Phase B extension)* — implements §3 loss.
- `src/scripts/train_alphazero.py` *(Phase F.1)* — CLI wrapper.
- `src/scripts/smoke_alphazero.py` *(Phase F.2)* — smoke A1..A6.

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

**Last updated:** 2026-04-30. Cross-link: see `_plans/stage4_plan.md` for engineering phases and `_progress/stage4_progress.md` for run status.
