# AM + AlphaGo Zero: MCTS-Guided Policy Improvement for Combinatorial Optimization

> **Revision 2026-10-07 (approved by Lejun).** The TSP chapter (Stages 0–5) is closed.
> Its outcome is in [TSP chapter outcome](#tsp-chapter-outcome-stages-05-closed-2026-10-07).
> The project now tests the thesis on the **stochastic Prize-Collecting TSP (SPCTSP)**.
> That is the new [Stage 6](#stage-6-stochastic-prize-collecting-tsp-revised-2026-10-07),
> which replaces the CVRP stretch goal. The original text of Stages 0–5 is kept below
> as the plan that was executed.

## Overview

This project combines two systems that are structurally more compatible than they first appear:

- **Attention Model (AM)** (Kool et al., ICLR 2019): A Transformer encoder-decoder that constructs solutions to routing problems (TSP, CVRP, etc.) autoregressively — selecting one node at a time via attention. Trained with REINFORCE and a greedy rollout baseline.
- **AlphaGo Zero** (Silver et al., Nature 2017): A self-play reinforcement learning system that uses Monte Carlo Tree Search (MCTS) as a policy improvement operator. A dual-headed network (policy + value) is trained to match MCTS search probabilities, creating a self-improving cycle.

The key observation is that both systems make **sequential decisions**: AM decodes one node at a time; AlphaGo Zero plays one move at a time. The action spaces map directly — "select next node in the tour" is analogous to "play next move on the board." The AM paper itself notes that its greedy rollout baseline is "analogous to AlphaGo self-play," but stops short of actually using MCTS. This project closes that loop.

## Core Thesis

**AM's REINFORCE training leaves significant performance on the table that MCTS-based policy improvement can recover, with better sample efficiency.**

Evidence supporting this thesis:

1. **REINFORCE is high-variance.** Each gradient update uses a single sampled trajectory and a scalar reward. The signal — "this trajectory was better/worse than the greedy baseline" — is noisy and requires many samples.

2. **AM already benefits from brute-force search at test time.** Sampling 1280 solutions and taking the best drops the TSP-100 gap from 4.53% to 2.26%. This reveals untapped potential in the model that a more intelligent search (MCTS) could exploit more efficiently.

3. **MCTS provides a richer training signal.** Instead of one scalar per trajectory, MCTS produces an improved action distribution at every decoding step. Training the policy to match these distributions is approximate policy iteration, which has stronger convergence properties than policy gradient methods.

4. **AlphaGo Zero demonstrates massive search amplification.** The ~2100 Elo gap between the raw network and the MCTS-guided player shows that search discovers improvements the network alone misses. Even a fraction of this effect applied to routing would be significant.

The proposed approach:
- Use AM's Transformer encoder-decoder as the backbone (replacing AlphaGo Zero's ResNet)
- Add a **value head** to estimate expected solution quality from partial states
- Replace REINFORCE with the **AlphaGo Zero training loop**: MCTS generates improved action distributions as training targets
- Train with the combined loss: value prediction (MSE) + policy distillation (cross-entropy) + regularization

The central research question is whether the sample efficiency gains from MCTS-based training outweigh the per-sample computational cost — and under what problem scales and budgets the tradeoff favors this approach over standard REINFORCE.

### Revised thesis (2026-10-07)

On deterministic TSP the answer was no, and the reason is structural. The greedy
rollout of a strong policy is already a cheap and almost exact value estimate, so:
- a value head can at best reproduce it, which saves compute but adds no information;
- the search's gains are short lookahead corrections that the network absorbs only in
  small part.

The thesis is therefore narrowed to problems where **a single rollout is a poor
estimate of a state's value**. The test case is stochastic transitions, where a
rollout sees one random future. The thesis becomes:

**Where outcomes are stochastic, a learned value function carries information that
single rollouts provide only by averaging many of them. MCTS guided by that value,
and training that distils the search, should then beat REINFORCE at matched compute.**

SPCTSP is the first test case. It is already supported by the AM reference code,
pretrained AM models exist for 20/50/100 nodes, and it changes only the
masking/state machinery. A node's real prize is revealed only on arrival, and the
tour may end only once the collected prize reaches 1. Measured on 2,000 random
instances (CPU, 2026-10-07), the pretrained AM greedy policy shows:

| size | mean cost | SD across instances | SD across prize realisations, same instance |
|---|---:|---:|---:|
| SPCTSP-20 | 3.254 | 0.39 | 0.23 |
| SPCTSP-50 | 4.645 | 0.31 | 0.18 |

The noise from the prize realisation alone is comparable to the spread across
instances. On TSP that noise is zero.

---

## Research Plan

### Stage 0: Reproduce AM Baseline
**Goal:** Get a working AM implementation that matches published results on TSP.

**Tasks:**
- Set up the codebase from `ref/attention-learn-to-route-master/`; verify it runs on CPU
- Train AM on TSP-20 (small, fast iteration) and confirm convergence
- Train AM on TSP-50 and TSP-100; compare greedy and sampling-1280 results against the paper
- Record training curves (loss, tour length vs. epoch) as our baseline reference

**Expected Outcome:**
- Reproduced AM results within ~1% of published numbers
- Baseline numbers: TSP-20 greedy ~3.85, TSP-50 greedy ~5.80, TSP-100 greedy ~8.12
- Measured wall-clock time and sample count to reach convergence (the "sample efficiency" baseline)
- A clean, modular AM codebase we can extend in later stages

**Key Metric:** Optimality gap (%) vs. Concorde at TSP-20/50/100

---

### Stage 1: Extend AM with a Value Head
**Goal:** Add a value head to the AM decoder and verify it can learn meaningful estimates of solution quality from partial states.

**Tasks:**
- Design the value head: small MLP on top of the decoder context (graph embedding + partial tour state) → scalar output
- Define the value target: normalized cost-to-go (total tour length minus cost so far, normalized by problem scale)
- Train with a joint loss: REINFORCE policy loss + MSE value loss (weighted by coefficient λ)
- Validate that the value head's predictions correlate with actual completion quality

**Expected Outcome:**
- Value head achieves reasonable R² (>0.7) in predicting final tour quality from partial states
- Policy performance is not degraded by the auxiliary value loss (shared backbone regularization may even help slightly)
- We understand what the value head finds easy/hard to predict (early steps vs. late steps, clustered vs. uniform instances)

**Key Metric:** Value prediction R², policy optimality gap unchanged or improved

---

### Stage 2: Implement MCTS for Routing
**Goal:** Build an MCTS module adapted from AlphaGo Zero's search, tailored to the sequential node-selection structure of routing problems.

**Tasks:**
- Implement the MCTS core: tree nodes, PUCT selection, expansion, backup
  - State = partial tour (ordered list of visited nodes + remaining capacity for CVRP)
  - Action = select next unvisited node (with feasibility masking)
  - Use the policy head `p(a|s)` as prior and value head `v(s)` for leaf evaluation
- Handle routing-specific details:
  - Feasibility masking in the tree (only expand valid next nodes)
  - No adversary — single-agent search (simpler than AlphaGo Zero)
  - Terminal value = negative normalized tour length (continuous, not ±1)
- Add Dirichlet noise at root for exploration: `P(s,a) = (1-ε)p_a + ε·η_a`, η ~ Dir(α)
- Tune α and ε for the routing domain (different from Go's 0.03/0.25)
- Implement temperature-based action selection: `π_a ∝ N(s,a)^(1/τ)`

**Expected Outcome:**
- A working MCTS that, given a trained AM+value network, produces solutions
- MCTS with 200 simulations/step should produce better tours than the greedy policy alone
- Verified correctness: all MCTS-produced tours are feasible

**Key Metric:** MCTS tour quality vs. greedy policy, at various simulation budgets (50, 100, 200, 400, 800)

---

### Stage 3: MCTS at Test Time Only (Search Amplification)
**Goal:** Validate that MCTS improves solution quality at test time, without changing the training procedure. This isolates the value of search from the value of the training loop change.

**Tasks:**
- Take the Stage 1 model (AM + value head, trained with REINFORCE)
- Apply MCTS at test time with varying simulation budgets
- Compare against AM's sampling-1280 baseline (brute-force search)
- Measure: solution quality vs. computation budget (forward passes)

**Expected Outcome:**
- MCTS outperforms greedy decoding significantly (target: 30-50% gap reduction on TSP-100)
- MCTS achieves comparable quality to sampling-1280 with fewer total forward passes (demonstrating search efficiency)
- Clear scaling curve: more simulations → better solutions, with diminishing returns

**Key Metric:** Optimality gap vs. number of forward passes (MCTS budget curve vs. sampling-K curve). This is the core "search efficiency" comparison.

---

### Stage 4: Full AlphaGo Zero Training Loop
**Goal:** Replace REINFORCE with the MCTS-based self-improvement cycle. This is the central contribution.

**Tasks:**
- Implement the training pipeline (3 components):
  1. **Data generation:** Current best model runs MCTS on random instances. Each instance produces training tuples `(s_t, π_t, z)` where `π_t` = MCTS visit distribution, `z` = final normalized tour length
  2. **Network training:** Sample mini-batches from replay buffer. Loss = `(z - v)² - π·log(p) + c||θ||²`
  3. **Evaluation & gating:** New checkpoint vs. current best on held-out instances. Adopt only if mean tour length is significantly better (paired t-test, α=0.05 — directly from AM's baseline update mechanism)
- Design choices to tune:
  - Replay buffer size (last N instances)
  - MCTS simulations per step during training (tradeoff: more sims = better targets but slower)
  - Temperature schedule (high early for exploration, low later for exploitation)
  - Gating threshold and evaluation set size
- Start with TSP-20, then scale to TSP-50

**Expected Outcome:**
- The self-improvement loop converges: tour quality improves over successive iterations
- **Sample efficiency:** Reaches AM-equivalent quality with fewer total training instances (the core thesis)
- **Ultimate quality:** Surpasses AM's best results at equal or greater training budget
- Training curves show the characteristic AlphaGo Zero pattern: rapid early improvement, gradual refinement

**Key Metrics:**
- Tour length vs. training instances (sample efficiency curve, compared to AM REINFORCE baseline)
- Tour length vs. wall-clock time (practical efficiency curve)
- Tour length vs. training iteration (self-improvement convergence)

---

### Stage 5: Systematic Experiments and Ablations
**Goal:** Understand what matters, what doesn't, and how the approach scales.

**Ablation studies:**
- **Value head contribution:** Full system vs. MCTS with policy-only (no value head, use rollout instead)
- **MCTS budget during training:** 50 vs. 200 vs. 800 simulations per step
- **Replay buffer size:** Small (recent data only) vs. large (more diversity)
- **Gating vs. no gating:** Does the evaluator prevent regression?
- **Training loss:** AlphaGo Zero loss vs. REINFORCE + value auxiliary loss (Stage 1 approach)

**Scaling experiments:**
- TSP-20 → TSP-50 → TSP-100: How does the advantage scale with problem size?
- Generalization: Train on TSP-50, test on TSP-100 (does MCTS training improve generalization?)
- Transfer to CVRP: Same architecture, different masking — does the approach transfer?

**Expected Outcome:**
- Clear understanding of which components drive the improvement
- Scaling trends that predict performance on larger instances
- Sufficient data for a paper's experiment section

**Key Deliverable:** A results table comparing all variants across TSP-20/50/100, with sample efficiency curves and ablation analysis.

---

### TSP chapter outcome (Stages 0–5, closed 2026-10-07)

**Reproduction and value head (Stages 0–2).** These held.
- AM was reproduced. Stage 1 (AM + value head) reached TSP-20 3.8394 and TSP-50 5.7999
  greedy without degrading the policy.
- The value head's R² was ≥ 0.996. R² proved to be the wrong metric, though: what
  matters is how well the head ranks sibling moves.

**Search at test time (Stage 3).** Search works.
- Rollout-leaf MCTS beats greedy decoding: TSP-20 K=100 −0.022; TSP-50 K=40 −0.063.

**The from-scratch training loop (Stage 4) never reached REINFORCE.**
- TSP-20: best 3.8486 vs 3.8394.
- TSP-50: best 5.93 vs 5.80, on different validation draws, but the gap is far
  larger than set-to-set noise.

**Why the value head looked broken (Stage 5).**
- On the training distribution, the head is well calibrated.
- On moves the policy did not take it is badly optimistic, so it ranks siblings
  worse than the policy prior does.
- Two remedies worked:
  - Distilling rollouts on those counterfactual moves (§V0) gave rollout-level
    ranking on TSP-20.
  - Adding endpoint and geometry inputs (§I Step 1) let search recover 70–91% of
    the rollout's gain on TSP-50.
- On TSP, a repaired head can therefore only replace a rollout that is already
  cheap and nearly exact.

**Student isolation (§I Step 2): the decisive test.**
- One round of distilling a rollout-MCTS teacher into Stage 1 TSP-50 beat
  continued REINFORCE at matched wall time. The best target gained 0.0131 over
  Stage 1; REINFORCE gained 0.0034, which was not significant.
- But the student kept only about 21% of the teacher's 0.0635 gain (95% CI
  17–24%), below the pre-registered 25% floor. Outcome: STOP.

**Conclusion.** On deterministic TSP, AlphaGo-Zero-style training does not
deliver the claimed advantage over REINFORCE. This is the "publishable negative
result about the computational tradeoff" anticipated in the summary below.

Records:
- [`_progress/stage5_progress.md`](_progress/stage5_progress.md)
- [`_progress/stage5_offpolicy_value_progress.md`](_progress/stage5_offpolicy_value_progress.md)
- [`_progress/value_evaluator_repair_colab_progress.md`](_progress/value_evaluator_repair_colab_progress.md)
- [`_progress/student_isolation_colab_progress.md`](_progress/student_isolation_colab_progress.md)

**Lessons that carry forward:**
1. Judge evaluators by sibling ranking (decision regret against a ground truth),
   split by horizon, never by R².
2. Train value heads on off-policy children, not only on states the policy visits.
3. Compare every training method with continued REINFORCE at matched wall time.
4. Fix decision rules before the run, and use ≥ 3 seeds for any claim (seed noise
   on TSP-20 is ~0.01).

---

### Stage 6: Stochastic Prize-Collecting TSP (revised 2026-10-07)
**Goal:** Test the revised thesis where single rollouts are noisy value estimates.

Each step below has a gate that is fixed before its run. A failed gate stops the
chapter early, at small cost. Detailed protocol:
[`_plans/stage6_spctsp_plan.md`](_plans/stage6_spctsp_plan.md).

**Tasks (in order, each gated):**
1. **Environment and baselines.**
   - Port SPCTSP (and the deterministic PCTSP as a control) into the codebase:
     problem, state, decoder context and value target.
   - Load and reproduce the pretrained AM SPCTSP-20/50 models, and train a Stage-1
     style AM + value head.
   - Evaluate with common random numbers: one fixed prize realisation per test
     instance, shared by every policy, plus expected cost over many realisations.
2. **Value-signal probe.** At on- and off-policy states, rank sibling moves against
   ground truth (the mean of many rollouts). Compare:
   - the policy prior;
   - a single rollout;
   - an average of a few rollouts;
   - the value head, trained the §V0 way.

   Gate: the head clearly out-ranks a single rollout. If one rollout already
   ranks as well as the head, the thesis fails here too and the chapter stops.
3. **Search at test time.** Online stochastic MCTS. Real prizes are known for
   visited nodes, and unrevealed prizes are sampled per simulation. Leaf
   evaluation is one rollout, a rollout average or the value head, compared at
   matched wall time. Gate: search beats greedy, and value-guided search is at
   least as good as rollout-guided search at equal time.
4. **Student isolation.** The Step 2 design, unchanged: one round of distillation
   vs. continued REINFORCE at matched wall time. Gate: it beats the control (95% CI
   below 0) and keeps ≥ 25% of the teacher's gain.
5. **Full loop.** Only if 2–4 pass. Run with ≥ 3 seeds, compare in GPU-hours with
   REINFORCE and with the pretrained AM, at SPCTSP-20 then -50.

**Expected Outcome:** One of two clean answers.
- (a) The thesis holds where rollouts are noisy, with measured conditions (noise
  level, problem size) under which it pays off.
- (b) It fails here too, extending the negative result beyond deterministic routing.

**Stretch (not planned):** stochastic-demand CVRP, if Stage 6 passes.

---

## Summary of Expected Progression

| Stage | What Changes | TSP-20 Gap Target | TSP-100 Gap Target |
|-------|-------------|-------------------|-------------------|
| 0 | Reproduce AM | ~0.34% (greedy) | ~4.53% (greedy) |
| 1 | + Value head | ~0.34% (unchanged) | ~4.5% (unchanged) |
| 3 | + MCTS test time | ~0.1% | ~2.0% (beat sampling-1280) |
| 4 | + MCTS training | ~0.05% | ~1.5% (fewer samples needed) |
| 5 | + Tuning/ablations | best achievable | best achievable |
| 6 | SPCTSP (revised 2026-10-07) | gated; see Stage 6 | — |

*Outcome of rows 0–5 (2026-10-07): Stages 0–3 met their goals. The Stage 4/5
training loop did not beat REINFORCE on TSP. See the TSP chapter outcome above.*

Each stage builds on the previous one, with a clear checkpoint and fallback. If Stage 3 (MCTS at test time) doesn't show improvement, we diagnose before proceeding. If Stage 4 (full loop) shows improvement but not sample efficiency, that's still a publishable negative result about the computational tradeoff.
