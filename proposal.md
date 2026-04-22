# AM + AlphaGo Zero: MCTS-Guided Policy Improvement for Combinatorial Optimization

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

### Stage 6: Extension to CVRP (Stretch Goal)
**Goal:** Demonstrate generality beyond TSP by applying the full pipeline to Capacitated Vehicle Routing.

**Tasks:**
- Adapt masking and state representation for CVRP (capacity constraints, depot returns)
- Adapt value head target (CVRP tour length normalization)
- Train and evaluate on CVRP-20/50/100

**Expected Outcome:**
- Competitive with AM's CVRP results
- Evidence that MCTS-based training is problem-agnostic (only masking changes)

---

## Summary of Expected Progression

| Stage | What Changes | TSP-20 Gap Target | TSP-100 Gap Target |
|-------|-------------|-------------------|-------------------|
| 0 | Reproduce AM | ~0.34% (greedy) | ~4.53% (greedy) |
| 1 | + Value head | ~0.34% (unchanged) | ~4.5% (unchanged) |
| 3 | + MCTS test time | ~0.1% | ~2.0% (beat sampling-1280) |
| 4 | + MCTS training | ~0.05% | ~1.5% (fewer samples needed) |
| 5 | + Tuning/ablations | best achievable | best achievable |

Each stage builds on the previous one, with a clear checkpoint and fallback. If Stage 3 (MCTS at test time) doesn't show improvement, we diagnose before proceeding. If Stage 4 (full loop) shows improvement but not sample efficiency, that's still a publishable negative result about the computational tradeoff.
