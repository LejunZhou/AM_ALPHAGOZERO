# AM_ALPHAGOZERO Paper Survey

A structured summary of papers relevant to the ml4scip project. Each entry captures the key innovation, architecture, training approach, and relevance to our unified framework.

---

## Schema

Each paper entry follows this schema:

| Field | Description |
|-------|-------------|
| **Title** | Full paper title |
| **Venue** | Conference/journal, year |
| **Authors** | First author et al. |
| **Repo** | GitHub URL (if public) |
| **Local** | Path under `ref/` |
| **Task** | SCIP plugin type and decision being learned |
| **Architecture** | Neural network type and key dimensions |
| **Training** | IL / RL / SL / hybrid, loss function, oracle |
| **Key Innovation** | The core contribution to reuse |
| **Tricks & Details** | Implementation details critical for reproduction |
| **Benchmarks** | Instance types and sizes used |
| **Hyperparameters** | Key training hyperparameters |
| **Relevance** | What we reuse vs. improve |
| **Cites** | Key papers this work builds on |
| **Cited By** | Later papers in this survey that build on this work |
| **Reproduction** | Status: `not started` / `in progress` / `reproduced` / `deferred` |
| **Our Implementation** | Path under current directory (once implemented) |
| **Known Gaps** | Differences between our reproduction and the original |

---

## Paper 1 — Attention Model (AM) for Routing Problems

| Field | Value |
|-------|-------|
| **Title** | Attention, Learn to Solve Routing Problems! |
| **Venue** | ICLR 2019 |
| **Authors** | Wouter Kool, Herke van Hoof, Max Welling |
| **Repo** | https://github.com/wouterkool/attention-learn-to-route |
| **Local** | `ref/attention-learn-to-route-master/` |
| **Task** | Constructive heuristic — autoregressive node selection for routing (TSP, CVRP, SDVRP, OP, PCTSP, SPCTSP) |
| **Architecture** | Transformer encoder-decoder. **Encoder:** 3 MHA layers, 8 heads, d_model=128, d_k=d_v=16, FF hidden=512 (ReLU), skip-connections + batch norm, no positional encoding (permutation-invariant). **Decoder:** context = [graph_embed ∥ first_node_embed ∥ last_node_embed], MHA glimpse (8 heads), single-head attention with tanh clipping C=10 for logits, masking for feasibility constraints. |
| **Training** | **RL** — REINFORCE with greedy rollout baseline. Policy gradient: ∇L = E[(L(π) − b(s)) ∇log p(π\|s)]. Baseline is a frozen copy of the policy (greedy decoding), updated each epoch only if the current policy is significantly better (paired t-test, α=5%, on 10k validation instances). First epoch uses exponential moving average baseline (β=0.8) for warmup. |
| **Key Innovation** | (1) Replaces Pointer Network / LSTM with input-order-invariant Transformer — attention replaces recurrence in both encoder and decoder. (2) Greedy rollout baseline for REINFORCE analogous to AlphaGo self-play — stable, low-variance, no critic network needed. (3) Single unified architecture handles 6 different routing problem types by only changing the masking / context. |
| **Tricks & Details** | • No positional encoding — graph is a set, not a sequence. • Tanh clipping (C=10) on logits to bound exploration. • Graph embedding = mean of all node embeddings. • Greedy rollout baseline is frozen per epoch; t-test gates updates to prevent baseline degradation. • Sampling 1280 solutions at test time and taking the best significantly improves results. • For split-delivery VRP, a special decoder handles partial demands. • Masking differs per problem (capacity for CVRP, prize budget for OP, stochastic penalties for SPCTSP). |
| **Benchmarks** | Uniform random instances: TSP (n=20,50,100), CVRP (n=20,50,100), SDVRP (n=20,50,100), OP (n=20,50,100), PCTSP (n=20,50,100), SPCTSP (n=20,50,100). Compared against Concorde (TSP optimal), LKH3, OR-Tools, Gurobi, and learned baselines (Pointer Network, RL-Vinyals, GCN). |
| **Hyperparameters** | Adam lr=1e-4, batch=512, 2500 gradient steps/epoch, 100 epochs (~12.8M training instances), 1280 samples at test time, d_model=128, N_layers=3, N_heads=8, FF_hidden=512, clip C=10, baseline t-test α=0.05, EMA β=0.8 (epoch 1 only). |
| **Relevance** | **Reuse:** (1) The Transformer encoder is the canonical graph embedding backbone — we adopt its architecture for encoding SCIP bipartite graphs. (2) The greedy-rollout REINFORCE baseline is our default RL training recipe. (3) Masking mechanism extends naturally to SCIP feasibility constraints. **Improve:** (1) Replace node-only features with bipartite (variable + constraint) graph features for MIP. (2) Extend decoder to branching / cutting / scheduling decisions. (3) Investigate GNN alternatives to capture constraint structure the Transformer ignores. |
| **Cites** | Vinyals et al. 2015 (Pointer Networks), Bello et al. 2017 (RL for CO), Vaswani et al. 2017 (Transformer), Nazari et al. 2018 (RL for VRP), Dai et al. 2017 (Structure2Vec + DQN for CO) |
| **Cited By** | (to be filled as more papers are surveyed) |
| **Reproduction** | `not started` |
| **Our Implementation** | — |
| **Known Gaps** | — |

### Key Results Summary

| Problem | n | Greedy (gap%) | Sampling 1280 (gap%) |
|---------|---|--------------|----------------------|
| TSP | 20 | 0.34% | 0.08% |
| TSP | 50 | 1.76% | 0.52% |
| TSP | 100 | 4.53% | 2.26% |
| CVRP | 20 | 1.34% | 0.44% |
| CVRP | 50 | 2.53% | 1.02% |
| CVRP | 100 | 4.01% | 1.73% |

Gaps measured against optimal (Concorde for TSP) or best known (LKH3/Gurobi for CVRP).

---

## Paper 2 — AlphaGo Zero

| Field | Value |
|-------|-------|
| **Title** | Mastering the game of Go without human knowledge |
| **Venue** | Nature, Vol 550, October 2017 |
| **Authors** | David Silver, Julian Schrittwieser, Karen Simonyan et al. (DeepMind) |
| **Repo** | — (no official public repo) |
| **Local** | — |
| **Task** | Game playing — learning to play Go tabula rasa via self-play RL + MCTS. Not a SCIP plugin, but provides the foundational RL+search training paradigm we adapt. |
| **Architecture** | **Dual-headed ResNet.** Input: 19×19×17 binary feature planes (8 history steps × 2 players + colour-to-play). **Residual tower:** 1 convolutional block (256 filters, 3×3, stride 1, BN, ReLU) + 19 or 39 residual blocks (each: conv 256→BN→ReLU→conv 256→BN→skip→ReLU). **Policy head:** conv 2 filters 1×1→BN→ReLU→FC to 362 (19²+1 pass). **Value head:** conv 1 filter 1×1→BN→ReLU→FC 256→ReLU→FC 1→tanh (output in [−1, 1]). Total depth: 39 or 79 parameterized layers + heads. |
| **Training** | **RL** — Self-play reinforcement learning with MCTS as policy improvement operator. No human data. **Loss:** l = (z − v)² − π^T log p + c‖θ‖² (MSE on value + cross-entropy on policy + L2 regularization). **Pipeline (3 async components):** (1) **Self-play:** best player α_θ* plays 25,000 games/iteration, 1,600 MCTS simulations/move (~0.4s/move). (2) **Optimization:** SGD with momentum on mini-batches of 2,048 positions sampled uniformly from last 500k games. (3) **Evaluator:** each new checkpoint plays 400 games vs current best; replaces it only if win rate >55%. |
| **Key Innovation** | **(1) MCTS as policy improvement operator inside training loop.** MCTS search probabilities π are much stronger than raw network policy p; training the network to match π creates a self-improving cycle (approximate policy iteration). **(2) Unified dual-head network** — single ResNet outputs both policy and value, providing regularization via shared representation (+600 Elo over separate networks). **(3) Tabula rasa learning** — no human data, no handcrafted features, no rollout policy; only game rules as domain knowledge. Surpasses all prior AlphaGo versions (100-0 vs AlphaGo Lee in 72h, 89-11 vs AlphaGo Master in 40 days). |
| **Tricks & Details** | • **PUCT selection:** a_t = argmax_a [Q(s,a) + c_puct · P(s,a) · √(Σ_b N(s,b)) / (1+N(s,a))]. • **Temperature:** τ=1 for first 30 moves (exploration), τ→0 thereafter (exploitation). • **Dirichlet noise at root:** P(s,a) = (1−ε)p_a + ε·η_a, where η∼Dir(0.03), ε=0.25 — ensures all moves can be tried. • **Dihedral augmentation:** random rotation/reflection of position during NN evaluation in MCTS. • **Training data augmented** with all 8 rotations/reflections. • **No positional encoding needed** — CNN structure matches grid. • **Resignation:** auto-tuned threshold v_resign keeping false-positive rate <5%; disabled in 10% of games to calibrate. • **Tree reuse:** subtree below played move is retained. • **Virtual loss** for parallel MCTS threads. • **Evaluator gating** (>55% win rate) prevents baseline regression — analogous to AM's t-test gating. |
| **Benchmarks** | 19×19 Go. Internal Elo tournament: AlphaGo Zero (20 blocks, 3 days) = 4,000+ Elo, defeating AlphaGo Lee 100-0. AlphaGo Zero (40 blocks, 40 days) = 5,185 Elo, defeating AlphaGo Master 89-11. Raw network without MCTS: 3,055 Elo. Compared against AlphaGo Fan (3,144), AlphaGo Lee (3,739), AlphaGo Master (4,858), Crazy Stone, Pachi, GnuGo. |
| **Hyperparameters** | SGD with momentum=0.9. LR schedule (in 1000s of steps): 0-200k→10⁻², 200-400k→10⁻², 400-600k→10⁻³, 600-700k→10⁻⁴, >700k→10⁻⁴. Mini-batch=2,048 (32/worker × 64 GPU workers). L2 reg c=10⁻⁴. MCTS: 1,600 simulations/move, c_puct tuned via Gaussian process optimization. Replay buffer: last 500k games. Evaluator: 400 games, >55% win threshold. 20-block run: 4.9M self-play games, 700k mini-batch updates, ~3 days. 40-block run: 29M self-play games, 3.1M mini-batch updates, ~40 days. |
| **Relevance** | **Reuse:** (1) The **MCTS-as-policy-improvement** paradigm is the core algorithmic idea for our project — using search to generate training targets that are stronger than the raw network, then distilling back. This is the "AlphaGo Zero" in our project name. (2) The **evaluator gating** mechanism (only adopt new policy if statistically better) parallels AM's greedy-rollout baseline t-test. (3) The **dual-head architecture** (shared backbone, separate policy/value heads) is directly applicable to MIP solvers where we need both a branching policy and a value estimate. (4) **Self-play / self-improvement loop** — we adapt this to generate improving SCIP solve trajectories. **Improve:** (1) Replace 2D CNN with GNN/Transformer to handle non-grid MIP bipartite graphs. (2) Replace full-game MCTS with lookahead search over SCIP branching decisions (partial tree, not full game tree). (3) Adapt the training pipeline to work with MIP feasibility/optimality rather than win/lose outcomes. |
| **Cites** | Silver et al. 2016 (AlphaGo Fan — Nature), He et al. 2016 (ResNets), Ioffe & Szegedy 2015 (Batch Norm), Coulom 2006 / Kocsis & Szepesvári 2006 (MCTS + UCB), Tesauro 1994 (TD-Gammon self-play), Sutton & Barto 1998 (RL textbook) |
| **Cited By** | Kool et al. 2019 (Paper 1 — AM borrows the self-play baseline idea); Silver et al. 2018 (AlphaZero — generalization to chess/shogi); Schrittwieser et al. 2020 (MuZero — learned model) |
| **Reproduction** | `not started` |
| **Our Implementation** | — |
| **Known Gaps** | — |

### MCTS + RL Training Loop (Detail)

The core training loop that makes AlphaGo Zero relevant to our project:

1. **Self-play data generation:** The current best network f_θ* guides MCTS (1,600 sims/move). MCTS outputs search probabilities π_a ∝ N(s,a)^(1/τ). Games produce training tuples (s_t, π_t, z_t) where z_t = ±1 is game outcome.

2. **Network optimization:** Sample mini-batches from replay buffer (last 500k games). Minimize:
   - **Value loss:** (z − v)² — predict game outcome
   - **Policy loss:** −π^T log p — match MCTS search probabilities
   - **Regularization:** c‖θ‖²

3. **Evaluation & gating:** New checkpoint plays 400 games vs current best at τ→0. Only adopted if win rate >55% (avoids noise-driven regression).

4. **Key insight — why MCTS improves the policy:** MCTS explores multiple lines of play and backs up value estimates, producing π that is much stronger than the raw network's p. Training p→π forces the network to internalize the search's discoveries. This creates a virtuous cycle: better network → better MCTS → better training targets → even better network.

### Key Results Summary

| Version | Blocks | Training | Elo | vs AlphaGo Lee | vs AlphaGo Master |
|---------|--------|----------|-----|----------------|-------------------|
| AG Zero (small) | 20 | 3 days, 4.9M games | ~4,000 | 100-0 | — |
| AG Zero (large) | 40 | 40 days, 29M games | 5,185 | — | 89-11 |
| Raw network (no MCTS) | 40 | — | 3,055 | — | — |

The ~2,100 Elo gap between raw network (3,055) and full MCTS player (5,185) quantifies the search amplification effect.

---

## Paper 3 — BQ-NCO (Bisimulation Quotienting for Neural CO)

| Field | Value |
|-------|-------|
| **Title** | BQ-NCO: Bisimulation Quotienting for Efficient Neural Combinatorial Optimization |
| **Venue** | NeurIPS 2023 |
| **Authors** | Darko Drakulic, Sofia Michel, Florian Mai, Arnaud Sors, Jean-Marc Andreoli (Naver Labs Europe) |
| **Repo** | (released per paper, URL placeholder in paper) |
| **Local** | `ref/BQ-NCO.pdf` |
| **Task** | Constructive heuristic for COPs: Euclidean TSP, ATSP, CVRP, OP, Knapsack. Generic MDP formulation framework. |
| **Architecture** | Transformer (9 layers, 12 heads, d=192, FF=512) with ReZero normalization, no positional encoding. Learnable origin/destination encodings added to node embeddings. **No encoder/decoder split** — entire model runs at every construction step. For ATSP: adds graph-conv layer using normalized cost matrix as edge weights and uses random node IDs. PerceiverIO variant gives linear attention. |
| **Training** | **Imitation learning** with cross-entropy loss on expert trajectories. Experts: Concorde (TSP), LKH (ATSP/CVRP), EA4OP (OP), DP (KP). 1M solutions, instances of size 100, 500 epochs, Adam lr=7.5e-4, batch=1024. Crucial trick: sample **sub-paths of random length n ∈ [4, N]** from each optimal solution — every sub-path is itself an optimal solution of a smaller sub-instance, yielding free data augmentation across sizes/distributions. |
| **Key Innovation — Partial Solution = New Sub-Problem (the core idea)** | **Bisimulation Quotienting (BQ).** Instead of representing the MDP state as `(instance, partial_solution)` (the "direct MDP" used by AM/POMO/etc.), BQ-NCO maps each partial solution `y` to the **tail sub-problem** `(f*y, X*y)` it induces, where `(f*y)(x)=f(y∘x)` and `X*y={x : y∘x ∈ X}`. The reduced MDP state **is itself a COP instance of the same type** — the original problem with already-chosen elements removed and parameters updated. This is a true bisimulation (proved in the paper): trajectories, rewards, and optimal policies are preserved. Many distinct `(instance, partial_solution)` pairs collapse to the same reduced state (e.g. any TSP partial tour ending at node e with unvisited set I gives the same sub-problem), exposing the problem's symmetry for free rather than forcing the network to learn it. |
| **How the sub-problem view works per COP** | • **TSP → path-TSP:** partial tour `x₁…xₖ` becomes a new path-TSP instance with origin=xₖ, same destination, unvisited nodes as customers. TSP is path-TSP with origin=destination. • **CVRP → path-CVRP:** partial solution becomes a path-CVRP instance with new origin=last node, **reduced remaining capacity** (full C minus cumulated demand served since last depot visit), unvisited customers. • **OP → path-OP:** new origin=last node, **remaining distance budget** decreased by traveled distance. • **KP:** picked items removed, **capacity updated** to C − Σ weights of picked items, remaining items form new KP instance. This "tail-recursion property" generalizes the Optimality Principle of Dynamic Programming — any DP-amenable COP satisfies it. |
| **Architectural consequence** | Because the state IS an instance, there is no encoder/decoder dichotomy. The same network runs on the current sub-instance at every step. Cost: O(N³) total (N steps × O(N²) attention) vs O(N²) for AM. Benefit: the network is *re-embedding the remaining sub-problem every step* — far stronger than AM-style frozen encoding, and explains why a single greedy rollout beats beam-search/sampling from AM/POMO on large instances. |
| **Tricks & Details** | • **ReZero** normalization over LayerNorm. • **k-NN pruning at inference** (k=250 nearest to origin) — slight quality change, big speedup. • **Expert trajectory ordering matters for CVRP**: sorting subtours by remaining capacity (last subtour has largest leftover) ~2× improvement over arbitrary order. • **Random node IDs** as input feature for ATSP (no coordinates available) — optionally added as extra feature for other problems to improve performance. • **Sub-path sampling** acts as implicit size/distribution augmentation during training. • **Ablation:** approximating by freezing lower layers and recomputing only the top layer (MDAM-style) degrades TSP100 gap from 0.35% → 8.18% — confirms full re-embedding of sub-problem is the critical factor. |
| **Benchmarks** | Trained on N=100, tested on N=100/200/500/1000 synthetic + TSPLib (up to 4461 nodes) + CVRPLib. Greedy rollout on TSP1000 gets 2.29% gap vs POMO's 40.60%, Sym-NCO's 37.51%. CVRP1000: 5.88% greedy vs POMO's 141%. |
| **Hyperparameters** | 9 layers, 12 heads, d_model=192, FF=512, ReZero; Adam lr=7.5e-4, decay 0.98/50 ep; batch=1024; 500 epochs; 1M training solutions of size 100; k-NN=250 at inference. |
| **Relevance to AM_ALPHAGOZERO** | **Reuse:** (1) The **sub-problem-as-state formulation** is directly applicable to MIP / SCIP — after a branching decision, the remaining MIP is a smaller MIP of the same type (tail-recursion via LP relaxation + bound tightening). This matches how SCIP itself works and suggests re-embedding the sub-MIP each node rather than freezing an initial encoder. (2) **Imitation from sub-trajectories**: sub-sequences of an optimal branch-and-bound trace are themselves training samples for smaller sub-MIPs — free augmentation. (3) **No encoder/decoder split** simplifies architecture; aligns with AlphaGo Zero's single dual-head network evaluating the current state. **Contrast with AlphaGo Zero path:** BQ-NCO shows IL on small instances + re-embedding ≫ RL with frozen encoder. Our project should consider whether a BQ-style state (current sub-MIP) + MCTS over branching actions is more sample-efficient than direct-MDP + RL. (4) The **bisimulation soundness proofs** (Prop. 1, 2) provide the formal justification for why treating the remaining sub-problem as the state loses no information. |
| **Cites** | Kool et al. 2019 (AM), Kwon et al. 2020 (POMO), Bresson & Laurent 2021 (TransformerTSP), Kim et al. 2022 (Sym-NCO), Bellman 1954 / Bertsekas 2012 (DP Optimality Principle), Vaswani et al. 2017 (Transformer), Jaegle et al. 2022 (PerceiverIO), Bachlechner et al. 2021 (ReZero). |
| **Cited By** | — |
| **Reproduction** | `not started` |
| **Our Implementation** | — |
| **Known Gaps** | — |

### Direct MDP vs BQ-MDP (schematic)

```
Direct MDP (AM, POMO, ...)          BQ-MDP (BQ-NCO)
---------------------------         -----------------------------
state  = (instance, partial x)      state  = sub-instance (f*x, X*x)
action = next construction step     action = next construction step
policy = π(a | instance, x)         policy = π(a | sub-instance)
encoder run ONCE per instance       model run EVERY step on sub-instance
must LEARN symmetry                  symmetry BUILT IN (many x → same sub-instance)
```

The central insight: **"partial solution" and "remaining sub-problem" are dual views**. BQ-NCO commits to the sub-problem view, which (i) collapses symmetric states, (ii) makes the state a first-class instance of the same COP, and (iii) forces the model to continuously re-evaluate the shrinking problem — closer in spirit to how AlphaGo Zero evaluates the current board position rather than the sequence of moves that led there.

---
