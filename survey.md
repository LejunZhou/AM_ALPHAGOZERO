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
| **Cited By** | Paper 3 (BQ-NCO — replaces direct-MDP with BQ-MDP on the same Transformer backbone); Paper 4 (GOAL — generalizes AM-style architecture to multi-task with adapters); Paper 5 (POMO — keeps the AM network unchanged but replaces REINFORCE-with-greedy-rollout training by a multi-starting-node shared-baseline REINFORCE). |
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
| **Cited By** | Paper 4 (GOAL) — same authors extend BQ-MDP from single-task to multi-task via disjoint union, reuse tail-subproblem sampling, inherit sub-instance-as-state principle. |
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

## Paper 4 — GOAL (Generalist Combinatorial Optimization Agent Learner)

| Field | Value |
|-------|-------|
| **Title** | GOAL: A Generalist Combinatorial Optimization Agent Learner |
| **Venue** | ICLR 2025 |
| **Authors** | Darko Drakulic, Sofia Michel, Jean-Marc Andreoli (Naver Labs Europe) — **same team as BQ-NCO (Paper 3)** |
| **Repo** | Released (URL in paper — Naver Labs) |
| **Local** | `ref/GOAL.pdf` |
| **Task** | **Multi-task** constructive heuristic across 8 training tasks spanning 4 families: routing (ATSP, CVRP, CVRPTW, OP), scheduling (JSSP, UMSP), packing (KP), graph (MVC). Fine-tuned to 8 new tasks: TRP, PCTSP, OCVRP, SDCVRP, SOP, MCLP, MIS, OSSP. |
| **Architecture** | **Single shared backbone** (9 transformer layers, D=D̄=128, FF=512, 8 heads, ReZero norm, ~2.1M params) + **light-weight task-specific input/output adapters** (few thousand params each). **Three architectural innovations:** (1) **Mixed-attention blocks** replacing vanilla attention — edges projected twice as (K′, Q′); score = ⟨K+K′ \| Q+Q′⟩ instead of ⟨K \| Q⟩. Incurs no overhead when edges are absent. (2) **Shared codebook**: adapters first project task features to low dim (ℓ=8 nodes, ℓ̄=4 edges), then a **shared** ℓ×D matrix (the codebook) lifts to backbone embedding dim — forces cross-task representation sharing. (3) **Multi-type transformer**: for problems with heterogeneous node types (e.g. JSSP's ops + machines), each layer expands into n² mixed-attention blocks (self + cross per type pair) **sharing the same parameters per layer** — switches between single-type and multi-type configurations at no parameter cost. |
| **Training** | **Imitation learning** (cross-entropy on expert trajectories) in multi-task mode. Each step samples one task, builds a batch by drawing **tail-subproblems** of training instances (directly inherited from BQ-NCO's sub-path trick). Single-class CE for sequential problems (ATSP/CVRP/…), multi-class CE for KP/MVC. AdamW lr=5e-4, decay 0.97/10 epochs, batch=256, 8× V100 for 7 days (~400 epochs). 1M oracle-labeled instances per task. Oracles: LKH (ATSP/CVRP), HGS (CVRPTW), A4OP (OP), ORTools (KP/JSSP), FastWVC (MVC), HiGHS (UMSP). |
| **Fine-tuning (two modes)** | (a) **Supervised** (minutes): train adapters from scratch with backbone open; only 128 labeled instances, 1 step per tail-subproblem. (b) **Unsupervised** (hours): ExIt-style apprentice/expert loop where the expert samples 128 solutions per instance from the apprentice, picks the best if sufficiently better, and feeds it back for imitation — similar to Corsini et al.'s self-labeling for JSSP. Both systematically beat training from scratch at equal compute. |
| **Key Innovation — relation to BQ-NCO** | GOAL **generalizes BQ-NCO's single-task BQ-MDP to a multi-task BQ-MDP via disjoint union** `⊔_{t∈T} Ω_t`. The paper proves that (i) the disjoint union of tail-recursive MDPs is itself tail-recursive, and (ii) its BQ-MDP is the disjoint union of component BQ-MDPs. Shared backbone = shared value/policy over the joint state space; task-specific adapters = learnable projections from each task's feature space into that shared space. Same sub-instance-as-state principle as BQ-NCO, now with task ID prepended and adapters per task. |
| **Tricks & Details** | • **Mixed attention vs MatNet / G2G ablation:** GOAL's form beats both — converges faster than G2G and to much lower gap than MatNet on ATSP. • **Codebook ablation:** no effect on training performance, but **dramatically stabilizes fine-tuning** across 10 runs (esp. PCTSP, OCVRP, MCLP) — sparsification prevents each task from specializing its own embedding subspace. • **Multi-type vs single-type ablation (JSSP):** multi-type reaches 4.6% gap vs single-type's 8.9% at 200 epochs on 100K instances. • **Per-task heuristics** (e.g. KNN pruning) used to keep O(N²) attention tractable at inference. • Multi-type blocks can be **unshared at fine-tune time** for extra flexibility while remaining shared at pre-train time for task-agnosticism. • Input always includes a random node ID feature ∈ [0,1] to disambiguate otherwise-identical nodes (cf. random features in GNNs). • Origin/destination tokens used per task exactly as in BQ-NCO. |
| **Benchmarks** | 8 training tasks at size 100 (10×10 JSSP, 100×20 UMSP). **Single-task GOAL is SOTA on 7/8 training tasks** (beats BQ-NCO, POMO, MatNet, MVMoE, RouteFinder, COMPASS, Gumbeldore) at greedy decoding. **Multi-task GOAL** is only slightly worse than single-task (e.g. ATSP100: 0.30% → 0.91%, CVRP100: 2.34% → 3.16%). **Generalization to N=1000**: ATSP100→1000 gap = 1.96% vs BQ-NCO 8.09%, MatNet collapses. **Fine-tuning beats from-scratch training** on all 8 new tasks. |
| **Hyperparameters** | Backbone: L=9 layers, D=D̄=128, FF=512, 8 heads, ReZero. Adapters: ℓ=8 (nodes), ℓ̄=4 (edges). Codebook: shared 8×128 + 4×128 linear. AdamW lr=5e-4, decay 0.97/10 ep, batch=256, ~400 epochs, 8× V100, 7 days. Fine-tune: supervised 128 instances × 1 step; unsupervised ExIt 128 samples/instance. |
| **Relevance to AM_ALPHAGOZERO** | **Reuse:** (1) The **adapter + shared-backbone + codebook pattern** is a recipe for a "foundation-model for CO" that we can reuse when extending from single SCIP plugin to multiple plugins (branching + cutting + node selection + restart). Each plugin = one "task", shares a backbone over SCIP state. (2) **Mixed attention** is the right drop-in for SCIP bipartite graphs: the SCIP coefficient matrix is edge-level info (variable↔constraint) and needs to participate at score level, not just as node features. (3) **Multi-type transformer** directly applicable — SCIP graphs are bipartite (variables vs constraints), and cut pools/branches add further heterogeneous node types. Using type-specific MMA blocks with shared parameters is the correct generalization of the standard bipartite-MIP GNN (Gasse'19). (4) **Low-dim codebook** is a simple but effective regularizer to force feature sharing across MIP problem classes (TSP-as-MIP, knapsack-as-MIP, …) — we should imitate this to pretrain on multiple SCIP problem distributions. (5) **Unsupervised ExIt fine-tuning** is our pathway when oracle solutions for a new MIP class are unavailable: apprentice rollouts + brute-force sample-best → imitate. (6) **Sub-instance-as-state** inherited from BQ-NCO — same lesson, now at multi-task scale. **Contrast / open questions:** GOAL is still pure IL; marrying it with AlphaGo-Zero-style MCTS + self-play to *improve past* the oracle (or to replace the oracle entirely for novel MIP classes) is exactly our project's research gap. |
| **Cites** | Drakulic et al. 2023 (**BQ-NCO — direct predecessor**; re-uses BQ-MDP, tail-subproblem sampling, sub-problem-as-state, path-problem framing for CVRP/OP), Kool et al. 2019 (AM), Kwon et al. 2020 (POMO), Kwon et al. 2021 (MatNet — mixed-attention predecessor), Henderson et al. 2023 (G2G attention), Zhou et al. 2024 (MVMoE), Berto et al. 2024 (RouteFinder), Liu et al. 2024 (Multi-task routing), Anthony et al. 2017 (ExIt — fine-tuning loop), Corsini et al. 2024 (self-labeling JSSP), Ibarz et al. 2022 (Generalist Neural Algorithmic Learner), Reed et al. 2022 (GATO), Bapna & Firat 2019 (NMT adapters), Bachlechner et al. 2021 (ReZero). |
| **Cited By** | — |
| **Reproduction** | `not started` |
| **Our Implementation** | — |
| **Known Gaps** | — |

### GOAL vs BQ-NCO at a glance

| Aspect | BQ-NCO (Paper 3) | GOAL (Paper 4) |
|--------|------------------|----------------|
| Scope | Single task per model | **One backbone, 8 tasks** + 8 fine-tune tasks |
| State | Sub-instance (BQ-MDP) | **Disjoint union of BQ-MDPs** + task ID |
| Attention | Vanilla + optional graph-conv for ATSP | **Mixed-attention** (edges into score) unified |
| Heterogeneous nodes | ad-hoc per problem | **Multi-type transformer** with shared params |
| Feature encoding | Per-problem input projection | **Low-dim task adapter + shared codebook** |
| Normalization | ReZero | ReZero (same) |
| Training | IL, 500 epochs, N=100 | IL, ~400 epochs, 8-task interleaving, N=100 |
| Size | 9L, d=192, 12H (~larger) | 9L, d=128, 8H (~2.1M — smaller) |
| Generalization | Strong at N=1000 | Comparable + cross-task transfer |
| Fine-tune recipe | N/A | Supervised (mins) + unsupervised ExIt (hrs) |

### Three architectural pillars of GOAL (how they interact)

```
            ┌─────────── task-specific ──────────┐
 instance → │ input adapter → low-dim (ℓ) projn  │
            └────────────────┬───────────────────┘
                             ▼
                 ┌─── shared codebook ℓ×D ───┐     ← forces cross-task
                 │   (lifts to backbone dim) │       representation sharing
                 └────────────┬──────────────┘
                              ▼
          ┌─── backbone (shared across all tasks) ───┐
          │   × L layers of Multi-head Mixed Atten.  │  ← score uses
          │     + FF + ReZero norm                    │    K+K', Q+Q'
          │   → if multi-type: n² MMA blocks,         │  ← type-aware
          │     same params per layer                  │    with param sharing
          └──────────────────┬────────────────────────┘
                             ▼
            ┌─── task-specific output adapter ───┐
            │   → action logits + task mask      │
            └────────────────────────────────────┘
```

The three pillars compose: **codebook** handles feature-space heterogeneity across tasks, **mixed attention** handles edge-vs-node heterogeneity within a task, and **multi-type transformer** handles node-type heterogeneity within multipartite problems. All three are orthogonal and all three are necessary (confirmed by ablations).

---

## Paper 5 — POMO (Policy Optimization with Multiple Optima)

| Field | Value |
|-------|-------|
| **Title** | POMO: Policy Optimization with Multiple Optima for Reinforcement Learning |
| **Venue** | NeurIPS 2020 |
| **Authors** | Yeong-Dae Kwon, Jinho Choo, Byoungjip Kim, Iljoo Yoon, Youngjune Gwon, Seungjai Min (Samsung SDS) |
| **Repo** | https://github.com/yd-kwon/POMO |
| **Local** | `ref/POMO.pdf` |
| **Task** | Constructive heuristic for CO — TSP, CVRP, 0-1 Knapsack. Same problem family as AM (Paper 1); POMO is a **training + inference recipe upgrade** on top of the same AM network. |
| **Architecture** | **Unchanged from AM** (Paper 1): Kool et al.'s Transformer encoder (3 MHA layers, 8 heads, d=128, FF=512) + autoregressive decoder with glimpse + single-head attention + tanh clipping C=10. The key move is that POMO reuses *the exact same network* — no architectural novelty — and wins purely by changing how trajectories are sampled and how the baseline is computed. Only change: the START-token scheme in the decoder is replaced by explicitly feeding N different first nodes in parallel (the decoder naturally handles this by stacking N queries). |
| **Training** | **RL — modified REINFORCE with shared baseline over N parallel rollouts.** For each instance, designate N starting nodes {a¹₁, …, aᴺ₁} and sample N full trajectories {τ¹, …, τᴺ} in parallel. Policy gradient: ∇J ≈ (1/N) Σᵢ (R(τⁱ) − b_shared(s)) ∇log p(τⁱ\|s), where **b_shared(s) = (1/N)Σⱼ R(τʲ)** — the *mean return across the N sibling rollouts of the same instance*. Adam lr=1e-4, weight decay 1e-6, batch=64 instances (so effective batch = 64·N trajectories), no LR decay (for simplicity; authors recommend one in practice). 1 epoch = 100k instances, generated on the fly. TSP100 ≈ 7 min/epoch on a Titan RTX; converges well by ~200 epochs (~1 day), fully converged by ~2000 epochs (~1 week). |
| **Key Innovation — multi-starting-node symmetry** | **Identifies the representational symmetry in sequential CO solutions**: a single optimal TSP tour has M equivalent cyclic starting-point rotations, and the network's `<START>` token forces it to pick one "canonical" starting node — which biases the learned policy. POMO's fix is conceptually simple but effective: **for each training instance, force N parallel rollouts from N distinct starting nodes** (for TSP, N=problem size so *every* node is a starting point). Each rollout targets a different sequence-representation of the same optimal solution. This is (i) an **entropy-maximization-by-construction** on the first action, (ii) an **on-policy data augmentation** that costs only more decoder queries (encoder runs once), and (iii) exposes the network to *many equivalent optima* rather than a single canonical one. Analogous to multi-crop evaluation in CV and to rotation-equivariant pretext tasks in self-supervised learning. |
| **Key Innovation — shared baseline** | **The baseline for trajectory τⁱ is the mean return of its N − 1 siblings from the same instance.** Contrast with AM's greedy-rollout baseline (a frozen copy of the policy, evaluated greedily). Properties that follow directly: (1) **Zero-mean advantages** — half the siblings are above, half below the group mean, instead of AM's systematically-negative advantages (sampled rollouts rarely beat a frozen greedy baseline). (2) **No baseline network / no frozen copy** — baseline is free, computed from the rollouts we already needed for the gradient. (3) **Low variance** — averaging over heterogeneous siblings of the *same* instance cancels instance difficulty out of the advantage. (4) **Resistant to local minima** — each τⁱ competes against N−1 *heterogeneous* siblings (different starting nodes → different trajectories even under the same policy), so premature convergence requires all N siblings to collapse together, which is heavily discouraged. Authors' ablation on the same AM network: replacing greedy-rollout baseline with POMO shared baseline cuts TSP100 greedy-gap from 3.51% → 1.07% — a pure training-recipe win. |
| **Key Innovation — multi-greedy inference + instance augmentation** | At test time: run **N deterministic greedy rollouts** from the N starting nodes and return the best. N greedy rollouts beat N sampled rollouts under the same budget in almost all regimes — search diversity comes from the *starting-node symmetry*, not from softmax randomness. **×8 instance augmentation** for 2D routing: apply the 8 unit-square dihedral transforms {(x,y),(y,x),(x,1-y),(1-y,x),(1-x,y),(y,1-x),(1-x,1-y),(1-y,1-x)} — the optimal tour is invariant, so take the best over 8×N greedy rollouts. On TSP100 this drops the gap from 0.46% → **0.14%** (state-of-the-art in 2020 for 2D routing construction methods at 1 min inference). |
| **Tricks & Details** | • **Encoder runs once per instance** regardless of N — POMO overhead lives only in the decoder. For TSP100 this means ~7 min/epoch for POMO vs ~6 min for plain REINFORCE, despite 100× more trajectories. • **Implementation of multi-rollout**: stack N decoder queries into one matrix; single attention call produces N parallel per-step distributions. • **Starting-node policy**: for TSP, use all M nodes (so N=M, perfectly symmetric). For CVRP, not all nodes are valid starts for optimal trajectories (depot/customer asymmetry + capacity leftovers — see their Fig. 4) — paper uses **all customer nodes naively** and still wins, leaving a learned "SelectStartNodes" network as future work. For KP, use every item as a first step. • **Single-trajectory eval mode**: to isolate the training-recipe gain, authors evaluate the POMO-trained net in plain single-greedy mode and still see the big gap (3.51% → 1.07% on TSP100) vs AM-trained net. Confirms the win is from training, not from multi-rollout inference. • **Gap reporting convention**: TSP ≤100 gaps are against Concorde / LKH3 optima; CVRP gaps are against LKH3 (no true optimum feasible at 10k instances). • **Implementation note**: CVRP uses the same TSP network, not a specialized one. |
| **Benchmarks** | 10,000 random instances per size, drawn as in Kool et al. 2019 (uniform in unit square, demands uniform in {1..9}, KP uniform weights/values in [0,1], capacity=25). **TSP**: n=20 (0.00% gap, 3s), n=50 (0.03%, 16s), n=100 (**0.14%**, 1m) all with ×8 aug. **CVRP** (gap vs LKH3): n=20 (0.21%), n=50 (0.45%), n=100 (0.32%) with ×8 aug — outperforms prior construction methods by wide margins, narrowing gap to improvement-based L2I. **KP** (no augmentation — no obvious symmetry): n=50 (0.007), n=100 (0.006), n=200 (0.008) — near-optimal. |
| **Hyperparameters** | Adam lr=1e-4, L2 weight decay 1e-6, batch=64 instances × N trajectories (N = problem size for TSP/KP, = all customers for CVRP), 100k instances/epoch, 200–2000 epochs. Network = AM defaults (3 enc layers, 8 heads, d=128, FF=512, tanh clip C=10). Inference: K=8 augmentations × N greedy rollouts. No LR schedule in the paper; authors recommend adding one. |
| **Relevance to AM_ALPHAGOZERO** | **Reuse:** (1) **Shared baseline is our default RL recipe over AM's greedy-rollout baseline** — zero-mean advantages, no baseline network, more stable, strictly better on every problem they tried. For SCIP plugin RL, we can compute the baseline as the mean return of N branchings started from different "entry actions" on the same MIP instance, as long as we can identify a symmetry that makes those entry actions comparable. (2) **Multi-start symmetry as a lens**: before training, ask "what are the sequence-representation symmetries of an optimal solution to our problem?" For branching, an optimal B&B proof is a *tree*, not a sequence, but the *order of exploring nodes* is partly free — that's our equivalent of POMO's cyclic rotations. For cutting-plane selection, different permutations of the selected cut set may give identical final LPs — another symmetry to exploit. (3) **Inference-time augmentation**: the dihedral-group trick only works for 2D coords, but the *pattern* generalizes — whenever the problem instance has a symmetry group G acting on features while leaving the objective invariant, run one greedy rollout per orbit element and take the best. SCIP problems have few coordinate-like symmetries, but MIP **variable permutations, constraint permutations, row/column scalings** are candidates. (4) **Architectural minimalism lesson**: POMO changes zero lines in the encoder/decoder and beats all specialized construction baselines of 2020 — a reminder that *training recipe gains can dominate architectural gains*. Before introducing multi-type MMA / codebooks / sub-instance re-embedding, make sure the RL baseline is at POMO level, not at AM level. (5) **Contrast with BQ-NCO / GOAL**: POMO is pure-RL on a non-bisimulation MDP; BQ-NCO is pure-IL on a bisimulation MDP. The two wins are additive in principle — BQ-NCO's sub-instance state + POMO-style shared baseline over N starting nodes would be a natural recipe for the project if we keep the AM/BQ backbone but go RL. (6) **AlphaGo-Zero analogy**: POMO's N parallel rollouts per instance with shared mean baseline is conceptually similar to AlphaGo Zero's MCTS generating many sibling trajectories from the same root and averaging — both derive the baseline from the cohort itself rather than from a frozen target. |
| **Cites** | Kool et al. 2019 (AM — the base network), Williams 1992 (REINFORCE), Bello et al. 2017 (NCO with RL, PtrNet baseline), Vinyals et al. 2015 (Pointer Networks), Vaswani et al. 2017 (Transformer), Rennie et al. 2017 (self-critical sequence training — conceptual ancestor of the shared baseline), Kool et al. 2019 ("Buy 4 REINFORCE samples, get a baseline for free!" — explicit precursor to shared baseline), Gidaris et al. 2018 (rotation-prediction self-supervised — conceptual ancestor of instance augmentation), Joshi et al. 2019 (GCN beam search), Wu et al. 2019 / Costa et al. 2020 (improvement heuristics baselines), Chen & Tian 2019 (NeuRewriter), Hottung & Tierney 2019 (NLNS), Lu et al. 2020 (L2I), Applegate et al. 2006 (Concorde), Helsgaun 2000 (LKH). |
| **Cited By** | Paper 3 (BQ-NCO — explicit baseline comparison; POMO's TSP1000 collapses to 40.60% gap under distribution shift, highlighting BQ-NCO's re-embedding advantage); Paper 4 (GOAL — compared as single-task baseline, beaten by GOAL on 7/8 tasks; POMO's symmetry idea is orthogonal to GOAL's multi-task sharing and could be stacked); Paper 6 (SGBS — uses POMO-pretrained AM as the base policy, reuses ×8 instance augmentation at inference, treats POMO-greedy as the rollout primitive inside simulation-guided beam search). |
| **Reproduction** | `not started` |
| **Our Implementation** | — |
| **Known Gaps** | — |

### Why the shared baseline works (schematic)

```
AM / greedy-rollout baseline (Paper 1)          POMO / shared baseline (Paper 5)
--------------------------------------          --------------------------------------
                               ┌── greedy(θ_frozen) ──►  b_AM      ┌── R(τ¹) ──┐
                               │                                    │
instance s ──► sample τ(θ) ────┤                                    │
                               │                                    ├── mean → b_shared(s)
                               └── R(τ)                              │
                                                                     │   each sibling sees
                                                                     │   advantage R(τⁱ) − mean
                                                                     ├── R(τ²) ──┤
                                                                     ... ...
                                                                     └── R(τᴺ) ──┘

• b_AM is typically > R(sampled τ)         • b_shared is literally the cohort mean → zero-mean advantage
  → advantages mostly NEGATIVE             • half above, half below → balanced credit assignment
• Baseline needs a frozen copy of policy   • Baseline is free (reuses rollouts)
• Baseline updates via t-test gating       • Baseline is instance-conditional by construction
• Symmetry is NOT exploited                • First-node symmetry is FORCED into the training distribution
```

### Shared baseline + multi-starting-node in one line

```python
# per instance s in batch:
#   pick N distinct starting nodes a¹₁, ..., aᴺ₁  (e.g. all nodes for TSP)
#   run decoder N times in parallel → trajectories τ¹, ..., τᴺ and log-probs
#   R = rewards(τ¹..τᴺ)                  # shape (N,)
#   b = R.mean()                         # shape (), shared across siblings
#   loss = −((R − b) * logp).mean()
```

### Relation to Paper 1 (AM) and Paper 3 (BQ-NCO)

| Aspect | AM (Paper 1) | POMO (Paper 5) | BQ-NCO (Paper 3) |
|--------|--------------|----------------|-------------------|
| MDP state | (instance, partial x) | (instance, partial x) — same as AM | sub-instance (BQ) |
| Training | RL, REINFORCE, greedy-rollout baseline | RL, REINFORCE, **shared baseline over N siblings** | IL with expert trajectories (oracle) |
| First-action bias | Yes (START token) | **No (N forced starts)** | No (sub-instance has a fixed new origin) |
| Encoder re-use at inference | Once per instance | Once per instance (cheap multi-rollout) | Re-run every step on the shrinking sub-instance |
| TSP100 greedy gap (paper) | 3.51% | **1.07%** (same net, only training changed) | 0.35% (IL + re-embedding) |
| TSP1000 zero-shot (greedy) | huge | 40.60% (collapses) | 2.29% |

**Takeaway:** POMO shows that the AM network has a lot of untapped capacity that was being wasted by a bad training recipe and a biased starting-node distribution. BQ-NCO shows that even with POMO-level training, the AM-style "encode-once" inference is leaving a much larger gap on the table, which re-embedding fixes. These two improvements are orthogonal: a BQ-style state with POMO-style shared-baseline RL is a natural unexplored combination for our project.

---

## Paper 6 — SGBS (Simulation-guided Beam Search)

| Field | Value |
|-------|-------|
| **Title** | Simulation-guided Beam Search for Neural Combinatorial Optimization |
| **Venue** | NeurIPS 2022 |
| **Authors** | Jinho Choo, Yeong-Dae Kwon (co-first), Jihoon Kim, Jeongwoo Jae, André Hottung, Kevin Tierney, Youngjune Gwon (Samsung SDS + Bielefeld University) |
| **Repo** | https://github.com/yd-kwon/SGBS |
| **Local** | `ref/SGBS.pdf` |
| **Task** | **Inference-time search procedure** for constructive neural CO — not a new model, not a new training recipe. Drops into any trained autoregressive policy π_θ (AM, POMO, MatNet, …) and replaces greedy / sampling / beam search / MCTS as the final decoding method. Evaluated on TSP, CVRP, and FFSP. |
| **Architecture** | **Unchanged** from the underlying policy network. Experiments use AM (Kool 2019) pretrained with POMO RL (Paper 5) for TSP/CVRP, and MatNet (Kwon 2021) for FFSP. SGBS only touches the inference-time decoding loop; zero new parameters. |
| **Training** | **None** — SGBS is a pure inference algorithm. The companion SGBS+EAS method (Alg. 2) does *test-time* fine-tuning of a small insertion-layer parameter ψ on the specific target instance, using an RL loss J_RL (REINFORCE-style with baseline) + an IL loss J_IL (imitate the incumbent s*_N). Pre-trained backbone θ is frozen during SGBS+EAS; only ψ is updated. |
| **Key Innovation — the three-phase loop** | **Beam search + MCTS-style rollouts + pre-pruning, fused into one breadth-first procedure.** At each depth d, for every node s_d currently in the beam (width β): **(1) Expansion (pre-pruning):** keep only the top-γ children of s_d by π_θ(a\|s_d) — not all children, not by cumulative sampling probability. Yields β×γ candidate child nodes. **(2) Simulation:** for each of the β×γ candidates, run **one greedy rollout** to a terminal state using π_θ. Record the terminal reward R(s_N). **(3) Pruning:** keep the β candidates with the highest *simulated* rewards R(s_N), not the β candidates with the highest prior probability. Repeat until all β nodes are terminal. Return the best terminal solution found. Two hyperparameters (β, γ) control width × depth of per-step exploration; cost scales linearly in β·γ. |
| **Relation to MCTS vs beam search** | **Not MCTS.** No tree reuse, no value backup, no UCB, no value network, no visit counts. A single greedy rollout per expanded node — no re-simulation, no averaging. Because simulations are fully parallel and deterministic, the whole procedure runs as β×γ batched forward passes through π_θ per depth level, making it GPU-friendly in a way MCTS is not. **Not ordinary beam search either.** Ranking is by *simulated terminal reward*, not by *cumulative path probability*. Authors explicitly ablate this (footnote 3): ranking the β×γ candidates by cumulative sampling probability (NLP-style beam search) gives worse results — "exploration (treating all beam nodes as equals) matters more than exploitation at the pruning stage." **The conceptual move** is to use the full rollout as a *heuristic value estimate* for otherwise-unknown child nodes, and to let this estimate — not the network's myopic prior — drive pruning. This is the MCTS simulation step without any of the MCTS bookkeeping. |
| **Key Innovation — SGBS + EAS hybrid** | SGBS is **deterministic**, so running it once uses the time budget and running it again gives nothing new; increasing β, γ yields quickly diminishing returns. To absorb a long test-time budget, SGBS is alternated with **EAS** (Efficient Active Search, Hottung 2022 — updates a small ψ injected into the frozen backbone per-instance). Per outer iteration: (1) run SGBS once to get a candidate solution s⁰_N; (2) sample M further rollouts {s¹_N, …, s^M_N} from π̃_{θ,ψ}; (3) update the incumbent s*_N as the best of all; (4) gradient ascent on ψ with J_RL (REINFORCE on sampled rollouts, baseline b°) + λ·J_IL (cross-entropy toward the incumbent). Synergy: **SGBS gives EAS a stronger incumbent to imitate** (escapes local optima that EAS-alone would get stuck in); **EAS gives SGBS a progressively better policy** (each SGBS call explores a different part of the space because ψ has moved). Paper reports SGBS occupies ~75% of wall-clock time in SGBS+EAS, but the remaining 25% of EAS updates more than doubles the quality gain. |
| **Three regimes where SGBS wins** | Paper explicitly dissects three cases (Fig. 2, CVRP100): **(a) Pre-trained model** (well-matched to test distribution): SGBS competitive with EAS, both beat sampling/beam/MCTS at equal 1.2K candidates/instance. **(b) Low-accuracy model** (distribution shift — CVRP100-trained model evaluated on CVRP200): sampling and beam search **underperform greedy** (!) because they blindly amplify the bad prior; SGBS wins decisively, because rollout-based pruning lets the network "correct itself on the fly" — low-probability-but-actually-good children can survive if their rollout reward is high. **(c) Fine-tuned model** (overconfident after EAS): beam search collapses because the top-γ children consume all probability mass at early depths; SGBS is robust because β and γ **hard-code an exploration width** independent of confidence calibration — this is why SGBS+EAS is strictly better than EAS alone. Authors frame this as a defense against the confidence-calibration problem (Guo et al. 2017). |
| **Tricks & Details** | • **Pruning uses simulated reward, not path probability** — the single most important design choice; NLP-style ranking fails (footnote 3). • **Single greedy rollout per candidate** is "good enough and more time-efficient" than multiple rollouts or nested SGBS — author's design principle is to keep the primitive cheap and compensate with larger β·γ. • **β × γ scales linearly**, all β·γ simulations are batched on the GPU. • **×8 instance augmentation** (from POMO) is stacked on top of SGBS for all TSP/CVRP experiments — orthogonal, free multiplier on solution diversity. • **Hyperparameters**: (β, γ) = (10, 10) for TSP, (4, 4) for CVRP, (5, 6) for FFSP. Authors report sensitivity is low within a reasonable range. • **SGBS+EAS training-saving trick:** over-training θ actually *hurts* SGBS+EAS — early-stopping during pretraining is required. They pretrain θ from scratch in only ~2 hours for CVRP100 and still match LKH3 quality after SGBS+EAS inference. Implication: **SGBS+EAS substitutes test-time compute for pretraining compute.** • **Zero backpropagation in SGBS itself** — keeps simulation batched and fast; only SGBS+EAS does gradient updates, and those are confined to a small ψ. |
| **Benchmarks** | **TSP100 (10K inst.)**: SGBS+EAS 0.024% gap in 15h vs POMO-greedy 0.144%, POMO-sampling 0.078%, EAS 0.044% (15h); **TSP200 (1K, zero-shot generalization)**: 0.196% vs EAS 0.302%. **CVRP100 (10K)**: 0.11% (30h) vs EAS 0.23% (30h), LKH3 0.53%; **CVRP200 (1K, zero-shot)**: 0.40% vs EAS 0.98%, LKH3 1.09% — *beats LKH3*. **FFSP20/50/100 (1K each, MatNet backbone)**: SGBS+EAS 0.14 / 0.21 / 0.34 gap (at 50/100/200 h) — same quality as EAS but faster, ~2–3× better than sampling. Across all three problems, SGBS alone (without EAS) already beats sampling by 2–3× at matched candidate budget. **Headline:** SGBS+EAS cuts EAS's optimality gap by 33–59% on CVRP and 33–45% on TSP at matched time budget. |
| **Hyperparameters** | Same pretraining as POMO / MatNet (the underlying base networks). Inference only: β, γ as above. EAS learning rate, insertion-layer size, λ (IL vs RL weight) inherited from Hottung 2022 (EAS-Lay variant — simpler, better for CVRP). No new hyperparameters introduced specifically by SGBS other than (β, γ). |
| **Relevance to AM_ALPHAGOZERO** | **Reuse:** (1) **SGBS is our low-effort inference upgrade**: for any autoregressive constructive policy we train (AM / BQ-NCO / GOAL / POMO-style), drop in SGBS at test time — no retraining, no parameter changes, small code delta, and it strictly dominates greedy / sampling / NLP-beam / MCTS at matched compute. This is the best cost/benefit search procedure in the literature and should be our default decoder until proven otherwise. (2) **The "rollout as value estimate" idea is the MCTS-lite connector to AlphaGo Zero**: SGBS is what you get when you keep AlphaGo Zero's idea that "full-depth simulation is a better evaluator than the raw policy" but drop the full MCTS machinery (backup, UCB, tree reuse) that doesn't vectorize on GPU. For our SCIP project, where branch-and-bound is a *search* in the first place, the right primitive may be closer to SGBS — partial node expansion scored by rollout — than to full MCTS over the B&B tree. (3) **SGBS+EAS as the runtime-amortization pattern**: the alternation of a *deterministic strong search* (SGBS) with a *stochastic per-instance parameter update* (EAS) is exactly the pattern we want for SCIP at inference: a few SCIP test-time gradient steps on the adapter layer using rollouts from SGBS-driven branching trees. The "SGBS gives EAS stronger incumbents to imitate, EAS gives SGBS progressively better priors" loop is a test-time analog of the AlphaGo Zero train-time loop, adapted to a single instance rather than a replay buffer. (4) **"Over-training hurts SGBS+EAS" observation** is valuable for our project roadmap: pretraining to convergence may be the wrong target if test-time adaptation is part of the plan. Early-stopping + test-time adaptation yields better end-to-end quality per GPU-hour than pretraining-to-convergence + fixed inference. For SCIP, this opens the door to lighter pretraining if we commit to per-instance adapter updates. (5) **Ranking by simulated reward, not path probability** is a concrete lesson for any search we build: the policy's cumulative probability is a biased estimator of terminal quality (especially under distribution shift), and a single rollout is a cheap unbiased one — use the rollout. **Contrast with AlphaGo Zero:** SGBS is *inference-only* and *no-value-network*; AlphaGo Zero is *training-time* and uses a learned value head to avoid full rollouts. A middle ground — SGBS where simulations are shortened by a learned value head once rollouts become expensive — is a natural extension and relevant to scaling our project to long SCIP traces. |
| **Cites** | Hottung et al. 2022 (EAS — fine-tuning partner), Kool et al. 2019 (AM — base network), Kwon et al. 2020 (POMO — pretraining recipe for TSP/CVRP experiments), Kwon et al. 2021 (MatNet — base network for FFSP experiments), Kocsis & Szepesvári 2006 / Coulom 2007 (MCTS — conceptual ancestor), Vinyals et al. 2015 (Pointer Networks, beam search usage), Nazari et al. 2018 (RL for VRP, beam search usage), Silver et al. 2016 (AlphaGo — MCTS + deep net for search, the broader inspiration), Cazenave 2012 / Baier & Winands 2012 (Monte Carlo Beam Search — closest prior art, but relies on random rollouts and no policy prior), Jooken et al. 2020 (MCTS variant with beam width, no pre-selection), Guo et al. 2017 (confidence calibration — explains the fine-tuned-model failure mode), Vidal 2022 (HGS — CVRP baseline), Helsgaun 2017 (LKH3), Applegate et al. 2006 (Concorde). |
| **Cited By** | — |
| **Reproduction** | `not started` |
| **Our Implementation** | — |
| **Known Gaps** | — |

### SGBS vs MCTS vs beam search (decision table)

| Aspect | Greedy | Sampling | NLP Beam | MCTS | **SGBS** |
|--------|--------|----------|----------|------|---------|
| Per-depth branching | 1 | N samples (flat) | β (by cum. prob.) | UCB-selected | **β·γ (top-γ per beam node)** |
| Ranking signal | max π | R of flat samples | cum. log π | Q + U (value + exploration) | **R of single greedy rollout** |
| Uses rollouts? | No | Only at leaves | No | Yes (many, averaged) | **Yes (one per candidate)** |
| Batch-friendly? | Trivial | Yes | Yes | No (sequential) | **Yes** |
| Value net needed? | No | No | No | Optional (often yes) | **No** |
| Resists bad priors? | No | No | No | Yes (via Q) | **Yes (via R)** |
| Determinism | Deterministic | Stochastic | Deterministic | Stochastic | **Deterministic** |
| Cost per instance | O(N) | O(N·K) | O(N·β) | high (simulations × tree bookkeeping) | **O(N·β·γ)** |

The core design principle: **replace the NLP-style "cumulative prior" ranking with a one-rollout heuristic value estimate, and keep everything batch-parallel.** That single swap captures most of MCTS's correction-on-the-fly benefit while preserving beam-search-level throughput on GPU.

### SGBS+EAS loop (schematic)

```
        ┌─────────────── per instance R, frozen θ, trainable ψ ─────────────┐
        │                                                                    │
        │   ┌── SGBS(π̃_{θ,ψ}, β, γ, R) ──► s⁰_N          (deterministic     │
        │   │                                              strong search)    │
        │   │                                                                │
  loop: │   ├── sample s¹_N, …, s^M_N ~ π̃_{θ,ψ}          (stochastic        │
        │   │                                              diverse rollouts) │
        │   │                                                                │
        │   ├── s*_N ← best of {s*_N, s⁰_N, s¹_N..s^M_N}  (update incumbent) │
        │   │                                                                │
        │   └── ψ ← ψ + α[ ∇J_RL + λ·∇J_IL ]              (per-instance      │
        │                                                   adapter update)  │
        │         where J_RL: REINFORCE on {s^i_N}                           │
        │               J_IL: cross-entropy toward s*_N                      │
        └────────────────────────────────────────────────────────────────────┘
```

SGBS without EAS is a one-shot decoder. SGBS+EAS turns it into a time-amortized solver where each outer iteration feeds the next a better policy and a better incumbent. Neither mechanism introduces a new learned component — only ψ (tens of thousands of params) is touched at test time, and SGBS itself is parameter-free.

### Relation to AM (Paper 1), POMO (Paper 5), BQ-NCO (Paper 3), and AlphaGo Zero (Paper 2)

| | AM (P1) | POMO (P5) | BQ-NCO (P3) | AG Zero (P2) | **SGBS (P6)** |
|---|---------|-----------|--------------|--------------|---------------|
| Role | Base network | Training recipe | State reformulation | Full pipeline | **Inference recipe** |
| What it touches | Architecture | Loss + sampling | MDP state | Model + search + RL | **Decoding only** |
| Requires training? | Yes | Yes | Yes | Yes (+ self-play) | **No** |
| Compatible with the others? | — | Stacks on AM | Stacks on AM | — | **Stacks on AM+POMO, AM+BQ, MatNet, GOAL backbone** |
| Search at inference | Greedy / sampling | Multi-start greedy + ×8 aug | Greedy rollout (re-embedding each step) | MCTS with value+policy net | **Beam of β × pre-pruned γ × rollout-pruned β** |
| Uses rollouts at inference? | Only in sampling | Multi-start greedy | Yes (each step is a full re-encode) | Yes (guided by value net) | **Yes (one per candidate, no value net)** |

**Takeaway for our project:** SGBS is the "free lunch" of this survey — it composes with every other paper we've reviewed and is the correct default inference method to pair with whatever training recipe and architecture we settle on. When we later introduce a value head (à la AlphaGo Zero), the natural extension of SGBS is to replace the full greedy rollout with a truncated rollout + value estimate, giving us a direct migration path from SGBS → AG-Zero-style search without changing the outer loop.

---
