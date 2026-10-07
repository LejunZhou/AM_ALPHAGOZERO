# Stage 6 — Stochastic Prize-Collecting TSP (SPCTSP)

Date: 2026-10-07. Approved by Lejun 2026-10-07: close the TSP chapter and move
the thesis to stochastic PCTSP. See proposal.md §Revised thesis and §Stage 6.
Progress: [`_progress/stage6_spctsp_progress.md`](../_progress/stage6_spctsp_progress.md).

## Question

Where a node's real prize is revealed only on arrival, is a learned value function
worth more than single rollouts? And does MCTS training built on it beat REINFORCE
at matched compute? The TSP chapter showed that on deterministic problems it is
not (the greedy rollout is already a near-exact value).

## Problem (as in the AM reference, `ref/attention-learn-to-route-master/problems/pctsp`)

- **Instance.** A depot and n nodes, uniform in the unit square. Each node has a
  penalty U(0, c_n·3/n) with c_20 = 2 and c_50 = 3, an expected prize
  U(0, 4/n), and a real prize U(0, 2 × expected).
- **Constraint.** The tour may return to the depot only once the collected *real*
  prize is ≥ 1, or once every node has been visited.
- **Information.** The policy sees expected prizes, plus the real prizes of nodes it
  has already visited.
- **Cost.** Tour length plus the penalties of unvisited nodes.
- **Stochasticity.** It enters only through the depot-legality mask, i.e. when the
  agent may stop. Therefore:
  - the greedy policy is deterministic given a prize realisation;
  - a rollout from a state sees one realisation of the unrevealed prizes.
- **Pretrained AM models.** `pretrained/pctsp_stoch_{20,50,100}` and
  `pctsp_det_{20,50,100}`. Their checkpoints include the optimizer and baseline
  state, which the Step 2-style REINFORCE control can continue from.
- **Measured (2026-10-07, CPU, 2,000 instances, one realisation).** AM greedy
  SPCTSP-20 3.254 and SPCTSP-50 4.645. The SD across prize realisations for the
  same instance is 0.23 / 0.18, against an SD across instances of 0.39 / 0.31.

## Evaluation protocol (all phases)

- **Common random numbers.** Each test instance carries R fixed prize realisations
  (default R = 16). Every policy is scored on the same realisations.
- **Paired statistics.** Per-instance means over the R realisations, paired across
  policies, with SE over instances. Fixed seeds: val and test are disjoint from any
  training draw.
- **Baselines.**
  - AM pretrained greedy.
  - Our AM + value (where trained).
  - The deterministic-PCTSP model run on the stochastic problem, as a sanity
    control.
  - Optional: AM's REOPT baseline.

  Best-of-N sampling is **not** a valid stochastic policy, because the choice
  would use prizes not yet revealed. It is reported only as an
  oracle-information reference, if at all.
- **Compute.** Every search or training comparison is at matched wall time on one
  GPU type. `sessions.jsonl` records the device per phase, as in Step 2.

## Phases and gates

Each gate is written here before its run. Thresholds marked *proposed* may be
changed by Lejun **before** the run, never after.

### P0 — Environment and parity (local CPU)

- **Port.** `am_baseline/problem/pctsp.py` (deterministic and stochastic problem;
  a batched state and a single-instance cloneable state for search), plus the
  PCTSP decoder context:
  - input embedding: depot and (x, y, expected prize, penalty);
  - context: current node and remaining prize to collect.
- **Value target.** The (bl-normalised) expected cost-to-go.
- **Load the AM pretrained weights** into the port.
- **Parity gate.** Greedy costs match the AM reference code on the same instances
  and realisations: max |Δ| ≤ 1e-5, and identical tours.
- **Unit tests.**
  - The mask never allows the depot before the prize is reached (unless all nodes
    are visited).
  - Real prizes stay hidden until arrival.
  - The cost reproduces AM's `_get_costs`.

### P1 — Value-signal probe (local CPU at n = 20; Colab at n = 50)

The cheapest test of the thesis. It reuses the sibling-ranking method of
`probe_sibling_ranking.py` / `probe_action_ranking.py` and splits every result by
horizon (prize still to collect, nodes left).

- **States.** Sampled from the frozen AM greedy policy:
  - on-policy: along the policy's own trajectories;
  - off-policy: after one random legal deviation.
- **Ground truth.** For every legal child, Q(s,a) is the mean cost-to-go over
  M = 256 independent prize realisations for the unrevealed nodes (child prize
  included), with greedy rollouts.
- **Estimators compared.**
  - the policy prior (argmax);
  - one rollout (one realisation);
  - the mean of k rollouts, k ∈ {2, 4, 16};
  - the value head, trained the §V0 way: a frozen policy, and MSE on **single-rollout
    labels of all legal children** of sampled states. Regression then averages out
    the realisation noise, which is the mechanism under test.
- **Metrics.**
  - Decision regret of each estimator's argmin against ground truth.
  - Spearman correlation with ground truth.
  - k_eq: the number of averaged rollouts whose regret equals the head's
    (interpolated).
  - Wall time per evaluation.
- **Gate G1a: is there room?** One-rollout regret must be ≥ 2× the 16-rollout
  regret (*proposed*). If not, realisation noise barely affects decisions, and the
  thesis has no room on SPCTSP. Stop.
- **Gate G1b: does the head use it?** On held-out instances (*proposed*):
  - head k_eq ≥ 4, and
  - head regret ≤ 0.5 × one-rollout regret.

  If k_eq < 2, stop. Between those values, report and decide with Lejun before
  P2.

### P2 — Search at test time

- **Online MCTS.** At each real step, search from the current state. Revealed prizes
  are fixed. Unrevealed prizes are re-sampled in each simulation (an open-loop
  tree; the depot-legality mask is evaluated per simulation).
- **Implementation.** First the Python reference at n = 20. A C++ port only after
  G2 passes at n = 20.
- **Leaf evaluators.** One rollout, the k-rollout mean, the value head, and
  prior-only.
- **Gate G2** (paired, test set, R realisations; thresholds fixed before the run):
  - search beats greedy on expected cost (95% CI < 0);
  - value-guided search is no worse than the best rollout-guided search at matched
    wall time (non-inferiority margin to be set from P1 noise levels).

### P3 — Student isolation (Step 2 design)

- **Setup.** Fixed teacher: AM + the P2 search. One round of distillation on
  identical teacher data. Targets: `gumbel_q`, `visits`, `best_tour`.
- **Control.** Continued REINFORCE from the pretrained checkpoint (optimizer +
  rollout baseline restored) at matched wall time.
- **Gate G3** (as on TSP, for comparability):
  - student − control 95% CI < 0, and
  - retention ≥ 0.25;
  - a screen on seed 0, then confirm on seeds 1–2.

### P4 — Full loop (only if G1–G3 pass)

- **Runs.** Iterated self-play with ≥ 3 seeds, SPCTSP-20 then -50.
- **Comparison.** Against continued REINFORCE and the pretrained AM, in GPU-hours.

## Engineering notes

- **Search code.** The C++ MCTS is TSP-specific. P1 needs no search code. P2 starts
  from the Python reference.
- **Runners.** Keep the Step 2 runner patterns: manifest signatures, resumable
  phases, embedded-snapshot Colab notebooks, CPU-runnable smoke profiles.
- **CPU-first.** Every phase runs on CPU at small scale. Colab is used only for
  scale-up.

## Out of scope

- Deterministic TSP. It is closed; §V1 (TSP-20 matched-wall vh search) is dropped.
- CVRP.
- Stochastic-demand CVRP is a stretch goal only if Stage 6 passes.
