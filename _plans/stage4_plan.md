# Stage 4 Plan: AlphaGo-Zero-Style MCTS Self-Improvement Loop on TSP-20

**Created:** 2026-04-29
**Predecessor:** Stage 3 (`_plans/stage3_plan.md`, `_progress/stage3_progress.md`) — closing.
**Reference:** Proposal §Stage 4 (`proposal.md:118-143`).
**Original plan file (Claude Code plans dir):** `C:\Users\Jun18\.claude\plans\i-think-we-have-snazzy-trinket.md`
**Status:** Approved 2026-04-29. Phase A in flight.

---

## Context

Stage 1 produced REINFORCE-trained AM checkpoints with auxiliary value heads (TSP-20/50/100). Stage 2 built and locked the canonical MCTS config (`c_puct=0.05, fpu=running_q, root_select=visits, tree_reuse=True, value_norm='bl'`). Stage 3 produced the search-efficiency headline (MCTS budget curves vs sampling-K vs AM-1280 reference at TSP-20/50/100, all pass gates met) and shipped three MCTS backends in `src/am_baseline/search/`: `python`, `cpp` (sequential C++), and `cpp_batch` (cross-instance batched, 2.20× / 3.09× speedups with exact cost preservation). Stage 3's final-Phase E off-policy R² probe found the value head is essentially in-distribution-accurate on MCTS-visited states (R² = 0.9949 vs in-distribution 0.9965), so distillation against MCTS targets is a refinement rather than a recovery from off-policy collapse.

**Stage 4 closes the proposal's central thesis (`proposal.md:13-15`):** AM's REINFORCE training leaves significant performance on the table that MCTS-based policy improvement can recover, with better sample efficiency. The deliverable is a **sample-efficiency curve** (val_avg_cost vs total training instances seen) showing Stage 4 reaches AM-equivalent quality with **fewer total training instances** than Stage 1's REINFORCE curve at the same x-axis, plus an **ultimate-quality** result where Stage 4's network *alone* (greedy decoding, no MCTS) matches Stage 3's Stage-1+MCTS-K=400 result.

**Why `cpp_batch` is the unlock.** Stage 3 measured Python MCTS at TSP-20 K=200 = 1469 s for 1000 instances; Stage 4 self-play at Python speeds is laptop-infeasible. Sequential cpp lands at 29.8 s; cpp_batch bs=64 should cut another ~3× on TSP-20. One Stage 4 iteration on TSP-20 (1000 self-play instances + 1 training pass) lands at ~30-60 s; 100 iterations finish overnight on the laptop GPU.

---

## Scope decisions (locked from clarification round)

1. **Graph size:** **TSP-20 only on the critical path.** TSP-50 (and TSP-100) follow in Stage 5 only if TSP-20 passes. This matches the proposal's "Start with TSP-20, then scale to TSP-50" wording and keeps compute tight (~3-6 h instead of ~30 h).
2. **Leaf evaluator: `value_head`** — *follow what AlphaGo Zero does.* The defining AlphaGo-Zero innovation vs AlphaGo Lee/Master is replacing the rollout policy with value-head-only leaf evaluation. Stage 3 E.1 confirmed off-policy R² = 0.9949 on MCTS-visited states, so value-head leaf eval is well-calibrated. Side benefit: ~5-10× faster per leaf eval than rollout, so K can scale up. **Note the trade-off vs Stage 3's headline:** Stage 2/3 showed rollout > value_head for *test-time* MCTS quality. Stage 4 expects the value head to *improve* as it trains on MCTS-distilled targets, closing the gap. A **rollout-leaf-eval ablation** (Phase G.1) is retained as the explicit comparison.
3. **Gating reject behavior:** **Keep candidate, continue training.** When the paired t-test rejects the candidate, the *self-play model* (the one running MCTS for new data) stays as the prior best, but the trainer's optimizer/weights continue evolving. Matches AlphaGo Zero's continuous-training design and keeps gradient continuity.
4. **Sample-efficiency baseline curve:** **Reuse Stage 1 checkpoints + W&B logs.** Stage 1 already saved per-epoch checkpoints with `val_avg_cost` logged. The headline plot uses `x = epoch × epoch_size` instances seen, `y = val_avg_cost` from existing logs. No Stage 1 re-training needed.
5. **Warm start from Stage 1 canonical checkpoint.** `outputs/tsp_20/stage1_tsp20_canonical_20260423T103541/epoch-99.pt` is the starting weights. From-scratch is a Stage 5 ablation. The encoder, decoder, and value head are already calibrated; the goal of Stage 4 is to push past the REINFORCE-trained ceiling.

---

## AGZ canonical mapping (explicit follows / deviates)

This table pins every Stage 4 design choice to the AlphaGo Zero paper (Silver et al., *Nature* 550, 2017). "Follow" = matches the paper directly; "Adapt" = scaled or modified for TSP / our compute; "Deviate" = intentional divergence with justification.

| # | Knob | AGZ canonical (citation) | Stage 4 default | Verdict | Rationale |
|---|------|---|---|---|---|
| 1 | **Search algorithm** | PUCT, U(s,a)=c·P(s,a)·√Σ_b N(s,b)/(1+N(s,a)); leaf NN-only (no rollout); virtual loss; tree reuse (Methods §Search algorithm, p.8) | `c_puct=0.05`, `leaf_eval='value_head'`, `tree_reuse=True`; PUCT identical | **Follow** | Stage 2 locked these via tuning; `c_puct=0.05` reflects sharp-AM-prior + smaller action space (N=20 vs Go's 362) |
| 2 | **MCTS sims/move (training)** | 1,600 (Empirical analysis, p.355) | 50 (pilot) → 100 (main) | **Adapt** | TSP-20 has ~20 plies/game vs Go's ~250; per-game compute scales linearly. K=100 × 20 plies = 2,000 leaf evals/game ≈ AGZ's 1,600/move at much smaller per-step branching. F.3 pilot will confirm |
| 3 | **Loss** | l = (z−v)² − π·log(p) + c·‖θ‖²; cross-entropy and MSE **weighted equally** (eq. 1, p.355; Methods §Optimization, p.7); z ∈ {−1, +1} broadcast across all plies | l = (z_t−v(s_t))² − π_t·log(p(·\|s_t)) + c·‖θ‖²; **z_t = per-state V_CURRENT cost-to-go** (not broadcast) | **Adapt (forced by domain)** | TSP cost is continuous; quantizing to ±1 discards the optimization signal. Existing MCTS leaf evaluator (`mcts.py:1-15`) computes `state.lengths/bl_val + v(state)` assuming v predicts remaining cost-to-go — broadcast-z would double-count. Reuses Stage 1's V_CURRENT shape from `value_targets_from_edges` (`utils/tensor_ops.py:57-78`) |
| 3b | **Policy target π_t** | π_t = σ_t = N(s,·)^{1/τ}/Σ — uses same τ as action selection; one-hot late-game in Go (Methods §Self-play, p.358) | **π_t = N(s,·)/Σ N (raw τ=1 always)**, decoupled from action-selection σ_t (which uses step30) | **Adapt (deliberate)** | TSP-20's late steps have legal_actions(t≥18) ≤ 2; strict-AGZ one-hot late targets carry no information beyond "the chosen action was legal", wasting a distillation signal. Decoupling preserves multimodal MCTS visit information when present. G.4 ablates strict-AGZ coupling |
| 4 | **Optimizer** | SGD with momentum=0.9; lr step-anneal {1e-2, 1e-2, 1e-3, 1e-4, 1e-4, 1e-4} across thousands-of-steps buckets {0–200, 200–400, 400–600, 600–700, 700–800, >800} (Methods §Optimization, p.7; Ext. Data Table 3, p.18) | **Adam, lr=1e-4 constant** | **Deviate** | Stage 1 warm-start was trained with Adam at 1e-4; switching optimizers from a converged checkpoint risks destabilizing fine-tuned weights. Total Stage 4 mini-batches (~20K) sit entirely in AGZ's final-bucket regime (lr=1e-4) anyway. SGD+momentum is the **F.4 fallback** if val plateaus early; logged as G.8 ablation if pursued |
| 5 | **Mini-batch size** | 2,048 (32 per worker × 64 workers, Methods §Optimization) | 512 | **Adapt** | Single-GPU; 4× smaller buffer → 4× smaller batch keeps grad-noise scale comparable |
| 6 | **Replay buffer** | last **500,000 games**, sampled uniformly (Methods §Optimization, p.7); 25K games/iter → ~20-iter window | last **200,000 instances** (~4M tuples); 1K/iter → 200-iter window | **Adapt + monitor** | Smaller absolute size; **but our reuse window is 10× wider in iters than AGZ's**. F.3 pilot will surface staleness; pilot will run with `buffer_capacity=50_000` (~50-iter window, closer to AGZ ratio) and main can scale up only if pilot is stable. KataGo-style power-law windowing (`shuffle.py:418-450`) is the Stage 5 fallback |
| 7 | **Self-play volume / iter** | 25,000 games/iter (Methods §Self-play, p.8) | 1,000 instances/iter | **Adapt** | Linear-scaled to single-machine compute; total Stage 4 = 100K instances vs AGZ's 4.9M games |
| 8 | **Train rows / data rows** | not stated explicitly; typical AGZ ratio ≈ 700K mini-batches × 2048 ÷ 4.9M games × 250 plies ≈ 1.2× | F.4: 200 steps × 512 ÷ (1K × 20) = **5.1×** | **Deviate (mild)** | Above AGZ-implicit but at KataGo's documented 4–8× cap (`SelfplayTraining.md` §`MAX_TRAIN_PER_DATA=8`). F.3 pilot watches val curve for over-fitting; reduce `train_steps_per_iter` to 100 if curve plateaus |
| 9 | **Temperature schedule (action selection σ_t)** | τ=1 for first **30 moves** of each game; τ→0 thereafter (Methods §Self-play, p.8). 30/~250 ≈ 12 % of game length | **`step30`** = τ=1 for first ⌈0.3·N⌉ tour-steps = 6/20 for TSP-20; τ→0 thereafter | **Adapt (proportional)** | Closest scaled analog of AGZ's "explore first 12%, exploit rest". `step50` and `const` are G.4 ablations |
| 10 | **Dirichlet root noise** | ε=0.25, α=0.03 for Go (362 actions); P(s,a)=(1−ε)p_a + εη_a (Methods §Self-play, p.8) | ε=0.25, α=10/N=0.5 for TSP-20 | **Adapt** | Community/KataGo heuristic α≈10/A reproduces AGZ's α≈0.028 at A=362; for TSP-20, 10/20=0.5. ε identical |
| 11 | **Gating** | 400 games at τ→0; promote if win rate **>55 %** (avoid "noise alone") (Methods §Evaluator, p.8) | paired t-test α=0.05 on val_size=10,000 (Stage 1 `RolloutBaseline.epoch_callback`) | **Adapt** | Stage 1's t-test signature is what we already trust; effectively a similar gate at TSP scale (lower variance per instance than per game). G.5.b ablation: replace t-test with explicit "≥55 % win rate AND mean cost improvement ≥ 0.0005" rule |
| 12 | **Reject behavior** | candidate dropped, best player retained for self-play; trainer continues with current weights (Methods §Self-play training pipeline, p.7-8) | identical (scope decision 3) | **Follow** | |
| 13 | **Symmetry augmentation** | random dihedral reflection/rotation at **every leaf evaluation** (8-fold group); also dataset augmentation at training (Methods §Search algorithm + §Domain knowledge, p.8) | **Not in F.4 default**; queued as **G.7 ablation** | **Deviate (deliberate)** | TSP analog = random 2D rotation + axis flip applied to coords before encoding; tour cost invariant. Cheap (~20 LOC) but expected small effect at TSP-20 (uniform iid coords already isotropic in expectation). Worth measuring once |
| 14 | **Eval-time policy** | MCTS at τ→0 with 1,600 sims for ratings (Methods §Evaluation, p.9) | **greedy** (no MCTS) for `val_avg_cost` | **Deviate (deliberate)** | Our acceptance criterion 2 is "Stage 4 network alone (no MCTS) matches Stage 3 K=400 MCTS" — measuring greedy is the **whole point**. AGZ-style MCTS-at-eval is a Stage 5 follow-up |
| 15 | **Resignation** | v_resign auto-tuned for FP < 5% (Methods §Self-play, p.8) | **none** | **Deviate (deliberate)** | TSP has no win/loss bit; "resign" doesn't map cleanly. Could analogize as "abort if predicted cost > threshold" but compute saving is small at TSP-20 |
| 16 | **Best-player tracking** | "best player so far αθ*" used for self-play (Methods §Self-play training pipeline) | identical (`self.best_model = deepcopy(model)` after gating accept) | **Follow** | |
| 17 | **L2 weight decay c** | 10^-4 (Methods §Optimization) | 1e-4 | **Follow** | |
| 18 | **Network architecture** | dual-headed; 20- or 40-block ResNet (Methods §Neural network architecture) | AM Transformer (encoder/decoder) + value head | **Deviate (project premise)** | Domain swap: graph-input + autoregressive policy needs Transformer, not ResNet. The proposal §Stage 1 thesis. AGZ's "dual–res > sep–conv" Elo gap (Fig. 4) is what we replicate via shared-backbone value+policy |

**Citations are to the paper PDF the user attached: pages 354–359 (main text + Methods) and Extended Data Table 3 (page 18 of the PDF).**

---

## Recommended approach (one paragraph)

A thin `MCTSCoach` orchestrator (`src/am_baseline/training/coach.py`) drives an iterative loop on TSP-20: **(1) generate** — the *best* model produces M=1000 random TSP-20 instances per iteration with `CppBatchMCTSSolver` configured for self-play (`leaf_eval='value_head'`, `temperature_schedule='step30'` — τ=1 for first ⌈0.3·N⌉ steps then τ=0, `dirichlet_epsilon=0.25`, `dirichlet_alpha=0.5` for N=20); **(2) distill** — push per-step (state_t, π_t, z_t) tuples into a flat-tensor replay buffer (capacity ~200K instances ≈ 4M per-step records), sample mini-batches, and train with `loss = (z_t − v_θ(s_t))² − π_t·log(p_θ(·|s_t)) + c·||θ||²` where **π_t = N(s_t,·)/Σ N (raw normalized visits, τ=1 always)** is the training target — distinct from the action-selection distribution σ_t which uses step30 — and **z_t = (tour_cost − lengths_t) / bl_val(x; θ★)** is the **per-state cost-to-go (V_CURRENT)** matching Stage 1's value-head training shape and `mcts.py:1-15` leaf-evaluator invariant; **(3) gate** — every G=5 iterations evaluate the candidate model via greedy rollout against the current best on a frozen 10K-instance validation set, paired t-test α=0.05, accept on win. Warm-start from Stage 1's canonical TSP-20 checkpoint. The visit-count exposure problem (MCTS currently discards root.N after picking an action) is solved with an opt-in `MCTSConfig.return_root_visits` flag plus a side-effect attribute `solver.root_visit_dists`, mirroring Stage 3's instrumentation pattern; the C++ side already retains the per-step `root` pointer and just needs to dump `root.n_visits` at the right moment.

---

## Phases

### Phase A — Visit-distribution exposure (foundation; ~1 day; no GPU)

**Goal:** Expose per-tour-step root visit counts π_t from all three backends (`python`, `cpp`, `cpp_batch`) without breaking existing callers. This is the single missing piece for Stage 4 distillation targets.

**A.1 Python solver hook** — `src/am_baseline/search/mcts.py`.
- Add `MCTSConfig.return_root_visits: bool = False` next to existing flags (`mcts.py:34-72`). ~2 LOC.
- In `MCTSSolver.solve_instance` after `_pick_root_action` (line 232) and *before* the tree-reuse advance (line 236-240): if `cfg.return_root_visits`, append `dict(root.N)` to `self.root_visit_dists`. Reset the list at the top of `solve_instance` (alongside the existing `fwd_count_*` resets at line 201-204). ~10 LOC.
- Initialize `self.root_visit_dists: list[dict[int, int]]` in `MCTSSolver.__init__`.

**A.2 C++ sequential** — `src/am_baseline/search/mcts_cpp/{mcts.hpp, mcts.cpp, bindings.cpp}`.
- Add `bool return_root_visits = false` to the C++ `Config` struct in `mcts.hpp`. ~2 LOC.
- In the C++ `Solver::solve_instance` per-tour-step loop, after `pick_root_action`, if `cfg_.return_root_visits`: serialize `root->n_visits` into a `std::vector<std::pair<int,int>>`, push it onto a member `std::vector<...> root_visits_per_step_`. Marshal as `py::list[py::list[tuple[int,int]]]` in the result dict. ~25 LOC C++ + ~3 LOC bindings.
- `solver.py:CppMCTSSolver.solve_instance` plumbs this back as `self.root_visit_dists = [dict(step) for step in result["root_visit_dists"]]`. ~5 LOC.

**A.3 C++ batched (`cpp_batch`)** — same pattern in `BatchSearch`, but the per-step root pointer is held per-instance in the cross-instance scheduler. Dump each tree's `root->n_visits` after that tree's `pick_root_action`; emit as `raw["root_visit_dists_per_instance"]: list[list[list[tuple[int,int]]]]`. Plumb through `_solve_chunk` to populate `self.root_visit_dists_per_instance: list[list[dict]]`. ~30 LOC C++ + ~10 LOC python.

**A.4 Validation runs** (no new compute):
- **A.4.a (legality, tree-reuse-on, the production config).** TSP-20 K=200 rollout MCTS on 20 instances with `return_root_visits=True`, `tree_reuse=True`. Per-step assertions: (i) `pi_t = N(s_t,·)/Σ N` sums to 1 within float tolerance; (ii) `pi_t[a] = 0` for every visited city (this is the *correctness* invariant — we never want to train the policy on a visited-city action); (iii) **`support(pi_t) ⊆ unvisited(s_t)`** (subset, not equality) — PUCT may legitimately leave some unvisited cities with N=0 at low K or under a sharp prior + small `c_puct`, so legal-but-unexplored actions get zero mass; (iv) `argmax(pi_t)` is a legal action (i.e., unvisited); (v) no inf/nan; (vi) `pi_t` is non-negative everywhere. **Do NOT assert `Σ N ≤ K + 1`** — under tree reuse, the root inherits its predecessor subtree's visits, so total root visits routinely exceed K. The cumulative bound is a feature, not a bug.
- **A.4.b (exact-count, tree-reuse-off).** Same instance with `tree_reuse=False, K=200`. Assert `Σ_a N(s_t, a) == K` exactly at every tour-step. (Both backends only increment `parent.N[a]` once per simulation in backup — `mcts.py:288`, `mcts_cpp/mcts.cpp:498` — and there is no stored separate "root visit" count, so total edge visits at the root equal the simulation count exactly.) This validates the visit-counting code is correct in isolation.
- **A.4.c (bit-equivalence python vs cpp — DETERMINISTIC SETTINGS ONLY).** Run the same instance with `python` and `cpp` backends, both with `return_root_visits=True`, `tree_reuse=True`, **`dirichlet_epsilon=0.0`** (no Dirichlet noise — priors are byte-identical from the shared model), **`temperature=0.0` for all steps** (greedy argmax — no multinomial action sampling). Under these settings both backends consume zero RNG state during search, so trees are identical iff their deterministic logic (PUCT, FPU, expand/backup, tree-reuse advance) agrees. Assert per-step visit dicts match exactly (integer counts, no fp drift). Matches Stage 3's existing canonical-cost bit-equivalence pattern.

  **Why the deterministic clamp is required.** With `dirichlet_epsilon>0`, Python's `np.random.dirichlet` (Mersenne Twister) and C++'s `std::gamma_distribution` produce different bit patterns even at the same seed → priors diverge → trees diverge despite both implementations being correct. With `temperature>0`, multinomial action sampling diverges similarly. **Equality of `pi_t` between backends under the production self-play config (Dirichlet on, `step30`) is NOT a correctness invariant** — only distributional equivalence is, which a smoke test does not check.
- **A.4.d (smoke harness extension).** Add A13 in `src/scripts/smoke_mcts.py` covering `return_root_visits=True` for both `value_head` and `rollout` leaf eval, both `python` and `cpp` backends, including a `cpp_batch` slice. Use the legality/support invariants from A.4.a (production config — applied to each backend *independently*, no cross-backend equality claim). One A13 sub-case toggles `tree_reuse=False` for the A.4.b exact-count check; another sub-case clamps `dirichlet_epsilon=0` and `temperature=0` for the A.4.c bit-equivalence check.

**Code reuse:** Builds on Stage 3 Phase A's `fwd_count_*` instrumentation pattern (mcts.py:113-115; solver.py:53-72).

**Wall-clock:** 1 day dev (mostly C++ marshalling); minutes of compute.

**Dependencies:** none.

---

### Phase B — Replay buffer + distillation training step (~1.5 days; no GPU)

**Goal:** Implement the data structure and training step that consume MCTS targets, in isolation from the coach loop.

**B.1 `MCTSReplayBuffer`** — new module `src/am_baseline/training/coach.py` (~150 LOC for this class).
- **Storage: flat dict-of-pre-allocated-tensors** (not Python deque-of-objects — millions of small Python tuples explode in tensor-header overhead at ~600 B/tensor × 4M tuples = 2.4 GB of headers alone).
  ```
  buffer = {
    # Per-instance (capacity_instances = 200_000):
    'coords':       (capacity_instances, N, 2)   float32,
    'bl_val':       (capacity_instances,)        float32,  # FROZEN at instance-push time, never refreshed
    'tour_cost':    (capacity_instances,)        float32,

    # Per-step (capacity_tuples = capacity_instances * N ≈ 4M):
    'pi':           (capacity_tuples, N)         float32,  # raw τ=1 normalized visits
    'visited':      (capacity_tuples, N)         bool,
    'first_a':      (capacity_tuples,)           int16,
    'prev_a':       (capacity_tuples,)           int16,
    'lengths':      (capacity_tuples,)           float32,  # cumulative cost so far
    'cost_to_go':   (capacity_tuples,)           float32,  # tour_cost - lengths
    'inst_idx':     (capacity_tuples,)           int32,    # back-pointer to coords row
  }
  ```
  Total ~520 MB pre-allocated. Fixed footprint, no Python-object overhead. Pattern matches KataGo's `python/shuffle.py` and AGZ-replicas.
- **Capacity:** `capacity_instances` (default 200K instances → ~4M tuples at TSP-20).
- **Eviction:** ring-buffer write head with `inst_idx % capacity_instances`; tuple slots `(inst_idx * N + step) % capacity_tuples`. Drops oldest instance + all its tuples atomically.
- **Auxiliary index `_step_index: list[np.ndarray]`** (length N). `_step_index[t]` is the array of tuple-slot indices currently filled with step==t. Updated on every push (append the new tuple slot) and on every eviction (drop the oldest tuple slot for that step). Enables O(1) stratified sampling without scanning the whole buffer.
- **All N per-step records are stored, including the final forced step.** No "skip when `legal_actions == 1`" — that mitigation conflicts with the dense layout (would require a `valid_mask` + rejection sampling or a variable-length `_step_index`). Late-step records carry near-zero policy gradient (CE between two near-identical one-hot distributions is small and finite), but the value-loss MSE is still informative. Matches AGZ Methods §Self-play, which stores tuples for every step up to termination.
- **Sampling — stratified by step (fixes mixed-step decoder bug).** Two methods: `sample(batch_size)` for the training loop (random step) and `sample_step(t, batch_size)` for tests/ablations (deterministic step):
  ```python
  def sample_step(self, step: int, batch_size: int) -> dict:
      """Deterministic-step variant — used by smoke tests and any future
      step-stratified analysis. Picks `batch_size` records uniformly from
      records currently filled at step==step.
      """
      step_indices = self._step_index[step]
      n_at_step = step_indices.shape[0]
      if n_at_step < batch_size:
          idx_within = np.random.randint(0, n_at_step, batch_size)        # with replacement (fresh buffer)
      else:
          idx_within = np.random.choice(n_at_step, batch_size, replace=False)
      idx = step_indices[idx_within]
      # Fancy-index per-step tensors at idx; per-instance tensors (coords, bl_val) at inst_idx[idx].
      z_batch = cost_to_go[idx] / bl_val[inst_idx[idx]]
      return {'state_i': step, 'coords': coords[inst_idx[idx]], ...}      # state_i is a SCALAR

  def sample(self, batch_size: int) -> dict:
      """Training-loop variant — picks one step uniformly per minibatch."""
      step = np.random.randint(0, self.N)
      return self.sample_step(step, batch_size)
  ```
  **Why stratified.** AM's decoder takes a single scalar `state.i` per `decode_step` call (`state.py:5-19`), with first-step branching that conditions on `state.i == 0` vs `> 0`. A uniform-over-tuples batch can mix step 3, 17, 9 records under one `state.i`, which silently produces wrong `log_p` distributions on the misaligned rows. Stratification picks one step per minibatch — every row's step matches the decoder's scalar `state.i`. Marginal distribution over (instance, step) across many train steps remains uniform; expectation of the gradient is unchanged; only per-batch variance is slightly higher (negligible at J=200 train steps × N=20 steps → ~10× coverage per step per iter). Pattern matches AGZ-style replicas (OpenSpiel, ELF OpenGo).
- **`z_batch = cost_to_go[idx] / bl_val[inst_idx[idx]]`** — both terms are frozen-at-generation, so the per-instance training target $z_t$ is **stationary across all training steps that draw this record**.
- **`bl_val` is frozen at instance-push time and never refreshed.** When `generate_self_play_batch(θ★, ...)` produces an instance, it computes `bl_val = cost(greedy_rollout(θ★, x))` once and writes it to the per-instance row alongside `tour_cost`. The owning θ★ may be superseded later by a gate accept; this does not invalidate the stored `bl_val`. Rationale: per-state $z_t$ + frozen `bl_val` makes the training target fully stationary per record, eliminating the moving-target concern (spec §3.5 Concern 1) without an `owner_id`-tagged refresh code path. The only "drift" is buffer-level: newer instances were generated under stronger θ★ and have smaller `bl_val`, so their $z_t$ is on a slightly different scale than older ones — but this is the *correct* mixture of "policy-iteration progress evidence" the loop is designed to learn from, not a bug.
- **Persistence:** `save(path)` writes the tensor dict + `n_filled_tuples` + `write_head_inst` + `write_head_tuple` via `torch.save`. **`_step_index` is NOT saved** — it is a runtime-cached projection of the dense slot layout and is deterministically rebuilt on `load()`. Rebuild logic exploits the locked invariant `tuple_slot = inst_idx * N + step`:
  ```python
  def load(self, path):
      d = torch.load(path)
      self.<tensor fields>...                      # restore data
      self.n_filled_tuples = d['n_filled_tuples']
      self.write_head_inst = d['write_head_inst']
      self.write_head_tuple = d['write_head_tuple']
      # Rebuild _step_index from the (slot % N == step) identity.
      slots = np.arange(self.capacity_tuples)
      step_per_slot = slots % self.N
      filled = self._filled_mask()                 # True for currently-valid slots
      self._step_index = [
          slots[(step_per_slot == t) & filled].astype(np.int32)
          for t in range(self.N)
      ]
  ```
  Cost: O(`capacity_tuples`) one-time on load — microseconds at TSP-20 (~4M slots). Rationale for rebuild over persist: eliminates a "saved index out of sync with saved data" failure mode if any future buffer change updates one path but not the other; keeps the save file smaller and schema-stable; makes `_step_index` purely a derived runtime cache. Sampling correctness is order-invariant within each step's index, so the rebuild produces an equivalent (not necessarily identical) index to what was in memory at save time.

**B.2 State-tensor reconstruction utility** in the same file (~50 LOC).
- Given a buffer record, reconstruct the `StateTSP` named-tuple needed by `model.decoder.decode_step`.
- Mirrors `solver.py:_state_from_snapshot` pattern (`mcts_cpp/solver.py:452-489`) — same fields, same dtype, same device handling.
- Critical: the gradient must flow through *the same compute graph* used during self-play, so `precompute_decoder` must be called on the encoded coords inside the train step, not cached.

**B.3 `train_step_alphazero`** — add to `src/am_baseline/training/trainer.py` alongside the existing `train_batch` (~100 LOC).

Signature: `train_step_alphazero(model, optimizer, batch, opts) → dict`.

```
# Pseudocode. batch is the dict returned by buffer.sample(B):
#   state_i  : scalar int (the single step value for this minibatch — stratified)
#   coords   : (B, N, 2) float32
#   visited  : (B, N) bool         — all aligned to state_i
#   first_a, prev_a, lengths : (B,)   — all aligned to state_i
#   pi       : (B, N) float32      — raw τ=1 normalized visits at step state_i
#   z        : (B,)   float32      — cost-to-go / bl_val at step state_i

encoded = model.encode(batch['coords'])                  # (B, N, embed_dim)
fixed = model.precompute_decoder(encoded)                # AttentionModelFixed
state = reconstruct_state(batch, i=batch['state_i'])     # StateTSP NamedTuple, state.i = scalar
log_p, mask, glimpse = model.decode_step(fixed, state, return_glimpse=True)
                                                        # log_p: (B, 1, N); mask: (B, 1, N); glimpse: (B, embed_dim)
log_p = log_p.squeeze(1)                                 # (B, N) — strip decoder's per-step axis
mask  = mask.squeeze(1)                                  # (B, N) — match pi shape, avoid (B,B,N) broadcast
v = model.value_head(glimpse)                            # (B,)

# Policy distillation cross-entropy. batch['pi'] is (B, N), masked-zero on visited cities.
# log_p has -inf at masked positions; replace with 0 to avoid 0 * -inf = NaN.
log_p_safe = torch.where(mask, torch.zeros_like(log_p), log_p)
policy_loss = -(batch['pi'] * log_p_safe).sum(dim=-1).mean()

# Value loss — per-state V_CURRENT (cost-to-go), matching Stage 1's target shape
# and the leaf-evaluator invariant in `mcts.py:1-15`.
# batch['z'] = cost_to_go[idx] / bl_val[inst_idx[idx]] computed at sample time.
value_loss = F.mse_loss(v, batch['z'])

# Total. L2 handled via optimizer.weight_decay (not in this loss).
loss = policy_loss + opts.lambda_v * value_loss
```

**Design choices:**
- **Per-step π distillation target.** π_t = N(s_t, ·) / Σ N (raw τ=1 normalized visits, *not* the τ-tempered action-selection distribution σ_t). See spec §4.2 — choice (B): action selection uses step30, training target stays τ=1 for richer late-game distillation signal.
- **Per-step V_CURRENT value target (NEW — was broadcast `z` in earlier plan rev).** z_t = (tour_cost − lengths_t) / bl_val matches Stage 1's `value_targets_from_edges` shape (`utils/tensor_ops.py:57-78`). Why: existing MCTS leaf evaluator computes `total_norm = state.lengths/bl_val + v(state)`, which assumes v predicts remaining cost-to-go. Training v on broadcast full-tour cost would double-count path cost at MCTS time. Per-state z_t makes Phase A's leaf evaluator a no-op vs Stage 3.
- **L2 via `optimizer.weight_decay`.** Cleaner than computing `c·||θ||²` inside the loss. Default `weight_decay=1e-4`.
- **`bl_val` from θ★, frozen at generation** — `generate_self_play_batch` in Phase C computes `bl_val(x) = cost(greedy_rollout(θ★, x))` at self-play time and stores it alongside `tour_cost` in the buffer's per-instance row. Same model that produced the tour produces the normalization. **No recomputation, no refresh** — when θ★ is superseded by a gate accept, existing buffer entries retain their original `bl_val`. The per-state $z_t = (\text{tour\_cost} - \text{lengths}_t)/\text{bl\_val}$ for any given buffer record is therefore **stationary across the record's entire lifetime**, eliminating the moving-target concern of broadcast-z designs.

**B.4 Smoke unit test** — `src/scripts/smoke_alphazero.py` (~150 LOC; staged across phases).
- **A1**: construct a 5-instance buffer manually (random pi_t, random z), run one `train_step_alphazero`, verify loss is finite, gradients flow into encoder + value_head + decoder.
- **A1.5 (stratification — deterministic per-step coverage).** Loop `for t in range(N): batch = buffer.sample_step(t, batch_size=8)` and assert every returned batch has all rows reporting `state_i == t` (catches wrong-row-routing). Also call `buffer.sample(batch_size=8)` once and assert all rows in that batch share the same `state_i` (the random-step path obeys stratification). Assert `pi_batch[r, a] == 0` for every visited city `a` per row `r` (correctness invariant — never train on visited-city actions); assert `pi_batch.sum(-1)` is 1 within float tolerance; assert `pi_batch >= 0` everywhere. **Do NOT** assert that every unvisited city has positive `pi_batch` mass — at low K or sharp priors, PUCT can leave legal actions unexplored (N=0 → π=0). **Probabilistic coverage of `sample()` over many random calls is intentionally not asserted** — at N=20 with 30 draws the miss probability is ~0.43; with 200 draws it's still nontrivial. Stratification correctness is established by `sample_step(t)` deterministic checks, not by waiting for the random path to enumerate steps.
- **A1.6 (resume — `_step_index` rebuild on load)**: push 50 instances into a fresh buffer, snapshot `_step_index`, call `buffer.save(tmp_path)` then construct a new buffer and `buffer.load(tmp_path)`. Assert each `_step_index[t]` (as a *set* of slot indices — rebuild order can differ from runtime-append order; sampling is set-uniform so order is irrelevant) equals the snapshot. Then loop `for t in range(N): buffer.sample_step(t, batch_size=4)` on the reloaded buffer and assert each returned batch satisfies the A1.5 invariants (all rows have `state_i == t`, π non-negative + sums-to-one + visited-mass-zero). This catches "save persisted data but not the index, sample crashes / returns garbage" deterministically.

**Code reuse:** value-target normalization machinery from `trainer.py:208-219` (the `bl_val.unsqueeze(-1)` Z pattern); decoder API from `attention_model.py:precompute_decoder, decode_step`.

**Wall-clock:** 1.5 days dev. Validation seconds.

**Dependencies:** none (parallel with A).

---

### Phase C — Self-play data generator (~1 day; no GPU)

**Goal:** Implement `generate_self_play_batch(model, M, graph_size, cfg, device) → list[InstanceRecord]` that calls `CppBatchMCTSSolver` with `return_root_visits=True` and packs results into buffer records.

**C.1 Self-play config preset** in `coach.py` (~25 LOC):

```python
def make_self_play_config(graph_size, n_simulations) -> MCTSConfig:
    return MCTSConfig(
        n_simulations=n_simulations,
        leaf_eval='value_head',                  # AlphaGo Zero canonical
        value_norm='bl',
        c_puct=0.05,
        temperature=1.0,                         # base τ for the schedule
        temperature_schedule='step30',           # τ=1 for first ⌈0.3·N⌉ steps, τ=0 after (AGZ-proportional)
        dirichlet_alpha=10.0/graph_size,         # 0.5 for N=20
        dirichlet_epsilon=0.25,                  # AlphaGo Zero standard
        fpu_mode='running_q', fpu_fallback=-1.0,
        root_select='visits', tree_reuse=True,
        return_root_visits=True,                 # exposes raw N(s_t, ·) for π_t target (τ=1 always)
    )
```

**Note on `value_norm='bl' + leaf_eval='value_head'`.** Stage 3's `MCTSConfig._validate_config` *rejects* this combination (mcts.py:118-162) because the value head was trained against `bl_val`-normalized targets but `sqrt_n` would create a scale mismatch. With `value_norm='bl'`, the combo is valid — confirmed by Stage 3 E.2 result. Stage 4 self-play uses this combo throughout.

**C.2 Generator function** in `coach.py` (~120 LOC).

```python
def generate_self_play_batch(best_model, M, graph_size, cfg, device) -> list[InstanceRecord]:
    instances = TSP.make_dataset(size=graph_size, num_samples=M)
    coords = torch.stack([inst for inst in instances]).to(device)   # (M, N, 2)

    # bl_val from θ★ (the model that will run MCTS) — frozen for the lifetime of this batch.
    # Cleaner than computing bl_val from the trainer's evolving θ; see spec §3.5 Concern 2.
    with torch.no_grad():
        best_model.set_decode_type('greedy')
        bl_costs, _ = best_model(coords)
        bl_val = bl_costs.cpu()                                     # (M,) float32

    solver = CppBatchMCTSSolver(best_model, cfg, device, mcts_batch_size=64)
    tour_costs, tours = solver.solve_batch(coords)
    visits = solver.root_visit_dists_per_instance                   # list[list[dict]]

    records = []
    for i in range(M):
        tour_cost = tour_costs[i].item()
        tour = tours[i].cpu().numpy()
        # Edge costs along the played tour (closing edge included as edge_costs[N-1]).
        edge_costs = compute_edge_costs(coords[i].cpu().numpy(), tour)  # (N,) float32
        # V_CURRENT at each step: cost of edges still to be traversed FROM s_t.
        # value_targets_from_edges in utils/tensor_ops.py:57-78 produces this exact shape;
        # equivalent to (tour_cost - lengths_t) for t ∈ {0..N-1}.
        cost_to_go = value_targets_from_edges(torch.from_numpy(edge_costs).unsqueeze(0)).squeeze(0).numpy()  # (N,)

        per_step = []
        for t in range(graph_size):
            visited_mask = mask_from_tour(tour[:t], graph_size)         # (N,) bool
            pi_t = normalize_visit_dict(visits[i][t], graph_size)       # (N,) float32, τ=1 normalized
            length_t = edge_costs[:max(0, t-1)].sum() if t > 0 else 0.0  # state.lengths at step t
            per_step.append({
                'visited': visited_mask, 'first': tour[0] if t > 0 else -1,
                'prev': tour[t-1] if t > 0 else -1, 'lengths': length_t,
                'pi': pi_t, 'cost_to_go': cost_to_go[t],
            })
        records.append(InstanceRecord(
            coords=coords[i].cpu(), bl_val=bl_val[i].item(), tour_cost=tour_cost,
            per_step=per_step,
        ))
    return records
```

**Tradeoff: value_head leaf eval per Phase scope decision 2.** Faster per-iter wall-clock; Stage 5 ablation covers rollout for comparison.

**C.3 Validation** — extend `smoke_alphazero.py` with A2: generate 10 TSP-20 instances with K=20, verify `pi_t` sums to 1, is zero on visited cities, and argmax aligns with the chosen tour for `temperature=0`. Verify value-head-leaf-eval+`value_norm='bl'` does not raise.

**Code reuse:** `TSP.make_dataset` (`tsp.py:51-70`); `CppBatchMCTSSolver` (`mcts_cpp/solver.py:562-648`).

**Wall-clock:** 1 day dev. Validation 1 min.

**Dependencies:** Phase A (visit-count exposure).

---

### Phase D — `MCTSCoach.learn` orchestrator (~1.5 days; no GPU)

**Goal:** Wire generate → train → gate into one iteration loop with checkpointing, W&B logging, and resume support.

**D.1 `MCTSCoach` class** in `coach.py` (~250 LOC).

```python
class MCTSCoach:
    def __init__(self, model, problem, opts, val_dataset, device):
        self.model = model                                # the trainer's working copy
        self.best_model = copy.deepcopy(model)            # the self-play / gating reference
        self.problem = problem
        self.opts = opts
        self.val_dataset = val_dataset
        self.buffer = MCTSReplayBuffer(capacity_instances=opts.buffer_capacity)
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=opts.lr_model, weight_decay=opts.weight_decay
        )
        # Reuse Stage 1's gating directly:
        from am_baseline.baseline.baselines import RolloutBaseline
        self.gating_baseline = RolloutBaseline(
            self.best_model, problem, opts, rollout_fn=rollout, epoch=0
        )
        self.iter_idx = 0
        self.total_instances_seen = 0
        self.logger = MetricsLogger(...)

    def learn(self, n_iterations):
        for self.iter_idx in range(n_iterations):
            t0 = time.time()
            cfg = make_self_play_config(self.opts.graph_size, self.opts.n_simulations_train)
            records = generate_self_play_batch(
                self.best_model, self.opts.M_instances, self.opts.graph_size, cfg, self.opts.device
            )
            self.buffer.add(records)
            self.total_instances_seen += self.opts.M_instances
            t1 = time.time()

            for step in range(self.opts.train_steps_per_iter):
                batch = self.buffer.sample(self.opts.batch_size)
                metrics = train_step_alphazero(self.model, self.optimizer, batch, self.opts)
                self.logger.log_step(metrics, self.iter_idx, step)
            t2 = time.time()

            val_cost = validate(self.model, self.val_dataset, self.opts)

            gated = False
            if (self.iter_idx + 1) % self.opts.gate_every == 0:
                gated = True
                accepted = self.gating_baseline.epoch_callback(self.model, epoch=self.iter_idx)
                if accepted:
                    self.best_model = copy.deepcopy(self.model)
                    self._save_checkpoint(tag=f"iter{self.iter_idx}_accepted")
                # NB: per scope decision 3, NO rollback on reject.

            self.logger.log_iteration(
                iter=self.iter_idx, total_instances=self.total_instances_seen,
                val_avg_cost=val_cost, gated=gated, accepted=accepted if gated else None,
                mcts_wall_s=t1-t0, train_wall_s=t2-t1,
            )
            self._save_checkpoint(tag=f"iter{self.iter_idx}")  # always save, for resume
```

**D.2 Logging extensions** — `src/am_baseline/training/logging.py`.
- New CSV `iterations.csv` with columns: `iter, total_instances, val_avg_cost, policy_loss_mean, value_loss_mean, mean_entropy_pi, gated, accepted, mcts_wall_s, train_wall_s, buffer_size`.
- New W&B step axis `iteration` plus a "sample-efficiency" custom plot (`x = total_instances`, `y = val_avg_cost`).
- ~50 LOC additive.

**D.3 Checkpoint format.** `outputs/tsp_20/stage4_<run_name>_<timestamp>/iter-{i}.pt` containing `{model, best_model, optimizer, buffer (separate file: buffer.pt), iter_idx, total_instances_seen, rng_state}`. Buffer saved separately to keep checkpoint files small. Resume via `--resume_from <iter>`.

**Code reuse:** `RolloutBaseline.epoch_callback` (`baselines.py:106-123`) handles the entire t-test gating verbatim — same paired t-test α=0.05 logic Stage 1 uses; `validate` (`trainer.py:13-19`) for the val curve; `MetricsLogger` (`logging.py`) for W&B plumbing.

**CLI / init-order trap (caught at review).** `RolloutBaseline.__init__` constructs and caches its validation dataset using `opts.val_size` *at construction time*; subsequent calls to `epoch_callback` do not re-read this value. Therefore: `MCTSCoach.__init__` must construct `RolloutBaseline(model, problem, opts, ...)` *after* `opts.val_size` has been finalized from the CLI. There is no Stage-4-specific `gate_val_size` flag — `opts.val_size` is the single source of truth for the gating dataset size, inherited from Stage 1's CLI conventions. Smoke tests that need a smaller validation set pass `--val_size 100` directly.

**Wall-clock:** 1.5 days dev.

**Dependencies:** Phases A, B, C.

---

### Phase E — Temperature schedule + Dirichlet noise wiring (~0.5 day; no GPU)

**Goal:** Lock the canonical exploration schedule. Most of this is already supported in `MCTSConfig`; the per-step temperature schedule needs new plumbing.

**E.1 Per-tour-step temperature schedule** — `src/am_baseline/search/mcts.py` and C++ mirror.
- Add `MCTSConfig.temperature_schedule: Optional[str] = None` accepting `{None, 'const', 'step30', 'step50'}`. Default `None` ≡ `'const'` ≡ existing scalar `cfg.temperature` behavior.
- `'step30'`: τ=`cfg.temperature` for first ⌈0.3·N⌉ tour-steps, τ=0 thereafter. **Closest analogue of AGZ's "first 30 of ~250 moves" (~12 %), scaled with safety margin to TSP's much shorter games (~30 % of plies vs AGZ's ~12 %).** This is the new pilot default.
- `'step50'`: τ=`cfg.temperature` for first ⌈0.5·N⌉ tour-steps, τ=0 thereafter. More exploratory; G.4 ablation lever.
- Python: update `_pick_root_action` (`mcts.py:413-442`) to read `state.i` (current step) and look up the schedule.
- C++ mirror: precompute a `std::vector<double> tau_per_step` of length N at solve_instance entry, filled per the schedule. Pass as part of Config. ~10 LOC C++.

**E.2 Dirichlet noise.** Already wired in `mcts.py:226-228, 444-460` and in C++. Just expose CLI flags in `train_alphazero.py`. Phase G ablation lever (G.3).

**Justification for `step30` as default:** TSP tour permutations have most policy entropy in the early steps (first city → second city has N-1 candidates with similar Q); the last few cities are forced or near-forced. AGZ's pattern is "explore early, exploit late" — first 12 % of game uses τ=1, rest τ=0. For TSP-20 the N-step game is 12.5× shorter than Go, so a fixed-percentage analog (`step30`) gives ~6 of 20 plies of exploration, leaving 14 deterministic plies. `step50` is the more aggressive variant we initially proposed; G.4 ablates the two against `const`.

**E.3 Validation** — extend smoke A3: TSP-20 K=50 self-play with `temperature_schedule='step30'`. **Two distributions to verify separately:**
- **Action-selection σ_t** (used to sample the played action): entropy decays sharply at step ⌈0.3·N⌉ = 6; collapses to one-hot (entropy = 0) thereafter.
- **Training target π_t = N/Σ N** (raw τ=1 normalized, stored in buffer): entropy stays bounded above zero throughout — only collapses to one-hot when MCTS visit counts themselves become degenerate (last 1-2 plies where legal_actions = 1). Per choice (B) in spec §4.2: π_t is decoupled from the τ-schedule.

**Wall-clock:** 0.5 day dev. Negligible compute.

**Dependencies:** Phase A.

---

### Phase F — TSP-20 pilot + main run (~3.5 h compute)

**Goal:** End-to-end Stage 4 run on TSP-20.

**F.1 New script `src/scripts/train_alphazero.py`** (~180 LOC).
- CLI mirrors `train.py` plus Stage 4 flags:
  - `--load_path` (path to Stage 1 checkpoint, **required**)
  - `--n_iterations` (default 100)
  - `--M_instances` (default 1000) — instances per iteration
  - `--n_simulations_train` (default 50) — K during self-play
  - `--buffer_capacity` (default 200000) — instance count
  - `--train_steps_per_iter` (default 200) — minibatch updates per iteration
  - `--batch_size` (default 512)
  - `--gate_every` (default 5)
  - **No `--gate_val_size`** — `RolloutBaseline.__init__` reads `opts.val_size` once at construction time and freezes the validation set; a dedicated `--gate_val_size` flag would be a silent no-op unless we monkey-patch `opts.val_size` before the baseline is built. Stage 4 reuses Stage 1's existing `--val_size` (default 10000) for the gating dataset; override with `--val_size 100` for smoke tests.
  - `--temperature_schedule {const,step30,step50}` (default `step30` — closest to AGZ Methods §Self-play, scaled to TSP plies)
  - `--dirichlet_epsilon` (default 0.25)
  - `--dirichlet_alpha_factor` (default 10.0; α = factor / N)
  - `--lambda_v` (default 1.0)
  - `--weight_decay` (default 1e-4)
  - `--lr_model` (default 1e-4)
  - `--leaf_eval {value_head,rollout}` (default `value_head`)
  - `--resume_from <path>` (resume from a checkpoint)
- Loads Stage 1 checkpoint; constructs `MCTSCoach`; calls `coach.learn(opts.n_iterations)`.

**F.2 Smoke battery `src/scripts/smoke_alphazero.py`** (~200 LOC, totaling A1-A6).
- A1 (Phase B): construct a 5-instance buffer, run `train_step_alphazero` once → finite loss, gradients flow.
- A2 (Phase C): generate 10 instances with K=20, verify π_t shape + sum.
- A3 (Phase E): self-play with `temperature_schedule='step30'`. Action-distribution σ_t entropy decays sharply at step ⌈0.3·N⌉=6 (collapses to one-hot thereafter); training-target π_t entropy stays bounded above zero throughout (because π_t is always τ=1 normalized, decoupled from σ_t per spec §4.2 choice B).
- A4 (tree_reuse=True, production config): legality/support/finiteness checks on `pi_t` at each tour-step — sums to 1, **support(pi_t) ⊆ unvisited(s_t)** (subset; legal-but-unexplored actions may have N=0 → π=0), `argmax` is unvisited, no inf/nan, non-negative. No cumulative-count bound (tree reuse makes `Σ_t Σ_a N(s_t, a)` grow above K·N legitimately). A4-strict (tree_reuse=False, K=50): `Σ_a N(s_t, a) == K` at every step (one increment per simulation, no stored root-node visit).
- A5: gating no-op when `gate_every > n_iterations` → no `epoch_callback` calls.
- A6: 3 iterations end-to-end with M=10, K=20 → no NaN, val_avg_cost finite, checkpoint round-trips through `--resume_from`.

**F.3 TSP-20 pilot run.**
- Warm-start from `outputs/tsp_20/stage1_tsp20_canonical_20260423T103541/epoch-99.pt`.
- Config: 20 iterations × M=1000 × K=50 × `train_steps_per_iter=100` × `batch_size=512` × `buffer_capacity=50_000` (~50-iter window, closer to AGZ's ~20-iter ratio than the 200K main-run default; surfaces staleness early if it exists).
- Per-iteration wall-clock estimate (RTX 4060): MCTS ~30 s (extrapolated from Stage 3 TSP-20 K=20 cpp_batch bs=64 = 8.1 s; K=50 ≈ 20 s) + train ~10 s = ~40 s/iter. **Total ~15 min.**
- Output: `outputs/tsp_20/stage4_pilot_<timestamp>/`.
- **Pilot pass conditions** (must all hold to proceed to F.4):
  - val_avg_cost at iter 20 is ≤ Stage 1's val_avg_cost (3.84443 from `_progress/stage1_progress.md` bs=2048, or 3.83943 from canonical bs=512) within noise.
  - No NaN losses across all iterations.
  - At least one gating call (gate_every=5, n_iterations=20 → 4 gates).
  - Sample-efficiency curve trends downward (val_avg_cost vs iter monotone non-increasing within ±0.001 noise band).

**F.4 TSP-20 main run** (after pilot passes).
- Same recipe scaled up: 100 iterations × M=1000 × K=100 × `train_steps_per_iter=200` × `batch_size=512`.
- Per-iteration wall-clock: MCTS ~60 s (K=100 cpp_batch bs=64 extrapolated) + train ~20 s = ~80 s/iter. **Total ~2.2 h.**
- Output: `outputs/tsp_20/stage4_main_<timestamp>/`.
- Gates every 5 iterations → 20 gating opportunities.

**F.5 Headline plot** — `src/scripts/plot_stage4.py` (~80 LOC).
- Reads `iterations.csv` from F.4 plus `outputs/tsp_20/stage1_tsp20_canonical_*/epochs.csv` from Stage 1.
- Output: `outputs/stage4/figures/sample_efficiency_tsp20.png` — `x = total_instances` (log scale), `y = val_avg_cost`, two curves: Stage 1 REINFORCE (epoch checkpoints) vs Stage 4 (per-iteration).
- Annotate with horizontal lines: Gurobi optimum (3.8279), Stage 1 final val_avg_cost, Stage 3 K=400 rollout MCTS (3.8312).

**Wall-clock:** ~15 min pilot + ~2.2 h main + plot = **~2.5 h compute total.**

**Dependencies:** Phases A, B, C, D, E.

---

### Phase G — TSP-20 ablations (~6-10 h compute, optional)

**Goal:** Measure the most-uncertain knobs. All run on TSP-20 with same warm-start. Each is a Stage 5 prerequisite.

| Ablation | Levers | Compute | Why |
|---|---|---|---|
| **G.1** Leaf eval: `rollout` vs `value_head` | F.4 recipe with `leaf_eval='rollout'` | ~3 h | The defining AlphaGo-Zero choice; head-to-head test |
| **G.2** Buffer capacity | 50K vs 200K vs 500K | 3× ~2 h | Catastrophic forgetting check |
| **G.3** Dirichlet ε ∈ {0, 0.15, 0.25, 0.4} | 4 small TSP-20 runs (50 iter) | 4× ~1 h | Optimal exploration mass |
| **G.4** Temperature schedule + target coupling: `const`, `step30` (default), `step50`; AND **strict-AGZ** (training target π_t uses same τ_t as σ_t — one-hot late targets) vs **decoupled** (π_t always τ=1 — F.4 default, choice B) | small TSP-20 runs | 4× ~1 h | Schedule efficacy + late-game distillation signal richness (TSP-20's deterministic late steps may make strict-AGZ one-hot a wasted signal vs richer τ=1 raw visits) |
| **G.5** Gating cadence: gate every 1/5/10 | 3 small runs | 3× ~1 h | Interaction with reject-policy |
| **G.6** Best-so-far per-instance value normalization (replaces original "cost-to-go vs broadcast z" — cost-to-go is now the F.4 default) | $z_t = (\text{tour\_cost} − \text{lengths}_t) / \min_\text{seen} \text{tour\_cost}(x)$ | ~1 h | Value-target normalization shape; tracks instance-level best instead of model's `bl_val` |
| **G.7** Symmetry augmentation at leaf eval | random 2D rotation+flip of coords pre-encoder | ~1 h | AGZ Methods §Search algorithm: dihedral aug at every leaf eval (8-fold). TSP analog is continuous SO(2) × {flip} |
| **G.8** Optimizer: Adam vs SGD+momentum 0.9 | F.4 recipe with SGD+momentum, lr=1e-3 | ~3 h | AGZ canonical optimizer; checks whether Adam-from-warmstart is leaving signal on the table |

Run only the ablations relevant to the user's interpretation of F.4 results. **G.1 is highest priority** as the explicit AlphaGo-Lee-vs-Zero head-to-head; **G.4** (temperature) and **G.7** (symmetry) are the cheapest paths to AGZ-canonical fidelity.

**Wall-clock:** 6-10 h on RTX 4060 (parallelizable on Modal).

**Dependencies:** Phase F closed.

---

## Acceptance criteria for Stage 4 closure

**TSP-20 main run (Phase F.4):**

1a. ✅ **Marginal improvement (warm-start signal).** Stage 4's final val_avg_cost is strictly better than its starting checkpoint by ≥ 0.001: `val_avg_cost(θ_iter100) ≤ 3.83843` (Stage 1 final 3.83943 minus 0.001 noise band). This is the AGZ-loop-improves-on-REINFORCE test, and is what Stage 4 is designed to demonstrate.

1b. ⚠️ **Strict total sample efficiency** (proposal-headline claim, requires careful interpretation). The combined (Stage 1 + Stage 4) sample-efficiency curve, plotted with x-axis = cumulative training instances seen across both stages, should reach Stage 1's final val_avg_cost (3.83943) at total instances < 128M (Stage 1 alone). Because Stage 4 warm-starts from the Stage 1 endpoint, this is trivially true at x = 128M + 1 (Stage 4 has only added improvements). The *meaningful* form is: at any matched x in the overlap region, Stage 1 + Stage 4 ≤ Stage 1 alone. Headline plot must annotate this clearly. **Strict from-scratch sample efficiency** (Stage 4 starting from random weights, comparing total instances to Stage 1's curve) is a Stage 5 follow-up — current Stage 4 *cannot* claim it.

2. ✅ **Ultimate quality.** Stage 4 final greedy val_avg_cost ≤ 3.8312 (Stage 3 K=400 rollout MCTS) — i.e., Stage 4's network alone (no MCTS at test time) matches Stage 3's search-augmented Stage 1 result. Equivalently, Stage 4 collapses the test-time gap between greedy and MCTS-K=400 rollout to within noise.

3. ✅ **Self-improvement.** val_avg_cost curve over iterations is monotone non-increasing within a ±0.001 noise band.

4. ✅ **Gating fires.** `gating_baseline.epoch_callback` returns `True` at least once over the 20 gating events.

**Reach (Stage 5 stretch):**
- TSP-20 final greedy val_avg_cost ≤ 3.8298 (= 0.05% gap vs Gurobi optimum 3.8279) — proposal target.
- Strict from-scratch sample efficiency (criterion 1b at random init) — needs Stage 5 from-scratch run.

---

## Code change inventory

| File | Action | Lines | Purpose |
|---|---|---|---|
| `src/am_baseline/search/mcts.py` | Edit | ~25 | Add `MCTSConfig.return_root_visits`, `MCTSConfig.temperature_schedule`; populate `solver.root_visit_dists` |
| `src/am_baseline/search/mcts_cpp/mcts.hpp` | Edit | ~10 | Add `return_root_visits`, `tau_per_step` to `Config` |
| `src/am_baseline/search/mcts_cpp/mcts.cpp` | Edit | ~50 | Emit per-step root visit dists; respect tau schedule in `pick_root_action` |
| `src/am_baseline/search/mcts_cpp/solver.py` | Edit | ~30 | Plumb `root_visit_dists` (+ `_per_instance` for cpp_batch) |
| `src/am_baseline/search/mcts_cpp/bindings.cpp` | Edit | ~5 | Expose new Config fields |
| `src/am_baseline/training/coach.py` | Create | ~500 | `MCTSCoach`, `MCTSReplayBuffer`, `make_self_play_config`, `generate_self_play_batch`, state reconstruction utility |
| `src/am_baseline/training/trainer.py` | Edit | ~100 | New `train_step_alphazero` |
| `src/am_baseline/training/logging.py` | Edit | ~50 | Add `log_iteration`, `iterations.csv` schema, sample-efficiency W&B plot |
| `src/am_baseline/config.py` | Edit | ~30 | Stage 4 fields: `n_iterations, M_instances, n_simulations_train, buffer_capacity, train_steps_per_iter, gate_every, temperature_schedule, dirichlet_*, weight_decay, leaf_eval` |
| `src/scripts/train_alphazero.py` | Create | ~180 | CLI wrapper; warm-start; resume; `coach.learn()` |
| `src/scripts/smoke_alphazero.py` | Create | ~200 | Smoke A1..A6 |
| `src/scripts/plot_stage4.py` | Create | ~80 | Sample-efficiency headline plot |
| `src/scripts/smoke_mcts.py` | Edit | ~40 | Add A13 `return_root_visits` smoke |
| `_progress/stage4_progress.md` | Create | (seed) | Progress tracker |
| `_plans/stage4_plan.md` | Create | (mirror) | Mirror of this plan |
| **Total** | | **~1300 new/edited** | |

---

## Compute budget

| Phase | Compute | Type |
|---|---|---|
| A — visit-count exposure | <0.1 h | Code; smoke only |
| B — replay buffer + loss | <0.1 h | Code |
| C — self-play generator | <0.1 h | Code; smoke only |
| D — coach orchestrator | <0.1 h | Code |
| E — temperature schedule | <0.1 h | Code; smoke only |
| F.3 — TSP-20 pilot | ~0.25 h | Critical path |
| F.4 — TSP-20 main | ~2.2 h | Critical path |
| F.5 — headline plot | <0.05 h | |
| G — ablations | 6-10 h | Optional |
| **Total Stage 4 critical path** | **~2.5 h GPU** | RTX 4060 sufficient |
| **Total dev** | **~5-6 days** | |

---

## Open design decisions to clarify with user (none blocking — defaults selected; raise during F.3 review if needed)

These are knobs that the F.3 pilot will surface evidence on. Defaults selected; surface in pilot review.

1. **Iteration sizing (M_instances × K × train_steps).** Default M=1000, K=100, train_steps=200 for the main run (≈ 5.1 train rows / data row, vs AGZ implicit ~1.2× and KataGo cap 8×). If F.3 pilot shows over-fitting on the buffer (val curve flat or rising), reduce train_steps_per_iter or grow M_instances. If the pilot is under-trained (val curve still steeply descending at iter 20), grow train_steps_per_iter.
2. **Buffer capacity.** F.3 default is **50K** (matches AGZ's ~20-iter window proportionally); F.4 main scales to **200K** only if F.3 shows the smaller window is fresh-but-noisy (i.e., val_avg_cost variance high across iters but mean trending down). If F.3 already shows gate-fail-streak ≥ 5 at 50K, the issue is *not* staleness — drop to G.5 first. KataGo-style power-law windowing (`shuffle.py:418-450`, `taper-window-exponent ≈ 0.65`) is the Stage 5 fallback.
3. **Temperature schedule choice.** F.3 default is `step30` (closer to AGZ's "first 12 % of game"). Ablation G.4 compares `const` / `step30` / `step50`.
4. **Optimizer.** Adam at lr=1e-4 is the F.4 default (matches Stage 1 warm-start). G.8 ablation tests SGD+momentum 0.9 with lr=1e-3 (AGZ-canonical) only if F.4 plateaus before acceptance criteria are met.

---

## Risks (top 7 with mitigations)

1. **`return_root_visits` C++ marshalling overhead.** At TSP-20 K=100 with 20 steps, that's 20 list-of-≤20 allocs per instance × 1000 instances/iter. Mitigation: return as a flat numpy array `(n_instances, n_steps, n_legal_actions)` of int32 (zero-padded); deserialize once on the Python side. Profile before/after Phase A; expected overhead < 1% of total iter time.
2. **Replay buffer staleness.** If gating rejects 5+ candidates in a row, the buffer's older slices were drawn under outdated policies. Mitigation: track `gate_fail_streak` in logs; F.3 will surface the worst-case. Stage 5 ablation: window-scaling with size proportional to total instances seen.
3. **π_t entropy collapse late in tour.** As legal_actions(t) → 1, the visit dist becomes degenerate (one-hot on the forced action). The CE term `−π_t · log p_θ(·|s_t)` is *not* ill-defined: the AM decoder's legality mask makes `log p_θ(·|s_t)` sharp on the same forced action, so CE is finite and small (essentially zero gradient on the policy at this step). Original plan revision included a "skip records with `legal_actions == 1`" mitigation, but **that conflicts with the dense `capacity_instances * N` ring-buffer layout** (skipping leaves uninitialized slots that sampling could hit, requiring either a `valid_mask` with rejection sampling or a packed `_step_index` with variable per-instance counts — both materially complicate the buffer). **Resolution: keep all N per-step records.** Late-step trivial CE is not a bug; AGZ stores tuples for every step up to termination too (Methods §Self-play). The wasted compute is ~5% of records carrying near-zero gradient — acceptable cost for a clean dense buffer. F.3 smoke confirms the loss stays finite and free of NaN at the trivial steps.
4. **Off-policy R² collapse on novel states.** Stage 3 E.1 found R²=0.9949 on MCTS-visited states drawn from the *Stage 1* model. If Stage 4 explores meaningfully different regions, value-head accuracy might drop, slowing distillation. Mitigation: log per-iteration value loss (B.3 dict output); if value loss diverges relative to Stage 1's, switch to rollout leaf eval (G.1) or best-so-far per-instance normalization (G.6) — the latter swaps the trainer's `bl_val` for `min_seen tour_cost(x)`, which is stationary across iterations and may stabilize the value target on hard instances. **Per-state cost-to-go is already the F.4 default; do not revive broadcast-z.**
5. **Gating false-rejects on small ε signal.** Early iterations may differ from best by < 0.001 — within the t-test's noise floor on val_size=10K. Mitigation: log val_avg_cost every iteration even when gating doesn't run, so the headline curve is monotone-trackable independent of gating cadence. Per scope decision 3, gating rejection does not roll back trainer weights — so false-rejects don't penalize learning, they just delay the "best" pointer update.
6. **value_head leaf eval exposes a quality gap.** Per Stage 2/3, value_head is worse than rollout for *test-time* MCTS quality. Stage 4 hopes the value head improves under MCTS-distillation pressure. If F.3 shows tour quality regressing, switch to G.1 (rollout) immediately.
7. **C++ tau-per-step plumbing edge cases.** `step50` means τ becomes 0 mid-tour — `_pick_root_action` must handle τ=0 (argmax) cleanly. Mitigation: existing code at `mcts.py:415` already branches on `temperature == 0`; just need to read the per-step value. C++ mirror has the same branch.

---

## Verification plan

End-to-end test, run in order:

1. **Phase A unit:** `python -m scripts.smoke_mcts --backend python` (extended A13 — `return_root_visits` python).
2. **Phase A C++ unit:** `python -m scripts.smoke_mcts --backend cpp` (extended A13 — cpp returns identical visit counts to python at fixed seed).
3. **Phase A cpp_batch unit:** `python -m scripts.smoke_mcts --backend cpp_batch` (extended A13 — cpp_batch returns identical visit counts to sequential cpp).
4. **Phase B-E unit:** `python -m scripts.smoke_alphazero` (A1-A6 all green).
5. **Phase F.3 pilot pass:** acceptance conditions in F.3 above.
6. **Phase F.4 main pass:** Stage 4 acceptance criteria (TSP-20 main run section).
7. **Reproducibility:** all CSVs deterministic at fixed seed=1234.

---

## Stage 5 follow-ups deferred from Stage 4

- **TSP-50 self-play.** Same recipe; ~20 h critical path on Modal A10. Pass gates: val_avg_cost ≤ 5.7999 (Stage 1 TSP-50) at fewer instances; ultimate greedy quality ≤ 5.7392 (Stage 3 K=100 rollout MCTS).
- **TSP-100 self-play.** Needs Stage 1 TSP-100 with full compute budget (current is reduced-compute) to be a fair apples-to-apples comparison with the AM paper's TSP-100 sampling-1280.
- **From-scratch training** (canonical "AlphaGo Zero" comparison) — does the Stage 1 warm-start advantage matter at convergence?
- **Window-scaled replay buffer** à la KataGo (`ref/KataGo-master/SelfplayTraining.md`).
- **Mixed leaf-eval schedule** — start with rollout, switch to value_head once off-policy R² > 0.99 (Stage 3 E.1 was 0.9949 already, so the threshold may be hit immediately).
- **Best-so-far per-instance value normalization** (G.6 above) — $z_t = (\text{tour\_cost} - \text{lengths}_t) / \min_\text{seen}\text{tour\_cost}(x)$, replacing $\text{bl\_val}$ with the per-instance running optimum. Stationary across iterations, no model dependency. (Note: the *broadcast-z vs per-state cost-to-go* question is resolved — per-state V_CURRENT cost-to-go is the F.4 default; broadcast-z would double-count path cost against the existing leaf evaluator. Do not revive.)
- **Multi-model arena** — pit each gating-accepted model against all prior accepted models (TrueSkill ranking) for an Elo-curve headline matching AlphaGo Zero's published figure.

---

## Notes

- Original plan (Claude Code plans dir): `C:\Users\Jun18\.claude\plans\i-think-we-have-snazzy-trinket.md` (this file).
- Stage 4 reuses the Stage 1 TSP-20 canonical checkpoint: `outputs/tsp_20/stage1_tsp20_canonical_20260423T103541/epoch-99.pt`.
- Stage 0 Gurobi reference for TSP-20: 3.8279 mean (1000 instances, seed=1234).
- Stage 1 reference val_avg_cost (TSP-20 canonical bs=512): 3.83943; bs=2048: 3.84443.
- Stage 3 reference test-time MCTS K=400 rollout: 3.8312 (gap 0.087% vs Gurobi).
- `RolloutBaseline.epoch_callback` (`src/am_baseline/baseline/baselines.py:106-123`) provides paired t-test α=0.05 gating. Stage 4 reuses it directly.
- `CppBatchMCTSSolver` (`src/am_baseline/search/mcts_cpp/solver.py:562-648`) is the canonical self-play backend; sequential `cpp` and `python` remain as references.

## AGZ paper citations (for the mapping table above)

All page numbers refer to the attached PDF (`AlphaGO ZERO.pdf`, *Nature* 550, 354–359 + Methods + Extended Data, 2017).

- **eq. 1, p.355** — `l = (z − v)² − π·log(p) + c·‖θ‖²`; the canonical Stage 4 loss.
- **Figs 1–2, p.355** — self-play loop, MCTS Select/Expand/Backup/Play.
- **Methods §Reinforcement learning, p.356** — policy-iteration framing; MCTS as policy improvement + evaluation operator.
- **Methods §Self-play training pipeline, p.357-358** — three-component pipeline (optimization / evaluator / self-play); best-player αθ* generates data while trainer continues.
- **Methods §Optimization, p.358** — SGD + momentum 0.9; weight decay c=10⁻⁴; mini-batch 2,048; 500K-game replay window.
- **Methods §Evaluator, p.358** — 400-game match, 55 % win threshold, τ→0, 1,600 sims.
- **Methods §Self-play, p.358** — 25K games/iter, 1,600 sims/move, τ=1 first 30 moves then τ→0, Dirichlet (ε=0.25, α=0.03), resignation v_resign auto-tuned for FP < 5 %.
- **Methods §Search algorithm, p.358** — PUCT formula U(s,a) = c·P(s,a)·√Σ_b N(s,b)/(1+N(s,a)); virtual loss; dihedral leaf augmentation; mini-batch of 8 NN evals.
- **Methods §Domain knowledge, p.357** — input is raw 19×19×17 plane stack; symmetry under rotation/reflection used for both data aug and search-time aug.
- **Methods §Neural network architecture, p.358-359** — 19/39 residual blocks, 256 filters, separate policy/value heads; "dual–res" outperforms "sep–conv" by ~1,200 Elo (Fig. 4).
- **Ext. Data Table 3, p.18 of PDF** — RL learning-rate schedule {1e-2, 1e-2, 1e-3, 1e-4, 1e-4, 1e-4} across thousands-of-steps buckets {0–200, 200–400, 400–600, 600–700, 700–800, >800}.

## KataGo cross-references

Tracked under `ref/KataGo-master/`:
- **`SelfplayTraining.md` §Asynchronous training** — gating is *optional*; "the whole loop works perfectly fine without it" — supports scope decision 3 and a no-gating G.5.c ablation.
- **`SelfplayTraining.md` example `synchronous_loop.sh`** — `MAX_TRAIN_PER_DATA=8`, `NUM_GAMES_PER_CYCLE=500`, `TAPER_WINDOW_SCALE=50000` — concrete reference for the train-rows / data-rows ratio cap and the Stage 5 power-law buffer.
- **`python/shuffle.py:418-450`** — power-law replay-buffer windowing implementation; window grows as `N^exponent` with `taper-window-exponent ≈ 0.65` and `expand-window-per-row ≈ 0.4`. Direct reference if F.3/F.4 surface staleness.
