# Stage 3 Plan: MCTS at Test Time vs. Sampling — Search-Efficiency Headline

**Created:** 2026-04-27
**Predecessor:** Stage 2 (`_plans/stage2_plan.md`, `_progress/stage2_progress.md`) — closed.
**Reference:** Proposal §Stage 3 (`proposal.md:100-115`).
**Status:** Approved 2026-04-27. Phase A in flight / pending kickoff.

---

## Context

Stage 1 produced a trained dual-head AM (policy + value) at TSP-20 and TSP-50. Stage 2 built a working MCTS layer on top and produced K-curves showing rollout-leaf-eval uniformly dominates value_head-leaf-eval at every matched K (+12-22pp gap reduction on TSP-20 and +17.6pp at TSP-50 K=100). Canonical MCTS config is locked: `c_puct=0.05, fpu=running_q, fpu_fallback=-1.0, root_select=visits, tree_reuse=True`.

Stage 3 answers the **search-efficiency** question the proposal frames at line 114: "Optimality gap vs. number of forward passes (MCTS budget curve vs. sampling-K curve)." That is — at fixed forward-pass budget, does MCTS beat AM's published sampling-1280 baseline? This isolates the value of **search** from any future training-loop change (Stage 4). Stage 2 already shows MCTS beats greedy; Stage 3 shows MCTS beats AM's strongest test-time-search baseline at fewer forward passes per instance.

The output is one headline figure per graph size (TSP-20, TSP-50, TSP-100): optimality gap on the y-axis, decode-step count on the x-axis (log-scale), three curves (sampling-K, MCTS-rollout-K, MCTS-value_head-K), with the AM-paper sampling-1280 reference point marked.

---

## Scope decisions (locked from clarification round)

1. **TSP-100 model:** Use the AM paper's released checkpoint at `ref/attention-learn-to-route-master/pretrained/tsp_100/epoch-99.pt` for the Stage 3 headline (no value head; rollout-only MCTS). Stage 1 TSP-100 training runs **in parallel** as a Stage 4 prerequisite — off the Stage 3 critical path.
2. **Optimality reference:** Compute Gurobi (TSP-50) and LKH3 (TSP-100) on our seed=1234, 1000-instance test set. The y-axis is "% gap to optimum." Side compute is small — extrapolating from Stage 0's measured TSP-20 base (Gurobi 8.1 s / LKH3 15.5 s for 1000 instances; `_progress/stage0_progress.md:59-60`): TSP-50 Gurobi ≈ 2 min – 1.5 h (variance-prone), TSP-50 LKH3 ≈ 2 min, TSP-100 LKH3 ≈ 10 min, TSP-100 Gurobi unreliable (skip). Total realistic side compute: **≤ 2 h** with Gurobi at TSP-50; **≤ 15 min** with LKH3 alone everywhere.
3. **Rollout K=400 at TSP-50:** **Skip.** TSP-50 rollout K-curve caps at K=200 (already shows diminishing returns at this size).
4. **Beam search:** Skip — defer to Stage 5 ablations if needed.
5. **Stage 2 cleanup items** (off-policy R², `value_norm='sqrt_n'`): folded into Phase E.

---

## Forward-pass accounting (the key metric)

Count `decode_step` calls only. Justification: one decode_step is the dominant per-call cost; encoder is amortized once per instance and shared across methods; value-head MLP is an order of magnitude cheaper than decode_step. Any constant-across-methods cost shifts the curves uniformly and obscures the comparison.

Per-method formulas (per instance, graph size N):

| Method | Decode-steps |
|---|---|
| Greedy | N |
| Sampling-K (`sample_many`) | K · N (one full N-step decode per replica) |
| MCTS-K, value_head leaf | ≤ N + K · N (instrumented to report empirical mean) |
| MCTS-K, rollout leaf | ≤ N + K · (N + d̄), d̄ = mean rollout depth (instrumented) |

Tree reuse, terminal shortcuts, and rollout depth all vary per instance — so we **measure, not derive**. Implementation: counter attributes on `MCTSSolver`, returned via opt-in flag (Phase A.1 below).

Plot footnote: "Forward-pass = `decode_step` calls per instance; encoder pass and value-head MLP excluded (constant across methods)."

**What we are NOT comparing:** wall-clock. Wall-clock is hardware-dependent (RTX 4060 vs A100 vs cloud GPU), and the current Python MCTS spends ~98% of its time in tree-walk overhead (Stage 2 decode-step micro-benchmark, `_progress/stage2_progress.md:296-299`) — overhead that disappears once Phase G's C++ port lands. A wall-clock comparison would conflate the algorithmic question (does search beat sampling?) with an implementation-language artifact. The proposal's key metric is forward-pass count for exactly this reason. Wall-clock is logged per run and tabulated in the progress doc's "Wall-clock / Resource Accounting" section for compute-budgeting purposes only — it is **not** the headline claim. After Phase G lands, a secondary "wall-clock budget curve" plot may be added for completeness; the forward-pass plot remains primary.

---

## Phases

### Phase A — Foundation (code + 1.5 h compute)

**A.1 Instrument forward-pass counts in MCTSSolver.**
- File: `src/am_baseline/search/mcts.py`.
- Add three counters on `MCTSSolver`: `fwd_count_decode`, `fwd_count_value`, `fwd_count_rollout`.
- Reset at start of `solve_instance`; increment at every `decode_step` call site (3 sites: `_populate_priors`, `_expand`, `_rollout_remaining_real`) and value-head call site (2 sites). Tag rollout-internal decode_step calls separately.
- Add `MCTSConfig.return_fwd_counts: bool = False`. When True, `solve_instance` returns `(cost, tour, counts_dict)`. Default False — preserves existing `(cost, tour)` signature, no test-suite breakage.
- Aggregate to batch mean inside `solve_batch`.

**A.2 Add `decode_steps` column to MCTS CSV output.**
- File: `src/scripts/run_mcts.py:185-198`.
- Set `cfg.return_fwd_counts=True`; write the per-instance `decode_steps` count to the CSV.
- New header: `idx, greedy_cost, mcts_cost, delta, gap_pct, decode_steps`.

**A.3 New runner: `src/scripts/run_sampling.py`.**
- Mirror `run_mcts.py`'s CLI: `--model, --graph_size, --val_size, --seed, --width, --batch_size, --output_csv, --no_cuda, --no_progress_bar`.
- Reuse `load_model` from `src/am_baseline/utils/misc.py` and `model.sample_many(input, batch_rep=width, iter_rep=1)` from `src/am_baseline/model/attention_model.py:108-114`.
- Chunk on `batch_size` (mirror the chunking in `src/scripts/evaluate.py:44`).
- Output CSV columns: `idx, greedy_cost, sample_cost, delta, gap_pct, decode_steps` (decode_steps = `width × graph_size`, computed analytically since sample_many doesn't have variable-length paths).

**A.4 Validation runs.**
- Sampling K=1 on TSP-20 (≡ greedy) — confirm output cost matches `run_mcts.py`'s greedy_cost column at the same seed within fp tolerance.
- Re-run **one** Stage-2 MCTS rollout config (TSP-20 K=200) with the new instrumentation — store as `outputs/stage3/tsp20_K200_rollout_instrumented.csv`. Sanity-check the empirical decode_step mean against `K · (N + d̄)` within 5%.

**Unblocks:** Phases B, C, D.

### Phase B — TSP-20 headline (≤ 0.5 day compute)

**B.1 Sampling sweep at TSP-20.** K ∈ {1, 100, 500}, seed=1234, val_size=1000 — `outputs/stage3/tsp20_sampling_K{K}.csv`. Wall-clock estimate: ≪ 30 min on RTX 4060 (sampling K=100 already runs in ~0.1 s for 20 instances; full 1000 at K=500 expected < 5 min). **K=1280 dropped at TSP-20** — greedy gap is already only ~0.30 %, so sampling-1280's published reference is overkill at this size and the curve already shows diminishing returns by K=500. K=1280 retained at TSP-50 and TSP-100 where it remains the primary AM-paper anchor.

**B.2 (Optional stretch)** Rollout K=800 at TSP-20 — `outputs/stage3/tsp20_K800_rollout_canonical.csv`. ~3 h. Defer if compute pressure.

**B.3 Aggregator.** New file `src/scripts/aggregate_stage3.py`. Reads:
- Stage 2 reusable: `outputs/stage2/tsp20_K{20,50,100,200,400,800}_canonical_v2.csv` (value_head), `outputs/stage2/tsp20_K{20,50,100,200,400}_rollout_canonical.csv` (rollout).
- Stage 3 new: `outputs/stage3/tsp20_sampling_K{1,100,500}.csv`, optionally rollout K=800.
- Existing Gurobi reference for TSP-20: 3.8279 (1000 instances seed=1234) from Stage 0.
- Output: `outputs/stage3/comparison_tsp20.csv` with columns `method, leaf_eval, K, decode_steps_mean, mean_cost, std_cost, gap_to_optimum_pct, gap_reduction_vs_greedy_pct, win_rate_vs_greedy, n_instances`.

**B.4 Plot.** New file `src/scripts/plot_stage3.py`. Output `outputs/stage3/figures/budget_curve_tsp20.png`. Three curves; AM-paper sampling-1280 marked.

**Pass gate (D.1 below).** Diagnose if not met before sinking compute into C/D.

### Phase C — TSP-50 headline (~10 h compute)

**C.1 Optima.** Run Gurobi on TSP-50 1000-instance seed=1234 dataset. Reuse `src/scripts/eval_baselines.py` (Stage 0's TSP-20 Gurobi run took 8.1 s; `_progress/stage0_progress.md:59`). Output: `outputs/baselines/tsp50_gurobi_seed1234.csv`. Expected wall-clock: **2 min – 1.5 h** (Gurobi MIP at N=50 is variance-prone). Mitigation: cap solver per-instance at 60 s and fall back to LKH3 (`elkai`) for any timeouts; LKH3 hit optimal at TSP-20 anyway. Pure-LKH3 fallback at TSP-50 ≈ 2 min for 1000 instances.

**C.2 Sampling sweep at TSP-50.** K ∈ {1, 100, 500, 1280}. ~1 h. (K=2560 dropped — AM paper's published anchor is sampling-1280, so K=1280 is the meaningful upper bracket; the K-curve diminishes past this point.)

**C.3 Rollout MCTS K-sweep at TSP-50.** K ∈ {20, 50, 200} (K=100 reused from Stage 2 `outputs/stage2/tsp50_K100_rollout_clean.csv`; K=400 skipped).
- K=20: ~0.5 h
- K=50: ~1.2 h
- K=200: ~4.7 h

**C.4 Value_head K=800 at TSP-50.** Single run to extend value_head curve to match TSP-20. ~3 h.

**C.5 Aggregator + plot.** Output `outputs/stage3/comparison_tsp50.csv` and `outputs/stage3/figures/budget_curve_tsp50.png`.

**Pass gate (D.2 below).**

### Phase D — TSP-100 headline (~13 h compute)

**D.1 Load released AM TSP-100 checkpoint.** Path: `ref/attention-learn-to-route-master/pretrained/tsp_100/epoch-99.pt`. Verify `load_model` path handling. **Note:** released checkpoint has no value head — only `leaf_eval='rollout'` is runnable. Sanity-check by setting `MCTSConfig.leaf_eval='value_head'` and asserting it errors as expected at `mcts.py:111`.

**D.2 LKH3 optima at TSP-100.** Run `eval_baselines.py` LKH3 path on 1000-instance seed=1234 set. Extrapolating LKH3's near-quadratic scaling from Stage 0's TSP-20 base (15.5 s / 1000 inst): expected wall-clock **≈ 10 min**. Skip Gurobi at TSP-100 — N=100 MIP is unreliable on consumer hardware. (Falls back to AM paper's published Concorde optima if `elkai` not available — flag in progress doc.)

**D.3 Sampling sweep at TSP-100.** K ∈ {1, 100, 500, 1280}. ~2-3 h. (K=2560 dropped per the same reasoning as TSP-50; sampling-1280 is the AM-paper reference.)

**D.4 Rollout MCTS K-sweep at TSP-100.** K ∈ {20, 50}. K=100 is no longer required for the current Stage 3 decision because K=20 and K=50 already close the strict TSP-100 pass gate; keep K=100/K=200 deferred unless a paper-quality curve later needs more high-budget points.
- K=20: ~2 h
- K=50: ~5 h
- Total: ~7 h overnight.

**D.5 Aggregator + plot.** Output `outputs/stage3/comparison_tsp100.csv` and `outputs/stage3/figures/budget_curve_tsp100.png`.

**Pass gate (D.3 below).**

### Phase E — Stage 2 cleanup (~2 h)

**E.1 Off-policy R² probe.** Re-run TSP-20 K=400 rollout MCTS with a value-head logging hook that records `(state, v_predicted, z_realized)` at every visited node. Compute R² on this off-policy distribution and compare to Stage 1's in-distribution R²=0.9965. Output: `outputs/stage3/value_head_offpolicy_r2_tsp20.csv`.

**E.2 `value_norm='sqrt_n'` rollout MCTS.** Single TSP-20 K=200 run with `value_norm='sqrt_n'`. Compare to bl baseline (Stage 2 number). If sqrt_n ≥ bl, document as the cleaner default for Stage 4 (no per-instance greedy pass needed).

### Phase F — Stage 1 TSP-100 training (parallel, off critical path)

**F.1** Train Stage 1 TSP-100 with value head, mirroring the TSP-50 setup at `outputs/tsp_50/stage1_tsp50_with_value_*`. Reuse `src/scripts/train.py`. Output checkpoint: `outputs/tsp_100/stage1_tsp100_with_value_<timestamp>/epoch-99.pt`. ~24-36 h. Runs overnight independently of A/B/C/D.

**F.2** When complete, optionally re-run Phase D rollout sweep on the new value-head'd checkpoint to add value_head MCTS curves to the TSP-100 figure (Stage 5 stretch — not required for Stage 3 closure).

### Phase G — C++ MCTS port (parallel engineering track; Stage 4 enabler)

**Motivation.** Stage 2's decode-step micro-benchmark (`_progress/stage2_progress.md:296-299`) decomposed MCTS wall-clock and found that **~98% of TSP-20 K=200 wall-clock is Python tree walk** (PUCT loop + `state.update` + `state.get_mask`), with only ~1.6 % spent in the leaf-eval forward pass. At TSP-50 the leaf-eval fraction grows to ~5 %, but Python tree walk is still dominant. This is the actual bottleneck for **Stage 4 self-play** (which will run thousands of MCTS games per epoch) and for **Stage 3's TSP-100 reach** (K=200 currently deferred at ~19 h).

**Approach.** pybind11 C++ MCTS that calls back into PyTorch for `decode_step` and `value_head` forwards. Justification: targets the actual hot loop (Python tree walk) while leaving the model forward in PyTorch (no need to port the AM model). Cython is the alternative; it's easier to integrate but typically gives 2-5× speedup vs C++/pybind11's 20-100×.

**G.1 Scaffold.**
- New subdirectory `src/am_baseline/search/mcts_cpp/` with `CMakeLists.txt`, `mcts.cpp`, `mcts.hpp`, `bindings.cpp`.
- Mirror the Python `MCTSNode` / `MCTSConfig` / `MCTSSolver` data structures in C++.
- Build via `pyproject.toml` extension or `setup.py` — wire into the existing `pip install -e .` flow.

**G.2 State mirror.** Port `StateTSP.update` and `StateTSP.get_mask` to C++ (small functions; ~50 LOC each). Keep the Python `StateTSP` as ground truth for tests.

**G.3 PUCT + tree.** Port `select_action`, `_simulate`, `_backup` to C++. The model-forward boundary stays in Python: C++ accumulates a leaf state, hands it to Python for `decode_step`, receives priors + value back, continues.

**G.4 Rollout.** Port `_rollout_remaining_real` to C++ with the model-forward boundary unchanged (each rollout decode_step still goes through Python torch).

**G.5 Bit-for-bit validation.**
- Smoke A1..A11 pass on C++ MCTS (re-purpose `src/scripts/smoke_mcts.py` with a `--backend=cpp` flag).
- TSP-20 K=200 rollout with same seed=1234 produces row-for-row CSV-identical results to Python MCTS within fp tolerance. Use the `outputs/stage2/tsp20_K200_rollout_canonical.csv` as the reference.
- TSP-50 K=100 rollout same check against `outputs/stage2/tsp50_K100_rollout_clean.csv`.
- Forward-pass instrumentation (Phase A.1) gives identical decode_step counts.

**G.6 Wall-clock benchmark.** Re-run TSP-20 K=200 rollout MCTS with both backends, report speedup. Target: ≥ 10× end-to-end on TSP-20, ≥ 5× on TSP-50 (rollout adds GPU-bound time that doesn't speed up).

**Risk-managed scope:** if the cross-language overhead per `decode_step` callback dominates (Python ↔ C++ marshalling on every leaf), the speedup may collapse. Mitigation: batched callbacks — accumulate N leaf states in C++ and call Python once with a batch. Defer this to G.7 if needed.

**G.7 (Stretch)** Batched leaf evaluation across simultaneous trees. Useful for Stage 4 self-play where thousands of trees run in parallel; not required for Stage 3.

**G.8 (Post-Phase-G extension)** Batched virtual-visit search within one C++ tree.
- Add `simulation_batch_size` with default `1` so the original sequential C++ backend remains the reference.
- For `simulation_batch_size>1`, collect multiple pending simulations with CPU-side virtual visits, then batch leaf/value/rollout evaluation through the existing Python/PyTorch model boundary.
- Keep true multithreaded tree mutation and full virtual loss deferred; this extension is single-threaded tree search plus batched neural inference.
- Finding after small TSP-50/TSP-100 sweeps: virtual visits alone are too weak for useful batching. Realized batch stays near `1.02–1.03` because deterministic PUCT repeatedly collides on the same pending leaf; collision overhead can erase or reverse any GPU batching gain.

**G.9 (Post-G.8 extension)** Virtual-loss batched search within one C++ tree.
- Add `virtual_loss_weight` and `virtual_loss_margin` to temporarily lower Q and inflate effective visits on pending edges, following KataGo's virtual-loss idea.
- Smoke result on TSP-50 K=20 val_size=100: virtual loss creates real pending batches (`6.67–18.35`) but badly fails the objective: wall-clock is ~12–14× slower and optimality gap regresses. It over-diversifies low-K search before backups arrive and destroys the decoder-cache advantage.
- Keep `simulation_batch_size=1` as canonical. If batching is revisited, prefer cross-instance/tree batching or a much more conservative selection-diversification rule rather than aggressive within-tree virtual loss.

**G.10 (Post-G.9 extension)** Cross-instance batched C++ MCTS.
- Add a separate `cpp_batch` backend that batches neural-network evaluator calls across independent C++ MCTS trees while leaving sequential `cpp` untouched as the correctness reference.
- Add a stateful C++ `BatchSearch` interface plus Python `CppBatchMCTSSolver`. The scheduler keeps up to `mcts_batch_size` active instances, collects one pending evaluator request per tree, batches compatible decoder calls by step, and returns results to C++ immediately.
- Preserve per-tree semantics: one pending simulation per tree, immediate backup, no within-tree virtual loss, no delayed multi-simulation backup. Keep decoder cache per instance because graph embeddings differ.
- Add CLI flags `--backend cpp_batch` and `--mcts_batch_size` (default `32`).
- Pilot scope: TSP-20 and TSP-50 only, `val_size=1000`, `K=20`, rollout leaf evaluation, `tree_reuse=True`; sweep `mcts_batch_size={16,32,64}` after smoke validation.
- Acceptance: K=0 smoke matches greedy exactly; small TSP-20/TSP-50 slices match sequential `cpp` within `1e-5`; full TSP-20/TSP-50 pilot runs produce valid tours and match sequential `cpp` mean cost within CSV precision; promote only if TSP-50 K=20 wall-clock improves meaningfully without mean-cost regression above `1e-5`.

**Effort estimate.** ~3-5 days of focused engineering for G.1-G.6. Independent of Phases A-F GPU compute — runs in parallel with Stage 3 experiments.

**Critical-path implications.** None for Stage 3 (Python MCTS is sufficient for the headline plot). For Stage 4 the C++ port is effectively a hard prerequisite — self-play wall-clock at Python speeds would put Stage 4 well outside laptop-feasible compute. Including Phase G in Stage 3 lets Stage 4 start unencumbered.

---

## Acceptance criteria

### D.1 — TSP-20 (Stage 1 canonical checkpoint)

- ✅ Some MCTS rollout-K achieves ≤ 0.1% gap vs Gurobi optimum (proposal target). Stage 2 K=400 already shows 71.3% gap reduction — if greedy gap is ~0.30%, MCTS-K=400 gap is ~0.087% → on target.
- ✅ At ~4000 decode_steps/inst, MCTS rollout-K beats sampling-K by ≥ 0.05% absolute gap. Crossover should be visible in the budget plot.
- ✅ Both rollout and value_head curves monotone-decreasing in K (allow one inversion at adjacent K).
- ✅ Sampling K=1 ≈ greedy within ~0.05% (multinomial-1 ≠ argmax exactly; Phase A.4 measured +0.019% drift on val_size=100).

### D.2 — TSP-50

- ✅ MCTS rollout K ∈ {100, 200} reaches ≤ 2.0% gap vs Gurobi optimum (proposal-style target).
- ✅ MCTS rollout at any K achieves ≤ sampling-1280's mean cost using ≤ 50% of sampling-1280's decode_steps. Stage 2 rollout K=100 = 5.7392 vs AM paper's sampling-1280 ≈ 5.72 — already passes by mean cost; need to verify decode_step ratio holds.
- ✅ Curves monotone decreasing.

### D.3 — TSP-100 (released AM checkpoint, rollout-only)

- ✅ MCTS rollout (any K) achieves ≥ 30% gap reduction vs greedy (proposal target: 30-50%). TSP-50 K=100 hit 59% so this should be comfortable.
- ✅ Rollout K=50 is sufficient for the current Stage 3 strict pass gate against the sampling-1280 anchor. K=100 is deferred rather than required.
- ✅ Curve monotone decreasing across the required K ∈ {20, 50}.

---

## Code changes — file inventory

| File | Action | Lines | Purpose |
|---|---|---|---|
| `src/am_baseline/search/mcts.py` | Edit | ~30 | Forward-pass counters; opt-in `return_fwd_counts` flag |
| `src/scripts/run_mcts.py` | Edit | ~5 | Add `decode_steps` to CSV output; set `return_fwd_counts=True` |
| `src/scripts/run_sampling.py` | Create | ~120 | Sampling-K runner with matched CSV schema |
| `src/scripts/aggregate_stage3.py` | Create | ~80 | Aggregate Stage 2 + Stage 3 CSVs into `comparison_tspN.csv` |
| `src/scripts/plot_stage3.py` | Create | ~80 | Headline budget-curve figure (matplotlib) |
| `src/scripts/_plot_style.py` | Create | ~30 | Shared matplotlib style for Stage 3+ figures |
| `src/am_baseline/search/mcts_cpp/` | Create | ~800 | Phase G — pybind11 C++ MCTS (mcts.cpp, mcts.hpp, bindings.cpp, CMakeLists.txt) |
| `pyproject.toml` or `setup.py` | Edit | ~10 | Phase G — wire C++ extension into `pip install -e .` |
| `_progress/stage3_progress.md` | Create | (seed) | Progress tracker |

No core-module refactor required. All new functionality is in `src/scripts/` or as additive instrumentation in `MCTSSolver`.

---

## Compute budget summary

| Phase | Compute | Type |
|---|---|---|
| A — instrumentation + sampling validation | ~1.5 h | Code + light run |
| B — TSP-20 headline | ~1.5 h (+3 h optional stretch) | Sampling sweep |
| C — TSP-50 headline | ~10 h MCTS + ≤ 1.5 h Gurobi (or 2 min LKH3) | Mostly overnight |
| D — TSP-100 headline | ~17 h MCTS + ~10 min LKH3 | Multi-night |
| E — Stage 2 cleanup | ~2 h | Quick |
| F — Stage 1 TSP-100 training | ~24-36 h | **Parallel, off critical path** |
| G — C++ MCTS port | ~3-5 days dev (no GPU) | **Parallel, off critical path; Stage 4 hard prereq** |
| **Total Stage 3 critical-path compute** | **~32 h + ≤ 2 h side compute (Gurobi/LKH3)** | Spread over 4-5 days of laptop GPU |

If Phase D's TSP-100 rollout K=200 is added later (currently deferred), add another ~19 h.

---

## Risks

1. **Sampling K=1 ≠ greedy cost.** `sample_many(batch_rep=1)` argmax-samples once, which might differ from `decode_type='greedy'` due to sampling vs argmax semantics on tied logits. Phase A.4 catches this; if it fires, document the discrepancy and use sampling K=1 as the sampling-curve anchor (greedy is already in CSVs separately).
2. **Tree-reuse decode_step accounting.** With `tree_reuse=True`, the second tour-step's root inherits children from the first step — those children have their decode_step count amortized. The instrumented total per instance must equal the **sum across tour-steps**, not per-step. Phase A.4's K · (N + d̄) sanity check guards against double-counting.
3. **Released TSP-100 checkpoint compatibility.** The reference AM repo uses slightly different module paths than `src/am_baseline/`. `load_model` may need a path-translation shim. If it doesn't load cleanly, fallback is to vendor the reference forward path or train Stage 1 TSP-100 first.
4. **Gurobi at TSP-50 wall-clock variance.** Stage 0 measured TSP-20 Gurobi at 8.1 s for 1000 instances (`_progress/stage0_progress.md:59`), so expected TSP-50 is short — but Gurobi MIP scaling is notoriously variance-prone (a single hard instance can dominate total wall-clock). Mitigation: cap per-instance solver time at 60 s and fall back to LKH3 (`elkai`) for any timeouts. Stage 0 verified LKH3 hit Gurobi-optimal on every TSP-20 instance, so the LKH3 fallback is sound.
5. **TSP-100 rollout K=100 wall-clock estimate is extrapolated.** Stage 2 numbers came from TSP-20/50; TSP-100 has 2× the cities so per-instance time may be larger than 9.7 h. Time-box K=100 at 12 h, then reassess K=200 inclusion.

6. **C++ MCTS correctness drift (Phase G).** Porting MCTS from Python to C++ introduces fp-arithmetic ordering risks (PUCT's argmax can flip on near-ties), state-mirror bugs (mismatched `update`/`get_mask` semantics), and pybind11-marshalling overhead that can erase the speedup. Mitigation: bit-for-bit validation against Python MCTS at fixed seed (G.5); smoke A1..A11 must pass before any benchmark claim; if marshalling overhead dominates, batched callback (G.7) is the fallback.

7. **Within-tree batching may not reduce time or preserve quality.** G.8 virtual visits did not create real batch width; G.9 virtual loss created batch width but over-diversified low-K search, reduced cache hits, and worsened both wall-clock and optimality gap. Mitigation: keep `simulation_batch_size=1` as the reference; only revisit batching through cross-instance/tree batching or a conservative rule with an explicit time+gap promotion gate.

---

## Verification

End-to-end test plan, executed in order:

1. **Smoke (Phase A done):** `python src/scripts/run_sampling.py --model outputs/tsp_20/stage1_tsp20_canonical_*/epoch-99.pt --graph_size 20 --val_size 100 --width 1 --output_csv /tmp/sampling_K1_smoke.csv` → mean cost matches `python src/scripts/run_mcts.py --model ... --K 0` greedy mean within 1e-4.
2. **Unit (Phase A done):** `python src/scripts/run_mcts.py --model ... --K 50 --output_csv /tmp/mcts_K50_smoke.csv` → CSV has `decode_steps` column; mean decode_steps lies in `[N, K·N + N]`.
3. **Phase B end-to-end:** `python src/scripts/aggregate_stage3.py --graph_size 20` → `outputs/stage3/comparison_tsp20.csv` has rows for greedy, sampling-{1,100,500}, mcts-rollout-{20,50,100,200,400}, mcts-value_head-{50,100,200,400,800}.
4. **Phase B headline:** `python src/scripts/plot_stage3.py --graph_size 20` → `outputs/stage3/figures/budget_curve_tsp20.png` shows three monotone-decreasing curves; rollout sits at-or-below sampling at matched decode_steps.
5. **Phase C/D pass gates:** acceptance criteria D.1, D.2, D.3 above.
6. **Reproducibility:** all CSVs deterministic at fixed seed=1234 (verify by re-running one configuration and diffing CSVs row-by-row, as Stage 2's `bln0tv1pg` re-run did).

---

## Open follow-ups (for Stage 4 / Stage 5, not in Stage 3 scope)

- **Stage 1 TSP-100** (Phase F) becomes a Stage 4 prerequisite. When it lands, optionally re-run Phase D rollout sweep with value_head leaf-eval.
- **C++ MCTS** (Phase G) becomes a Stage 4 hard prerequisite. Stage 4 self-play at Python speeds is laptop-infeasible; the port unblocks Stage 4 launch and may also enable TSP-100 K=200 within Stage 3 if it lands before Phase D wraps.
- **Beam search baseline** as a Stage 5 ablation if a paper-quality comparison is wanted.
- **Larger MCTS K at TSP-100** (K=200, K=400) if Phase D's K=100 result motivates it (and especially after Phase G's speedup).
- **Off-policy R² probe** result feeds Stage 4's training-data design: if R² collapses on MCTS-visited states, distillation against MCTS targets is essential.
