# Research diagnosis: MCTS-guided policy improvement (2026-09-25)

Plan: [research_diagnosis_20260925_plan.md](../_plans/research_diagnosis_20260925_plan.md).
Status: audit complete; suggested implementation changes and training experiments
below have **not** been performed. Existing local changes were preserved.

## Assessment

Do not abandon the learned-value/search idea on the present evidence. There are
two concrete defects to address before judging it: a missing value-normalization
setting in Stage 4 evaluation, and a value representation that provably loses
necessary state information. The evaluation correction alone changes a learned
value evaluator from worse than greedy to substantially better than greedy on a
matched 1,000-instance test. Rollouts still perform better, and from-scratch
search-supervised greedy policies still lag REINFORCE.

Stop broad learning-rate, search-budget, and mixing sweeps. The next useful work
is a small controlled experiment on value representation and decision quality,
followed by a separate test of how well the policy learns the teacher's
improvements. Keep TSP-20 as a diagnostic, use TSP-50 to test whether a fix scales,
and choose a new problem only after distinguishing evaluator failure from
teacher/student failure.

## Scope and evidence

Reviewed README, proposal, the stage plans/progress and algorithm specification,
model/decoder/value code, Python MCTS and relevant C++ conversion paths,
self-play/replay/loss/gating code, evaluation scripts, local checkpoints,
iterations logs, matched comparison NPZs, and existing sibling-ranking CSVs.

The primary reproducible audit is
[`eval_logs/research_diagnosis_20260925.py`](eval_logs/research_diagnosis_20260925.py);
its output is
[`eval_logs/research_diagnosis_20260925.json`](eval_logs/research_diagnosis_20260925.json).
An additional matched greedy checkpoint evaluation is saved in
[`eval_logs/research_diagnosis_20260925_greedy.json`](eval_logs/research_diagnosis_20260925_greedy.json).

Conda is unavailable on this Mac. Used the existing `.venv` with PyTorch 2.10.0,
CPU, one PyTorch thread. No dependencies were installed. The C++ extension is
not built here; new MCTS measurements use the Python reference implementation.
No training, GPU/cloud jobs, or solver reruns were launched. Existing Gurobi
references were read from saved NPZs, not independently certified again.

## 1. Confirmed evaluation normalization defect

`src/scripts/val_stage4_mcts.py::_build_mcts_config` (around line 352) supplies
`value_norm='bl'` but omits `value_target_norm`. Therefore `MCTSConfig` silently
uses `value_target_norm='bl'` even when `train_args` says `none`. This also occurs
with `--match_train`.

These settings represent different quantities:

- `value_norm`: denominator for backed-up path and leaf costs.
- `value_target_norm`: units of the trained head's output.

F.6.1.6 is trained on raw remaining cost (`none`), as specified by
`modal_run_train_alphazero.py::run_f616_400iter_step_decay -> _f61_args` and the
existing progress records. Its locally copied checkpoint lacks sibling
`args.json` and does not embed this metadata. The configuration needs to be
recorded explicitly when evaluating this checkpoint.

If the normalizer is B, the correct total-cost estimate is

`path_cost / B + raw_value / B`.

The evaluation script instead computes

`path_cost / B + raw_value`.

This changes the relative weight of known path cost versus remaining cost, the
exploration scale, and the meaning of backup values at different depths. It is
not a harmless common offset or uniform rescaling of the total score.

### Matched CPU reproduction

Checkpoint: F.6.1.6 `iter-361_accepted.pt`, key `best_model`.
TSP-20, 1,000 instances, seed 20260430, K=40, c_puct=0.05, no noise, no
temperature, tree reuse. Only head-output normalization changes between the two
value rows. Dataset generation was checked bit-for-bit against `TSP.make_dataset`.

| Evaluator | Mean tour cost | Difference from greedy | Search loop seconds |
|---|---:|---:|---:|
| Greedy | 3.863356 | — | — |
| Value head, incorrect `bl` interpretation | 3.869028 | +0.005672 | 16.40 |
| Value head, correct `none` interpretation | 3.838671 | -0.024685 | 22.14 |
| Greedy-rollout leaves | 3.834016 | -0.029340 | 75.47 |

Correct minus incorrect: -0.030357, paired-instance approximate 95% CI
[-0.034066, -0.026648]. Correct value minus rollout: +0.004655, CI
[+0.003526, +0.005784]. These quantify test-instance uncertainty for one fixed
checkpoint, not variation across training seeds.

The corrected head captures about 84% of the rollout improvement over greedy in
this check. Its Python search loop is about 3.4 times faster than rollout here;
this is not a production C++/GPU throughput or training-speed claim. Timings
exclude the common batched greedy normalizer computation.

This provides a concrete explanation for much of the discrepancy between Stage
5 C.3 and H.3: `eval_tsp20_mix_lambda_sweep.py` explicitly propagates
`value_target_norm`, unlike `val_stage4_mcts.py`. Historical executions were not
reconstructed byte-for-byte, so the audit does not claim to reproduce every old
number. However, old C.3 results cannot establish that correctly scaled value
search is useless. The training coach already propagates `value_target_norm`
correctly; this evaluation defect alone does not explain the training plateau.

Required follow-up: pass the checkpoint's output-normalization setting through
all evaluation paths, add an explicit override for legacy checkpoints without
metadata, and save architecture/normalization metadata in future checkpoints.
Fail explicitly when units cannot be determined. Do not assume all checkpoints
use either `bl` or `none`.

## 2. Confirmed value-representation limitation

The value head sees only the decoder glimpse (`model/value_head.py:23`). For
each attention head, `decoder.py:183-189` computes a masked softmax-weighted sum
of fixed node value embeddings. The first/current endpoints affect the query,
but there is no direct endpoint or query connection into the value MLP.

With one unvisited city u, softmax over the only legal key is identically one.
Thus the glimpse depends on u and the fixed encoded graph, but is independent
of the first city f and current city c. The true remaining cost is exactly

`V(s) = distance(c,u) + distance(u,f)`.

Changing c or f changes this target while leaving the value input unchanged.
No MLP depth, width, or amount of training on the same glimpse can represent all
these values correctly. This directly qualifies Stage 1's design rationale
that the glimpse necessarily retains all current-state information.

### Reproduction

One fixed TSP-20 graph (seed 20260925), same unvisited city, all 342 distinct
ordered pairs of visited start/current endpoints. Every state was built from a
valid prefix; exact remaining costs were also checked against actual terminal
completion including the closing edge.

- Maximum glimpse difference across endpoint pairs: **0.0** in both checkpoints.
- True remaining cost range: **0.458365 to 1.506348**.
- Stage 1 predicts the same raw value **0.855639** for every pair.
- F.6.1.6 predicts the same raw value **0.545525** for every pair.
- Best possible constant-prediction RMSE on this constructed set: **0.225244**.

This is a structural counterexample, not an estimate of average error on the
training distribution and not a proof that the entire plateau has this cause.
The forced final action itself needs no ranking, but its estimated cost enters
backups for earlier decisions. A search can eventually reach terminal states
and recover exact values, so this defect also does not imply MCTS can never
improve on the policy.

Smallest representation experiment: retain the glimpse and add explicit
first/current embeddings or endpoint coordinates, an unvisited-set summary,
and remaining-node count. Use exact completion at trivial tails (one city first;
small Held-Karp tails as a separate ablation). Start with a frozen policy and
detached policy features to isolate whether the evaluator improves decisions.

## 3. The greedy performance gap is real, but narrower claims are needed

Fresh CPU greedy evaluation on the same 1,000-instance sets (seed 20260430),
with Stage 4 key `best_model`:

| Problem | Checkpoint | Mean tour cost | Mean per-instance gap to saved optimum |
|---|---|---:|---:|
| TSP-20 | Stage 1 REINFORCE + auxiliary value | 3.842391 | 0.292% |
| TSP-20 | Stage 4 K10 rollout, no value loss | 3.866808 | 0.919% |
| TSP-20 | Stage 4 K40 rollout chain, iter-199 | 3.858945 | 0.722% |
| TSP-20 | Stage 4 F.6.1.6 | 3.863356 | 0.830% |
| TSP-20 | Stage 4 warm-start, small learning rate | 3.841851 | 0.278% |
| TSP-50 | Stage 1 plain AM, value disabled | 5.811708 | 2.141% |
| TSP-50 | Stage 1 AM + auxiliary value | 5.800891 | 1.952% |
| TSP-50 | Stage 4 K25 rollout, no value loss | 5.944604 | 4.479% |
| TSP-50 | Stage 4 K50 rollout chain, iter-199 | 5.923993 | 4.115% |

These are the available named checkpoints, not every best historical working
model. Differences from old validation means reflect dataset/checkpoint-key
choices. The TSP-20 Stage 1 baseline includes auxiliary value training and must
not be mislabeled plain AM. Its fresh costs agree with the saved comparison
array to at most 9.54e-7. The warm-start gain is small: -0.000540 with paired SE
0.000499 on this set, so this check does not establish a repeatable advantage.

There is also a useful distinction between greedy and sampling performance:
the saved TSP-20 K10 comparison has Stage 4 sampling-1280 = **3.832634** versus
Stage 1 sampling-1280 = **3.833698**, on the same instances. Paired difference
-0.001063, SE 0.000457, two-sided p=0.020 for this fixed run. Stage 4 greedy is
worse on the same set. Good tours therefore exist in the learned distribution;
the result is consistent with an issue in the distribution of probability over
tours or in distillation. It is not proof of general superiority, a same-time
advantage, or an identified causal mechanism. TSP-50 does not show this reversal.

## 4. What the existing value diagnostics establish

Reaggregated the existing sibling CSVs (100 TSP-20 instances per row):

| Checkpoint, own greedy states | Optimal child selected by prior | By rollout | By value head |
|---|---:|---:|---:|
| Stage 1 | 95.78% | 98.83% | 62.83% |
| F.6.1.6 | 93.00% | 97.00% | 80.00% |

The TSP-20 reference uses exact Held-Karp completion. These diagnostics support
the claim that the head ranks counterfactual children poorly. They do **not**
imply that combining it with priors and deeper search can only degrade results;
the corrected MCTS experiment is a counterexample to that assertion.

The saved aleatoric probe gives mean MSE 0.0071128 = 0.0016852 completion
variance + 0.0054276 squared conditional-mean error. The latter is about 76% of
this decomposition. It includes approximation/representation error and any
target/distribution mismatch; it does not by itself distinguish their causes.
Finite completion samples also introduce estimation uncertainty.

Recommended metrics: decision regret, rank/choice quality on visited tree
leaves and siblings, error relative to decision margins, and total tour quality
per second. Report errors by remaining-node count. Global R² can be dominated
by predictable variation across tour depths and is insufficient for search.

Caveats on existing Stage 5 I interpretation:

- For TSP-50 states with more than 20 legal children, LKH labels are heuristic.
  Subtracting two upper bounds does not yield an upper bound on true regret.
  Call these estimated-reference regrets, not certified regrets or optima.
- Top-three-by-prior scores are a useful restricted diagnostic, not a proven
  description of every action PUCT visits.
- Each checkpoint's own states differ, so cross-checkpoint ranking differences
  do not isolate the effect of value training on a shared state distribution.

## 5. Why a plateau and rollout cost remain plausible

For deterministic TSP and a frozen deterministic completion policy, one full
rollout returns the exact cost of that policy's completion. A learned evaluator
must buy speed while preserving the small differences between candidate moves;
it cannot remove Monte Carlo noise from an already deterministic completion.
It need not approximate optimal completion to help, but its target policy must
be specified consistently.

Current replay stores states on the committed self-play tour and its realized
cost-to-go. Search evaluates additional child states and deeper leaves. Training
on committed paths alone need not teach the head to rank these counterfactual
states. Record and label actual search states, including poor alternatives, and
use held-out instances for diagnostics. Freeze the completion policy during
each label-generation period to separate label drift from fitting error.

Current policy targets are raw normalized visit counts, even when exploration
noise was active. At small K, these counts reflect exploration and finite
sampling as well as action quality. Cross-entropy does not guarantee that the
new policy's greedy tour improves, particularly with function approximation,
unvisited actions, replay distribution shift, and multiple equivalent tours.
The small-budget issue is directly relevant to
[Gumbel AlphaZero](https://openreview.net/forum?id=bERaNdoegnO), which redesigns
search/policy improvement when root actions are not all visited. Its guarantees
should not be transferred to an inaccurate evaluator or neural distillation.

Value loss also backpropagates through the shared policy representation. The
rollout/lambda_v ablation is evidence of a harmful auxiliary objective in that
recipe, not proof that value gradients are always noise. Indeed the fresh
TSP-50 Stage 1 value-enabled checkpoint beats its plain-AM counterpart in this
single run. Use frozen/detached features before another joint-loss sweep.

Finally, `leaf_eval='mix'` evaluates both the head and a complete rollout,
including at lambda=0 or 1. This is true in Python and C++ request construction.
It is an evaluator-blending experiment, not a strategy for avoiding rollouts.
H.3's lambda-dependent timing cannot be explained simply by adding an MLP only
when lambda>0: the MLP is computed at both endpoints, while search paths change.

## 6. Efficiency and baseline accounting corrections

The canonical Stage 1 args specify 1,280,000 fresh instances **per epoch**, for
100 epochs: **128 million**, not 1.28 million total. At batch size 512 this is
250,000 optimizer steps. Several later progress comparisons use the per-epoch
count as the full-run denominator. The old 6.4x sample-efficiency statement is
therefore not a valid calculation or a matched-quality result.

For example, 200 Stage 4 iterations at M=1,000 use 200,000 new graphs but also
many tree simulations, completions, and replay updates. With 200 training steps
per iteration and batch size 512, that is 20.48 million sampled state records
for SGD. Graph count alone is not compute cost; synthetic uniform TSP instances
are cheap. Log fresh graphs, evaluated decoder rows, complete rollouts,
optimizer work, total wall time, peak memory, and hardware separately.

Do not compare searched Stage 4 against greedy Stage 1 as evidence of a better
training method. Compare greedy-to-greedy and matched inference-time frontiers,
with separate matched training budgets. Use paired test instances, at least
three training seeds for a promising final variant, and a test set separate from
hyperparameter selection. A validation-set SE is not a training-seed SD.

The minimum baseline set should include plain AM and AM with the same value
features/loss where relevant; greedy, best-of-K and beam-search inference;
[POMO](https://arxiv.org/abs/2010.16011) for symmetry-aware routing RL; and
[Gumbeldore](https://arxiv.org/html/2403.15180v2), which already trains NCO policies
from self-generated improved solutions on TSP, CVRP and JSSP. Recent
[MACSIM (ICLR 2026)](https://openreview.net/forum?id=6KrETIaOYD) also studies the
generation cost and symmetry issues of self-improvement. These works make
"self-supervised NCO via improved samples" alone an insufficient novelty claim.

## 7. Suggested next experiment sequence

1. **Make evaluation units explicit.** Fix the omission described above and
   reproduce correct normalization in the C++ batch evaluator before updating
   historical conclusions. Keep the fixed checkpoint and test instances.
2. **Isolate the value representation.** On a frozen policy, compare the current
   glimpse head, an endpoint/count/unvisited-aware head with detached policy
   features, and rollout. Use counterfactual sibling/tree-state labels from the
   same frozen completion policy. Add exact one-city tails to all relevant
   conditions, or evaluate that shortcut separately. Compare MSE plus action
   regret and search quality per second. Widening the current MLP is not the
   appropriate control for missing information.
3. **Isolate the teacher from distillation.** For fixed training instances and
   compute, compare current visit targets to imitation of the best complete
   tour, retaining the greedy incumbent in the candidate set. A teacher selected
   this way cannot be worse than the incumbent on that instance; the trained
   student still has no such guarantee. Compare with ordinary sampling and
   Gumbeldore-style sampling without replacement. Canonicalize or augment tour
   rotations/reversal consistently. If keeping tree search, test a small-budget
   Gumbel/sequential-halving allocation as a distinct experiment.
4. **Only then test the loop.** One short TSP-20 pilot followed by TSP-50 if the
   frozen-policy result improves the quality/time frontier. Advance to multiple
   training seeds only for a promising variant. If the teacher is better but
   greedy student is not, change targets/representation; if the teacher cannot
   beat equal-time sampling, change the search method or problem.

For reducing rollout cost, prefer an explicit allocation rule: exact short
tails, learned values elsewhere, and a bounded number of rollouts where the
decision margin is small or a calibrated error estimate is large. This remains
a hypothesis until validated. A residual against a cheap geometric bound or
heuristic may reduce target scale, but a residual against a freshly computed
full neural rollout does not itself save that rollout.

## 8. Whether to change problems

The near-solved TSP-20 regime is useful for diagnosing errors, but gives little
headroom for a final greedy-quality claim. Simply increasing to TSP-100
increases branching and rollout cost without addressing the measured defects.

If the purpose is **amortizing expensive expected-cost evaluation with a learned
value**, my preferred next routing pilot is stochastic PCTSP. It is already in
the bundled AM reference code, and the
[original AM paper](https://arxiv.org/abs/1803.08475) includes stochastic PCTSP,
so it offers a baseline and less migration work. Compare a deterministic version
first, then add stochastic prizes to isolate uncertainty. Use expected realized
cost under the same observation information, shared scenario draws for paired
action comparisons, explicit chance transitions, and held-out scenarios. Never
give search future outcomes unavailable to the policy. More stochasticity also
makes value learning harder, so first establish a value-versus-multiple-rollout
quality/time advantage before building the full training loop.

If the purpose is **search-supervised combinatorial construction with stronger
long-horizon interactions**, small JSSP is a reasonable second option: job
precedence and machine contention create delayed scheduling consequences, and
problem bounds can inform residual values. This is a research hypothesis, not a
claim of easier learning. Gumbeldore already reports strong JSSP results, so it
must be a comparison. Branch-and-bound is a larger reformulation of the action
space and objective; it is not the economical next test of this codebase.

A proposed sharper research question is: can a state representation and search
allocation trained for **decision-relevant remaining cost** preserve rollout
improvements with lower evaluation cost, and can those improvements be retained
in the greedy policy? This is a recommendation, not an edit to proposal.md.

## Verification and remaining work

- [x] Reaggregated existing sibling metrics and inspected exact/heuristic labels.
- [x] Recomputed matched greedy costs for nine local checkpoints on CPU.
- [x] Proved and reproduced endpoint information loss on valid partial tours.
- [x] Reproduced normalization omission and its effect over 1,000 test instances.
- [x] Checked Python/C++ scale and mixed-evaluation code paths by inspection.
- [x] Checked primary papers for relevant baselines and alternatives.
- [x] Updated README status and linked corrections from Stage 5 progress.
- [x] Audit script compilation, saved-artifact consistency, local report links,
  and `git diff --check` for the edited tracked documents passed.
- [ ] Implement evaluation fix and metadata handling (follow-up, not this audit).
- [ ] Run the representation/target ablations above (proposed research work).

No result here establishes that either a representation change or a problem
pivot will improve end-to-end training. The confirmed defects and controlled
measurements identify a more informative next experiment than further tuning
the current recipe.
