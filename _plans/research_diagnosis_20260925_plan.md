# Research diagnosis: MCTS-guided policy improvement (2026-09-25)

## Objective

Assess why search-supervised Attention Model training has not surpassed the
REINFORCE baseline, why learned leaf values underperform rollouts, and whether
to improve the method on TSP or test a different problem. Separate verified
implementation facts and measured results from causal hypotheses.

## Scope

- Review README, proposal, existing plans/progress, current implementation, and
  locally available experiment artifacts.
- Check search/value/target conventions, baseline comparability, and whether
  value diagnostics measure action ranking rather than only global fit.
- Run bounded CPU diagnostics when existing artifacts and the environment permit.
- Check primary literature for relevant alternatives and experimental design.
- Record an evidence-based diagnosis and a small decision-oriented experiment
  sequence in the matching progress file; update README status if needed.

This is an audit, not a new training campaign. Preserve existing changes and
untracked probes. Any proposed research-direction revision remains a
recommendation; do not edit proposal.md in this task.

## Work sequence

1. Inventory available code, logs, checkpoints, and prior decisions.
2. Trace value features/targets, MCTS backup, exploration, replay, and losses.
3. Reconcile reported outcomes with raw local artifacts and bounded checks.
4. Compare plausible improvements and candidate problems using primary sources.
5. Document conclusions, evidence limits, and explicit continue/pivot criteria.

Completed 2026-09-25. Findings and follow-up experiments are recorded in
`../_progress/research_diagnosis_20260925_progress.md`. New measurements cover
matched greedy checkpoint evaluation, a value-input counterexample, and a
1,000-instance evaluation-normalization comparison. Main training/search code
and proposal.md were not modified.

## Verification

Use existing raw metrics and matched per-instance results wherever available.
Report hardware and dataset differences, missing checkpoints, and untested
causal explanations. Avoid treating one-seed outcomes or validation standard
errors as training-seed significance. Run `git diff --check` on audit edits.
