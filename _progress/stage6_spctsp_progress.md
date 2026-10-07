# Stage 6 — Stochastic PCTSP — progress

Plan: [`_plans/stage6_spctsp_plan.md`](../_plans/stage6_spctsp_plan.md).

## Status

**PLANNED 2026-10-07.** Lejun approved closing the TSP chapter and moving to
stochastic PCTSP (proposal.md revised the same day). No code has been written yet.
The next step is P0: environment port and parity.

| Phase | Status | Gate |
|---|---|---|
| P0 environment + parity | not started | AM-reference parity ≤ 1e-5, identical tours |
| P1 value-signal probe | not started | G1a: room exists; G1b: head k_eq ≥ 4 (proposed) |
| P2 search at test time | not started | G2: thresholds set before the run |
| P3 student isolation | not started | G3: beats control and retention ≥ 0.25 |
| P4 full loop | blocked on G1–G3 | — |

## Pre-work measurement (2026-10-07, Mac CPU)

The pretrained AM SPCTSP models from `ref/attention-learn-to-route-master/pretrained/`
load and run with the project `.venv` (torch 2.10). Greedy decoding was run on
2,000 instances from `generate_instance` (torch seed 0), then re-run on the same
instances with a second prize realisation (seed 1):

| model | mean cost (realisation 1 / 2) | SD across instances | SD across realisations, same instance | wall |
|---|---|---:|---:|---:|
| pctsp_stoch_20 | 3.2544 / 3.2450 | 0.39 | 0.23 | 0.7 s |
| pctsp_stoch_50 | 4.6445 / 4.6492 | 0.31 | 0.18 | 3.3 s |

The realisation noise of a whole episode is large. Whether it changes *decisions*
(sibling ranking at a state) is what gate G1a measures; whole-episode noise is an
upper bound on that.
