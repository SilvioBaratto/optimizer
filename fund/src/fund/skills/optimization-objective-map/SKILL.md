---
name: optimization-objective-map
description: |
  Map a ConstraintSet.objective (protection, income, growth, max) to the optimizer
  family and risk measure the mean_risk candidate runs with, then run
  compute_candidates and report truthfully what each of the four candidates applied.
  Use when the allocator is about to call compute_candidates and must state how the
  user's inputs relate to it. Does NOT emit weights and never chooses a
  candidate — the optimizer computes the weights, the user chooses.
---

# Optimization objective → candidate map

The allocator's mapping step: work out which optimizer FAMILY and risk measure the
objective in the user's inputs sets for the `mean_risk` candidate, run the optimizer for all four
candidates, and report truthfully what each applied.

## When to use
After the universe is filtered and a ConstraintSet exists. The economist's views
are background context and are not an input here.

## Load-bearing rule
The LLM emits ONLY structured inputs. NEVER weights, mu, or covariance, and never a
choice between candidates. `compute_candidates` computes the weights, and its only
model-facing input is the universe.

## Procedure
1. Read the objective + risk measure + bounds from the ConstraintSet; reference.md
   gives the optimizer family each objective implies.
2. Call `compute_candidates(universe)`. The run date, the portfolio, the base
   currency and the ConstraintSet are bound by the tool; any extra argument is
   silently dropped. It refuses a universe holding a ticker outside the portfolio's
   eligible universe, naming each offender.
3. Describe each candidate as what was applied (next section), in id order — `hrp`,
   `mean_risk`, `min_risk`, `risk_parity` — naming each only by its method and
   settings, and cite the `risk_aversion` the `mean_risk` settings report.
4. Report every candidate, the infeasible ones with their breaches. Candidates can be
   identical (a protection or variance setting makes `mean_risk` equal `min_risk`;
   tight caps push `hrp` and `risk_parity` towards equal weight): say so plainly.
   Never rank them or single one out; hand the whole set to `risk-limits-check`
   BEFORE any execution.

## What the tool applies
Every candidate is fitted on one returns panel under one shared configuration from
the ConstraintSet: `min_weights`, `max_weights` (tightened by the K&E position cap),
`budget`, and `prior_config` (the mean and covariance estimators of the ConstraintSet's
`moments_estimator`, the same ones `estimate_moments` is shown with). The
`mean_risk` candidate solves a long-only MeanRisk problem from the mapped config:
`objective`, `risk_measure`, `risk_aversion` (rescaled to the risk measure when it is
a first-order one such as CVaR), `cardinality`, `cvar_beta`, `cdar_beta`, `l1_coef`,
`l2_coef` and `risk_free_rate` (quoted per year, compounded to the panel's own bars
per year). With no resolved ConstraintSet it falls back to MINIMIZE_RISK over VARIANCE
(minimum variance). `min_risk` is minimum variance under the same bounds, `hrp`
is hierarchical risk parity on variance and `risk_parity` is equal risk
contribution on variance; the last two cannot enforce `cardinality`, so a
ConstraintSet with one reports them infeasible. Each candidate also reports how far
it moves the book from the current holdings, against the portfolio's drift band (L1).

## Intent the tool does not apply today
A robust uncertainty set, a turnover penalty tied to the portfolio's drift band (L1),
and the economist's views have no input on the tool. They stay recorded in the
decision as the user's stated intent; reference.md shows what they map to. Do not present
any of them as applied in your rationale.

## Boundaries
- Never emit weights/mu/cov; never invent a knob the ConstraintSet already fixes.
- Never choose, rank or single out a candidate: the user chooses at the gate.
- Variance estimators store `variance_` (1-D), not a full covariance matrix — do not
  swap them into a prior that needs a matrix.

## Deeper detail
The objective→cone table (with chapter refs) is in `reference.md`; runnable
optimizer-config skeletons per objective are in `factories.md` (illustrative — the
tool builds its own config from the ConstraintSet and does not accept them). Theory: start at the curated map
`optimizer-theory/openwiki/optimization/index.md`, then
`optimizer-theory/openwiki/optimization/convex-optimization.md` for the cone /
duality / KKT derivation; the full chapters are `14` (cones) + `02`, `16`, `17`,
`19`, `20` under `optimizer-theory/docs/`.
