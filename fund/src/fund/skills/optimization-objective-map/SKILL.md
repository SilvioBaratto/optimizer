---
name: optimization-objective-map
description: |
  Map a ConstraintSet.objective (protection, income, growth, max) to the optimizer
  family and risk measure it implies, then run optimize_portfolio and report truthfully
  what the tool applied. Use when the allocator is about to call optimize_portfolio and
  must state how the profile's objective relates to it. Does NOT emit weights — the
  optimizer computes the weights.
---

# Optimization objective → cone map

The allocator's mapping step: work out which optimizer FAMILY and risk measure the
profile's objective implies, run the optimizer, and report truthfully what it applied.

## When to use
After the universe is filtered and a ConstraintSet exists. The economist's views
are advisory and are not an input here.

## Load-bearing rule
The LLM emits ONLY structured inputs. NEVER weights, mu, or covariance.
`optimize_portfolio` computes the weights, and its only model-facing input is the
universe.

## Procedure
1. Read the objective + risk measure + bounds from the ConstraintSet; reference.md
   gives the optimizer family each objective implies.
2. Call `optimize_portfolio(universe)`. The run date, the portfolio, the base currency
   and the ConstraintSet are bound by the tool; any extra argument is silently
   dropped.
3. Describe the result as what was applied (next section) and cite the
   `risk_aversion` the tool reports having run with.
4. Hand weights + metrics to `risk-limits-check` BEFORE any execution.

## What the tool applies
The tool solves a long-only, fully invested MeanRisk problem from the ConstraintSet's
mapped config: `objective`, `risk_measure`, `risk_aversion` (rescaled to the risk
measure when it is a first-order one such as CVaR), `min_weights`, `max_weights`
(tightened by the K&E position cap), `budget`, `cardinality`, `cvar_beta`,
`cdar_beta`, `l1_coef`, `l2_coef` and `risk_free_rate`. With no resolved profile it
falls back to MINIMIZE_RISK over VARIANCE (minimum variance).

## Intent the tool does not apply today
A robust uncertainty set, a turnover penalty tied to the mandate's drift band (L1),
and the moments or views from `estimate_moments` have no input on the tool. They stay
recorded in the decision as the client's intent; reference.md shows what they map to.
Do not present any of them as applied in your rationale.

## Boundaries
- Never emit weights/mu/cov; never invent a knob the ConstraintSet already fixes.
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
