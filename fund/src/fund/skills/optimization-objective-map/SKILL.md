---
name: optimization-objective-map
description: |
  Map a chosen ConstraintSet.objective to the correct optimizer cone / risk-measure
  and factory config. Use when the allocator must turn an objective (min-risk,
  max-utility, max-ratio, risk-budget) into a concrete optimizer configuration before
  optimize_portfolio. Does NOT emit weights — the optimizer computes the weights.
---

# Optimization objective → cone map

The allocator's core step: pick the optimizer FAMILY and risk measure that match the
profile's objective, emit the factory config, and let the optimizer solve.

## When to use
After universe + moments + views are ready and a ConstraintSet exists.

## Load-bearing rule
The LLM emits ONLY a structured optimizer config. NEVER weights, mu, or covariance.
`optimize_portfolio` computes the weights from this config.

## Procedure
1. Read the objective + risk measure + bounds from the ConstraintSet.
2. Map objective → cone / risk measure (table in reference.md; call skeletons in
   factories.py). Long-only, sum-to-one, no leverage by default.
3. Apply the cardinality cap + min-weight, a robust uncertainty set if the profile is
   prudent, and any turnover penalty tied to the executor's no-trade band.
4. Call `optimize_portfolio(returns_asof, universe, constraints)`; hand weights +
   metrics to `risk-limits-check` BEFORE any execution.

## Boundaries
- Never emit weights/mu/cov; never invent a knob the ConstraintSet already fixes.
- Variance estimators store `variance_` (1-D), not a full covariance matrix — do not
  swap them into a prior that needs a matrix.

## Deeper detail
The objective→cone table (with chapter refs) is in `reference.md`; exact factory-call
skeletons per objective are in `factories.py`. Theory: `14` (cones) + `02`, `16`,
`17`, `19`, `20`, and `OPTIMIZER-OBLIGATIONS.md §1`/`§3`. (The roadmap said "§7";
§7 is the out-of-scope section — corrected here.)
