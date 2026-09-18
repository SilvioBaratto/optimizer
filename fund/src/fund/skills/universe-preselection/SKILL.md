---
name: universe-preselection
description: |
  Filter the full DB universe (~8898 names) down to a tractable candidate set using
  liquidity, factor-score, drop-correlated and sector-cap enums before optimization.
  Use when the allocator must scope the universe for a run. Emits filter enums and
  thresholds only — never a hand-picked ticker list and never weights.
---

# Universe pre-selection

The allocator's first step: shrink ~8898 names to a well-conditioned candidate set so
the optimizer is stable. The LLM chooses the FILTERS; the pipeline applies them.

## When to use
Start of the allocator step, before moments + optimization.

## Load-bearing rule
The LLM emits filter enums/thresholds (liquidity floor, factor score, correlation
cap, sector caps). `universe_filter` / the pre-selection pipeline picks the names.
Never hand-pick tickers and never emit weights.

## Procedure
1. Choose filters from the ConstraintSet + regime (see reference.md): liquidity
   floor, factor tilts, drop-correlated threshold, sector caps, K&E exclusions.
2. Call `universe_filter(criteria)` for the candidate list (big frames stay in the
   tool; only the summary returns).
3. Call `estimate_moments(returns_asof, universe, config)` for mu/cov on the subset.
4. Hand the candidate set + moments to `optimization-objective-map`.

## Boundaries
- Never bypass the filter to name assets directly; never emit weights.
- Honour the ESG hard exclusions from the ConstraintSet (they cannot be re-admitted).

## Deeper detail
Filter enum table, factor-selection theory, and the pipeline gotchas live in
`reference.md` (theory `09`, `13`, `05`, `27`, `08`).
