---
name: universe-preselection
description: |
  Prune the candidate tickers handed to the allocator with the pre-selection stack
  (data cleaning, incomplete-history, zero-variance and correlated-asset drops)
  before moments and optimization. Use when the allocator must scope a run's
  universe. Passes the tickers it was given - never invents a ticker, never emits
  weights.
---

# Universe pre-selection

The allocator's first step: prune the candidate tickers to a well-conditioned set so
the optimizer is stable. The LLM passes the candidates in; the pipeline decides which
survive.

## When to use
Start of the allocator step, before moments + optimization.

## Load-bearing rule
`universe_filter` picks the surviving names. The LLM only forwards the candidate
tickers it was handed; it never adds, drops or invents a ticker and never emits
weights.

## Procedure
1. Take the candidate tickers from the task / the economist's handoff. No tool lists
   the investable universe, so if none were handed over, stop and report that to the
   PM instead of guessing names.
2. Call `universe_filter(universe)` with those tickers. It returns the survivors in
   requested order (big frames stay in the tool; only the ticker list returns). Names
   the client's mandate forbids come back under `mandate_excluded`: report them to the
   PM. An empty list is a valid outcome: report it, do not retry with other names.
3. Call `estimate_moments(universe)` with the survivors for mu/cov (Ledoit-Wolf
   covariance + empirical mean over history up to the run date). Tickers it flags as
   `missing` had no instrument row (an unknown or mistyped ticker) or no priced days.
   The tools skip them on their own, so do not edit the list by hand: report the
   `missing` names to the PM and never substitute other tickers.
4. Hand the survivors + moments to `optimization-objective-map`.

## What the filter applies
Fixed defaults, no thresholds to choose: data validation and outlier treatment,
incomplete-history and zero-variance drops, and a 0.95 pairwise-correlation cap. The
tool then applies the client's mandate from the ConstraintSet: names in an
ESG-excluded sector or breaching the K&E instrument restrictions (and any whose sector
it cannot classify) are removed and listed as `mandate_excluded`. The tools take only
`universe`; any other argument is silently dropped, so never pass thresholds,
criteria or a config.

## Boundaries
- Never bypass the filter to name assets directly; never emit weights.
- There is no liquidity floor (no volume data is loaded), factor tilt or sector cap
  in the filter. Never tell the PM one was applied.
- Honour the mandate screen: never re-admit a name listed as `mandate_excluded`.

## Deeper detail
Pipeline stages, what is not modelled, factor-selection theory and the pipeline
gotchas live in `reference.md` (theory `09`, `13`, `05`, `27`, `08`).
