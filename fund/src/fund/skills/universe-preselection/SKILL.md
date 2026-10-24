---
name: universe-preselection
description: |
  Scope the DB universe (~8898 names) down to a tractable candidate set: list it with
  `list_universe`, then prune the candidates with the pre-selection stack (data
  cleaning, incomplete-history, zero-variance and correlated-asset drops) before
  moments and optimization. Use when the allocator must scope a run's universe.
  Tickers always come from `list_universe` or the task's handoff, never from memory,
  and the skill never emits weights.
---

# Universe pre-selection

The allocator's first step: shrink the investable universe to a well-conditioned
candidate set so the optimizer is stable. The LLM chooses WHICH listed names to carry
forward; the pipeline prunes them and the optimizer weights them.

## When to use
Start of the allocator step, before moments + optimization.

## Load-bearing rule
`universe_filter` picks the surviving names. The LLM only forwards candidates it
took from `list_universe` (which already removes the client's ESG-excluded sectors)
or was handed by the task; it never adds, drops or invents a ticker by hand and never
emits weights. A ticker the database cannot price comes back as `missing`.

## Procedure
1. Choose the scope from the ConstraintSet + regime (see reference.md): sector tilts,
   size floor, K&E restrictions. If the task handed over candidate tickers, start from
   those.
2. Call `list_universe(sector, limit, offset)` to page through the investable names
   (largest market cap first; `total` is the full count) and collect the candidates.
   `sector` takes the ConstraintSet's GICS names (`information_technology`) or the
   stored names (`Technology`); an unknown name returns an error listing the valid
   ones, never an empty page. Sector and market cap are the latest snapshot, so for a
   past run date the size ranking is today's.
3. Call `universe_filter(universe)` with the candidates. It returns the survivors in
   requested order (big frames stay in the tool; only the ticker list returns). Names
   the client's mandate forbids (complex or leveraged products for a restricted client,
   an unclassifiable instrument) come back under `mandate_excluded`: report them to the
   PM. An empty list is a valid outcome: report it, do not retry with other names.
4. Call `estimate_moments(universe)` with the survivors for mu/cov (Ledoit-Wolf
   covariance + empirical mean over history up to the run date). Tickers it flags as
   `missing` had no instrument row (an unknown or mistyped ticker) or no priced days.
   The tools skip them on their own, so do not edit the list by hand: report the
   `missing` names to the PM and never substitute other tickers.
5. Hand the survivors + moments to `optimization-objective-map`.

## What the filter applies
Fixed defaults, no thresholds to choose: data validation and outlier treatment,
incomplete-history and zero-variance drops, and a 0.95 pairwise-correlation cap. The
tool then applies the client's mandate from the ConstraintSet: names in an
ESG-excluded sector or breaching the K&E instrument restrictions (and any whose sector
it cannot classify) are removed and listed as `mandate_excluded`. The tools take only
`universe`; any other argument is silently dropped, so never pass thresholds,
criteria or a config. `list_universe` takes only `sector`, `limit` and `offset`.

## Boundaries
- Never bypass `list_universe` and the filter to name assets directly; never emit
  weights.
- There is no liquidity floor (no volume data is loaded), factor tilt or sector cap
  in the filter. Never tell the PM one was applied.
- Honour the mandate screen: never re-admit a name listed as `mandate_excluded`.

## Deeper detail
Pipeline stages, what is not modelled, factor-selection theory and the pipeline
gotchas live in `reference.md` (theory `09`, `13`, `05`, `27`, `08`).
