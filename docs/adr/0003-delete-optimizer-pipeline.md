# ADR-0003: Delete `optimizer.pipeline`; the library is composable primitives

- Status: Accepted
- Date: 2026-09-15

## Context

The `optimizer` library shipped a fixed end-to-end runner, `optimizer.pipeline`, with
`run_full_pipeline` and its relatives. Opinionated, database-connected orchestration (FX
conversion, delisting correction, universe and factor selection, rebalancing decisions,
persistence) does not belong in a database-agnostic library, and the `fund/` bridge was
about to own exactly that (commit `6d6c3e43`, 2026-09-15).

## Decision

Delete the module. The library is a set of composable, database-agnostic primitives with
no built-in orchestration. The removed public symbols are `run_full_pipeline`,
`run_full_pipeline_with_selection`, `optimize`, `backtest`, `tune_and_optimize`,
`compute_net_backtest_returns` and `PortfolioResult`. `build_portfolio_pipeline`, the
pre-selection to optimiser composition, moved to `optimizer.pre_selection`. Callers
compose `build_portfolio_pipeline(optimizer)` with `validation.run_cross_val`.

## Alternatives considered

- **Keep the runner and give it database hooks.** The commit states the reason against
  it: "keeping optimizer DB-agnostic". No other alternative is recorded.

## Consequences

- This is a breaking change for any user of the library. `CHANGELOG.md` lists it under
  `Removed (BREAKING — library)`, yet the version is still `0.1.0` and no release has
  been cut since, so a release is overdue (standard 5.1.1).
- Orchestration lives in `fund/`, which may import both `optimizer` and `portopt_db`
  (ADR-0002), so the library keeps no database dependency.
