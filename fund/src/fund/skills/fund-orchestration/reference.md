# Fund orchestration — reference

On-demand support for `fund-orchestration`. Points to the theory; never copies it.

## Pipeline order (one round)

| # | Step | Agent | Emits (structured) | Tool surface |
|---|---|---|---|---|
| 0 | Profile | profiler | ConstraintSet | mifid mapping (once / annual) |
| 1 | Regime + views | economist | regime label, ViewSet | get_macro_series, get_regime_stats |
| 2 | Pre-select + candidates | allocator | surviving tickers (the whole eligible universe when listed), the four candidates in id order | universe_filter, estimate_moments, compute_candidates |
| 3 | Validate | risk | passed / violations per candidate (BLOCKING) | risk_check, backtest (per candidate id) |
| 4 | Execute | executor | trades per feasible candidate; the adviser chooses one at the PM's gate (HITL) | none (PM: get_portfolio_history, place_orders) |

Universe build precedes selection: a stale universe caps everything downstream.
The profile/universe is fixed for the whole run (walk-forward cannot vary
constraints per fold) — a run is single-regime.

## Guardrails
- Round cap = `max_pm_rounds`; graph `recursion_limit`. On hit → the harness finalises
  the run INCOMPLETE with no ticket (no gate is raised).
- Risk gate is blocking (MiFID / ESG). `risk_check` judges one candidate of the
  latest set by id (it takes no weights) and records nothing; the verdict keyed to
  the chosen weights' digest is recorded when the adviser names a candidate, and
  `place_orders` places nothing without that choice and a passing verdict. HITL
  default gates `place_orders`.
- No agent chooses a candidate: the gate shows all four in id order and the adviser
  names one, or rejects the run.
- Subagents are stateless: delegate with complete instructions, collect the report.

## Theory
- The full signal → view → optimizer → weights flow (in this pipeline views are
  advisory and never reach the optimizer):
  `optimizer-theory/openwiki/workflows/from-conditional-forecasts-to-weights.md`.
- When and how to rebalance (round/turnover discipline):
  `optimizer-theory/openwiki/workflows/portfolio-revision.md`.
- Load-bearing contract, held inline (not in the theory tree): the LLM chooses
  structured inputs, the optimizer computes the weights — no agent ever emits a
  weight, a `mu`, or a covariance.
