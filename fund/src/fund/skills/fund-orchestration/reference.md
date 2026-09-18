# Fund orchestration — reference

On-demand support for `fund-orchestration`. Points to the theory; never copies it.

## Pipeline order (one round)

| # | Step | Agent | Emits (structured) | Tool surface |
|---|---|---|---|---|
| 0 | Profile | profiler | ConstraintSet | mifid mapping (once / annual) |
| 1 | Regime + views | economist | regime label, ViewSet | get_macro_series |
| 2 | Universe + optimize | allocator | filters, optimizer config, weights | universe_filter, estimate_moments, optimize_portfolio |
| 3 | Validate | risk | passed / violations (BLOCKING) | risk_check, backtest |
| 4 | Execute | executor | paper orders (HITL) | place_orders |

Universe build precedes selection: a stale universe caps everything downstream.
The profile/universe is fixed for the whole run (walk-forward cannot vary
constraints per fold) — a run is single-regime.

## Guardrails
- Round cap = `max_pm_rounds`; graph `recursion_limit`. On hit → run INCOMPLETE + HITL.
- Risk gate is blocking (MiFID / ESG). HITL default gates `place_orders`.
- Subagents are stateless: delegate with complete instructions, collect the report.

## Theory
- Module flow + layering: `ARCHITECTURE.md`.
- Build order / what the optimizer must expose: `OPTIMIZER-OBLIGATIONS.md §6`.
- Load-bearing ("LLM chooses, optimizer computes"): `deep_agent.md` Principi trasversali.
