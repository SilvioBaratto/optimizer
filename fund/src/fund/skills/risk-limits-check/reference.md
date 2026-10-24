# Risk-limits check — reference

On-demand support for `risk-limits-check`. Points to the theory; never copies it.

## Drawdown ceilings (from loss capacity)
The profiler emits escalating tiers on the client's one-year loss tolerance `L`:
`nu1 = 0.5*L` (soft warning), `nu2 = L` (hard ceiling), `nu3 = min(1.5*L, 1.0)`
(absolute stop). `backtest` compares its single `max_drawdown` with all three and
returns the breached tiers in `drawdown_tiers`; `risk_check` applies the same
comparison, so nu2 / nu3 block it and nu1 is only a warning.
The theory states the three ceilings as separate constraints on max, average and CDaR
drawdown (`M <= nu1*C`, `A <= nu2*C`, `Delta_alpha <= nu3*C`: `20:31`, `20:42`,
`20:66`, ceiling framing at `20:109`) — background for the measures, not what the
tool evaluates; do not read nu2 / nu3 as average-drawdown or CDaR limits. Roadmap
anchor: `20:75`.

Basis gap: the optimizer enforces these ceilings on the additive cumulative-return
path (skfolio), while `backtest` reports `max_drawdown` on the compounded wealth path
(`drawdown_basis: "compounded"`). The figures differ by the compounding effect over
the drawdown window, so a backtest reading close to a ceiling is not by itself a
breach.

## Coherent risk & tail
- Coherent risk axioms (monotonicity, subadditivity, ...): `04:18`.
- CVaR as a coherent objective: `16:28`, `16:121`.
- Risk attribution / budgets (Euler, existence): `23:70`, `23:204`.

## Robustness & stress
- Robust uncertainty set (prudent profiles), radius/geometry/norm in 1.0: `15:80`.
- Stress plausibility (Mahalanobis) + reverse stress: `24:54`, `24:78`, `24:116`.

## Validation discipline
- Walk-forward, `shuffle=False`; single-regime (constraints fixed per run).
- Data-snooping / overfitting haircuts: `10:14`, `10:64`, `10:84`.
