# Risk-limits check — reference

On-demand support for `risk-limits-check`. Points to the theory; never copies it.

## Drawdown ceilings (from loss capacity)
The profiler emits escalating tiers on the client's one-year loss tolerance `L`:
`nu1 = 0.5*L` (soft warning), `nu2 = L` (hard ceiling), `nu3 = min(1.5*L, 1.0)`
(absolute stop). `backtest` compares `worst_rolling_loss` with all three and
returns the breached tiers in `drawdown_tiers`; `risk_check` applies the same
comparison, so nu2 / nu3 block it and nu1 is only a warning.
`worst_rolling_loss` is the worst compounded peak-to-trough decline inside any window
of one year of the panel's calendar (about 261 bars on a weekday calendar): `L` is a
one-year figure, so a book that bled slowly over several years is not blocked for a
multi-year drawdown no single year contains. The full-history peak-to-trough stays in
the reply as `full_sample_max_drawdown`, for information. With less than a year of
common history the one-year loss cannot be measured, and the verdict is
`drawdown_unverified`. `L = 0` makes every tier 0 and fails any book with a loss at
all; the gate names it `zero_loss_tolerance` (a profile problem, not an allocation
one).
The theory states the three ceilings as separate constraints on max, average and CDaR
drawdown (`M <= nu1*C`, `A <= nu2*C`, `Delta_alpha <= nu3*C`: `20:32`, `20:42`,
`20:46`; drawdown caps as constraints at `20:136`) — background for the measures, not
what the tool evaluates; do not read nu2 / nu3 as average-drawdown or CDaR limits.
Here `C = 1`: the tiers are unitless fractions of capital, so compare
`worst_rolling_loss` to them directly, never to `mandate.capital`.

Basis: the optimizer does not enforce these ceilings (they have no `MeanRiskConfig`
field). `risk_check` compares them with `worst_rolling_loss`, the compounded
wealth-path decline (`drawdown_basis: "compounded"`), so a reading just past a ceiling
is a breach on the basis the ceilings are judged on.

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
