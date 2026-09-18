# Risk-limits check — reference

On-demand support for `risk-limits-check`. Points to the theory; never copies it.

## Drawdown ceilings (from loss capacity)
`M <= nu1*C`, `A <= nu2*C`, `Delta_alpha <= nu3*C` — max / average / CDaR drawdown
capped by the capacity multipliers in the ConstraintSet. Theory: `20:31`, `20:42`,
`20:66`, and the ceiling framing at `20:109`. Roadmap anchor: `20:75`.

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
