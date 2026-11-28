# Optimization objective → cone map — reference

On-demand support for `optimization-objective-map`. Points to the theory; never copies.

## Input objective → optimizer family

`ConstraintSet.objective` takes one of the four input values below; the MeanRisk
objective each maps to is fixed by `ConstraintSet.to_mean_risk_config()`. The risk
measure is a separate ConstraintSet field (`variance`, `semi_variance`, `cvar`, `cdar`,
`max_drawdown`) and pairs with any row.

| ConstraintSet.objective | Optimizer family | Risk measure | Cone / form | Theory |
|---|---|---|---|---|
| `protection` | MeanRisk (MINIMIZE_RISK) | variance / CVaR / CDaR | QP / LP | `02:84`, `16:75`, `20:66` |
| `income`, `growth` | MeanRisk (MAXIMIZE_UTILITY, risk_aversion = `a_gamma`) | variance / semivariance | QP | `02:14`, `14:126` |
| `max` | MeanRisk (MAXIMIZE_RATIO) | variance (Sharpe) / CVaR | SOCP | `02:121`, `14:80` |

## Not reachable from the inputs
No `ConstraintSet.objective` value selects these optimizer families; never name one as
the inputs' objective.
- RiskBudgeting (ERC / budgets), variance, log-barrier: `17:37`, `17:107`.
- MaximumDiversification, variance, QP: `19:58`.

## Risk measure → cone
- Variance → QP; Sharpe / tangency → SOCP: `14:80`, `14:126`.
- CVaR → Rockafellar-Uryasev LP: `16:28`, `16:75`.
- Drawdown (MaxDD / CDaR) → LP: `20:42`, `20:66`.
- Why the cone matters (KKT / duality): `14:154`, `14:206`.

## Limits
- Applied by `compute_candidates` (the `mean_risk` candidate; the bounds bind all
  four): the ConstraintSet bounds `min_weights`,
  `max_weights` (tightened by the K&E position cap) and `budget` — long-only
  `w >= 0`, `sum(w) = 1`, no leverage (D17) — plus the mapped objective, risk measure,
  `risk_aversion`, tail betas, L1/L2 regularisation, `cardinality` and the
  ConstraintSet's moment estimators (`prior_config`, from `moments_estimator`).
- Risk aversion `A` is quoted for variance utility; the tool rescales it for
  CVaR / CDaR / MaxDD so max_utility lands at the risk level that variance-utility
  `A` implies (neither collapsing into min_risk nor drifting to a riskier book).
- `max` (MAXIMIZE_RATIO) assumes a tangency portfolio exists, which holds only if
  the risk-free rate is below the expected return of the minimum-variance portfolio
  (`02:185`). When the estimated returns leave no such portfolio, say so; do not
  describe the result as a max-Sharpe book.
- Recorded intent, never applied today: the robust uncertainty set for prudent
  inputs: `15:80` (radius/geometry/norm in 1.0), and the turnover penalty.
- Objective / cone obligations: the curated map
  `optimizer-theory/openwiki/optimization/index.md` (which family solves which
  objective) and `optimizer-theory/openwiki/optimization/convex-optimization.md`
  (why the cone is the binding form).
