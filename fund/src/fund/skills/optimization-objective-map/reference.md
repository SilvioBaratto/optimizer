# Optimization objective → cone map — reference

On-demand support for `optimization-objective-map`. Points to the theory; never copies.

## Objective → optimizer family / risk measure

| ConstraintSet.objective | Optimizer family | Risk measure | Cone / form | Theory |
|---|---|---|---|---|
| min_risk | MeanRisk (MINIMIZE_RISK) | variance / CVaR / CDaR | QP / LP | `02:84`, `16:75`, `20:66` |
| max_utility | MeanRisk (MAXIMIZE_UTILITY, risk_aversion=A) | variance / semivariance | QP | `02:14`, `14:126` |
| max_ratio | MeanRisk (MAXIMIZE_RATIO) | variance (Sharpe) / CVaR | SOCP | `02:121`, `14:80` |
| risk_budget | RiskBudgeting (ERC / budgets) | variance | log-barrier | `17:37`, `17:107` |
| max_diversification | MaximumDiversification | variance | QP | `19:58` |

## Risk measure → cone
- Variance → QP; Sharpe / tangency → SOCP: `14:80`, `14:126`.
- CVaR → Rockafellar-Uryasev LP: `16:28`, `16:75`.
- Drawdown (MaxDD / CDaR) → LP: `20:42`, `20:66`.
- Why the cone matters (KKT / duality): `14:154`, `14:206`.

## Guardrails baked into the config
- Bounds: long-only `w >= 0`, `sum(w) = 1`, no leverage (D17).
- Cardinality cap + min-weight per profile (avoid micro-weights).
- Robust uncertainty set for prudent profiles: `15:80` (radius/geometry/norm in 1.0).
- Objective / cone obligations: the curated map
  `optimizer-theory/openwiki/optimization/index.md` (which family solves which
  objective) and `optimizer-theory/openwiki/optimization/convex-optimization.md`
  (why the cone is the binding form).
