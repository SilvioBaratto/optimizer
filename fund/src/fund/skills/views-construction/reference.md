# Views construction — reference

On-demand support for `views-construction`. Points to the theory; never copies it.

## ViewSet → Black-Litterman

| Concept | How it enters BL | Theory |
|---|---|---|
| Equilibrium prior Π* | reverse-optimised market weights | `03:111` |
| View (P, Q) | relative/absolute expected-return statements | `03:149` |
| Confidence → Omega | higher confidence → smaller Omega variance | `03:123`, `03:149` |
| Posterior (mu, Sigma) | the optimizer computes it, not the LLM | `03:94` |

## Regime → view bridge
`21:152` (regime → BL view), `21:114` (four economically-interpretable regimes).

## Factor views
- Factor-model structure and covariance: `05:71`, `05:121`.
- Signal → expected-return discipline (alpha = vol × IC × score): `12:42`, `12:68`.
- Config gotchas (repo): views use `tuple[str, ...]` (hashable) and embed a
  `MomentEstimationConfig`; factor views reference factor names and route through
  `TimeSeriesFactorModel.fit(X, factors=...)` (skfolio 1.0 keyword-only `factors=`).
