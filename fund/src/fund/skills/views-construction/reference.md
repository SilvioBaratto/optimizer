# Views construction — reference

On-demand support for `views-construction`. Points to the theory; never copies it.

## ViewSet → Black-Litterman (bridge exists, not wired in a run)

`ViewSet.to_black_litterman_config` renders the views, but no bound tool takes the
result, so the table below describes the bridge, not what a run does to the weights.

| Concept | How it enters BL | Theory |
|---|---|---|
| Equilibrium prior Π* | theory: reverse-optimised market weights; the bridge never passes `market_weights`, so skfolio's `EquilibriumMu` uses equal weights (1/N) | `03:111` |
| View (P, Q) | relative/absolute expected-return statements | `03:149` |
| Confidence → Omega | higher confidence → smaller Omega variance | `03:123`, `03:149` |
| Posterior (mu, Sigma) | computed by a Black-Litterman run, never by the LLM | `03:94` |

## Regime → view bridge
`21:152` (regime → BL view), `21:114` (four economically-interpretable regimes).

## Factor views
- Factor-model structure and covariance: `05:71`, `05:121`.
- Signal → expected-return discipline (alpha = vol × IC × score): `12:42`, `12:68`.
- Config gotchas (repo): views use `tuple[str, ...]` (hashable) and embed a
  `MomentEstimationConfig`; factor views reference factor names and would need a
  `TimeSeriesFactorModel.fit(X, factors=...)` prior (skfolio 1.0 keyword-only
  `factors=`). That prior is not built: the bridge always builds `EquilibriumMu` +
  a covariance estimator, and `ViewKind.FACTOR` is metadata only, so a factor-name
  view would fail to parse against that asset-only prior.
