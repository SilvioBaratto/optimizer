---
name: skfolio
description: |
  Load proactively whenever the user works with skfolio or portfolio optimization — building portfolios, backtesting allocation strategies, estimating expected returns or covariance, applying Black-Litterman or Entropy Pooling views, running walk-forward or combinatorial-purged cross-validation, tuning hyperparameters, or stress-testing with synthetic data. Do not wait to be asked; apply this skill automatically whenever the user mentions portfolio weights, efficient frontier, risk parity, HRP, mean-variance, CVaR optimization, factor models, covariance shrinkage, or any sklearn-style portfolio workflow. Covers skfolio 0.20.1 (April 2026): MeanRisk, RiskBudgeting, HRP/HERC/NCO/SchurComplementary, StackingOptimization, BlackLitterman, TimeSeriesFactorModel (replaces deprecated FactorModel), EntropyPooling, OpinionPooling, SyntheticData, regime-adjusted EW covariance, online learning (online_predict, OnlineGridSearch), covariance forecast evaluation, cross-sectional regression, and cross-sectional preprocessing transformers (CSStandardScaler, CSWinsorizer, CSGaussianRankScaler, CSPercentileRankScaler, CSTanhShrinker).
allowed-tools:
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - Bash
  - WebFetch
  - WebSearch
  - mcp__ide__getDiagnostics
---

# skfolio Portfolio Optimization

Expert guidance for **skfolio** — a portfolio optimization and risk management framework built on scikit-learn. Every estimator exposes `fit` / `predict` / `get_params`, composes into `Pipeline`, and plugs into `GridSearchCV`.

**Covers skfolio 0.20.1 (April 2026).** Major additions since 0.15.x:

- `skfolio.preprocessing` cross-sectional transformers — `BaseCSTransformer`, `CSStandardScaler`, `CSWinsorizer`, `CSGaussianRankScaler`, `CSPercentileRankScaler`, `CSTanhShrinker` (0.20.0)
- `skfolio.linear_model` — cross-sectional WLS regression (0.19.0)
- Online learning — `online_predict`, `online_score`, `OnlineGridSearch`, `OnlineRandomizedSearch` (0.18.0)
- Covariance forecast evaluation — `covariance_forecast_evaluation`, `CovarianceForecastComparison` (0.18.0)
- Variance estimators — `EmpiricalVariance`, `EWVariance`, `RegimeAdjustedEWVariance` (0.17.0)
- `RegimeAdjustedEWCovariance` with STVU multiplier (0.17.0)
- `SchurComplementary` hierarchical allocator (0.17.0)
- `TimeSeriesFactorModel` replaces deprecated `FactorModel` (0.17.0)
- `partial_fit` on `EmpiricalPrior`, `MeanRisk`, `EWMu`, `EWCovariance` (0.17.0–0.18.0)

## Where to look

Keep this file open for orientation and gotchas. For deep detail jump into a topic file:

| You're working on... | Read |
|---|---|
| Picking / configuring an optimizer (MeanRisk, HRP, HERC, NCO, Schur, Stacking, ...) | `references/optimization.md` |
| Choosing a prior (BlackLitterman, EntropyPooling, TimeSeriesFactorModel, ...) | `references/priors.md` |
| Expected returns, variance, covariance estimators (incl. regime-adjusted EW) | `references/moments.md` |
| Distance, clustering, pre-selection, uncertainty sets | `references/distance_clustering.md` |
| WalkForward / CombinatorialPurgedCV / GridSearchCV / metadata routing | `references/model_selection.md` |
| `partial_fit`-based workflows & covariance forecast evaluation | `references/online_learning.md` |
| Cross-sectional regression (`CSLinearRegression`) | `references/linear_model.md` |
| Cross-sectional preprocessing (CSStandardScaler, CSWinsorizer, ...) | `references/preprocessing.md` |
| Copulas & stress testing | `references/distributions.md` |
| `Portfolio`, `MultiPeriodPortfolio`, `Population` return types | `references/portfolio.md` |
| Worked end-to-end examples (20 patterns) | `PATTERNS.md` |

## Official documentation

Always cross-check against upstream — the library evolves fast.

| Topic | URL |
|---|---|
| API Reference | https://skfolio.org/api.html |
| User guide index | https://skfolio.org/user_guide/index.html |
| Online Learning | https://skfolio.org/user_guide/online_learning.html |
| Variance | https://skfolio.org/user_guide/variance.html |
| Releases / changelog | https://github.com/skfolio/skfolio/releases |
| Examples gallery | https://skfolio.org/auto_examples/index.html |

## Architecture

```
skfolio/
├── optimization/        # Portfolio optimization models
├── prior/               # Prior return distribution estimators
├── moments/             # Mu, variance & covariance estimators
├── linear_model/        # Cross-sectional regression (0.19.0+)
├── distance/            # Codependence & distance estimators
├── cluster/             # Hierarchical clustering
├── uncertainty_set/     # Mu & covariance uncertainty sets
├── pre_selection/       # Asset filtering transformers
├── model_selection/     # CV, backtesting, online learning, forecast eval
├── metrics/             # Scoring functions
├── preprocessing/       # prices_to_returns + cross-sectional transformers (0.20.0+)
├── distribution/        # Copulas & univariate distributions
├── datasets/            # Sample datasets
├── portfolio/           # Portfolio & MultiPeriodPortfolio
├── population/          # Population of portfolios
└── measures/            # Risk/performance enums
```

## Decision guide

Use these starting points — they cover 90% of real use cases. Each row points at the matching reference file for the full API.

### Pick an optimizer

| You want to... | Start with | Why |
|---|---|---|
| Maximize risk-adjusted return (Sharpe, Sortino, ...) | `MeanRisk(objective_function=MAXIMIZE_RATIO, risk_measure=...)` | Most flexible; 15 convex risk measures, full constraints |
| Minimize tail risk (CVaR, EVaR, CDaR) | `MeanRisk(objective_function=MINIMIZE_RISK, risk_measure=CVAR)` | CVaR is convex and well-behaved |
| Allocate **risk** equally across assets (ERC) | `RiskBudgeting()` | Default budget is equal risk contribution |
| Robust to covariance estimation error | `HierarchicalRiskParity()` or `SchurComplementary(gamma=0.5)` | No matrix inversion; SchurComplementary tunes HRP↔MVP |
| Robust to mu/cov uncertainty (worst-case) | `MeanRisk(mu_uncertainty_set_estimator=..., covariance_uncertainty_set_estimator=...)` or `DistributionallyRobustCVaR` | Worst-case over bootstrap ball or Wasserstein ball |
| Track a benchmark | `BenchmarkTracker(tracking_error_target=0.01)` | Pass benchmark as `y` in `fit(X, y=...)` |
| Ensemble multiple strategies | `StackingOptimization(estimators=[...], final_estimator=...)` | Feeds base optimizers into a final allocator |
| Baseline comparison | `EqualWeighted()` or `InverseVolatility()` | Every serious backtest should compare against these |

Details: `references/optimization.md`.

### Pick a prior

| Situation | Prior |
|---|---|
| Just use history | `EmpiricalPrior(mu_estimator=..., covariance_estimator=...)` |
| Have analyst views on returns or relative performance | `BlackLitterman(views=[...])` |
| Have views on variance / correlation / skew / CVaR / groups | `EntropyPooling(mean_views=..., variance_views=..., correlation_views=..., ...)` |
| Reduce dimensionality via factors | `TimeSeriesFactorModel(factor_prior_estimator=...)` — pass factor returns as `y` |
| Combine multiple expert opinions | `OpinionPooling(estimators=[...], opinion_probabilities=[...])` |
| Stress test with synthetic scenarios | `SyntheticData(distribution_estimator=VineCopula(), sample_args=dict(conditioning={...}))` |

Details: `references/priors.md`.

### Pick a covariance estimator

| Situation | Covariance |
|---|---|
| Small universe, plenty of data | `EmpiricalCovariance` |
| Large universe, small sample (N > T/4) | `LedoitWolf` or `OAS` (shrinkage) |
| Want to adapt to changing vol regimes | `EWCovariance(half_life=40)` |
| Regime-shift adaptive, fast vol, stable correlations | `RegimeAdjustedEWCovariance(half_life=23, corr_half_life=50)` |
| Reduce noise in correlation matrix | `DenoiseCovariance` or `DetoneCovariance` (RMT) |
| Sparse precision structure | `GraphicalLassoCV` |
| Use options-implied vol | `ImpliedCovariance()` (needs metadata routing) |
| Heavy-tailed, outlier-prone returns | `GerberCovariance` |

Before committing, **compare candidates** with `online_covariance_forecast_evaluation` — see `references/online_learning.md`.

### Pick a CV / evaluation strategy

| Goal | Use |
|---|---|
| Single time-ordered backtest | `WalkForward(test_size=60, train_size=252)` + `cross_val_predict` |
| Multiple testing paths, purging, embargo | `CombinatorialPurgedCV` + `cross_val_predict` |
| Monte Carlo over asset subsets and windows | `MultipleRandomizedCV` |
| Fast walk-forward with incremental `partial_fit` | `online_predict` / `online_score` (needs EW estimators) |
| Compare covariance forecasts, optimizer-agnostic | `online_covariance_forecast_evaluation` |

Details: `references/model_selection.md` and `references/online_learning.md`.

## Data preparation — the one rule that matters

Always feed **linear returns**, not log returns:

```python
from skfolio.preprocessing import prices_to_returns
X = prices_to_returns(prices)   # linear returns by default
```

Why: linear returns aggregate across assets (portfolio return = weighted sum). Log returns aggregate across time but **not** across assets, which silently breaks every optimizer that assumes weighted-sum portfolio returns. For multi-year horizons use `EmpiricalPrior(is_log_normal=True, investment_horizon=...)` instead of ad-hoc √T scaling.

`X` must be a pandas `DataFrame` with tickers as columns and a `DatetimeIndex`.

## Key constraints & gotchas

These cut across every optimizer and prior — internalize them before you start.

1. **Linear returns only** as input `X` (see above).
2. **`shuffle=False`** in any `KFold` or `train_test_split` — shuffling financial series causes lookahead leakage.
3. **Metadata routing is opt-in:** `set_config(enable_metadata_routing=True)` **before** using `.set_fit_request()`.
4. **`TimeSeriesFactorModel` uses `fit(X, y)`** where `X` = asset returns and `y` = factor returns. `FactorModel` is deprecated in 0.17.0 — migrate the import; the contract is unchanged.
5. **`BenchmarkTracker` uses `fit(X, y)`** where `y` = benchmark returns.
6. **Group constraints** need a `groups` dict (`{"Tech": ["AAPL", "MSFT"], ...}`) plus `linear_constraints=["Tech <= 0.4"]`.
7. **Nested parameter tuning** uses `__` syntax: `"prior_estimator__mu_estimator__half_life"`. Discover them via `model.get_params()`.
8. **`CombinatorialPurgedCV` returns `Population`**, not `MultiPeriodPortfolio`. Use `optimal_folds_number()` to pick fold sizes.
9. **`Pipeline` works** for pre-selection + optimization with standard `fit`, but **not** inside `online_predict` / `OnlineGridSearch` — those require a single estimator that implements `partial_fit`.
10. **Variance estimators** (`EmpiricalVariance`, `EWVariance`, `RegimeAdjustedEWVariance`) store results in `variance_` and are **not** drop-in replacements for covariance estimators inside priors that need a full `covariance_` matrix.
11. **Regime-adjusted EW** defaults to clipping the regime multiplier to `(0.7, 1.6)` — widen `regime_multiplier_clip` for fast-moving regimes.
12. **Cross-sectional regression** (`CSLinearRegression`) expects `X: (T, N, K)`, `y: (T, N)`, `sample_weight: (T, N)`. Zero-weight pairs are excluded and may contain NaN.
13. **Covariance forecast evaluation is optimizer-agnostic** — rank covariance estimators with `CovarianceForecastComparison` **before** plugging one into a prior.
14. **Cross-sectional transformers** (`CSStandardScaler`, `CSWinsorizer`, `CSGaussianRankScaler`, `CSPercentileRankScaler`, `CSTanhShrinker`) operate **across assets per period** (axis=1), not across time. Use as feature preprocessors for `CSLinearRegression` or factor signals; they preserve `(T, N)` shape and skip NaN per row.

## Common imports cheat sheet

```python
from skfolio import RiskMeasure, RatioMeasure, PerfMeasure, ExtraRiskMeasure
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.prior import EmpiricalPrior, BlackLitterman, TimeSeriesFactorModel
from skfolio.moments import LedoitWolf, EWMu, EWCovariance, RegimeAdjustedEWCovariance
from skfolio.model_selection import WalkForward, cross_val_predict, online_predict
from skfolio.preprocessing import prices_to_returns
from skfolio.datasets import load_sp500_dataset, load_factors_dataset
```

For the full import surface including naive models, uncertainty sets, copulas, and datasets, see the matching reference file.

## Implementation patterns

20 worked end-to-end examples — basic mean-variance, Black-Litterman, factor models, HRP, risk budgeting, stacking, pre-selection pipelines, walk-forward, hyperparameter tuning, robust optimization, synthetic data, opinion pooling, custom scoring, metadata routing, regime-adjusted covariance, online learning, covariance forecast evaluation, SchurComplementary, cross-sectional regression, full production pipeline — in `PATTERNS.md`.
