---
name: skfolio
description: |
  Load proactively whenever the user works with skfolio or portfolio optimization — building portfolios, backtesting allocation strategies, estimating expected returns or covariance, applying Black-Litterman or Entropy Pooling views, running walk-forward or combinatorial-purged cross-validation, tuning hyperparameters, or stress-testing with synthetic data. Do not wait to be asked; apply this skill automatically whenever the user mentions portfolio weights, efficient frontier, risk parity, HRP, mean-variance, CVaR optimization, factor models, covariance shrinkage, or any sklearn-style portfolio workflow. Covers skfolio 1.0: MeanRisk (with the new fallback / raise_on_failure resilience layer + FailedPortfolio), RiskBudgeting, HRP/HERC/NCO/SchurComplementary, StackingOptimization, BlackLitterman, TimeSeriesFactorModel + the new characteristics/BARRA-style CharacteristicsFactorModel (skfolio.descriptor, skfolio.factor_exposure, skfolio.containers.AssetPanel), EntropyPooling, OpinionPooling, SyntheticData, regime-adjusted EW covariance, online learning (online_predict, OnlineGridSearch), covariance forecast evaluation, cross-sectional regression, orthogonal uncertainty sets, and cross-sectional preprocessing transformers. Knows the 1.0 breaking changes: EW estimators use half_life (not alpha), factor models fit(X, factors=...) (not y), WalkForward expand_train (not expend_train), UncertaintySet radius/geometry/norm (not k/sigma), annualization_factor and non_dominated_sort renames.
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

**Covers skfolio 1.0.**

### ⚠️ 1.0 breaking changes (see `references/*` + the migration guide)

| Change | Old (0.20.x) | New (1.0) |
|---|---|---|
| EW estimators decay param | `EWMu(alpha=0.2)` | `EWMu(half_life=3.11)` — `alpha` **removed** (raises `TypeError`); `half_life = -1/log2(1-alpha)` |
| Factor-model fit contract | `fit(X, y=factors)` | `fit(X, factors=factors)` — **keyword-only** |
| `FactorModel` meaning | the prior estimator | the fitted **container** (loading matrix + moments); use `TimeSeriesFactorModel` / `CharacteristicsFactorModel` as the estimator |
| `WalkForward` expanding flag | `expend_train` | `expand_train` (typo fixed) |
| `UncertaintySet` fields | `.k`, `.sigma` | `.radius`, `.geometry`, + new `.norm` (1=diamond, 2=ellipsoid default, inf=box) |
| Deprecated → new (FutureWarning, removed 2.0) | `annualized_factor`, `non_denominated_sort` | `annualization_factor`, `non_dominated_sort` |

### New in 1.0 (additive)

- **Optimizer resilience layer** — `fallback` (estimator / list / `"previous_weights"`) + `raise_on_failure` on every optimizer; `skfolio.portfolio.FailedPortfolio` sentinel; fitted `fallback_` / `fallback_chain_` / `error_`
- **Characteristics (BARRA-style) factor stack** — `skfolio.prior.CharacteristicsFactorModel`, `skfolio.descriptor` (~46 fundamental/price descriptors), `skfolio.factor_exposure` (`GlobalFactor`, `OneHotCategoricalFactors`, `FixedWeightedFactor`, `DerivedFactor`), `skfolio.containers.AssetPanel`, `skfolio.datasets.make_synthetic_characteristics`
- **Orthogonal uncertainty sets** — `OrthogonalMuUncertaintySet`, `OrthogonalCovarianceUncertaintySet`, `CompactCovarianceUncertaintySet` (require a factor-model prior)
- **`MeanRisk` tracking constraints** — `target_weights`, `max_tracking_error`
- **`WalkForward` calendar frequencies** — `freq` (e.g. `"WOM-3FRI"`), `freq_offset`, `previous`, `reduce_test`, `purged_size`
- **Native NaN-aware estimators** — `active_mask` / `estimation_mask`, full-universe `ReturnDistribution` (non-investable → NaN, optimizer solves subset then expands)

Carried over from the 0.16–0.20 line: cross-sectional preprocessing transformers, `skfolio.linear_model` cross-sectional WLS, online learning, covariance forecast evaluation, variance estimators, `RegimeAdjustedEWCovariance`, `SchurComplementary`, `partial_fit`.

## Where to look

Keep this file open for orientation and gotchas. For deep detail jump into a topic file:

| You're working on... | Read |
|---|---|
| Picking / configuring an optimizer (MeanRisk, HRP, HERC, NCO, Schur, Stacking) + the fallback/resilience layer | `references/optimization.md` |
| Choosing a prior (BlackLitterman, EntropyPooling, TimeSeriesFactorModel, ...) | `references/priors.md` |
| Characteristics/BARRA factor models — `CharacteristicsFactorModel`, descriptors, factor exposures, `AssetPanel` | `references/factor_models.md` |
| Expected returns, variance, covariance estimators (incl. `half_life`, regime-adjusted EW) | `references/moments.md` |
| Distance, clustering, pre-selection, uncertainty sets (incl. orthogonal) | `references/distance_clustering.md` |
| WalkForward / CombinatorialPurgedCV / GridSearchCV / metadata routing | `references/model_selection.md` |
| `partial_fit`-based workflows & covariance forecast evaluation | `references/online_learning.md` |
| Cross-sectional regression (`CSLinearRegression`) | `references/linear_model.md` |
| Cross-sectional preprocessing (CSStandardScaler, CSWinsorizer, ...) | `references/preprocessing.md` |
| Wide format, `AssetPanel`, NaN-aware `active_mask` / changing universes | `references/data_representation.md` |
| Copulas & stress testing | `references/distributions.md` |
| `Portfolio`, `MultiPeriodPortfolio`, `Population`, `FailedPortfolio` return types | `references/portfolio.md` |
| Worked end-to-end examples (23 patterns) | `PATTERNS.md` |

## Official documentation

Always cross-check against upstream — the library evolves fast.

| Topic | URL |
|---|---|
| API Reference | https://skfolio.org/api.html |
| **1.0 Migration guide** | https://skfolio.org/user_guide/migration.html |
| User guide index | https://skfolio.org/user_guide/index.html |
| Factor models | https://skfolio.org/user_guide/factor_models.html |
| Data representation | https://skfolio.org/user_guide/data_representation.html |
| Online Learning | https://skfolio.org/user_guide/online_learning.html |
| Releases / changelog | https://github.com/skfolio/skfolio/releases |
| Examples gallery | https://skfolio.org/auto_examples/index.html |

## Architecture

```
skfolio/
├── optimization/        # Portfolio optimization models (+ fallback/resilience layer)
├── prior/               # Prior estimators (incl. CharacteristicsFactorModel); FactorModel = fitted container
├── moments/             # Mu, variance & covariance estimators
├── descriptor/          # NEW 1.0 — fundamental/price descriptors (BARRA-style)
├── factor_exposure/     # NEW 1.0 — factor-exposure estimators (GlobalFactor, FixedWeightedFactor, ...)
├── containers/          # NEW 1.0 — AssetPanel (wide multi-field cross-sectional data)
├── linear_model/        # Cross-sectional regression
├── distance/            # Codependence & distance estimators
├── cluster/             # Hierarchical clustering
├── uncertainty_set/     # Mu & covariance uncertainty sets (+ orthogonal / compact)
├── pre_selection/       # Asset filtering transformers
├── model_selection/     # CV, backtesting, online learning, forecast eval
├── metrics/             # Scoring functions (make_scorer)
├── preprocessing/       # prices_to_returns + cross-sectional transformers
├── distribution/        # Copulas & univariate distributions
├── datasets/            # Sample datasets (+ make_synthetic_characteristics)
├── portfolio/           # BasePortfolio, Portfolio, MultiPeriodPortfolio, FailedPortfolio
├── population/          # Population of portfolios
├── utils/               # sorting (non_dominated_sort), tools (half_life_to_decay_factor)
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
| Track a benchmark | `BenchmarkTracker()` (benchmark as `y` in `fit(X, y=...)`) | No constructor tracking arg; for an explicit cap use `MeanRisk(target_weights=..., max_tracking_error=0.01)` |
| Ensemble multiple strategies | `StackingOptimization(estimators=[...], final_estimator=...)` | Feeds base optimizers into a final allocator |
| Baseline comparison | `EqualWeighted()` or `InverseVolatility()` | Every serious backtest should compare against these |

Details: `references/optimization.md`.

### Pick a prior

| Situation | Prior |
|---|---|
| Just use history | `EmpiricalPrior(mu_estimator=..., covariance_estimator=...)` |
| Have analyst views on returns or relative performance | `BlackLitterman(views=[...])` |
| Have views on variance / correlation / skew / CVaR / groups | `EntropyPooling(mean_views=..., variance_views=..., correlation_views=..., ...)` |
| Reduce dimensionality via observed factor returns | `TimeSeriesFactorModel(factor_prior_estimator=...)` — pass factor returns as `factors=` (keyword, 1.0) |
| Build a BARRA-style model from fundamentals/characteristics | `CharacteristicsFactorModel(factors=[...])` — `fit(characteristics=AssetPanel)` (see `references/factor_models.md`) |
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

0. **1.0 breaking changes** (top of mind when upgrading from 0.20.x): EW estimators take `half_life`, not `alpha` (passing `alpha` raises `TypeError`); factor models fit with `factors=` keyword, not positional `y`; `FactorModel` is now the fitted container, use `TimeSeriesFactorModel` / `CharacteristicsFactorModel`; `WalkForward` uses `expand_train`, not `expend_train`; `UncertaintySet` fields are `radius` / `geometry` / `norm`, not `k` / `sigma`; `annualized_factor` → `annualization_factor` and `non_denominated_sort` → `non_dominated_sort` (both emit `FutureWarning`, removed in 2.0).
1. **Linear returns only** as input `X` (see above).
2. **`shuffle=False`** in any `KFold` or `train_test_split` — shuffling financial series causes lookahead leakage.
3. **Metadata routing is opt-in:** `set_config(enable_metadata_routing=True)` **before** using `.set_fit_request()`.
4. **`TimeSeriesFactorModel` uses `fit(X, factors=...)`** (keyword-only) where `X` = asset returns and `factors` = factor returns. `CharacteristicsFactorModel` uses `fit(characteristics=AssetPanel)`.
5. **`BenchmarkTracker` uses `fit(X, y)`** where `y` = benchmark returns. (Or set `target_weights` + `max_tracking_error` directly on `MeanRisk`.)
6. **Group constraints** need a `groups` dict (`{"Tech": ["AAPL", "MSFT"], ...}`) plus `linear_constraints=["Tech <= 0.4"]`.
7. **Nested parameter tuning** uses `__` syntax: `"prior_estimator__mu_estimator__half_life"`. Discover them via `model.get_params()`.
8. **`CombinatorialPurgedCV` returns `Population`**, not `MultiPeriodPortfolio`. Use `optimal_folds_number()` to pick fold sizes.
9. **`Pipeline` works** for pre-selection + optimization with standard `fit`, but **not** inside `online_predict` / `OnlineGridSearch` — those require a single estimator that implements `partial_fit`.
10. **Variance estimators** (`EmpiricalVariance`, `EWVariance`, `RegimeAdjustedEWVariance`) store results in `variance_` and are **not** drop-in replacements for covariance estimators inside priors that need a full `covariance_` matrix.
11. **Regime-adjusted EW** defaults to clipping the regime multiplier to `(0.7, 1.6)` — widen `regime_multiplier_clip` for fast-moving regimes.
12. **Cross-sectional regression** (`CSLinearRegression`) expects `X: (T, N, K)`, `y: (T, N)`, `cs_weights: (T, N)` (the weight kwarg is `cs_weights`, not `sample_weight`). Zero-weight pairs are excluded and may contain NaN.
13. **Covariance forecast evaluation is optimizer-agnostic** — rank covariance estimators with `CovarianceForecastComparison` **before** plugging one into a prior.
14. **Cross-sectional transformers** (`CSStandardScaler`, `CSWinsorizer`, `CSGaussianRankScaler`, `CSPercentileRankScaler`, `CSTanhShrinker`) operate **across assets per period** (axis=1), not across time. Use as feature preprocessors for `CSLinearRegression` or factor signals; they preserve `(T, N)` shape and skip NaN per row.

## Common imports cheat sheet

```python
from skfolio import RiskMeasure, RatioMeasure, PerfMeasure, ExtraRiskMeasure
from skfolio import Portfolio, MultiPeriodPortfolio, Population   # top-level container imports
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.prior import EmpiricalPrior, BlackLitterman, TimeSeriesFactorModel, CharacteristicsFactorModel
from skfolio.moments import LedoitWolf, EWMu, EWCovariance, RegimeAdjustedEWCovariance  # EW → half_life=
from skfolio.model_selection import WalkForward, cross_val_predict, online_predict      # WalkForward(expand_train=...)
from skfolio.preprocessing import prices_to_returns
from skfolio.datasets import load_sp500_dataset, load_factors_dataset, make_synthetic_characteristics
from skfolio.portfolio import FailedPortfolio                     # optimizer resilience sentinel
# characteristics factor stack:
from skfolio.containers import AssetPanel
from skfolio.descriptor import BookToPrice, EWMomentum, LogMarketCap
from skfolio.factor_exposure import GlobalFactor, OneHotCategoricalFactors, FixedWeightedFactor
```

For the full import surface including naive models, uncertainty sets, copulas, and datasets, see the matching reference file.

## Implementation patterns

Worked end-to-end examples — basic mean-variance, Black-Litterman, factor models, HRP, risk budgeting, stacking, pre-selection pipelines, walk-forward, hyperparameter tuning, robust optimization, synthetic data, opinion pooling, custom scoring, metadata routing, regime-adjusted covariance, online learning, covariance forecast evaluation, SchurComplementary, cross-sectional regression, full production pipeline, **plus 1.0: optimizer resilience/fallback, CharacteristicsFactorModel, orthogonal uncertainty sets** — in `PATTERNS.md`.
