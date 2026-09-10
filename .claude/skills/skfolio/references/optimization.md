# Optimization Models

All optimizers are scikit-learn estimators: `model.fit(X)` learns, `model.predict(X)` returns a `Portfolio`. Weights live in `weights_` after fit.

## Imports

```python
from skfolio.optimization import (
    # Naive
    EqualWeighted, InverseVolatility, Random,
    # Convex
    MeanRisk, BenchmarkTracker, RiskBudgeting,
    MaximumDiversification, DistributionallyRobustCVaR,
    # Clustering
    HierarchicalRiskParity, HierarchicalEqualRiskContribution,
    NestedClustersOptimization, SchurComplementary,
    # Ensemble
    StackingOptimization,
    # Enum
    ObjectiveFunction,
)
```

## Resilience layer (NEW in 1.0)

Every optimizer now accepts a failure-handling layer. Applies to walk-forward / online backtests where a single rebalance can be infeasible.

```python
model = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    fallback=None,          # single estimator | list of estimators | "previous_weights"
    raise_on_failure=True,  # default; False → warn + return FailedPortfolio
)
```

- **`fallback`** — on primary-fit failure try, in order: a single estimator, a list of estimators (first that succeeds wins), or the literal `"previous_weights"` (reuse the last good allocation).
- **`raise_on_failure`** (default `True`) — `True` re-raises after fallbacks exhausted; `False` emits a warning and `predict()` returns a `FailedPortfolio`.
- **Fitted attrs**: `fallback_` (`BaseOptimization | "previous_weights" | None`), `fallback_chain_` (`list[tuple[str, str]]` of attempts+outcomes), `error_` (`str | list[str] | None`).
- **`skfolio.portfolio.FailedPortfolio`** — sentinel returned by `predict()` on failure; carries `optimization_error` and `fallback_chain`. See `portfolio.md`.

```python
from skfolio.optimization import MeanRisk, ObjectiveFunction

model = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    fallback="previous_weights",
    raise_on_failure=False,
)
model.fit(X_train)
portfolio = model.predict(X_test)   # Portfolio, or FailedPortfolio on failure
print(model.fallback_chain_)
```

## MeanRisk

The primary convex optimizer. Solves four objective functions over any convex risk measure.

```python
model = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,  # max Sharpe
    risk_measure=RiskMeasure.CVAR,
    min_weights=0.0,           # long-only
    max_weights=0.15,          # cap per asset
    budget=1.0,                # fully invested
    prior_estimator=EmpiricalPrior(),
    mu_uncertainty_set_estimator=None,          # robust mean-risk (see distance_clustering.md)
    covariance_uncertainty_set_estimator=None,  # applied only for VARIANCE risk / max_variance
    l1_coef=0.0,
    l2_coef=0.0,
    transaction_costs=0.0,
    management_fees=0.0,
    groups=None,               # {"Tech": ["AAPL", "MSFT"], ...}
    linear_constraints=None,   # ["Tech <= 0.4", "Tech >= Health"]
    left_inequality=None,      # Aw <= b
    right_inequality=None,
    target_weights=None,       # NEW 1.0 — weight-based tracking target
    max_tracking_error=None,   # NEW 1.0 — tracking-error cap
    fallback=None,             # NEW 1.0
    raise_on_failure=True,     # NEW 1.0
)
```

Robust mean-risk is expressed on `MeanRisk` itself via `mu_uncertainty_set_estimator` / `covariance_uncertainty_set_estimator` — there is no separate wrapper class. Covariance uncertainty is only applied when `risk_measure=RiskMeasure.VARIANCE` (or `max_variance` is set).

### ObjectiveFunction

| Value | Description |
|---|---|
| `MINIMIZE_RISK` | Minimize the risk measure |
| `MAXIMIZE_RETURN` | Maximize expected return |
| `MAXIMIZE_UTILITY` | Maximize return − risk_aversion × risk |
| `MAXIMIZE_RATIO` | Maximize return / risk (e.g., Sharpe) |

### Efficient frontier

`MeanRisk(risk_measure=..., efficient_frontier_size=30)` makes `predict(X)` return a `Population` of 30 portfolios (one per frontier point). See `portfolio.md`.

### RiskMeasure (convex — usable with MeanRisk)

| Value | Description |
|---|---|
| `VARIANCE` | Portfolio variance |
| `SEMI_VARIANCE` | Downside variance |
| `STANDARD_DEVIATION` | Portfolio volatility |
| `SEMI_DEVIATION` | Downside deviation |
| `MEAN_ABSOLUTE_DEVIATION` | Mean absolute deviation |
| `FIRST_LOWER_PARTIAL_MOMENT` | First lower partial moment |
| `CVAR` | Conditional Value at Risk |
| `EVAR` | Entropic Value at Risk |
| `WORST_REALIZATION` | Worst-case scenario |
| `CDAR` | Conditional Drawdown at Risk |
| `MAX_DRAWDOWN` | Maximum drawdown |
| `AVERAGE_DRAWDOWN` | Average drawdown |
| `EDAR` | Entropic Drawdown at Risk |
| `ULCER_INDEX` | Ulcer index |
| `GINI_MEAN_DIFFERENCE` | Gini mean difference |

### ExtraRiskMeasure (non-convex — scoring only)

`VALUE_AT_RISK`, `DRAWDOWN_AT_RISK`, `ENTROPIC_RISK_MEASURE`, `FOURTH_CENTRAL_MOMENT`, `FOURTH_LOWER_PARTIAL_MOMENT`, `SKEW`, `KURTOSIS`.

### RatioMeasure

`SHARPE_RATIO`, `SORTINO_RATIO`, `CALMAR_RATIO`, `CVAR_RATIO` (+ `ANNUALIZED_SHARPE_RATIO`, ...).

### PerfMeasure

`MEAN`, `ANNUALIZED_MEAN`.

## RiskBudgeting

Allocates a risk budget across assets (equal risk contribution by default).

```python
model = RiskBudgeting(
    risk_measure=RiskMeasure.CVAR,
    risk_budget=None,          # None ⇒ equal contribution
    prior_estimator=EmpiricalPrior(),
    min_weights=0.0, max_weights=1.0,
)
```

## MaximumDiversification

Maximizes the diversification ratio. (Class name is `MaximumDiversification`.)

```python
model = MaximumDiversification(prior_estimator=EmpiricalPrior())
```

## DistributionallyRobustCVaR

Minimizes worst-case CVaR within a Wasserstein ball.

```python
model = DistributionallyRobustCVaR(
    risk_aversion=1.0,
    wasserstein_ball_radius=0.02,
    prior_estimator=EmpiricalPrior(),
)
```

## HierarchicalRiskParity (HRP)

Hierarchical clustering with recursive bisection.

```python
model = HierarchicalRiskParity(
    risk_measure=RiskMeasure.CVAR,
    prior_estimator=EmpiricalPrior(),
    distance_estimator=PearsonDistance(),
    hierarchical_clustering_estimator=HierarchicalClustering(),
)
```

## HierarchicalEqualRiskContribution (HERC)

Top-down dendrogram division for equal risk contribution.

```python
model = HierarchicalEqualRiskContribution(
    risk_measure=RiskMeasure.CDAR,
    distance_estimator=PearsonDistance(),
    hierarchical_clustering_estimator=HierarchicalClustering(),
)
```

## NestedClustersOptimization (NCO)

Inner and outer optimization via clustering.

```python
model = NestedClustersOptimization(
    inner_estimator=MeanRisk(),
    outer_estimator=MeanRisk(),
    distance_estimator=PearsonDistance(),
    clustering_estimator=HierarchicalClustering(),
    cv=None, n_jobs=None,
)
```

## SchurComplementary

Schur-complement-inspired hierarchical allocator that interpolates between HRP (`gamma=0`) and minimum-variance (`gamma→1`).

```python
from skfolio.optimization import SchurComplementary

model = SchurComplementary(
    gamma=0.5,                 # 0 → HRP, 1 → MVP
    keep_monotonic=True,       # guard against ill-conditioned cov
    prior_estimator=EmpiricalPrior(),
    distance_estimator=PearsonDistance(),
    hierarchical_clustering_estimator=HierarchicalClustering(),
    min_weights=0.0, max_weights=1.0,
)
```

Tune `gamma` via `GridSearchCV` to trade off HRP robustness against MVP efficiency.

## StackingOptimization

Ensemble that feeds outputs of several optimizers into a final allocator.

```python
model = StackingOptimization(
    estimators=[
        ("hrp", HierarchicalRiskParity()),
        ("meanrisk", MeanRisk()),
    ],
    final_estimator=MeanRisk(),
    cv=None, n_jobs=None,
)
```

## BenchmarkTracker

Minimizes tracking error vs. a benchmark return series.

```python
model = BenchmarkTracker(risk_measure=RiskMeasure.VARIANCE)
model.fit(X, y=benchmark_returns)       # y is required
```

> Tracking-style constraints can now also be expressed directly on `MeanRisk` via `target_weights` + `max_tracking_error`.

## Naive Models

```python
EqualWeighted()                # 1/N
InverseVolatility()            # inverse-vol weighting
Random()                       # random Dirichlet weights
```
