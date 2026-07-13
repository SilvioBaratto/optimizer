# Optimization Models

All optimizers are scikit-learn estimators: `model.fit(X)` learns, `model.predict(X)` returns a `Portfolio`.

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
    NestedClustersOptimization, SchurComplementary,   # v0.17.0+
    # Ensemble
    StackingOptimization,
    # Enum
    ObjectiveFunction,
)
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
    mu_uncertainty_set_estimator=None,
    covariance_uncertainty_set_estimator=None,
    l1_coef=0.0,
    l2_coef=0.0,
    transaction_costs=0.0,
    management_fees=0.0,
    groups=None,               # {"Tech": ["AAPL", "MSFT"], ...}
    linear_constraints=None,   # ["Tech <= 0.4", "Tech >= Health"]
    left_inequality=None,      # Aw <= b
    right_inequality=None,
)
```

### ObjectiveFunction

| Value | Description |
|---|---|
| `MINIMIZE_RISK` | Minimize the risk measure |
| `MAXIMIZE_RETURN` | Maximize expected return |
| `MAXIMIZE_UTILITY` | Maximize return − risk_aversion × risk |
| `MAXIMIZE_RATIO` | Maximize return / risk (e.g., Sharpe) |

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
| `MAXIMUM_DRAWDOWN` | Maximum drawdown |
| `AVERAGE_DRAWDOWN` | Average drawdown |
| `EDAR` | Entropic Drawdown at Risk |
| `ULCER_INDEX` | Ulcer index |
| `GINI_MEAN_DIFFERENCE` | Gini mean difference |

### ExtraRiskMeasure (non-convex — scoring only)

| Value | Description |
|---|---|
| `VALUE_AT_RISK` | VaR |
| `DRAWDOWN_AT_RISK` | Drawdown VaR |
| `ENTROPIC_RISK_MEASURE` | Entropic risk |
| `FOURTH_CENTRAL_MOMENT` | Kurtosis proxy |
| `FOURTH_LOWER_PARTIAL_MOMENT` | Downside kurtosis |
| `SKEW` | Portfolio skewness |
| `KURTOSIS` | Portfolio kurtosis |

### RatioMeasure

| Value | Description |
|---|---|
| `SHARPE_RATIO` | Return / StdDev |
| `SORTINO_RATIO` | Return / Downside deviation |
| `CALMAR_RATIO` | Return / Max drawdown |
| `CVAR_RATIO` | Return / CVaR |

### PerfMeasure

| Value | Description |
|---|---|
| `MEAN` | Mean return |
| `ANNUALIZED_MEAN` | Annualized mean |

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

Maximizes the diversification ratio.

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
    hierarchical_clustering_estimator=HierarchicalClustering(),
)
```

## SchurComplementary (v0.17.0+)

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
)
```

## BenchmarkTracker

Minimizes tracking error vs. a benchmark return series.

```python
model = BenchmarkTracker(tracking_error_target=0.01)
model.fit(X, y=benchmark_returns)       # y is required
```

## Naive Models

```python
EqualWeighted()                # 1/N
InverseVolatility()            # inverse-vol weighting
Random(n_portfolios=100)       # random portfolios
```
