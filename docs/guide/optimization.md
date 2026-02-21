# Optimization

Comprehensive guide to the portfolio optimization module. This module provides 10+ optimizer models spanning convex programming, hierarchical clustering, ensemble methods, robust formulations, and naive baselines. Every model follows the same pattern: **frozen `@dataclass` config** + **factory function** + **`str, Enum` types**.

---

## Architecture

All optimizers produce sklearn-compatible estimators that expose `fit(X)` / `predict(X)` and compose into `sklearn.pipeline.Pipeline`. The config/factory split enforces a strict boundary:

- **Config** (`@dataclass(frozen=True)`) -- holds only primitives, enums, and nested frozen dataclasses (serializable, hashable).
- **Factory function** -- accepts the config plus any non-serializable objects (prior estimators, numpy arrays, constraint matrices) as keyword arguments.

```python
from optimizer.optimization import MeanRiskConfig, build_mean_risk

# Config: serializable, hashable, suitable for storage/logging
config = MeanRiskConfig.for_max_sharpe()

# Factory: builds the skfolio estimator, accepts non-serializable kwargs
model = build_mean_risk(config, prior_estimator=my_prior)
model.fit(X)
portfolio = model.predict(X)
```

---

## Model Overview

| Model | Config | Factory | Category |
|-------|--------|---------|----------|
| Mean-Risk | `MeanRiskConfig` | `build_mean_risk()` | Convex |
| Risk Budgeting | `RiskBudgetingConfig` | `build_risk_budgeting()` | Convex |
| Max Diversification | `MaxDiversificationConfig` | `build_max_diversification()` | Convex |
| HRP | `HRPConfig` | `build_hrp()` | Hierarchical |
| HERC | `HERCConfig` | `build_herc()` | Hierarchical |
| NCO | `NCOConfig` | `build_nco()` | Hierarchical |
| Benchmark Tracker | `BenchmarkTrackerConfig` | `build_benchmark_tracker()` | Convex |
| Equal Weighted | `EqualWeightedConfig` | `build_equal_weighted()` | Naive |
| Inverse Volatility | `InverseVolatilityConfig` | `build_inverse_volatility()` | Naive |
| Stacking | `StackingConfig` | `build_stacking()` | Ensemble |
| Robust Mean-Risk | `RobustConfig` | `build_robust_mean_risk()` | Robust |
| DR-CVaR | `DRCVaRConfig` | `build_dr_cvar()` | Robust |
| Regime-Blended | `RegimeRiskConfig` | `build_regime_blended_optimizer()` | Regime |

---

## Enums

### ObjectiveFunctionType

Controls what the convex optimizer maximizes or minimizes.

| Value | Description |
|-------|-------------|
| `MINIMIZE_RISK` | Minimize the chosen risk measure subject to constraints |
| `MAXIMIZE_RETURN` | Maximize expected return subject to a risk budget |
| `MAXIMIZE_UTILITY` | Maximize \( \mu^\top w - \frac{\lambda}{2} \rho(w) \) where \( \lambda \) is `risk_aversion` |
| `MAXIMIZE_RATIO` | Maximize the return/risk ratio (e.g. Sharpe ratio) |

### RiskMeasureType

Fifteen convex risk measures available for `MeanRisk`, `RiskBudgeting`, `BenchmarkTracker`, and other convex optimizers.

| Value | Mathematical Definition |
|-------|------------------------|
| `VARIANCE` | \( \sigma^2 = w^\top \Sigma w \) |
| `SEMI_VARIANCE` | Variance computed only on below-mean returns |
| `STANDARD_DEVIATION` | \( \sigma = \sqrt{w^\top \Sigma w} \) |
| `SEMI_DEVIATION` | Standard deviation of below-mean returns |
| `MEAN_ABSOLUTE_DEVIATION` | \( \text{MAD} = \mathbb{E}[\lvert r_p - \mu_p \rvert] \) |
| `FIRST_LOWER_PARTIAL_MOMENT` | \( \text{FLPM} = \mathbb{E}[\max(0, \tau - r_p)] \) |
| `CVAR` | \( \text{CVaR}_\alpha = -\frac{1}{1-\alpha}\int_0^{1-\alpha} F^{-1}(u)\,du \) |
| `EVAR` | Entropic Value at Risk (tightest upper bound on CVaR from Chernoff inequality) |
| `WORST_REALIZATION` | \( \max_t(-r_{p,t}) \) -- the worst single-period loss |
| `CDAR` | Conditional Drawdown at Risk (CVaR applied to the drawdown distribution) |
| `MAX_DRAWDOWN` | Maximum peak-to-trough decline |
| `AVERAGE_DRAWDOWN` | Mean of the drawdown series |
| `EDAR` | Entropic Drawdown at Risk |
| `ULCER_INDEX` | \( \sqrt{\frac{1}{T}\sum_t d_t^2} \) where \( d_t \) is the drawdown at time \( t \) |
| `GINI_MEAN_DIFFERENCE` | \( \text{GMD} = \frac{1}{T^2}\sum_{i \neq j}\lvert r_i - r_j \rvert \) |

### ExtraRiskMeasureType

Seven non-convex risk measures available exclusively for hierarchical methods (HRP, HERC) which do not require convexity.

| Value | Description |
|-------|-------------|
| `VALUE_AT_RISK` | \( \text{VaR}_\alpha = -F^{-1}(1-\alpha) \) |
| `DRAWDOWN_AT_RISK` | VaR applied to the drawdown distribution |
| `ENTROPIC_RISK_MEASURE` | Entropic risk measure |
| `FOURTH_CENTRAL_MOMENT` | \( \mathbb{E}[(r - \mu)^4] \) -- captures kurtosis |
| `FOURTH_LOWER_PARTIAL_MOMENT` | Fourth moment of below-mean returns |
| `SKEW` | Third standardized central moment |
| `KURTOSIS` | Fourth standardized central moment |

### DistanceType

Distance metrics for hierarchical clustering in HRP, HERC, and NCO.

| Value | Description |
|-------|-------------|
| `PEARSON` | \( d_{ij} = \sqrt{\frac{1}{2}(1 - \rho_{ij})} \) (default) |
| `KENDALL` | Kendall rank correlation distance |
| `SPEARMAN` | Spearman rank correlation distance |
| `COVARIANCE` | Covariance-based distance (requires a covariance estimator) |
| `DISTANCE_CORRELATION` | Non-linear distance correlation (captures non-linear dependencies) |
| `MUTUAL_INFORMATION` | Information-theoretic distance |

### LinkageMethodType

Linkage methods for agglomerative hierarchical clustering.

| Value | Description |
|-------|-------------|
| `WARD` | Minimize within-cluster variance (default, requires Euclidean distance) |
| `SINGLE` | Nearest-neighbor linkage |
| `COMPLETE` | Farthest-neighbor linkage |
| `AVERAGE` | Average linkage (UPGMA) |
| `WEIGHTED` | Weighted average linkage (WPGMA) |
| `CENTROID` | Centroid linkage |
| `MEDIAN` | Median linkage |

### RatioMeasureType

Ratio measures for scoring and ensemble quantile selection. Includes 18 standard skfolio ratio measures plus a custom `INFORMATION_RATIO` (active return / tracking error).

---

## Sub-Configs

### DistanceConfig

Configures the distance estimator used by hierarchical methods.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `distance_type` | `DistanceType` | `PEARSON` | Distance metric |
| `absolute` | `bool` | `False` | Apply absolute transformation to correlation matrix |
| `power` | `float` | `1.0` | Power transformation exponent |
| `threshold` | `float` | `0.5` | Distance correlation threshold (only for `DISTANCE_CORRELATION`) |

```python
from optimizer.optimization import DistanceConfig, DistanceType

# Spearman distance (robust to outliers)
dist_cfg = DistanceConfig(
    distance_type=DistanceType.SPEARMAN,
    absolute=True,
)
```

### ClusteringConfig

Configures hierarchical clustering used by HRP, HERC, and NCO.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_clusters` | `int or None` | `None` | Maximum number of flat clusters. `None` uses the Two-Order Difference Gap Statistic heuristic |
| `linkage_method` | `LinkageMethodType` | `WARD` | Linkage method for the dendrogram |

```python
from optimizer.optimization import ClusteringConfig, LinkageMethodType

cluster_cfg = ClusteringConfig(
    max_clusters=5,
    linkage_method=LinkageMethodType.COMPLETE,
)
```

---

## 1. Mean-Risk Optimization (MeanRiskConfig)

The workhorse of the module. Solves the general convex mean-risk program:

\[
\begin{aligned}
&\underset{w}{\text{minimize}} && \rho(w) \\
&\text{subject to} && \mu^\top w \geq \text{target}, \quad \mathbf{1}^\top w = b, \quad w_{\min} \leq w \leq w_{\max}
\end{aligned}
\]

where \( \rho \) is any convex risk measure from `RiskMeasureType`.

### Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `objective` | `ObjectiveFunctionType` | `MINIMIZE_RISK` | Objective function |
| `risk_measure` | `RiskMeasureType` | `VARIANCE` | Risk measure |
| `risk_aversion` | `float` | `1.0` | Risk-aversion coefficient for `MAXIMIZE_UTILITY` |
| `efficient_frontier_size` | `int or None` | `None` | Number of points on the efficient frontier (`None` = single portfolio) |
| `min_weights` | `float or None` | `0.0` | Lower bound on asset weights |
| `max_weights` | `float or None` | `1.0` | Upper bound on asset weights |
| `budget` | `float or None` | `1.0` | Portfolio budget (sum of weights) |
| `max_short` | `float or None` | `None` | Maximum short position |
| `max_long` | `float or None` | `None` | Maximum long position |
| `cardinality` | `int or None` | `None` | Maximum number of assets |
| `transaction_costs` | `float` | `0.0` | Linear transaction costs |
| `management_fees` | `float` | `0.0` | Linear management fees |
| `max_tracking_error` | `float or None` | `None` | Maximum tracking error vs benchmark |
| `l1_coef` | `float` | `0.0` | L1 regularization coefficient (promotes sparsity) |
| `l2_coef` | `float` | `0.0` | L2 regularization coefficient (shrinks weights toward zero) |
| `risk_free_rate` | `float` | `0.0` | Risk-free rate for ratio objectives |
| `cvar_beta` | `float` | `0.95` | CVaR confidence level |
| `evar_beta` | `float` | `0.95` | EVaR confidence level |
| `cdar_beta` | `float` | `0.95` | CDaR confidence level |
| `edar_beta` | `float` | `0.95` | EDaR confidence level |
| `solver` | `str` | `"CLARABEL"` | CVXPY solver name |
| `solver_params` | `dict or None` | `None` | Additional solver parameters |
| `prior_config` | `MomentEstimationConfig or None` | `None` | Inner prior configuration |

### Presets

```python
from optimizer.optimization import MeanRiskConfig

# Minimum-variance portfolio
config = MeanRiskConfig.for_min_variance()

# Maximum Sharpe ratio
config = MeanRiskConfig.for_max_sharpe()

# Maximum utility with custom risk aversion
config = MeanRiskConfig.for_max_utility(risk_aversion=2.0)

# Minimum CVaR at 99% confidence
config = MeanRiskConfig.for_min_cvar(beta=0.99)

# Efficient frontier with 30 points
config = MeanRiskConfig.for_efficient_frontier(size=30)
```

### Factory: build_mean_risk()

```python
from optimizer.optimization import MeanRiskConfig, build_mean_risk

# Basic usage
model = build_mean_risk(MeanRiskConfig.for_max_sharpe())
model.fit(X)
portfolio = model.predict(X)

# With prior estimator and factor constraints
from optimizer.moments import build_prior, MomentEstimationConfig
from optimizer.factors import build_factor_exposure_constraints

prior = build_prior(MomentEstimationConfig.for_shrunk_denoised())
constraints = build_factor_exposure_constraints(...)

model = build_mean_risk(
    config,
    prior_estimator=prior,
    factor_exposure_constraints=constraints,
    previous_weights=old_weights,  # for transaction cost optimization
)
```

**Factory kwargs** (non-serializable, not stored in config):
- `prior_estimator` -- skfolio `BasePrior` instance
- `factor_exposure_constraints` -- `FactorExposureConstraints` (injects `left_inequality` / `right_inequality`)
- `previous_weights` -- numpy array for turnover-aware optimization
- `groups` -- asset group labels
- `linear_constraints` -- additional linear constraints

### Short-Selling Example

```python
config = MeanRiskConfig(
    objective=ObjectiveFunctionType.MAXIMIZE_RATIO,
    risk_measure=RiskMeasureType.VARIANCE,
    min_weights=-0.3,    # allow up to 30% short per asset
    max_weights=0.5,     # max 50% long per asset
    max_short=0.5,       # total short exposure <= 50%
    max_long=1.5,        # total long exposure <= 150%
    budget=1.0,          # net exposure = 100%
)
```

### Cardinality-Constrained Example

```python
config = MeanRiskConfig(
    objective=ObjectiveFunctionType.MINIMIZE_RISK,
    risk_measure=RiskMeasureType.VARIANCE,
    cardinality=15,      # at most 15 assets
    l1_coef=0.001,       # L1 regularization for sparsity
)
```

---

## 2. Risk Budgeting (RiskBudgetingConfig)

Risk parity and generalized risk budgeting. Each asset contributes a pre-specified share of total portfolio risk:

\[
w_i \frac{\partial \rho(w)}{\partial w_i} = b_i \cdot \rho(w), \quad i = 1, \ldots, n
\]

where \( b_i \) is the risk budget for asset \( i \) (summing to 1). When \( b_i = 1/n \) for all \( i \), this is **risk parity**.

### Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `risk_measure` | `RiskMeasureType` | `VARIANCE` | Risk measure |
| `min_weights` | `float or None` | `0.0` | Lower bound on asset weights |
| `max_weights` | `float or None` | `1.0` | Upper bound on asset weights |
| `risk_free_rate` | `float` | `0.0` | Risk-free rate |
| `cvar_beta` | `float` | `0.95` | CVaR confidence level |
| `evar_beta` | `float` | `0.95` | EVaR confidence level |
| `cdar_beta` | `float` | `0.95` | CDaR confidence level |
| `edar_beta` | `float` | `0.95` | EDaR confidence level |
| `solver` | `str` | `"CLARABEL"` | CVXPY solver name |
| `solver_params` | `dict or None` | `None` | Additional solver parameters |
| `prior_config` | `MomentEstimationConfig or None` | `None` | Inner prior configuration |

### Presets

```python
from optimizer.optimization import RiskBudgetingConfig

# Equal risk contribution (risk parity) with variance
config = RiskBudgetingConfig.for_risk_parity()

# Risk parity with CVaR
config = RiskBudgetingConfig.for_cvar_parity(beta=0.95)

# Risk parity with CDaR
config = RiskBudgetingConfig.for_cdar_parity(beta=0.95)
```

### Factory: build_risk_budgeting()

The `risk_budget` array is passed as a factory kwarg because numpy arrays are not hashable in frozen dataclasses.

```python
import numpy as np
from optimizer.optimization import RiskBudgetingConfig, build_risk_budgeting

# Equal risk parity (default when risk_budget=None)
model = build_risk_budgeting(RiskBudgetingConfig.for_risk_parity())
model.fit(X)

# Custom risk budgets: 60% risk to equities, 40% to bonds
budgets = np.array([0.15, 0.15, 0.15, 0.15, 0.20, 0.20])
model = build_risk_budgeting(
    RiskBudgetingConfig.for_cvar_parity(),
    risk_budget=budgets,
)
```

**Gotcha**: When `risk_budget=None`, skfolio assigns equal budgets (1/n per asset). You do not need to manually construct the equal-weight array.

---

## 3. Maximum Diversification (MaxDiversificationConfig)

Maximizes the diversification ratio:

\[
\text{DR}(w) = \frac{w^\top \sigma}{\sqrt{w^\top \Sigma w}}
\]

where \( \sigma \) is the vector of individual asset volatilities and \( \Sigma \) is the covariance matrix.

### Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `min_weights` | `float or None` | `0.0` | Lower bound on asset weights |
| `max_weights` | `float or None` | `1.0` | Upper bound on asset weights |
| `budget` | `float or None` | `1.0` | Portfolio budget |
| `max_short` | `float or None` | `None` | Maximum short position |
| `max_long` | `float or None` | `None` | Maximum long position |
| `cardinality` | `int or None` | `None` | Maximum number of assets |
| `l1_coef` | `float` | `0.0` | L1 regularization |
| `l2_coef` | `float` | `0.0` | L2 regularization |
| `risk_free_rate` | `float` | `0.0` | Risk-free rate |
| `solver` | `str` | `"CLARABEL"` | CVXPY solver |
| `solver_params` | `dict or None` | `None` | Solver parameters |
| `prior_config` | `MomentEstimationConfig or None` | `None` | Prior configuration |

### Usage

```python
from optimizer.optimization import MaxDiversificationConfig, build_max_diversification

config = MaxDiversificationConfig(l2_coef=0.01)
model = build_max_diversification(config)
model.fit(X)
```

---

## 4. Hierarchical Risk Parity -- HRP (HRPConfig)

Hierarchical Risk Parity (Lopez de Prado, 2016) avoids matrix inversion entirely. It builds a hierarchical clustering dendrogram from asset distances, then allocates risk by recursively bisecting the dendrogram and inverse-variance weighting each split.

### Algorithm Steps

1. Compute a distance matrix from asset returns (e.g. Pearson correlation distance)
2. Build a hierarchical clustering dendrogram using a linkage method
3. Quasi-diagonalize the covariance matrix according to the dendrogram ordering
4. Recursively bisect the dendrogram, allocating weights inversely proportional to cluster risk at each split

### Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `risk_measure` | `RiskMeasureType` | `VARIANCE` | Convex risk measure |
| `extra_risk_measure` | `ExtraRiskMeasureType or None` | `None` | Non-convex risk measure (overrides `risk_measure` when set) |
| `min_weights` | `float or None` | `0.0` | Lower bound on asset weights |
| `max_weights` | `float or None` | `1.0` | Upper bound on asset weights |
| `distance_config` | `DistanceConfig or None` | `None` | Distance estimator configuration |
| `clustering_config` | `ClusteringConfig or None` | `None` | Clustering configuration |
| `prior_config` | `MomentEstimationConfig or None` | `None` | Prior configuration |

### Presets

```python
from optimizer.optimization import HRPConfig

config = HRPConfig.for_variance()
config = HRPConfig.for_cvar()
```

### Usage with Custom Distance and Clustering

```python
from optimizer.optimization import (
    HRPConfig,
    build_hrp,
    DistanceConfig,
    DistanceType,
    ClusteringConfig,
    LinkageMethodType,
    ExtraRiskMeasureType,
)

config = HRPConfig(
    risk_measure=RiskMeasureType.CVAR,
    distance_config=DistanceConfig(
        distance_type=DistanceType.SPEARMAN,
        absolute=True,
    ),
    clustering_config=ClusteringConfig(
        max_clusters=5,
        linkage_method=LinkageMethodType.COMPLETE,
    ),
)

model = build_hrp(config)
model.fit(X)
```

### Using Non-Convex Risk Measures

HRP and HERC support non-convex risk measures via `extra_risk_measure`. When set, it overrides `risk_measure`:

```python
config = HRPConfig(
    extra_risk_measure=ExtraRiskMeasureType.VALUE_AT_RISK,
)
```

---

## 5. Hierarchical Equal Risk Contribution -- HERC (HERCConfig)

HERC (Thomas et al., 2018) extends HRP by equalizing risk contributions within each cluster, similar to risk budgeting but applied to the hierarchical tree structure. Unlike HRP, HERC can use a solver for the intra-cluster allocation step.

### Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `risk_measure` | `RiskMeasureType` | `VARIANCE` | Convex risk measure |
| `extra_risk_measure` | `ExtraRiskMeasureType or None` | `None` | Non-convex risk measure (overrides `risk_measure`) |
| `min_weights` | `float or None` | `0.0` | Lower bound |
| `max_weights` | `float or None` | `1.0` | Upper bound |
| `solver` | `str` | `"CLARABEL"` | CVXPY solver |
| `solver_params` | `dict or None` | `None` | Solver parameters |
| `distance_config` | `DistanceConfig or None` | `None` | Distance configuration |
| `clustering_config` | `ClusteringConfig or None` | `None` | Clustering configuration |
| `prior_config` | `MomentEstimationConfig or None` | `None` | Prior configuration |

### Presets

```python
from optimizer.optimization import HERCConfig

config = HERCConfig.for_variance()
config = HERCConfig.for_cvar()
```

### Usage

```python
from optimizer.optimization import HERCConfig, build_herc

config = HERCConfig(
    risk_measure=RiskMeasureType.CVAR,
    clustering_config=ClusteringConfig(max_clusters=4),
)
model = build_herc(config)
model.fit(X)
```

---

## 6. Nested Clusters Optimization -- NCO (NCOConfig)

NCO (Lopez de Prado, 2019) addresses the instability of mean-variance by decomposing the optimization into intra-cluster and inter-cluster stages:

1. Cluster assets using hierarchical clustering
2. Run an **inner optimizer** within each cluster
3. Run an **outer optimizer** across the cluster-level portfolios

### Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `quantile` | `float` | `0.5` | Quantile for portfolio selection across CV folds |
| `n_jobs` | `int or None` | `None` | Number of parallel jobs |
| `distance_config` | `DistanceConfig or None` | `None` | Distance configuration |
| `clustering_config` | `ClusteringConfig or None` | `None` | Clustering configuration |

### Usage

The inner and outer estimators are passed as factory kwargs because they are not serializable:

```python
from optimizer.optimization import (
    NCOConfig,
    build_nco,
    build_mean_risk,
    MeanRiskConfig,
)

inner = build_mean_risk(MeanRiskConfig.for_min_variance())
outer = build_mean_risk(MeanRiskConfig.for_max_sharpe())

config = NCOConfig(quantile=0.5)
model = build_nco(
    config,
    inner_estimator=inner,
    outer_estimator=outer,
)
model.fit(X)
```

---

## 7. Benchmark Tracker (BenchmarkTrackerConfig)

Minimizes tracking error against a benchmark index. The benchmark returns are passed as `y` in `fit(X, y)`.

### Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `risk_measure` | `RiskMeasureType` | `STANDARD_DEVIATION` | Risk measure for tracking error |
| `min_weights` | `float or None` | `0.0` | Lower bound |
| `max_weights` | `float or None` | `1.0` | Upper bound |
| `max_short` | `float or None` | `None` | Maximum short |
| `max_long` | `float or None` | `None` | Maximum long |
| `cardinality` | `int or None` | `None` | Maximum assets |
| `transaction_costs` | `float` | `0.0` | Transaction costs |
| `management_fees` | `float` | `0.0` | Management fees |
| `l1_coef` | `float` | `0.0` | L1 regularization |
| `l2_coef` | `float` | `0.0` | L2 regularization |
| `risk_free_rate` | `float` | `0.0` | Risk-free rate |
| `solver` | `str` | `"CLARABEL"` | CVXPY solver |
| `solver_params` | `dict or None` | `None` | Solver parameters |
| `prior_config` | `MomentEstimationConfig or None` | `None` | Prior configuration |

### Usage

```python
from optimizer.optimization import BenchmarkTrackerConfig, build_benchmark_tracker

config = BenchmarkTrackerConfig(
    cardinality=50,       # replicate benchmark with at most 50 stocks
    l1_coef=0.001,        # sparse tracking
)
model = build_benchmark_tracker(config)

# benchmark_returns is a 1-D array/Series aligned with X
model.fit(X, y=benchmark_returns)
portfolio = model.predict(X)
```

**Gotcha**: Benchmark returns must be passed as `y` in `fit(X, y)`, not as part of the config or factory kwargs.

---

## 8. Equal Weighted (EqualWeightedConfig)

The naive 1/N allocation. Assigns identical weight \( w_i = 1/N \) to each asset. No estimation is required, making it immune to estimation error. Serves as a strong baseline that is surprisingly hard to beat out of sample (DeMiguel et al., 2009).

### Usage

```python
from optimizer.optimization import EqualWeightedConfig, build_equal_weighted

model = build_equal_weighted(EqualWeightedConfig())
model.fit(X)
```

`EqualWeightedConfig` has no parameters.

---

## 9. Inverse Volatility (InverseVolatilityConfig)

Weights each asset inversely proportional to its estimated volatility:

\[
w_i = \frac{1/\hat{\sigma}_i}{\sum_{j=1}^{N} 1/\hat{\sigma}_j}
\]

The volatility estimates come from the diagonal of the covariance matrix provided by the prior estimator.

### Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prior_config` | `MomentEstimationConfig or None` | `None` | Prior configuration (covariance estimator determines volatility) |

### Usage

```python
from optimizer.optimization import InverseVolatilityConfig, build_inverse_volatility
from optimizer.moments import MomentEstimationConfig

config = InverseVolatilityConfig(
    prior_config=MomentEstimationConfig.for_shrunk_denoised(),
)
model = build_inverse_volatility(config)
model.fit(X)
```

---

## 10. Stacking Optimization (StackingConfig)

Ensemble method that combines multiple sub-optimizers via a meta-optimizer. Each sub-optimizer produces a portfolio, and the meta-optimizer allocates across those portfolios.

### Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `quantile` | `float` | `0.5` | Quantile for portfolio selection across CV folds |
| `quantile_measure` | `RatioMeasureType` | `SHARPE_RATIO` | Ratio measure for quantile selection |
| `n_jobs` | `int or None` | `None` | Number of parallel jobs |
| `cv` | `int or None` | `None` | Cross-validation folds (`None` = no CV) |

### Usage

The `estimators` list and `final_estimator` are passed as factory kwargs:

```python
from optimizer.optimization import (
    StackingConfig,
    build_stacking,
    build_mean_risk,
    build_hrp,
    MeanRiskConfig,
    HRPConfig,
    RatioMeasureType,
)

sub_optimizers = [
    ("min_var", build_mean_risk(MeanRiskConfig.for_min_variance())),
    ("max_sharpe", build_mean_risk(MeanRiskConfig.for_max_sharpe())),
    ("hrp", build_hrp(HRPConfig.for_variance())),
]

meta = build_mean_risk(MeanRiskConfig.for_min_variance())

config = StackingConfig(
    quantile=0.5,
    quantile_measure=RatioMeasureType.SHARPE_RATIO,
    cv=5,
)
model = build_stacking(
    config,
    estimators=sub_optimizers,
    final_estimator=meta,
)
model.fit(X)
```

**Default estimators**: When `estimators=None`, the factory defaults to `[("mean_risk", MeanRisk()), ("hrp", HierarchicalRiskParity())]`.

---

## Robust Variants

### 11. Robust Mean-Risk (RobustConfig)

Hedges against estimation error in the expected return vector by constructing an ellipsoidal uncertainty set around the sample mean and optimizing for the worst-case expected return within that set.

#### Uncertainty Set for Expected Returns

The ellipsoidal uncertainty set is:

\[
U_\mu = \left\{ \mu : (\mu - \hat{\mu})^\top S_\mu^{-1} (\mu - \hat{\mu}) \leq \kappa^2 \right\}
\]

where:
- \( \hat{\mu} \) is the estimated mean vector (sample or shrinkage)
- \( S_\mu = \hat{\Sigma} / T \) is the estimation error covariance of the sample mean
- \( \kappa \) is the robustness parameter (larger values produce more conservative, diversified portfolios)

The worst-case expected return within \( U_\mu \) is:

\[
\min_{\mu \in U_\mu} \mu^\top w = \hat{\mu}^\top w - \kappa \cdot \| S_\mu^{1/2} w \|_2
\]

The penalty term \( \kappa \cdot \| S_\mu^{1/2} w \|_2 \) grows with the portfolio's exposure to estimation uncertainty, naturally encouraging diversification.

#### Kappa-Confidence Level Mapping

The parameter \( \kappa \) relates to the chi-squared confidence level via:

\[
\kappa^2 = \chi^2_{n}(\beta) \quad \Longleftrightarrow \quad \beta = F_{\chi^2_n}(\kappa^2)
\]

where \( n \) is the number of assets. Since \( n \) is only known at fit time, the conversion from \( \kappa \) to `confidence_level` is deferred to the `fit()` call.

#### Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `kappa` | `float` | `1.0` | Ellipsoidal uncertainty radius for \( \mu \). `kappa=0` recovers standard MeanRisk exactly |
| `cov_uncertainty` | `bool` | `False` | Also apply covariance uncertainty set |
| `cov_uncertainty_method` | `str` | `"bootstrap"` | `"bootstrap"` (stationary block bootstrap) or `"empirical"` (formula-based) |
| `B` | `int` | `500` | Number of bootstrap resamples (only for `"bootstrap"` method) |
| `block_size` | `int` | `21` | Expected block length for stationary bootstrap (~1 trading month) |
| `bootstrap_alpha` | `float` | `0.05` | Significance level for covariance uncertainty ellipsoid |
| `mean_risk_config` | `MeanRiskConfig or None` | `None` | Embedded mean-risk configuration |

#### Presets

| Preset | kappa | cov_uncertainty | Use Case |
|--------|-------|-----------------|----------|
| `for_conservative()` | 2.0 | `False` | High estimation uncertainty (short history, non-stationary) |
| `for_moderate()` | 1.0 | `False` | Balanced trade-off |
| `for_aggressive()` | 0.5 | `False` | Closer to standard MeanRisk |
| `for_bootstrap_covariance()` | 1.0 | `True` | Hedges against both mean and covariance estimation error |

#### Usage

```python
from optimizer.optimization import RobustConfig, build_robust_mean_risk, MeanRiskConfig

# Conservative: strong robustness
model = build_robust_mean_risk(RobustConfig.for_conservative())
model.fit(X)

# kappa=0: identical to standard MeanRisk (no penalty)
baseline = build_robust_mean_risk(RobustConfig(kappa=0.0))

# Robust max-Sharpe with bootstrap covariance uncertainty
config = RobustConfig(
    kappa=1.5,
    cov_uncertainty=True,
    cov_uncertainty_method="bootstrap",
    B=1000,
    block_size=21,
    mean_risk_config=MeanRiskConfig.for_max_sharpe(),
)
model = build_robust_mean_risk(config)
model.fit(X)
```

#### Standalone Bootstrap Covariance Utility

The module also exposes `bootstrap_covariance_uncertainty()` for standalone analysis of covariance estimation uncertainty:

```python
from optimizer.optimization import bootstrap_covariance_uncertainty

result = bootstrap_covariance_uncertainty(
    returns,
    B=500,
    block_size=21,
    alpha=0.05,
    seed=42,
)
print(f"Frobenius-norm confidence radius: {result.delta:.4f}")
print(f"Sample covariance shape: {result.cov_hat.shape}")
print(f"Bootstrap samples shape: {result.cov_samples.shape}")  # (500, n, n)
```

The Frobenius-norm confidence set is \( \{ \Sigma : \| \Sigma - \hat{\Sigma} \|_F \leq \delta \} \) where \( \delta \) is the \( (1-\alpha) \) quantile of bootstrap Frobenius distances.

---

### 12. Distributionally Robust CVaR (DRCVaRConfig)

Minimizes the worst-case CVaR over all probability distributions within a Wasserstein ball of radius \( \varepsilon \) centered at the empirical distribution:

\[
\min_w \sup_{P \in \mathcal{B}_\varepsilon(\hat{P})} \text{CVaR}_\alpha^P(w)
\]

The tractable SOCP reformulation (Esfahani and Kuhn, 2018) is solved via skfolio's `DistributionallyRobustCVaR`, which exposes this as a risk-aversion utility:

\[
\max_w \; \hat{\mu}^\top w - \lambda \cdot \sup_{P \in \mathcal{B}_\varepsilon(\hat{P})} \text{CVaR}_\alpha^P(w)
\]

#### Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `epsilon` | `float` | `0.001` | Wasserstein ball radius. Larger values = more conservative. `epsilon=0` = standard CVaR |
| `alpha` | `float` | `0.95` | CVaR confidence level |
| `risk_aversion` | `float` | `1.0` | Risk-aversion coefficient \( \lambda \) (ignored when `epsilon=0`) |
| `norm` | `int` | `2` | Wasserstein norm order. **Only L2 is supported** |
| `min_weights` | `float or None` | `0.0` | Lower bound |
| `max_weights` | `float or None` | `1.0` | Upper bound |
| `budget` | `float or None` | `1.0` | Portfolio budget |
| `max_short` | `float or None` | `None` | Maximum short |
| `max_long` | `float or None` | `None` | Maximum long |
| `risk_free_rate` | `float` | `0.0` | Risk-free rate |
| `solver` | `str` | `"CLARABEL"` | CVXPY solver. `MOSEK` preferred for large instances |
| `solver_params` | `dict or None` | `None` | Solver parameters |
| `prior_config` | `MomentEstimationConfig or None` | `None` | Prior configuration |

#### Presets

| Preset | epsilon | Description |
|--------|---------|-------------|
| `for_conservative()` | `0.01` | Wider ball, more robust against tail risk misspecification |
| `for_standard()` | `0.001` | Moderate hedge against distribution misspecification |

#### Dispatch Behavior

The factory `build_dr_cvar()` dispatches to different skfolio classes based on epsilon:

- **epsilon = 0**: Returns `MeanRisk(MINIMIZE_RISK, CVAR)` -- identical to standard empirical CVaR minimization.
- **epsilon > 0**: Returns `DistributionallyRobustCVaR` -- solves the Wasserstein DRO reformulation.

#### Usage

```python
from optimizer.optimization import DRCVaRConfig, build_dr_cvar

# Conservative DRO-CVaR
model = build_dr_cvar(DRCVaRConfig.for_conservative())
model.fit(X)

# epsilon=0 -> standard CVaR (exact equivalence)
baseline = build_dr_cvar(DRCVaRConfig(epsilon=0.0))

# Custom: 99% CVaR, wider Wasserstein ball
config = DRCVaRConfig(
    epsilon=0.05,
    alpha=0.99,
    risk_aversion=2.0,
)
model = build_dr_cvar(config)
model.fit(X)
```

**Gotcha**: Only `norm=2` (L2 Wasserstein) is supported. Setting any other value raises `ValueError` at config construction via `__post_init__` validation.

---

### 13. Regime-Blended Optimization (RegimeRiskConfig)

HMM-driven regime-conditional risk measure selection and risk budgeting. Uses a fitted Hidden Markov Model to select the risk measure based on the current market regime.

#### Blended Risk Measure

The probability-weighted blended risk is:

\[
\rho_t(w) = \sum_s \gamma_T(s) \cdot \rho_s(w)
\]

where \( \gamma_T(s) = P(z_T = s \mid r_{1:T}) \) is the filtered state probability from the HMM, and \( \rho_s \) is the regime-specific risk measure for state \( s \).

#### Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `regime_measures` | `tuple[RiskMeasureType, ...]` | (required) | One risk measure per HMM state. Must match `HMMResult.n_states` |
| `hmm_config` | `HMMConfig` | `HMMConfig()` | HMM hyper-parameters |
| `cvar_beta` | `float` | `0.95` | CVaR confidence level |

#### Presets

| Preset | States | Risk Measures | Description |
|--------|--------|---------------|-------------|
| `for_calm_stress()` | 2 | Variance, CVaR | Low-vol regime uses variance; stress regime uses CVaR |
| `for_calm_stress_drawdown()` | 2 | Variance, CDaR | Low-vol regime uses variance; stress regime uses CDaR |
| `for_three_regimes()` | 3 | Variance, MAD, CVaR | Calm/normal/stress with increasing tail sensitivity |

#### Blended Optimizer

Because skfolio's `MeanRisk` requires a single convex risk measure, `build_regime_blended_optimizer()` selects the risk measure of the **dominant regime** (the state with the highest current probability):

```python
from optimizer.moments import HMMConfig, fit_hmm
from optimizer.optimization import RegimeRiskConfig, build_regime_blended_optimizer

# Fit HMM
hmm_result = fit_hmm(returns, HMMConfig(n_states=2))

# Build optimizer using dominant regime's risk measure
config = RegimeRiskConfig.for_calm_stress()
model = build_regime_blended_optimizer(config, hmm_result)
model.fit(X)
```

#### Blended Risk Computation

For analytics and monitoring, `compute_blended_risk_measure()` computes the full probability-weighted blended risk:

```python
from optimizer.optimization import compute_blended_risk_measure

risk = compute_blended_risk_measure(
    returns,
    weights,
    hmm_result,
    regime_measures=(RiskMeasureType.VARIANCE, RiskMeasureType.CVAR),
    cvar_beta=0.95,
)
```

Regimes with fewer than 5 observations fall back to full-sample risk computation.

#### Regime-Conditional Risk Budgets

`build_regime_risk_budgeting()` computes a probability-weighted blended budget vector and passes it to `build_risk_budgeting()`:

\[
b_t = \sum_s \gamma_T(s) \cdot b_s
\]

```python
import numpy as np
from optimizer.optimization import (
    RiskBudgetingConfig,
    build_regime_risk_budgeting,
)

# Per-regime budget vectors
calm_budget = np.array([0.25, 0.25, 0.25, 0.25])     # equal in calm
stress_budget = np.array([0.10, 0.10, 0.40, 0.40])    # tilt to safe assets in stress

model = build_regime_risk_budgeting(
    RiskBudgetingConfig.for_risk_parity(),
    hmm_result,
    regime_budgets=[calm_budget, stress_budget],
)
model.fit(X)
```

---

## Common Patterns

### Passing Prior Estimators

All convex optimizers accept a `prior_estimator` factory kwarg. When `None`, the factory checks `config.prior_config` and builds a prior from it. If both are `None`, skfolio's default empirical prior is used.

```python
from optimizer.moments import MomentEstimationConfig, build_prior
from optimizer.optimization import MeanRiskConfig, build_mean_risk

# Option 1: via config (serializable)
config = MeanRiskConfig(
    prior_config=MomentEstimationConfig.for_shrunk_denoised(),
)
model = build_mean_risk(config)

# Option 2: via factory kwarg (non-serializable, takes precedence)
prior = build_prior(MomentEstimationConfig.for_adaptive())
model = build_mean_risk(config, prior_estimator=prior)
```

### Factor Exposure Constraints

`build_mean_risk()` accepts a `factor_exposure_constraints` kwarg that injects `left_inequality` and `right_inequality` matrices:

```python
from optimizer.factors import build_factor_exposure_constraints
from optimizer.optimization import MeanRiskConfig, build_mean_risk

constraints = build_factor_exposure_constraints(
    factor_scores=scores,
    target_exposures=targets,
    tolerance=0.1,
)
model = build_mean_risk(
    MeanRiskConfig.for_min_variance(),
    factor_exposure_constraints=constraints,
)
```

Explicit `left_inequality` / `right_inequality` entries in kwargs take precedence over the constraints object.

### Pipeline Integration

All optimizers are sklearn-compatible and compose into pipelines:

```python
from sklearn.pipeline import Pipeline
from optimizer.optimization import build_mean_risk, MeanRiskConfig

pipe = Pipeline([
    ("optimizer", build_mean_risk(MeanRiskConfig.for_max_sharpe())),
])
pipe.fit(X)
```

Nested parameter access uses sklearn's `__` notation:

```python
# Access nested parameters for tuning
pipe.get_params()["optimizer__l2_coef"]

# Set parameters for grid search
param_grid = {
    "optimizer__l2_coef": [0.0, 0.001, 0.01],
    "optimizer__risk_aversion": [0.5, 1.0, 2.0],
}
```

---

## Solver Notes

All convex optimizers default to `solver="CLARABEL"`, an open-source interior-point solver. For large instances or specific problem structures:

| Solver | License | Best For |
|--------|---------|----------|
| `CLARABEL` | Open source (Apache 2.0) | General-purpose default |
| `MOSEK` | Commercial | Large-scale SOCP/SDP, DR-CVaR |
| `SCS` | Open source | Large sparse problems |
| `ECOS` | Open source | Small to medium conic programs |

Pass solver parameters via `solver_params`:

```python
config = MeanRiskConfig(
    solver="MOSEK",
    solver_params={"MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-10},
)
```

---

## Gotchas and Tips

1. **Non-serializable objects are factory kwargs, not config fields.** Prior estimators, risk budget arrays, inner/outer estimators (NCO), and estimator lists (Stacking) must be passed to the factory function, not stored in the config.

2. **`kappa=0` and `epsilon=0` recover standard models exactly.** `RobustConfig(kappa=0.0)` produces the same result as `build_mean_risk()`. `DRCVaRConfig(epsilon=0.0)` produces the same result as `MeanRisk(MINIMIZE_RISK, CVAR)`.

3. **HRP/HERC support non-convex risk measures; convex optimizers do not.** Use `ExtraRiskMeasureType` only with `HRPConfig` and `HERCConfig`. When `extra_risk_measure` is set, it overrides `risk_measure`.

4. **Benchmark returns are `y`, not part of the config.** For `BenchmarkTracker`, always call `model.fit(X, y=benchmark_returns)`.

5. **Regime measures must match HMM states.** `len(config.regime_measures)` must equal `hmm_result.n_states`, or a `ConfigurationError` is raised.

6. **DR-CVaR only supports L2 norm.** Setting `norm` to anything other than `2` raises `ValueError` at construction time.

7. **Stacking defaults.** When `estimators=None`, the factory defaults to `[("mean_risk", MeanRisk()), ("hrp", HierarchicalRiskParity())]` with skfolio defaults.

8. **Cardinality constraints make the problem mixed-integer.** Using `cardinality` may significantly increase solve time. Consider L1 regularization (`l1_coef`) as a convex relaxation alternative.

9. **Transaction costs require `previous_weights`.** The `transaction_costs` field in `MeanRiskConfig` penalizes turnover relative to `previous_weights`, which must be passed as a factory kwarg.

10. **ClusteringConfig `max_clusters=None` uses automatic selection.** The Two-Order Difference Gap Statistic heuristic determines the optimal number of clusters automatically.
