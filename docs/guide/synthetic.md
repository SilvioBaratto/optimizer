# Synthetic Data

The synthetic module generates synthetic return scenarios using vine copula models. It enables scenario generation for portfolio stress testing, Monte Carlo simulation, and conditional what-if analysis by modeling the full joint distribution of asset returns including tail dependencies.

## Overview

Traditional mean-variance optimization assumes normally distributed returns, which underestimates the probability of extreme co-movements. Vine copulas address this by decomposing the multivariate return distribution into:

1. **Marginal distributions** — fitted independently per asset (capturing skewness, kurtosis)
2. **Bivariate copulas** — capturing pairwise dependence structure (including tail dependence)

The copulas are organized in a vine (tree) structure that efficiently represents high-dimensional dependencies. The resulting model can generate synthetic scenarios that preserve the empirical dependence structure, including fat tails and asymmetric tail dependence.

## Vine Copula Configuration

```python
from optimizer.synthetic import VineCopulaConfig

config = VineCopulaConfig(
    fit_marginals=True,
    max_depth=4,
    log_transform=False,
    dependence_method=DependenceMethodType.KENDALL_TAU,
    selection_criterion=SelectionCriterionType.AIC,
    independence_level=0.05,
    n_jobs=None,
    random_state=None,
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `fit_marginals` | `bool` | `True` | Whether to fit univariate marginal distributions |
| `max_depth` | `int` | 4 | Maximum depth of the vine tree structure |
| `log_transform` | `bool` | `False` | Apply log transformation before fitting |
| `dependence_method` | `DependenceMethodType` | `KENDALL_TAU` | Pairwise dependence measure for tree construction |
| `selection_criterion` | `SelectionCriterionType` | `AIC` | Information criterion for copula family selection |
| `independence_level` | `float` | 0.05 | Significance level for independence testing |
| `n_jobs` | `int` or `None` | `None` | Number of parallel jobs |
| `random_state` | `int` or `None` | `None` | Seed for reproducibility |

### Dependence Methods

| Method | Description |
|--------|-------------|
| `KENDALL_TAU` | Rank-based concordance measure; robust to outliers |
| `MUTUAL_INFORMATION` | Information-theoretic dependence; captures nonlinear relationships |
| `WASSERSTEIN_DISTANCE` | Optimal transport distance between marginals |

### Selection Criteria

| Criterion | Description |
|-----------|-------------|
| `AIC` | Akaike Information Criterion — balances fit and complexity |
| `BIC` | Bayesian Information Criterion — penalizes complexity more than AIC |

## Synthetic Data Configuration

```python
from optimizer.synthetic import SyntheticDataConfig

config = SyntheticDataConfig(
    n_samples=1_000,
    vine_copula_config=VineCopulaConfig(),
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_samples` | `int` | 1,000 | Number of synthetic scenarios to generate |
| `vine_copula_config` | `VineCopulaConfig` or `None` | `None` | Vine copula configuration; ignored when `distribution_estimator` is passed directly |

### Presets

| Preset | n_samples | Vine Config | Use Case |
|--------|-----------|-------------|----------|
| `for_scenario_generation(10_000)` | 10,000 | Default | Large-sample Monte Carlo simulation |
| `for_stress_test(10_000)` | 10,000 | BIC + max_depth=6 | Deep tree for tail dependence capture |

## Building and Using Synthetic Data

### Basic scenario generation

```python
from optimizer.synthetic import SyntheticDataConfig, build_synthetic_data

config = SyntheticDataConfig.for_scenario_generation(n_samples=10_000)
synthetic_prior = build_synthetic_data(config)

# Use as prior estimator in optimization
from optimizer.optimization import MeanRiskConfig, build_mean_risk

optimizer = build_mean_risk(
    MeanRiskConfig.for_max_sharpe(),
    prior_estimator=synthetic_prior,
)
optimizer.fit(returns)
portfolio = optimizer.predict(returns)
```

### Stress testing with conditioning

Conditional sampling generates scenarios where specific assets are fixed at extreme values:

```python
config = SyntheticDataConfig.for_stress_test(n_samples=10_000)

# Condition on a market crash: SPY drops 10%
synthetic_prior = build_synthetic_data(
    config,
    sample_args={"conditioning": {"SPY": -0.10}},
)

# Optimize under stress scenario
optimizer = build_mean_risk(
    MeanRiskConfig.for_min_cvar(),
    prior_estimator=synthetic_prior,
)
```

### Building just the vine copula

```python
from optimizer.synthetic import VineCopulaConfig, build_vine_copula

vine = build_vine_copula(VineCopulaConfig(
    max_depth=6,
    selection_criterion=SelectionCriterionType.BIC,
))
```

## Code Examples

### Scenario-based portfolio optimization

```python
from optimizer.synthetic import SyntheticDataConfig, build_synthetic_data
from optimizer.optimization import MeanRiskConfig, build_mean_risk
from optimizer.pipeline import run_full_pipeline

# Build synthetic prior from historical data
config = SyntheticDataConfig.for_scenario_generation(n_samples=50_000)
prior = build_synthetic_data(config)

# Optimize using synthetic scenarios
optimizer = build_mean_risk(
    MeanRiskConfig.for_min_cvar(beta=0.95),
    prior_estimator=prior,
)
result = run_full_pipeline(prices=prices, optimizer=optimizer)
```

### Stress test: sector crash

```python
# What if financials drop 15%?
prior = build_synthetic_data(
    SyntheticDataConfig.for_stress_test(),
    sample_args={"conditioning": {
        "JPM": -0.15,
        "BAC": -0.15,
        "GS": -0.15,
    }},
)
```

## Gotchas and Tips

!!! tip "Use BIC for stress tests"
    The `for_stress_test` preset uses BIC instead of AIC for copula selection. BIC penalizes complexity more heavily, producing simpler copula structures that are less likely to overfit — important when extrapolating to tail events.

!!! tip "Deeper trees capture more tail dependence"
    Increasing `max_depth` allows the vine to model higher-order dependencies between assets. The default (4) is sufficient for most equity portfolios; stress tests benefit from `max_depth=6`.

!!! warning "Computational cost scales with n_samples and assets"
    Fitting a vine copula to 50+ assets with deep trees can be slow. Use `n_jobs=-1` for parallelism and consider reducing `max_depth` for large universes.

!!! tip "Conditioning dict for stress tests"
    Pass conditioning values via `sample_args={"conditioning": {"TICKER": value}}` to the factory. The synthetic prior then generates scenarios conditioned on those asset returns being fixed at the specified values.

## Quick Reference

| Task | Code |
|------|------|
| Scenario generation | `SyntheticDataConfig.for_scenario_generation(10_000)` |
| Stress test config | `SyntheticDataConfig.for_stress_test(10_000)` |
| Build prior | `build_synthetic_data(config)` |
| Conditional stress | `build_synthetic_data(config, sample_args={"conditioning": {"SPY": -0.10}})` |
| Build vine copula | `build_vine_copula(VineCopulaConfig())` |
