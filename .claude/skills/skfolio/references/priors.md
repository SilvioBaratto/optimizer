# Prior Estimators

A **prior** produces a `ReturnDistribution` (mu, covariance, returns, sample_weight, cholesky) that downstream optimizers consume. Swap priors to inject views, stress tests, factor structure, or synthetic scenarios without changing the optimizer.

## Imports

```python
from skfolio.prior import (
    EmpiricalPrior,
    BlackLitterman,
    TimeSeriesFactorModel,    # v0.17.0+ — replaces deprecated FactorModel
    SyntheticData,
    EntropyPooling,
    OpinionPooling,
    LoadingMatrixRegression,
    ReturnDistribution,
)
```

## EmpiricalPrior

Historical distribution with pluggable mu/covariance estimators.

```python
prior = EmpiricalPrior(
    mu_estimator=ShrunkMu(),
    covariance_estimator=LedoitWolf(),
    is_log_normal=False,
    investment_horizon=None,  # set with is_log_normal=True for multi-year
)
```

## BlackLitterman

Bayesian model — market equilibrium prior combined with analyst views.

```python
prior = BlackLitterman(
    views=[
        "AAPL == 0.10",           # absolute
        "MSFT - GOOG == 0.03",    # relative
    ],
    tau=0.05,
    prior_estimator=EmpiricalPrior(mu_estimator=EquilibriumMu()),
)
```

**View syntax:**
- Absolute: `"TICKER == value"` or `"TICKER >= value"`
- Relative: `"TICKER1 - TICKER2 == value"`

## TimeSeriesFactorModel (v0.17.0+)

Factor model — reduces dimensionality via common factors. **Replaces deprecated `FactorModel`.** The constructor signature and `fit(X, y)` contract are unchanged; only the import needs updating.

```python
from skfolio.prior import TimeSeriesFactorModel

prior = TimeSeriesFactorModel(
    loading_matrix_estimator=LoadingMatrixRegression(),
    factor_prior_estimator=EmpiricalPrior(),
)
# X = asset returns, y = factor returns
model.fit(X_train, y=factor_returns_train)
```

**Black-Litterman Factor Model** — chain a BL prior inside the factor prior:

```python
prior = TimeSeriesFactorModel(
    factor_prior_estimator=BlackLitterman(
        views=["MTUM == 0.10", "QUAL - VLUE == 0.04"],
        tau=0.05,
    ),
)
```

> `FactorModel` still works in 0.17.x+ but emits a deprecation warning. Migrate imports.

## SyntheticData

Generates synthetic scenarios from a fitted distribution. Ideal for stress tests.

```python
prior = SyntheticData(
    distribution_estimator=VineCopula(),
    n_samples=10_000,
    sample_args=dict(conditioning={"AAPL": -0.10}),  # stress: AAPL drops 10%
)
```

## EntropyPooling

Adjusts baseline probabilities to satisfy views while minimizing KL divergence.

```python
prior = EntropyPooling(
    mean_views=["JPM == -0.002", "PG >= LLY", "BAC >= prior(BAC) * 1.2"],
    variance_views=["BAC == prior(BAC) * 4"],
    correlation_views=["(BAC,JPM) == 0.80", "(BAC,JNJ) <= prior(BAC,JNJ) * 0.5"],
    skew_views=["BAC == -0.05"],
    cvar_views=["GE == 0.08"],
    cvar_beta=0.95,
    groups={"Financials": ["BAC", "JPM"], "Healthcare": ["JNJ", "LLY"]},
    prior_estimator=EmpiricalPrior(),
)
```

### View types

| Type | Syntax | Example |
|---|---|---|
| Mean | `"TICKER == value"` | `"JPM == -0.002"` |
| Mean relative | `"TICKER1 >= TICKER2"` | `"PG >= LLY"` |
| Mean vs prior | `"TICKER >= prior(TICKER) * factor"` | `"BAC >= prior(BAC) * 1.2"` |
| Variance | `"TICKER == prior(TICKER) * factor"` | `"BAC == prior(BAC) * 4"` |
| Correlation | `"(T1,T2) == value"` | `"(BAC,JPM) == 0.80"` |
| Correlation vs prior | `"(T1,T2) <= prior(T1,T2) * factor"` | `"(BAC,JNJ) <= prior(BAC,JNJ) * 0.5"` |
| Skew | `"TICKER == value"` | `"BAC == -0.05"` |
| CVaR | `"TICKER == value"` | `"GE == 0.08"` |
| Group mean | `"Group1 == factor * Group2"` | `"Financials == 2 * Growth"` |

## OpinionPooling

Combines multiple expert distributions into a consensus prior.

```python
prior = OpinionPooling(
    estimators=[
        ("expert_1", EntropyPooling(mean_views=["AAPL == 0.001"])),
        ("expert_2", EntropyPooling(mean_views=["AAPL == -0.001"])),
    ],
    opinion_probabilities=[0.4, 0.5],  # remaining 0.1 → base prior
    prior_estimator=EmpiricalPrior(),
)
```
