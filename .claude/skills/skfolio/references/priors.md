# Prior Estimators

A **prior** produces a `ReturnDistribution` (mu, covariance, returns, sample_weight, factor_model) that downstream optimizers consume. Swap priors to inject views, stress tests, factor structure, or synthetic scenarios without changing the optimizer. The fitted object lives on `return_distribution_`.

## Imports

```python
from skfolio.prior import (
    EmpiricalPrior,
    BlackLitterman,
    TimeSeriesFactorModel,       # time-series factor model (estimator)
    CharacteristicsFactorModel,  # NEW 1.0 — cross-sectional / BARRA-style (see factor_models.md)
    FactorModel,                 # NEW 1.0 meaning — fitted CONTAINER, not an estimator
    SyntheticData,
    EntropyPooling,
    OpinionPooling,
    LoadingMatrixRegression,
    ReturnDistribution,
)
```

## ⚠️ 1.0 breaking changes for factor models

1. **Factor returns are keyword-only `factors=`** — the old positional `y` no longer works:
   ```python
   # 0.20.x (BROKEN)   model.fit(X_train, y_train)
   # 1.0               model.fit(X_train, factors=factors_train)
   ```
   This propagates through metadata routing to a nested factor prior inside an optimizer.
2. **`FactorModel` is repurposed.** In 1.0 `skfolio.prior.FactorModel` is the *fitted result container* (loading matrix, factor moments, idiosyncratic covariance, exposures, factor/idio returns) produced by `TimeSeriesFactorModel` / `CharacteristicsFactorModel` — it is **no longer an estimator you instantiate as `prior_estimator`**. Use `TimeSeriesFactorModel()` (or `CharacteristicsFactorModel()`) as the estimator.

## ReturnDistribution

```python
ReturnDistribution(
    mu,           # (n_assets,)
    covariance,   # (n_assets, n_assets)
    returns,      # (n_observations, n_assets)
    sample_weight=None,
    factor_model=None,   # FactorModel | None; the matrix sqrt is the read-only property `covariance_sqrt`
)
```

Read moments/scenarios from a fitted prior's `return_distribution_`. In 1.0 this can span the *full universe* with non-investable assets marked NaN; optimizers solve the investable subset then expand (see `data_representation.md`).

## EmpiricalPrior

Historical distribution with pluggable mu/covariance estimators.

```python
prior = EmpiricalPrior(
    mu_estimator=ShrunkMu(),
    covariance_estimator=LedoitWolf(),
    is_log_normal=False,
    investment_horizon=None,  # set with is_log_normal=True for multi-year projection
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

**View syntax:** absolute `"TICKER == value"` / `"TICKER >= value"`; relative `"TICKER1 - TICKER2 == value"`.

## TimeSeriesFactorModel

Time-series factor model — estimates asset exposures by regressing on observed factor returns. Reduces dimensionality via common factors.

```python
from skfolio.prior import TimeSeriesFactorModel

prior = TimeSeriesFactorModel(
    loading_matrix_estimator=LoadingMatrixRegression(),  # default; LassoCV-based
    factor_prior_estimator=EmpiricalPrior(),
)
# X = asset returns, factors = factor returns (KEYWORD)
model.fit(X_train, factors=factors_train)
```

**Black-Litterman factor model** — chain a BL prior inside the factor prior; views reference **factor** names:

```python
prior = TimeSeriesFactorModel(
    factor_prior_estimator=BlackLitterman(
        views=["MTUM == 0.10", "QUAL - VLUE == 0.04"],
        tau=0.05,
    ),
)
model.fit(X_train, factors=factors_train)
```

## CharacteristicsFactorModel (NEW in 1.0)

Cross-sectional (BARRA-style) factor model driven by fundamental/price *descriptors* over an `AssetPanel`. Full treatment in **`factor_models.md`**. Sketch:

```python
from skfolio.prior import CharacteristicsFactorModel

model = CharacteristicsFactorModel(
    factors=[("market", global_factor), ("value", value_factor), ...],
    neutralize_against={"non_linear_size": ["size"]},
    exposure_lag=1,
    factor_prior_estimator=EmpiricalPrior(...),
)
model.fit(characteristics=panel)   # panel is a skfolio.containers.AssetPanel
```

## SyntheticData

Generates synthetic scenarios from a fitted distribution. Ideal for stress tests.

```python
prior = SyntheticData(
    distribution_estimator=VineCopula(),
    n_samples=10_000,
    sample_args=dict(conditioning={"AAPL": -0.10}),  # stress: AAPL drops 10%
)
```

Composable: usable standalone (`fit(X)`), as a `prior_estimator` inside `EntropyPooling`, or as a `factor_prior_estimator` inside `TimeSeriesFactorModel`. Re-stress a nested instance via `set_params(factor_prior_estimator__sample_args=...)`.

## EntropyPooling

Adjusts baseline scenario probabilities (`sample_weight`) to satisfy views while minimizing KL divergence. Stackable on top of another prior via `prior_estimator=`.

```python
prior = EntropyPooling(
    mean_views=["JPM == -0.002", "PG >= LLY", "BAC >= prior(BAC) * 1.2"],
    variance_views=["BAC == prior(BAC) * 4"],
    correlation_views=["(BAC,JPM) == 0.80", "(BAC,JNJ) <= prior(BAC,JNJ) * 0.5"],
    skew_views=["BAC == -0.05"],
    cvar_views=["GE == 0.08"],
    cvar_beta=0.95,
    groups={"Financials": ["BAC", "JPM"], "Healthcare": ["JNJ", "LLY"]},
    prior_estimator=EmpiricalPrior(),   # or a SyntheticData / factor prior to pool on top of
)
```

Fitted diagnostics: `relative_entropy_`, `effective_number_of_scenarios_`.

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

Combines multiple expert distributions into a consensus prior. Supports linear and logarithmic pooling with an optional robust KL-penalty.

```python
prior = OpinionPooling(
    estimators=[
        ("expert_1", EntropyPooling(mean_views=["AAPL == 0.001"])),
        ("expert_2", EntropyPooling(mean_views=["AAPL == -0.001"])),
    ],
    opinion_probabilities=[0.4, 0.5],  # need NOT sum to 1; residual mass → base prior
    prior_estimator=EmpiricalPrior(),
)
```

## Stress testing

Fit a prior (or read `VineCopula.sample(..., conditioning=...)`), then evaluate an existing allocation on the stressed distribution by passing it to `optimizer.predict(stressed_return_distribution)` — no refit needed.
