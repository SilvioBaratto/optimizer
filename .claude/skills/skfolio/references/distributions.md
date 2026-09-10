# Distributions & Stress Testing

Copulas and univariate distributions for synthetic-data generation, stress testing, and conditional scenarios. These feed `SyntheticData` (see `priors.md`), which downstream optimizers consume like any other prior.

## Imports

```python
from skfolio.distribution import (
    # Copulas
    VineCopula, GaussianCopula, StudentTCopula,
    ClaytonCopula, GumbelCopula, JoeCopula, IndependentCopula,
    # Univariate
    Gaussian, StudentT, JohnsonSU, NormalInverseGaussian,
)
```

## VineCopula

Multivariate dependence model — fits a regular-vine structure where each node is a bivariate copula. Handles tail dependence and asymmetric co-movement that Gaussian copulas miss.

```python
from skfolio.distribution import VineCopula, StudentT, JohnsonSU

copula = VineCopula(
    copula_candidates=None,                            # None = default candidate set (or pass a list of copula classes)
    marginal_candidates=[StudentT, JohnsonSU],         # univariate marginals
)
copula.fit(X)
samples = copula.sample(n_samples=10_000)
```

**Available copulas:** `GaussianCopula`, `StudentTCopula`, `ClaytonCopula`, `GumbelCopula`, `JoeCopula`, `IndependentCopula`

**Univariate marginals:** `Gaussian`, `StudentT`, `JohnsonSU`, `NormalInverseGaussian`

## Stress Testing via SyntheticData

Pass `sample_args=dict(conditioning={...})` to generate scenarios conditional on asset or factor shocks — useful for CCAR-style stress tests.

```python
from skfolio.prior import SyntheticData

prior = SyntheticData(
    distribution_estimator=VineCopula(),
    n_samples=10_000,
    sample_args=dict(conditioning={"AAPL": -0.10}),   # AAPL drops 10%
)
```

### Stressed factor model

```python
from skfolio.prior import TimeSeriesFactorModel, SyntheticData

factor_prior = SyntheticData(
    distribution_estimator=VineCopula(),
    n_samples=10_000,
    sample_args=dict(conditioning={"MTUM": -0.10}),    # momentum crashes
)

prior = TimeSeriesFactorModel(factor_prior_estimator=factor_prior)
model = MeanRisk(risk_measure=RiskMeasure.CVAR, prior_estimator=prior)
model.fit(X_train, factors=factors_train)              # 1.0: keyword `factors=`, not `y`
```

> **1.0:** factor returns are passed as the keyword-only `factors=` (was positional `y`), and the estimator is `TimeSeriesFactorModel` — `FactorModel` is now a fitted *container*, not an estimator. See `priors.md` / `factor_models.md`.
