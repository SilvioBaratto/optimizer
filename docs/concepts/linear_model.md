# Cross-Sectional Linear Regression

## When to use

`CSLinearRegression` fits a per-period linear regression of a cross-
section of returns on a cross-section of factor scores. Use it when
you want the regression slope as your information coefficient (gives
both direction and magnitude per period) instead of a rank
correlation, or when running cross-sectional risk-factor regressions
inside a research pipeline.

The skfolio reference is in
`~/.claude/skills/skfolio/references/linear_model.md`.

## Shape contract

The estimator expects a panel of `(T, N, K)` for `X`, `(T, N)` for
`y`, and `(T, N)` for `sample_weight`:

| Input | Shape | Meaning |
|-------|-------|---------|
| `X` | `(T, N, K)` | `T` periods, `N` assets, `K` regressors. |
| `y` | `(T, N)` | Forward returns per period and asset. |
| `cs_weights` | `(T, N)` | Optional per-pair weights. Zero-weight pairs are excluded; NaNs are allowed. |

Output: `coef_` of shape `(T, K)` (slopes per period), `intercept_`
of shape `(T,)` if `fit_intercept=True`.

## API surface

`CSLinearRegressionConfig` carries primitive-only fields:

| Field | Default | Forwarded to skfolio? |
|-------|---------|------------------------|
| `fit_intercept` | `True` | Yes (`fit_intercept`) |
| `weighted` | `False` | No — caller hint signalling that `cs_weights` will be supplied at fit time. |
| `min_observations` | `10` | No — caller-side quality gate consumed by downstream factor-IC code. |

Two presets:

```python
CSLinearRegressionConfig.for_default()
CSLinearRegressionConfig.for_weighted()
```

## Composition pattern

Pass the Config to the factory; fit on the panel:

```python
import numpy as np

from optimizer.linear_model import (
    CSLinearRegressionConfig,
    build_cs_linear_regression,
)


rng = np.random.default_rng(0)
T, N, K = 60, 8, 3
X = rng.normal(size=(T, N, K))
beta_true = rng.normal(size=K)
y = (X @ beta_true).reshape(T, N) + 0.1 * rng.normal(size=(T, N))

estimator = build_cs_linear_regression(CSLinearRegressionConfig.for_default())
estimator.fit(X, y)

print("per-period slopes shape:", estimator.coef_.shape)  # (T, K)
```

## Factor IC integration

`compute_ic_series(..., use_cs_regression=True, cs_config=...)` in
`optimizer.factors` uses the slope coefficient as the IC. The default
(`use_cs_regression=False`) preserves the original Spearman-rank IC
behaviour byte-for-byte.

## See also

- skfolio reference:
  `~/.claude/skills/skfolio/references/linear_model.md`.
- `optimizer.factors.compute_ic_series` for the opt-in CS-regression
  IC path.
