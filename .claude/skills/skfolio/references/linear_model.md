# Cross-Sectional Regression (v0.19.0+)

`skfolio.linear_model` provides vectorized **cross-sectional** weighted-least-squares regression for panel data — one independent regression per observation. Useful for Fama-MacBeth-style factor-return extraction, characteristic-based return models, and any setting where the asset universe changes over time.

## Imports

```python
from skfolio.linear_model import CSLinearRegression, CSLinearRegressorWrapper
```

## Shapes

- `X` — features, shape `(T, N, K)` (T observations × N assets × K features)
- `y` — targets, shape `(T, N)`
- `sample_weight` — optional, shape `(T, N)`; zeros drop an asset for that observation
- `coef_` after `fit` — shape `(T, K)`; one coefficient vector per observation
- `intercept_` — shape `(T,)` when `fit_intercept=True`

## CSLinearRegression

Solves **β_t = argmin_β Σ_i w_{t,i} (y_{t,i} − X_{t,i}^T β)²** independently for each t.

```python
import numpy as np
from skfolio.linear_model import CSLinearRegression

T, N, K = 252, 50, 3
rng = np.random.default_rng(42)
X = rng.standard_normal((T, N, K))
y = rng.standard_normal((T, N))
w = np.ones((T, N))                    # zero rows are excluded per period

model = CSLinearRegression(fit_intercept=True)
model.fit(X, y, sample_weight=w)

preds = model.predict(X)               # (T, N)
r2 = model.score(X, y)
print(model.coef_.shape, model.intercept_.shape)
```

**Key behaviors**
- Fully vectorized — no Python loop over observations.
- Handles asset universes that change over time via `sample_weight=0` for absent assets.
- Zero-weight pairs may contain NaN (they are excluded from the fit).
- Non-zero-weight pairs must be finite.

## CSLinearRegressorWrapper

Adapts any scikit-learn `Regressor` to the cross-sectional contract — call its `fit` independently per observation. Use when you need Ridge, Lasso, or tree-based models as the per-period estimator.

```python
from sklearn.linear_model import Ridge
from skfolio.linear_model import CSLinearRegressorWrapper

model = CSLinearRegressorWrapper(estimator=Ridge(alpha=1.0))
model.fit(X, y, sample_weight=w)
```
