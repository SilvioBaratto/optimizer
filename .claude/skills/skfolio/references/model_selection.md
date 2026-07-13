# Model Selection & Hyperparameter Tuning

Standard cross-validation, backtesting, hyperparameter search, and metadata routing. For incremental `partial_fit`-based workflows see `online_learning.md`.

## Imports

```python
from skfolio.model_selection import (
    WalkForward, CombinatorialPurgedCV, MultipleRandomizedCV,
    cross_val_predict, optimal_folds_number,
)
```

## WalkForward

Rolling/expanding window CV for time series.

```python
cv = WalkForward(
    test_size=60,      # test observations per fold
    train_size=252,    # None ⇒ expanding window
)

pred = cross_val_predict(MeanRisk(), X, cv=cv)
print(pred.sharpe_ratio)   # pred is a MultiPeriodPortfolio
```

## CombinatorialPurgedCV

Multiple paths with purging and embargoing to prevent leakage from nearby folds (López de Prado).

```python
cv = CombinatorialPurgedCV(
    n_folds=10,
    n_test_folds=8,
    purge_size=5,
    embargo_size=5,
)

pred = cross_val_predict(MeanRisk(), X, cv=cv)
# pred is a Population of MultiPeriodPortfolio
print(pred.summary())
```

Use `optimal_folds_number(n_observations, n_test_folds, ...)` to pick `n_folds` that balance path count against fold size.

## MultipleRandomizedCV

Monte Carlo evaluation with asset subsampling and temporal windows.

```python
cv = MultipleRandomizedCV(
    walk_forward=WalkForward(test_size=60, train_size=252),
    n_subsamples=10,
    asset_subset_size=10,
    window_size=None,
)
```

Supports `get_n_splits()` (0.16.0+).

## cross_val_predict

- `MultiPeriodPortfolio` for single-path CV (`KFold`, `WalkForward`)
- `Population` for multi-path CV (`CombinatorialPurgedCV`, `MultipleRandomizedCV`)

```python
pred = cross_val_predict(model, X, cv=cv)
```

---

## Hyperparameter Tuning

Uses scikit-learn's `GridSearchCV` and `RandomizedSearchCV`. For online/walk-forward tuning see `online_learning.md` (`OnlineGridSearch`).

### Nested parameter syntax

Double-underscore `__` reaches nested estimator params. Discover them via `model.get_params()`.

```python
param_grid = {
    "prior_estimator__mu_estimator__alpha": [0.001, 0.01, 0.1],
    "risk_measure": [RiskMeasure.SEMI_VARIANCE, RiskMeasure.CVAR],
}
```

### make_scorer

```python
from skfolio.metrics import make_scorer
from skfolio import RatioMeasure

scoring = make_scorer(RatioMeasure.SORTINO_RATIO)

def custom(pred):                           # receives a Portfolio
    return pred.mean - 2 * pred.variance
scoring = make_scorer(custom)
```

### GridSearchCV

```python
from sklearn.model_selection import GridSearchCV, KFold

grid = GridSearchCV(
    estimator=MeanRisk(),
    param_grid=param_grid,
    cv=KFold(n_splits=5, shuffle=False),    # shuffle=False for time series
    scoring=scoring,
    n_jobs=-1,
)
grid.fit(X)
best = grid.best_estimator_
```

### RandomizedSearchCV

```python
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats

rd = RandomizedSearchCV(
    estimator=MeanRisk(),
    param_distributions={"l2_coef": stats.loguniform(0.01, 1)},
    n_iter=50,
    cv=KFold(n_splits=5, shuffle=False),
    n_jobs=-1,
)
```

---

## Metadata Routing

Passes extra arrays (e.g., implied volatility) through nested estimators. Required for `ImpliedCovariance`, `BenchmarkTracker` with benchmark returns, etc.

```python
from sklearn import set_config
set_config(enable_metadata_routing=True)

from skfolio.moments import ImpliedCovariance
from skfolio.prior import EmpiricalPrior
from skfolio.optimization import MeanRisk

model = MeanRisk(
    prior_estimator=EmpiricalPrior(
        covariance_estimator=ImpliedCovariance()
            .set_fit_request(implied_vol=True)
    )
)
model.fit(X, implied_vol=implied_vol)
```

Three steps:
1. `set_config(enable_metadata_routing=True)` — enable globally.
2. `.set_fit_request(<param>=True)` — declare the metadata on the consumer.
3. `model.fit(X, <param>=value)` — pass it at fit time; routing threads it through.

Metadata propagates through `GridSearchCV` splits automatically.
