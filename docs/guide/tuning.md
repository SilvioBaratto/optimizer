# Tuning

The tuning module wraps sklearn's `GridSearchCV` and `RandomizedSearchCV` with temporal cross-validation defaults that prevent look-ahead bias. It enforces walk-forward validation by default, ensuring that hyperparameter selection respects the time-series nature of financial data.

## Overview

Hyperparameter tuning for portfolio optimization requires special care: standard k-fold CV would use future returns to select parameters, introducing look-ahead bias. The tuning module addresses this by coupling sklearn's search algorithms with temporal cross-validation from the [validation](validation.md) module.

Because the portfolio pipeline is a single sklearn `Pipeline` object, all nested parameters are accessible via the double-underscore `__` notation (e.g., `"optimizer__l2_coef"`, `"drop_correlated__threshold"`).

## Grid Search

Exhaustive search over a specified parameter grid with temporal CV.

### Configuration

```python
from optimizer.tuning import GridSearchConfig
from optimizer.validation import WalkForwardConfig
from optimizer.scoring import ScorerConfig

config = GridSearchConfig(
    cv_config=WalkForwardConfig.for_quarterly_rolling(),
    scorer_config=ScorerConfig.for_sharpe(),
    n_jobs=None,
    return_train_score=False,
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cv_config` | `WalkForwardConfig` | default (quarterly rolling) | Temporal cross-validation strategy |
| `scorer_config` | `ScorerConfig` | default (Sharpe ratio) | Portfolio scoring function |
| `n_jobs` | `int` or `None` | `None` | Parallel jobs; `-1` uses all cores |
| `return_train_score` | `bool` | `False` | Compute training scores (slower) |

### Presets

| Preset | CV Config | n_jobs | Description |
|--------|-----------|--------|-------------|
| `for_quick_search()` | Monthly rolling | -1 | Fast evaluation, all cores |
| `for_thorough_search()` | Quarterly expanding | -1 | Comprehensive with train scores |

## Randomized Search

Samples parameter configurations from specified distributions rather than exhaustive enumeration. Preferred when the parameter space is large or continuous.

### Configuration

```python
from optimizer.tuning import RandomizedSearchConfig

config = RandomizedSearchConfig(
    n_iter=50,
    cv_config=WalkForwardConfig.for_quarterly_rolling(),
    scorer_config=ScorerConfig.for_sharpe(),
    n_jobs=None,
    random_state=42,
    return_train_score=False,
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_iter` | `int` | 50 | Number of random parameter samples |
| `cv_config` | `WalkForwardConfig` | default | Temporal CV strategy |
| `scorer_config` | `ScorerConfig` | default (Sharpe) | Scoring function |
| `n_jobs` | `int` or `None` | `None` | Parallel jobs |
| `random_state` | `int` or `None` | `None` | Seed for reproducibility |
| `return_train_score` | `bool` | `False` | Compute training scores |

### Presets

| Preset | n_iter | CV Config | Description |
|--------|--------|-----------|-------------|
| `for_quick_search(20)` | 20 | Monthly rolling | Fast random sampling |
| `for_thorough_search(100)` | 100 | Quarterly expanding | Comprehensive search |

## Nested Parameter Addressing

The sklearn `Pipeline` flattens all transformer and optimizer parameters, making them tunable via the double-underscore `__` notation. The step names come from `build_portfolio_pipeline()`:

```
validate__max_abs_return
outliers__winsorize_threshold
outliers__remove_threshold
impute__sector_mapping
drop_correlated__threshold
optimizer__risk_measure
optimizer__l2_coef
optimizer__prior_estimator__mu_estimator__alpha
```

### Discovering tunable parameters

```python
from optimizer.pipeline import build_portfolio_pipeline
from optimizer.optimization import MeanRiskConfig, build_mean_risk

optimizer = build_mean_risk(MeanRiskConfig.for_max_sharpe())
pipeline = build_portfolio_pipeline(optimizer)

# List all tunable parameters
for name, value in sorted(pipeline.get_params().items()):
    print(f"{name}: {value}")
```

## Code Examples

### Grid search over regularization

```python
from optimizer.pipeline import build_portfolio_pipeline, tune_and_optimize
from optimizer.optimization import MeanRiskConfig, build_mean_risk
from optimizer.tuning import GridSearchConfig
from skfolio.preprocessing import prices_to_returns

X = prices_to_returns(prices)
optimizer = build_mean_risk(MeanRiskConfig.for_max_sharpe())
pipeline = build_portfolio_pipeline(optimizer)

param_grid = {
    "optimizer__l2_coef": [0.0, 0.01, 0.05, 0.1],
}

result = tune_and_optimize(
    pipeline, X,
    param_grid=param_grid,
    tuning_config=GridSearchConfig.for_quick_search(),
)
print(f"Best L2 coef: {result.pipeline.get_params()['optimizer__l2_coef']}")
```

### Grid search over multiple parameters

```python
param_grid = {
    "optimizer__l2_coef": [0.0, 0.01, 0.1],
    "drop_correlated__threshold": [0.85, 0.90, 0.95],
    "outliers__winsorize_threshold": [2.5, 3.0, 3.5],
}

result = tune_and_optimize(
    pipeline, X,
    param_grid=param_grid,
    tuning_config=GridSearchConfig(n_jobs=-1),
)
```

### Randomized search with distributions

```python
from scipy.stats import uniform, loguniform
from optimizer.tuning import RandomizedSearchConfig

param_distributions = {
    "optimizer__l2_coef": loguniform(1e-4, 1e-1),
    "drop_correlated__threshold": uniform(0.80, 0.15),  # [0.80, 0.95]
}

result = tune_and_optimize(
    pipeline, X,
    param_grid=param_distributions,
    tuning_config=RandomizedSearchConfig.for_thorough_search(n_iter=50),
)
```

### Using build functions directly

```python
from optimizer.tuning import build_grid_search_cv, build_randomized_search_cv

# Grid search
gs = build_grid_search_cv(pipeline, param_grid, config=GridSearchConfig())
gs.fit(X)
print(f"Best score: {gs.best_score_:.4f}")
print(f"Best params: {gs.best_params_}")

# Randomized search
rs = build_randomized_search_cv(pipeline, param_distributions, config=RandomizedSearchConfig())
rs.fit(X)
```

## Gotchas and Tips

!!! warning "Temporal CV is enforced by default"
    Both `GridSearchConfig` and `RandomizedSearchConfig` default to walk-forward validation. Do not override this with standard `KFold` — it introduces look-ahead bias.

!!! tip "Use double-underscore notation for nested parameters"
    Pipeline parameters are addressed as `"step_name__parameter"`. For deeply nested parameters, chain underscores: `"optimizer__prior_estimator__mu_estimator__alpha"`.

!!! tip "Grid search vs randomized search"
    Use grid search when the parameter space is small and discrete. Use randomized search when exploring continuous distributions or when the grid would be too large. Randomized search with `n_iter=50` often finds good parameters faster than exhaustive grid search.

!!! warning "Computation cost"
    Each combination is evaluated across all walk-forward folds. With 4 folds, 3 parameters, and 4 values each: 4^3 * 4 = 256 fits. Use `n_jobs=-1` for parallelism and start with `for_quick_search()`.

## Quick Reference

| Task | Code |
|------|------|
| Quick grid search | `GridSearchConfig.for_quick_search()` |
| Thorough grid search | `GridSearchConfig.for_thorough_search()` |
| Quick random search | `RandomizedSearchConfig.for_quick_search(n_iter=20)` |
| Thorough random search | `RandomizedSearchConfig.for_thorough_search(n_iter=100)` |
| Tune + optimize | `tune_and_optimize(pipeline, X, param_grid={...})` |
| Build grid search | `build_grid_search_cv(pipeline, param_grid)` |
| List tunable params | `pipeline.get_params().keys()` |
