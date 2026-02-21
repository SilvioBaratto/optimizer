# Validation

The validation module provides cross-validation strategies designed specifically for financial time series. Unlike standard k-fold CV which randomly shuffles data, these methods respect the temporal ordering of observations to prevent look-ahead bias — a critical requirement when backtesting portfolio strategies.

## Overview

Standard cross-validation assumes observations are i.i.d., which is violated by financial returns that exhibit autocorrelation, volatility clustering, and regime changes. The three validation strategies in this module address this by enforcing temporal ordering:

- **Walk-Forward** — rolling or expanding window that mimics real-time portfolio management
- **Combinatorial Purged CV (CPCV)** — generates a population of backtest paths with purging and embargoing
- **Multiple Randomized CV** — dual randomization across time and assets for robustness testing

All validators follow the frozen-config + factory pattern: a `@dataclass(frozen=True)` config holds serializable parameters, and a factory function builds the skfolio cross-validator.

## Walk-Forward

Walk-Forward validation partitions the time series into successive train/test windows that move forward in time. This is the most common and intuitive method — it directly simulates how a portfolio manager would use the model in practice.

```
|-------- train --------|-- test --|
                    |-------- train --------|-- test --|
                                        |-------- train --------|-- test --|
```

### Configuration

```python
from optimizer.validation import WalkForwardConfig

config = WalkForwardConfig(
    test_size=63,       # ~1 quarter of trading days
    train_size=252,     # ~1 year of trading days
    purged_size=0,      # observations purged between train/test
    expend_train=False, # False = rolling, True = expanding
    reduce_test=False,  # allow shorter final test window
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `test_size` | `int` | 63 | Trading days per test window |
| `train_size` | `int` | 252 | Trading days per training window (initial size when expanding) |
| `purged_size` | `int` | 0 | Observations excised between train and test |
| `expend_train` | `bool` | `False` | `True` = expanding window, `False` = rolling window |
| `reduce_test` | `bool` | `False` | Allow shorter final test window to avoid data waste |

### Presets

| Preset | test_size | train_size | Window Type |
|--------|-----------|------------|-------------|
| `for_monthly_rolling()` | 21 | 252 | Rolling |
| `for_quarterly_rolling()` | 63 | 252 | Rolling |
| `for_quarterly_expanding()` | 63 | 252 | Expanding |

### Rolling vs Expanding

- **Rolling window** (`expend_train=False`): Training window has fixed size and slides forward. Better when the market regime changes over time — older data may not be representative.
- **Expanding window** (`expend_train=True`): Training window grows as data accumulates. Better when more data always improves estimation — the estimator benefits from a longer history.

## Combinatorial Purged Cross-Validation (CPCV)

CPCV generates a combinatorial population of backtest paths from all possible selections of test folds, with purging and embargoing to prevent information leakage. Developed by Marcos Lopez de Prado, it provides a distribution of backtest performance rather than a single path.

```python
from optimizer.validation import CPCVConfig

config = CPCVConfig(
    n_folds=10,       # non-overlapping temporal blocks
    n_test_folds=8,   # blocks per test set in each combination
    purged_size=0,    # observations purged at train/test boundary
    embargo_size=0,   # observations embargoed after each test block
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_folds` | `int` | 10 | Number of non-overlapping temporal blocks |
| `n_test_folds` | `int` | 8 | Blocks assigned to test set per combination |
| `purged_size` | `int` | 0 | Observations purged at each train-test boundary |
| `embargo_size` | `int` | 0 | Observations embargoed after each test block |

### Presets

| Preset | n_folds | n_test_folds | Paths | Use Case |
|--------|---------|--------------|-------|----------|
| `for_statistical_testing()` | 12 | 2 | C(12,2) = 66 | Significance testing with high statistical power |
| `for_small_sample()` | 6 | 2 | C(6,2) = 15 | Shorter time series |

### Purging and Embargoing

- **Purging**: Removes observations immediately adjacent to the train-test boundary to prevent information leakage from autocorrelated returns.
- **Embargoing**: Removes observations immediately following each test block to prevent the model from learning patterns that persist into the test period.

## Multiple Randomized CV

Dual randomization across both temporal windows and asset subsets to test strategy robustness along both dimensions. Each trial randomly selects a time window and a subset of assets, then runs walk-forward validation within that subsample.

```python
from optimizer.validation import MultipleRandomizedCVConfig

config = MultipleRandomizedCVConfig(
    walk_forward_config=WalkForwardConfig(),
    n_subsamples=10,       # number of random trials
    asset_subset_size=10,  # assets drawn per trial
    window_size=None,      # None = full sample
    random_state=42,
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `walk_forward_config` | `WalkForwardConfig` | default | Inner walk-forward configuration |
| `n_subsamples` | `int` | 10 | Number of random trials |
| `asset_subset_size` | `int` | 10 | Assets drawn per trial |
| `window_size` | `int` or `None` | `None` | Temporal window length; `None` = full sample |
| `random_state` | `int` or `None` | `None` | Seed for reproducibility |

### Preset

| Preset | n_subsamples | asset_subset_size | Use Case |
|--------|-------------|-------------------|----------|
| `for_robustness_check(20, 10)` | 20 | 10 | Standard robustness testing |

## Running Cross-Validation

The `run_cross_val` function is the main entry point for executing cross-validation:

```python
from optimizer.validation import WalkForwardConfig, run_cross_val

cv_config = WalkForwardConfig.for_quarterly_rolling()
cv_result = run_cross_val(pipeline, X, cv=cv_config, y=None, n_jobs=None)
```

When no `cv` argument is provided, `run_cross_val` defaults to quarterly rolling walk-forward validation.

### Computing Optimal Folds

```python
from optimizer.validation import compute_optimal_folds

n_folds = compute_optimal_folds(n_observations=1260, min_train=252, min_test=63)
```

## Code Examples

### Walk-forward backtest

```python
from optimizer.optimization import MeanRiskConfig, build_mean_risk
from optimizer.pipeline import run_full_pipeline
from optimizer.validation import WalkForwardConfig

optimizer = build_mean_risk(MeanRiskConfig.for_max_sharpe())
result = run_full_pipeline(
    prices=prices,
    optimizer=optimizer,
    cv_config=WalkForwardConfig.for_quarterly_rolling(),
)

# Out-of-sample performance
print(f"OOS Sharpe: {result.backtest.sharpe_ratio:.3f}")
print(f"OOS Max DD: {result.backtest.max_drawdown:.3f}")
```

### CPCV for backtest overfitting detection

```python
from optimizer.validation import CPCVConfig, build_cpcv, run_cross_val

cpcv = build_cpcv(CPCVConfig.for_statistical_testing())
population = run_cross_val(pipeline, X, cv=cpcv)

# population is a Population object — analyze distribution of paths
for path in population:
    print(f"Path Sharpe: {path.sharpe_ratio:.3f}")
```

### Robustness testing with randomized CV

```python
from optimizer.validation import MultipleRandomizedCVConfig, build_multiple_randomized_cv

config = MultipleRandomizedCVConfig.for_robustness_check(
    n_subsamples=20,
    asset_subset_size=15,
)
cv = build_multiple_randomized_cv(config)
population = run_cross_val(pipeline, X, cv=cv)
```

## Gotchas and Tips

!!! warning "Never use standard k-fold CV"
    Standard `KFold` or `StratifiedKFold` will randomly assign future data to the training set, creating look-ahead bias. Always use temporal validation methods for financial time series.

!!! tip "Default is quarterly rolling"
    `run_cross_val()` defaults to quarterly rolling walk-forward (`test_size=63`, `train_size=252`) when no `cv` is passed. This is a sensible default for daily equity returns.

!!! tip "CPCV returns Population, not MultiPeriodPortfolio"
    Walk-forward returns a `MultiPeriodPortfolio` (single path). CPCV returns a `Population` (collection of paths). Handle them differently when extracting metrics.

!!! warning "Purging is important for high-frequency data"
    For daily equity data, `purged_size=0` is usually fine. For intraday data or when using features with look-ahead windows (e.g., rolling averages), increase `purged_size` to cover the window length.

## Quick Reference

| Task | Code |
|------|------|
| Quarterly rolling backtest | `WalkForwardConfig.for_quarterly_rolling()` |
| Monthly rolling backtest | `WalkForwardConfig.for_monthly_rolling()` |
| Expanding window | `WalkForwardConfig.for_quarterly_expanding()` |
| CPCV statistical test | `CPCVConfig.for_statistical_testing()` |
| Robustness check | `MultipleRandomizedCVConfig.for_robustness_check()` |
| Run CV | `run_cross_val(pipeline, X, cv=config)` |
| Default CV | `run_cross_val(pipeline, X)` → quarterly rolling |
