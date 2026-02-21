# Pipeline Overview

The pipeline module orchestrates the full portfolio construction workflow — from raw price data through preprocessing, optimization, validation, and rebalancing into a single function call. It composes sklearn-compatible transformers and skfolio optimizers into a unified `Pipeline` object that can be cross-validated, tuned, and serialized.

## Architecture

The optimizer library follows a linear data-flow architecture:

```
prices → returns → [preprocess → pre-select → optimize] → backtest → weights
```

The conversion from prices to returns happens **outside** the sklearn pipeline (it changes data semantics from levels to differences), while everything inside the brackets is a single `sklearn.pipeline.Pipeline` object.

```
┌─────────────────────────────────────────────────────────────┐
│                     run_full_pipeline()                      │
│                                                              │
│  prices ──→ prices_to_returns() ──→ returns DataFrame        │
│                                         │                    │
│           ┌─────────────────────────────┤                    │
│           │  build_portfolio_pipeline() │                    │
│           │                             │                    │
│           │  validate ──→ outliers ──→ impute                │
│           │      ──→ SelectComplete ──→ DropZeroVariance     │
│           │      ──→ DropCorrelated ──→ [SelectKExtremes]    │
│           │      ──→ optimizer (skfolio)                     │
│           └─────────────────────────────┘                    │
│                                         │                    │
│  backtest (walk-forward CV) ←───────────┘                    │
│  fit full data → final weights                               │
│  rebalancing check (if previous_weights)                     │
│                                                              │
│  → PortfolioResult                                           │
└──────────────────────────────────────────────────────────────┘
```

### Why prices_to_returns() runs outside

The sklearn pipeline convention requires that `fit(X)` and `transform(X)` operate on the same kind of data. Price-to-return conversion changes the data semantics (levels become differences, one row is consumed), so it runs before pipeline construction. Inside the pipeline, every transformer receives and returns a return DataFrame.

### Flattened pipeline for parameter access

`build_portfolio_pipeline()` flattens the pre-selection sub-pipeline steps into the top-level pipeline so that `get_params()` exposes all nested parameters for hyperparameter tuning:

```python
from optimizer.pipeline import build_portfolio_pipeline
from optimizer.optimization import MeanRiskConfig, build_mean_risk

optimizer = build_mean_risk(MeanRiskConfig.for_max_sharpe())
pipeline = build_portfolio_pipeline(optimizer)

# All pre-selection + optimizer params are accessible
print(pipeline.get_params().keys())
# dict_keys(['validate__max_abs_return', 'outliers__winsorize_threshold',
#             'drop_correlated__threshold', 'optimizer__risk_measure', ...])
```

## Core Functions

### run_full_pipeline

The primary entry point. Converts prices to returns, builds the pipeline, optionally backtests, fits on the full dataset, and checks rebalancing thresholds:

```python
from optimizer.optimization import MeanRiskConfig, build_mean_risk
from optimizer.pipeline import run_full_pipeline
from optimizer.validation import WalkForwardConfig

optimizer = build_mean_risk(MeanRiskConfig.for_max_sharpe())
result = run_full_pipeline(
    prices=price_df,
    optimizer=optimizer,
    cv_config=WalkForwardConfig.for_quarterly_rolling(),
)

print(result.weights)              # pd.Series: ticker → weight
print(result.summary)              # dict: sharpe_ratio, max_drawdown, ...
print(result.backtest.sharpe_ratio) # out-of-sample Sharpe
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `prices` | `pd.DataFrame` | Price matrix (dates x tickers) |
| `optimizer` | skfolio optimizer | From any `build_*()` factory |
| `pre_selection_config` | `PreSelectionConfig` or `None` | Data cleaning config |
| `sector_mapping` | `dict[str, str]` or `None` | Ticker → sector for imputation |
| `cv_config` | `WalkForwardConfig` or `None` | `None` skips backtesting |
| `previous_weights` | `ndarray` or `None` | For rebalancing analysis |
| `rebalancing_config` | `ThresholdRebalancingConfig` / `HybridRebalancingConfig` or `None` | Rebalancing strategy |
| `current_date` | `pd.Timestamp` or `None` | For hybrid rebalancing |
| `last_review_date` | `pd.Timestamp` or `None` | For hybrid rebalancing |
| `y_prices` | `pd.DataFrame` or `None` | Benchmark/factor prices |
| `n_jobs` | `int` or `None` | Parallel jobs for backtesting |

### run_full_pipeline_with_selection

Extends `run_full_pipeline` with upstream stock selection. When `fundamentals` is provided, the function:

1. Screens the universe for investability
2. Computes and standardizes factor scores
3. Applies macro regime tilts (optional)
4. Computes composite score and selects stocks
5. Delegates to `run_full_pipeline()` on the selected tickers

```python
from optimizer.pipeline import run_full_pipeline_with_selection
from optimizer.factors import SelectionConfig, CompositeScoringConfig

result = run_full_pipeline_with_selection(
    prices=price_df,
    optimizer=optimizer,
    fundamentals=fundamentals_df,
    volume_history=volume_df,
    scoring_config=CompositeScoringConfig(),
    selection_config=SelectionConfig(n_stocks=50),
    cv_config=WalkForwardConfig.for_quarterly_rolling(),
)
```

When `fundamentals=None`, all selection steps are skipped and the function delegates directly to `run_full_pipeline()`.

### Lower-level composable functions

For more control, use the individual building blocks:

```python
from optimizer.pipeline import optimize, backtest, tune_and_optimize, build_portfolio_pipeline
from skfolio.preprocessing import prices_to_returns

# Manual pipeline composition
X = prices_to_returns(prices)
pipeline = build_portfolio_pipeline(optimizer)

# Option 1: Just optimize (no backtest)
result = optimize(pipeline, X)

# Option 2: Backtest first, then optimize
bt = backtest(pipeline, X, cv_config=WalkForwardConfig.for_quarterly_rolling())
result = optimize(pipeline, X)
result.backtest = bt

# Option 3: Tune hyperparameters then optimize
result = tune_and_optimize(
    pipeline, X,
    param_grid={"optimizer__l2_coef": [0.0, 0.01, 0.1]},
)
```

## PortfolioResult

All pipeline functions return a `PortfolioResult` dataclass:

| Field | Type | Description |
|-------|------|-------------|
| `weights` | `pd.Series` | Final asset weights (ticker → weight) |
| `portfolio` | skfolio `Portfolio` | In-sample portfolio with `.sharpe_ratio`, `.max_drawdown`, `.composition` |
| `backtest` | `MultiPeriodPortfolio` / `Population` / `None` | Out-of-sample results; `None` when backtesting was skipped |
| `pipeline` | sklearn `Pipeline` | The fitted pipeline, reusable for `predict()` on new data |
| `summary` | `dict[str, float]` | Key metrics: `mean`, `annualized_mean`, `variance`, `standard_deviation`, `sharpe_ratio`, `sortino_ratio`, `max_drawdown`, `cvar` |
| `rebalance_needed` | `bool` or `None` | Whether drift exceeds thresholds; `None` when no previous weights |
| `turnover` | `float` or `None` | One-way turnover vs previous weights |

## Transaction Cost Deduction

For net-of-cost backtest analysis, use `compute_net_backtest_returns`:

```python
from optimizer.pipeline import compute_net_backtest_returns

net_returns = compute_net_backtest_returns(
    gross_returns=backtest_returns,
    weight_changes=weight_change_df,
    cost_bps=10.0,  # 10 basis points per unit of turnover
)
```

## Code Examples

### Minimal pipeline

```python
from optimizer.optimization import MeanRiskConfig, build_mean_risk
from optimizer.pipeline import run_full_pipeline

optimizer = build_mean_risk(MeanRiskConfig.for_min_variance())
result = run_full_pipeline(prices=prices, optimizer=optimizer)
print(result.weights)
```

### With rebalancing

```python
from optimizer.rebalancing import ThresholdRebalancingConfig
import numpy as np

result = run_full_pipeline(
    prices=prices,
    optimizer=optimizer,
    previous_weights=np.array([0.25, 0.25, 0.25, 0.25]),
    rebalancing_config=ThresholdRebalancingConfig(threshold=0.05),
)
print(f"Rebalance needed: {result.rebalance_needed}")
print(f"Turnover: {result.turnover:.4f}")
```

### With stock selection

```python
from optimizer.factors import SelectionConfig, CompositeScoringConfig
from optimizer.universe import InvestabilityScreenConfig

result = run_full_pipeline_with_selection(
    prices=prices,
    optimizer=optimizer,
    fundamentals=fundamentals,
    volume_history=volume,
    investability_config=InvestabilityScreenConfig.for_developed_markets(),
    scoring_config=CompositeScoringConfig(),
    selection_config=SelectionConfig(n_stocks=50),
    cv_config=WalkForwardConfig.for_quarterly_rolling(),
)
```

### Hyperparameter tuning

```python
from optimizer.pipeline import tune_and_optimize, build_portfolio_pipeline
from skfolio.preprocessing import prices_to_returns

X = prices_to_returns(prices)
pipeline = build_portfolio_pipeline(optimizer)

result = tune_and_optimize(
    pipeline, X,
    param_grid={
        "optimizer__l2_coef": [0.0, 0.01, 0.1],
        "drop_correlated__threshold": [0.90, 0.95],
    },
)
print(f"Best params: {result.pipeline.get_params()}")
```

## Gotchas and Tips

!!! warning "prices_to_returns() is not in the pipeline"
    The price-to-return conversion runs **outside** the sklearn pipeline. Do not add it as a pipeline step — it changes data dimensionality (drops one row) which breaks cross-validation fold alignment.

!!! warning "previous_weights alignment"
    When `previous_weights` is passed to `run_full_pipeline()`, the function auto-aligns them on the post-pre-selection universe and re-normalizes. If pre-selection drops assets, their previous weights are set to zero and the remainder is rescaled to sum to 1.

!!! tip "Benchmark returns via y_prices"
    For `BenchmarkTracker` or any model that requires `fit(X, y)`, pass benchmark prices via `y_prices`. They are converted to returns alongside asset prices.

!!! tip "Sector mapping"
    Sector mapping is injected as a plain `dict[str, str]` (ticker → sector label), not queried from a database. Assets not in the mapping are assigned to an `"__unmapped__"` sector.

## Quick Reference

| Task | Code |
|------|------|
| Basic optimization | `run_full_pipeline(prices, optimizer)` |
| With backtest | `run_full_pipeline(prices, optimizer, cv_config=WalkForwardConfig())` |
| With rebalancing | `run_full_pipeline(prices, optimizer, previous_weights=w, rebalancing_config=cfg)` |
| With stock selection | `run_full_pipeline_with_selection(prices, optimizer, fundamentals=df)` |
| Manual pipeline | `build_portfolio_pipeline(optimizer)` then `optimize(pipeline, X)` |
| Tune + optimize | `tune_and_optimize(pipeline, X, param_grid={...})` |
| Net-of-cost returns | `compute_net_backtest_returns(gross, changes, cost_bps=10)` |
