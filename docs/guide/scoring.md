# Scoring

The scoring module wraps skfolio ratio measures and custom scoring functions into callables compatible with sklearn cross-validation and hyperparameter tuning. It provides a consistent interface for evaluating portfolio performance during model selection.

## Overview

When using `GridSearchCV` or `RandomizedSearchCV` with portfolio optimizers, you need a scoring function that evaluates the quality of each portfolio. The scoring module maps ratio measure names to sklearn-compatible scorer callables via the frozen-config + factory pattern.

## ScorerConfig

```python
from optimizer.scoring import ScorerConfig

config = ScorerConfig(
    ratio_measure=RatioMeasureType.SHARPE_RATIO,
    greater_is_better=None,  # auto-detected from measure
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ratio_measure` | `RatioMeasureType` or `None` | `SHARPE_RATIO` | Built-in ratio measure; `None` for custom scorer |
| `greater_is_better` | `bool` or `None` | `None` | Whether higher scores are better; auto-detected when `None` |

### Presets

| Preset | Ratio Measure | Use Case |
|--------|---------------|----------|
| `ScorerConfig.for_sharpe()` | Sharpe Ratio | General-purpose risk-adjusted return |
| `ScorerConfig.for_sortino()` | Sortino Ratio | Downside-risk-focused evaluation |
| `ScorerConfig.for_calmar()` | Calmar Ratio | Drawdown-focused evaluation |
| `ScorerConfig.for_cvar_ratio()` | CVaR Ratio | Tail-risk-focused evaluation |
| `ScorerConfig.for_information_ratio()` | Information Ratio | Active return vs benchmark |
| `ScorerConfig.for_custom()` | `None` | Custom callable passed to factory |

## Available Ratio Measures

All 19 ratio measures available in `RatioMeasureType`:

| Measure | Description |
|---------|-------------|
| `SHARPE_RATIO` | Excess return / standard deviation |
| `ANNUALIZED_SHARPE_RATIO` | Annualized Sharpe ratio |
| `SORTINO_RATIO` | Excess return / downside deviation |
| `ANNUALIZED_SORTINO_RATIO` | Annualized Sortino ratio |
| `MEAN_ABSOLUTE_DEVIATION_RATIO` | Return / mean absolute deviation |
| `FIRST_LOWER_PARTIAL_MOMENT_RATIO` | Return / first lower partial moment |
| `VALUE_AT_RISK_RATIO` | Return / VaR |
| `CVAR_RATIO` | Return / CVaR |
| `ENTROPIC_RISK_MEASURE_RATIO` | Return / entropic risk |
| `EVAR_RATIO` | Return / EVaR |
| `WORST_REALIZATION_RATIO` | Return / worst realization |
| `DRAWDOWN_AT_RISK_RATIO` | Return / drawdown-at-risk |
| `CDAR_RATIO` | Return / CDaR |
| `CALMAR_RATIO` | Return / max drawdown |
| `AVERAGE_DRAWDOWN_RATIO` | Return / average drawdown |
| `EDAR_RATIO` | Return / EDaR |
| `ULCER_INDEX_RATIO` | Return / ulcer index |
| `GINI_MEAN_DIFFERENCE_RATIO` | Return / Gini mean difference |
| `INFORMATION_RATIO` | Active return / tracking error (custom) |

## Building Scorers

```python
from optimizer.scoring import ScorerConfig, build_scorer

# Built-in ratio measure
scorer = build_scorer(ScorerConfig.for_sharpe())

# Information ratio (requires benchmark)
scorer = build_scorer(
    ScorerConfig.for_information_ratio(),
    benchmark_returns=benchmark_returns,
)

# Custom scoring function
def my_scorer(portfolio):
    return portfolio.annualized_mean / portfolio.max_drawdown

scorer = build_scorer(ScorerConfig.for_custom(), custom_func=my_scorer)
```

## Code Examples

### Using with hyperparameter tuning

```python
from optimizer.scoring import ScorerConfig
from optimizer.tuning import GridSearchConfig

# Grid search scored by Sortino ratio
tuning_config = GridSearchConfig(
    scorer_config=ScorerConfig.for_sortino(),
    n_jobs=-1,
)
```

### Using with cross-validation

```python
from optimizer.scoring import ScorerConfig, build_scorer
from optimizer.validation import WalkForwardConfig, run_cross_val

scorer = build_scorer(ScorerConfig.for_calmar())
# scorer is a callable compatible with sklearn CV
```

## Gotchas and Tips

!!! tip "Information Ratio requires benchmark"
    The `INFORMATION_RATIO` is not a native skfolio ratio measure — it is implemented as a custom scorer (active return / tracking error). You must pass `benchmark_returns` to `build_scorer()`.

!!! tip "Scorer sign convention"
    All built-in ratio measures follow the sklearn convention where `greater_is_better=True`. The scorer returns positive values for good portfolios and the search maximizes the score.

!!! tip "Default scorer"
    When no `ScorerConfig` is provided to tuning, the default is Sharpe ratio — a reasonable choice for most equity portfolio strategies.

## Quick Reference

| Task | Code |
|------|------|
| Sharpe scorer | `build_scorer(ScorerConfig.for_sharpe())` |
| Sortino scorer | `build_scorer(ScorerConfig.for_sortino())` |
| Calmar scorer | `build_scorer(ScorerConfig.for_calmar())` |
| CVaR ratio scorer | `build_scorer(ScorerConfig.for_cvar_ratio())` |
| Information ratio | `build_scorer(ScorerConfig.for_information_ratio(), benchmark_returns=bm)` |
| Custom scorer | `build_scorer(ScorerConfig.for_custom(), custom_func=fn)` |
