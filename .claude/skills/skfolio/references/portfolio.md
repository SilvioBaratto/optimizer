# Portfolio, MultiPeriodPortfolio, Population

These are the output types of `predict` / `cross_val_predict` / `online_predict`. They carry performance properties and plotting helpers so you rarely need to compute metrics by hand.

## Portfolio

Returned by `model.predict(X)` on a single fit.

| Property | Description |
|---|---|
| `returns` | Return series (pd.Series) |
| `cumulative_returns` | Cumulative return series |
| `mean` | Mean return |
| `annualized_mean` | Annualized mean return |
| `variance` | Portfolio variance |
| `standard_deviation` | Portfolio volatility |
| `sharpe_ratio` | Return / volatility |
| `sortino_ratio` | Return / downside deviation |
| `calmar_ratio` | Return / max drawdown |
| `cvar` | Conditional Value at Risk |
| `max_drawdown` | Maximum drawdown |
| `weights` | Asset weights (np.ndarray) |
| `composition` | DataFrame of weights with tickers |

Methods: `summary()`, `plot_cumulative_returns()`, `plot_composition()`, `plot_returns()`, `plot_returns_distribution()`, `plot_rolling_measure()`.

## MultiPeriodPortfolio

Sequence of portfolios across rebalancing periods — returned by `cross_val_predict(cv=WalkForward(...))` and `online_predict`. Has the same performance properties as `Portfolio` computed on the concatenated return path, plus iteration over individual periods.

## Population

Collection of portfolios — returned by `cross_val_predict(cv=CombinatorialPurgedCV(...))` or built manually to compare strategies.

| Method | Description |
|---|---|
| `summary()` | Summary statistics for all portfolios |
| `plot_cumulative_returns()` | Overlay cumulative returns |
| `plot_composition()` | Compare compositions |
| `plot_frontier()` | Efficient frontier |
| `filter()` | Filter by criteria |
| `sort()` | Sort by measure |

Use `Population` to compare optimizers head-to-head on the same data:

```python
from skfolio.population import Population

pop = Population([hrp_pf, meanrisk_pf, erc_pf])
pop.summary()
pop.plot_cumulative_returns()
```
