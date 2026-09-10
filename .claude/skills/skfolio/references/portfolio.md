# BasePortfolio, Portfolio, MultiPeriodPortfolio, Population, FailedPortfolio

Output types of `predict` / `cross_val_predict` / `online_predict`. They carry performance/risk properties and plotting helpers so you rarely compute metrics by hand. Import from the top level: `from skfolio import Portfolio, MultiPeriodPortfolio, Population`.

## BasePortfolio

Shared base holding the returns-based metric machinery. Constructor: `BasePortfolio(returns, observations, annualization_factor=252.0, fitness_measures=[PerfMeasure.MEAN, RiskMeasure.VARIANCE], ...)` — `observations` is **required** (no default).

- Comparison operators (`==`, `>=`, `>`) and `dominates()` use `fitness_measures`.
- `np.asarray(portfolio)` returns the returns vector.
- Methods: `summary()`, `dominates(other)`, `rolling_measure(measure=...)`, plots.

## Portfolio

Returned by `model.predict(X)` on a single fit. Returns = `R·wᵀ − transaction_costs − management_fees`.

| Property | Description |
|---|---|
| `returns` / `cumulative_returns` | Return series |
| `mean` / `annualized_mean` | Mean return |
| `variance` / `standard_deviation` | Risk |
| `sharpe_ratio` / `annualized_sharpe_ratio` | Return / volatility |
| `sortino_ratio`, `calmar_ratio` | Ratio measures |
| `cvar`, `max_drawdown` | Tail / drawdown risk |
| `weights` | Asset weights (np.ndarray) |
| `composition` | DataFrame of weights with tickers |

Methods: `summary()`, `contribution(measure=...)`, `get_weight(asset)`, `plot_cumulative_returns()`, `plot_composition()`, `plot_contribution()`, `plot_returns()`, `plot_returns_distribution()`, `plot_rolling_measure()`.

⚠️ **1.0 rename:** `annualized_factor` → `annualization_factor` (default `252.0`). Old name deprecated (`FutureWarning`, removed in 2.0).

## MultiPeriodPortfolio

Sequence of portfolios across rebalancing periods — returned by `cross_val_predict(cv=WalkForward(...))` and `online_predict`. Same performance properties as `Portfolio`, computed on the concatenated return path, plus iteration over periods.

## FailedPortfolio (NEW in 1.0)

`from skfolio.portfolio import FailedPortfolio`. Sentinel returned by `predict()` when an optimizer fails and `raise_on_failure=False`. Carries `optimization_error` and `fallback_chain`. Lets a walk-forward / online backtest continue past an infeasible rebalance instead of crashing. See the resilience layer in `optimization.md`.

## Population

Collection of portfolios — returned by multi-path `cross_val_predict` (`CombinatorialPurgedCV`, `MultipleRandomizedCV`) or by `MeanRisk(efficient_frontier_size=...)`. Subclasses `list` (slicing returns a `Population`).

| Method | Description |
|---|---|
| `summary()` | Summary statistics for all portfolios |
| `composition()` | Compositions |
| `quantile(measure=..., q=...)` | Single Portfolio at a measure quantile |
| `non_dominated_sort(first_front_only=False)` | Multi-objective fronts |
| `plot_cumulative_returns()` | Overlay cumulative returns |
| `plot_measures(x=, y=, z=, show_fronts=True)` | Measure scatter / efficient frontier |
| `plot_composition()` / `plot_distribution(measure_list=...)` | Compare |
| `set_portfolio_params(compounded=True)` | Mutate held portfolios |

⚠️ **1.0 rename:** `non_denominated_sort` → `non_dominated_sort` (both the `Population` method and the standalone `skfolio.utils.sorting.non_dominated_sort` function; the standalone module also exports `dominate`). Old name deprecated (`FutureWarning`, removed in 2.0).

```python
from skfolio import Population

pop = Population([hrp_pf, meanrisk_pf, erc_pf])
pop.summary()
pop.plot_cumulative_returns()
fronts = pop.non_dominated_sort()
```
