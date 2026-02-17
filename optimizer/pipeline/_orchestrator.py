"""End-to-end portfolio orchestration functions."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from skfolio.preprocessing import prices_to_returns
from sklearn.pipeline import Pipeline

from optimizer.pipeline._builder import build_portfolio_pipeline
from optimizer.pipeline._config import PortfolioResult
from optimizer.pre_selection._config import PreSelectionConfig
from optimizer.rebalancing._config import ThresholdRebalancingConfig
from optimizer.rebalancing._rebalancer import (
    compute_turnover,
    should_rebalance,
)
from optimizer.tuning._config import GridSearchConfig
from optimizer.tuning._factory import build_grid_search_cv
from optimizer.validation._config import WalkForwardConfig
from optimizer.validation._factory import build_walk_forward, run_cross_val

# ---------------------------------------------------------------------------
# Low-level composable functions
# ---------------------------------------------------------------------------


def backtest(
    pipeline: Pipeline,
    X: pd.DataFrame,
    *,
    cv_config: WalkForwardConfig | None = None,
    y: pd.DataFrame | None = None,
    n_jobs: int | None = None,
) -> Any:
    """Run walk-forward backtest on a portfolio pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        A fitted-ready sklearn Pipeline (from ``build_portfolio_pipeline``).
    X : pd.DataFrame
        Return matrix (observations x assets).
    cv_config : WalkForwardConfig or None
        Walk-forward configuration.  Defaults to quarterly rolling.
    y : pd.DataFrame or None
        Benchmark or factor returns for models that require ``fit(X, y)``.
    n_jobs : int or None
        Number of parallel jobs.

    Returns
    -------
    MultiPeriodPortfolio or Population
        Out-of-sample portfolio predictions.
    """
    cv = build_walk_forward(cv_config)
    return run_cross_val(pipeline, X, cv=cv, y=y, n_jobs=n_jobs)


def optimize(
    pipeline: Pipeline,
    X: pd.DataFrame,
    *,
    y: pd.DataFrame | None = None,
) -> PortfolioResult:
    """Fit pipeline on full data and return final weights.

    Parameters
    ----------
    pipeline : Pipeline
        A fitted-ready sklearn Pipeline.
    X : pd.DataFrame
        Return matrix (observations x assets).
    y : pd.DataFrame or None
        Benchmark or factor returns.

    Returns
    -------
    PortfolioResult
        Weights, in-sample portfolio, and fitted pipeline.
    """
    if y is not None:
        pipeline.fit(X, y)
    else:
        pipeline.fit(X)

    portfolio = pipeline.predict(X)
    weights = _extract_weights(portfolio)
    summary = _extract_summary(portfolio)

    return PortfolioResult(
        weights=weights,
        portfolio=portfolio,
        pipeline=pipeline,
        summary=summary,
    )


def tune_and_optimize(
    pipeline: Pipeline,
    X: pd.DataFrame,
    param_grid: dict[str, list[Any]],
    *,
    tuning_config: GridSearchConfig | None = None,
    y: pd.DataFrame | None = None,
) -> PortfolioResult:
    """Tune hyperparameters via grid search, then optimise.

    Parameters
    ----------
    pipeline : Pipeline
        A fitted-ready sklearn Pipeline.
    X : pd.DataFrame
        Return matrix (observations x assets).
    param_grid : dict
        Parameter grid for ``GridSearchCV``.  Keys use sklearn
        double-underscore notation for nested parameters.
    tuning_config : GridSearchConfig or None
        Grid search configuration.  Defaults to quarterly
        walk-forward with Sharpe ratio scoring.
    y : pd.DataFrame or None
        Benchmark or factor returns.

    Returns
    -------
    PortfolioResult
        Weights from the best estimator, with backtest from CV.
    """
    gs = build_grid_search_cv(pipeline, param_grid, config=tuning_config)

    if y is not None:
        gs.fit(X, y)
    else:
        gs.fit(X)

    best_pipeline = cast(Pipeline, gs.best_estimator_)
    portfolio = best_pipeline.predict(X)
    weights = _extract_weights(portfolio)
    summary = _extract_summary(portfolio)

    return PortfolioResult(
        weights=weights,
        portfolio=portfolio,
        pipeline=best_pipeline,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# High-level end-to-end function
# ---------------------------------------------------------------------------


def run_full_pipeline(
    prices: pd.DataFrame,
    optimizer: Any,
    *,
    pre_selection_config: PreSelectionConfig | None = None,
    sector_mapping: dict[str, str] | None = None,
    cv_config: WalkForwardConfig | None = None,
    previous_weights: npt.NDArray[np.float64] | None = None,
    rebalancing_config: ThresholdRebalancingConfig | None = None,
    y_prices: pd.DataFrame | None = None,
    n_jobs: int | None = None,
) -> PortfolioResult:
    """End-to-end: prices → validated weights + backtest + rebalancing.

    This is the single entry point for producing a portfolio from
    raw price data.  It:

    1. Converts prices to linear returns.
    2. Builds the full pipeline (pre-selection + optimiser).
    3. Backtests via walk-forward (if ``cv_config`` is provided).
    4. Fits on full data to produce final weights.
    5. Checks rebalancing thresholds (if ``previous_weights`` given).

    Parameters
    ----------
    prices : pd.DataFrame
        Price matrix (dates x tickers).
    optimizer : BaseOptimization
        A skfolio optimiser instance (e.g. from ``build_mean_risk()``).
    pre_selection_config : PreSelectionConfig or None
        Pre-selection configuration.
    sector_mapping : dict[str, str] or None
        Ticker → sector mapping for imputation.
    cv_config : WalkForwardConfig or None
        Walk-forward backtest configuration.  ``None`` skips
        backtesting.
    previous_weights : ndarray or None
        Current portfolio weights for rebalancing analysis.
    rebalancing_config : ThresholdRebalancingConfig or None
        Threshold configuration for rebalancing decisions.
    y_prices : pd.DataFrame or None
        Benchmark or factor price series.  Converted to returns
        alongside asset prices.
    n_jobs : int or None
        Number of parallel jobs for backtesting.

    Returns
    -------
    PortfolioResult
        Complete result with weights, portfolio metrics, optional
        backtest, and rebalancing signals.

    Examples
    --------
    >>> from optimizer.optimization import MeanRiskConfig, build_mean_risk
    >>> from optimizer.validation import WalkForwardConfig
    >>> from optimizer.pipeline import run_full_pipeline
    >>>
    >>> optimizer = build_mean_risk(MeanRiskConfig.for_max_sharpe())
    >>> result = run_full_pipeline(
    ...     prices=price_df,
    ...     optimizer=optimizer,
    ...     cv_config=WalkForwardConfig.for_quarterly_rolling(),
    ... )
    >>> print(result.weights)
    >>> print(result.summary)
    >>> print(result.backtest.sharpe_ratio)  # out-of-sample
    """
    # 1. Prices → returns
    X = cast(pd.DataFrame, prices_to_returns(prices))
    y: pd.DataFrame | None = (
        cast(pd.DataFrame, prices_to_returns(y_prices))
        if y_prices is not None
        else None
    )

    # 2. Build pipeline
    pipeline = build_portfolio_pipeline(
        optimizer=optimizer,
        pre_selection_config=pre_selection_config,
        sector_mapping=sector_mapping,
    )

    # 3. Backtest (optional)
    bt = None
    if cv_config is not None:
        bt = backtest(pipeline, X, cv_config=cv_config, y=y, n_jobs=n_jobs)

    # 4. Fit on full data → final weights
    result = optimize(pipeline, X, y=y)
    result.backtest = bt

    # 5. Rebalancing analysis (optional)
    if previous_weights is not None:
        prev_series = (
            previous_weights
            if isinstance(previous_weights, pd.Series)
            else pd.Series(previous_weights, index=X.columns)
        )
        # Align on the new weight universe (pre-selection may drop assets)
        aligned_prev = prev_series.reindex(result.weights.index, fill_value=0.0)
        # Re-normalise so they sum to the same budget
        prev_sum = aligned_prev.sum()
        if prev_sum > 0:
            aligned_prev = aligned_prev / prev_sum

        prev_arr = aligned_prev.to_numpy()
        new_arr = result.weights.to_numpy()
        result.turnover = compute_turnover(prev_arr, new_arr)
        result.rebalance_needed = should_rebalance(
            prev_arr,
            new_arr,
            config=rebalancing_config,
        )

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_weights(portfolio: Any) -> pd.Series:
    """Extract asset weights as a named Series from a skfolio Portfolio."""
    composition = portfolio.composition
    # composition is a DataFrame with a single column and asset-name index
    return composition.iloc[:, 0]


def _extract_summary(portfolio: Any) -> dict[str, float]:
    """Extract key metrics from a skfolio Portfolio."""
    attrs = [
        "mean",
        "annualized_mean",
        "variance",
        "standard_deviation",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "cvar",
    ]
    summary: dict[str, float] = {}
    for attr in attrs:
        val = getattr(portfolio, attr, None)
        if val is not None:
            summary[attr] = float(val)
    return summary
