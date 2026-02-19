"""End-to-end portfolio orchestration functions."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from skfolio.preprocessing import prices_to_returns
from sklearn.pipeline import Pipeline

from optimizer.factors._config import (
    GROUP_WEIGHT_TIER,
    CompositeScoringConfig,
    FactorConstructionConfig,
    FactorGroupType,
    FactorIntegrationConfig,
    GroupWeight,
    RegimeTiltConfig,
    SelectionConfig,
    StandardizationConfig,
)
from optimizer.factors._construction import compute_all_factors
from optimizer.factors._regime import apply_regime_tilts, classify_regime
from optimizer.factors._scoring import compute_composite_score
from optimizer.factors._selection import select_stocks
from optimizer.factors._standardization import standardize_all_factors
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
from optimizer.universe._config import InvestabilityScreenConfig
from optimizer.universe._factory import screen_universe
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


def run_full_pipeline_with_selection(
    prices: pd.DataFrame,
    optimizer: Any,
    *,
    fundamentals: pd.DataFrame | None = None,
    volume_history: pd.DataFrame | None = None,
    financial_statements: pd.DataFrame | None = None,
    analyst_data: pd.DataFrame | None = None,
    insider_data: pd.DataFrame | None = None,
    macro_data: pd.DataFrame | None = None,
    investability_config: InvestabilityScreenConfig | None = None,
    factor_config: FactorConstructionConfig | None = None,
    standardization_config: StandardizationConfig | None = None,
    scoring_config: CompositeScoringConfig | None = None,
    selection_config: SelectionConfig | None = None,
    regime_config: RegimeTiltConfig | None = None,
    integration_config: FactorIntegrationConfig | None = None,
    sector_mapping: dict[str, str] | None = None,
    pre_selection_config: PreSelectionConfig | None = None,
    cv_config: WalkForwardConfig | None = None,
    previous_weights: npt.NDArray[np.float64] | None = None,
    rebalancing_config: ThresholdRebalancingConfig | None = None,
    y_prices: pd.DataFrame | None = None,
    current_members: pd.Index | None = None,
    ic_history: pd.DataFrame | None = None,
    n_jobs: int | None = None,
) -> PortfolioResult:
    """End-to-end: fundamentals + prices → stock selection → optimization.

    Extends :func:`run_full_pipeline` with upstream stock pre-selection:

    1. Screen universe for investability (if ``fundamentals`` provided).
    2. Compute and standardize factor scores.
    3. Apply macro regime tilts (if ``macro_data`` + ``regime_config``).
    4. Compute composite score and select stocks.
    5. Run existing ``run_full_pipeline`` on selected tickers.

    Parameters
    ----------
    prices : pd.DataFrame
        Price matrix (dates x tickers).
    optimizer : BaseOptimization
        A skfolio optimiser instance.
    fundamentals : pd.DataFrame or None
        Cross-sectional data indexed by ticker (market_cap, ratios).
        If ``None``, skips screening and factor selection.
    volume_history : pd.DataFrame or None
        Volume matrix (dates x tickers).
    financial_statements : pd.DataFrame or None
        Statement-level data for screening.
    analyst_data : pd.DataFrame or None
        Analyst recommendation data for factor construction.
    insider_data : pd.DataFrame or None
        Insider transaction data for factor construction.
    macro_data : pd.DataFrame or None
        Macro indicators for regime classification.
    investability_config : InvestabilityScreenConfig or None
        Universe screening configuration.
    factor_config : FactorConstructionConfig or None
        Factor construction parameters.
    standardization_config : StandardizationConfig or None
        Factor standardization parameters.
    scoring_config : CompositeScoringConfig or None
        Composite scoring parameters.
    selection_config : SelectionConfig or None
        Stock selection parameters.
    regime_config : RegimeTiltConfig or None
        Regime tilt parameters.
    integration_config : FactorIntegrationConfig or None
        Factor-to-optimization bridge parameters.
    sector_mapping : dict[str, str] or None
        Ticker -> sector mapping.
    pre_selection_config : PreSelectionConfig or None
        Return-data pre-selection configuration.
    cv_config : WalkForwardConfig or None
        Walk-forward backtest configuration.
    previous_weights : ndarray or None
        Current portfolio weights for rebalancing.
    rebalancing_config : ThresholdRebalancingConfig or None
        Rebalancing threshold configuration.
    y_prices : pd.DataFrame or None
        Benchmark or factor price series.
    current_members : pd.Index or None
        Currently selected tickers for hysteresis.
    ic_history : pd.DataFrame or None
        IC history for IC-weighted scoring.
    n_jobs : int or None
        Number of parallel jobs.

    Returns
    -------
    PortfolioResult
        Complete result with weights, metrics, backtest, and
        rebalancing signals.
    """
    selected_prices = prices

    if fundamentals is not None:
        vol = volume_history if volume_history is not None else pd.DataFrame()

        # 1. Screen universe for investability
        investable = screen_universe(
            fundamentals=fundamentals,
            price_history=prices,
            volume_history=vol,
            financial_statements=financial_statements,
            config=investability_config,
            current_members=current_members,
        )

        investable_fundamentals = fundamentals.loc[
            fundamentals.index.intersection(investable)
        ]
        investable_prices = prices[
            prices.columns.intersection(investable)
        ]

        # 2. Compute factors
        investable_vol = (
            vol[vol.columns.intersection(investable)]
            if len(vol) > 0
            else None
        )
        raw_factors = compute_all_factors(
            fundamentals=investable_fundamentals,
            price_history=investable_prices,
            volume_history=investable_vol,
            analyst_data=analyst_data,
            insider_data=insider_data,
            config=factor_config,
        )

        # 3. Standardize
        sector_labels = (
            pd.Series(sector_mapping).reindex(investable_fundamentals.index)
            if sector_mapping
            else None
        )
        standardized, coverage = standardize_all_factors(
            raw_factors,
            config=standardization_config,
            sector_labels=sector_labels,
        )

        # 4. Regime tilts (optional)
        has_regime = (
            macro_data is not None
            and regime_config is not None
            and regime_config.enable
        )
        if has_regime:
            regime = classify_regime(macro_data)

            # Build base group weights from config
            _scoring = scoring_config or CompositeScoringConfig()
            base_weights: dict[FactorGroupType, float] = {}
            for group in FactorGroupType:
                tier = GROUP_WEIGHT_TIER[group]
                base_weights[group] = (
                    _scoring.core_weight
                    if tier == GroupWeight.CORE
                    else _scoring.supplementary_weight
                )

            apply_regime_tilts(base_weights, regime, regime_config)

        # 5. Composite score
        composite = compute_composite_score(
            standardized, coverage,
            config=scoring_config,
            ic_history=ic_history,
        )

        # 6. Select stocks
        selected = select_stocks(
            composite,
            config=selection_config,
            current_members=current_members,
            sector_labels=sector_labels,
            parent_universe=investable,
        )

        selected_prices = prices[prices.columns.intersection(selected)]

    # 7. Delegate to existing pipeline
    return run_full_pipeline(
        prices=selected_prices,
        optimizer=optimizer,
        pre_selection_config=pre_selection_config,
        sector_mapping=sector_mapping,
        cv_config=cv_config,
        previous_weights=previous_weights,
        rebalancing_config=rebalancing_config,
        y_prices=y_prices,
        n_jobs=n_jobs,
    )


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
