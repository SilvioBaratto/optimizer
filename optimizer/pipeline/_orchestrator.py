"""End-to-end portfolio orchestration functions."""

from __future__ import annotations

import logging
from copy import deepcopy
from math import sqrt
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from skfolio.preprocessing import prices_to_returns
from sklearn.pipeline import Pipeline

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.factors._config import (
    GROUP_WEIGHT_TIER,
    CompositeScoringConfig,
    FactorConstructionConfig,
    FactorGroupType,
    FactorIntegrationConfig,
    GroupWeight,
    PublicationLagConfig,
    RegimeTiltConfig,
    SelectionConfig,
    StandardizationConfig,
)
from optimizer.factors._construction import compute_all_factors
from optimizer.factors._regime import apply_regime_tilts, classify_regime
from optimizer.factors._scoring import compute_composite_score
from optimizer.factors._selection import select_stocks
from optimizer.factors._standardization import standardize_all_factors
from optimizer.fx._config import FxConfig, FxConversionMode
from optimizer.fx._decomposition import decompose_fx_returns
from optimizer.fx._factory import build_fx_converter
from optimizer.optimization._config import RatioMeasureType
from optimizer.pipeline._builder import build_portfolio_pipeline
from optimizer.pipeline._config import PortfolioResult
from optimizer.pre_selection._config import PreSelectionConfig
from optimizer.preprocessing._delisting import (
    apply_delisting_returns as _apply_delisting_rets,
)
from optimizer.rebalancing._config import (
    HybridRebalancingConfig,
    ThresholdRebalancingConfig,
)
from optimizer.rebalancing._rebalancer import (
    compute_turnover,
    should_rebalance,
    should_rebalance_hybrid,
)
from optimizer.tuning._config import GridSearchConfig, RandomizedSearchConfig
from optimizer.tuning._factory import build_grid_search_cv, build_randomized_search_cv
from optimizer.universe._config import InvestabilityScreenConfig
from optimizer.universe._factory import screen_universe
from optimizer.validation._config import WalkForwardConfig
from optimizer.validation._factory import build_walk_forward, run_cross_val

logger = logging.getLogger(__name__)

_MIN_SELECTED_STOCKS = 5

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
    tuning_config: GridSearchConfig | RandomizedSearchConfig | None = None,
    y: pd.DataFrame | None = None,
    risk_free_rate: float = 0.0,
) -> PortfolioResult:
    """Tune hyperparameters via grid or randomized search, then optimise.

    Parameters
    ----------
    pipeline : Pipeline
        A fitted-ready sklearn Pipeline.
    X : pd.DataFrame
        Return matrix (observations x assets).
    param_grid : dict
        Parameter grid for ``GridSearchCV`` or distributions for
        ``RandomizedSearchCV``.  Keys use sklearn double-underscore
        notation for nested parameters.
    tuning_config : GridSearchConfig or RandomizedSearchConfig or None
        Search configuration.  Defaults to quarterly walk-forward
        with Sharpe ratio scoring (grid search).
    y : pd.DataFrame or None
        Benchmark or factor returns.
    risk_free_rate : float
        Daily risk-free rate for consistent Sharpe scoring (issue #272).
        When non-zero and the scorer uses Sharpe ratio, the scorer
        config is updated to use this rate.

    Returns
    -------
    PortfolioResult
        Weights from the best estimator, with backtest from CV.
    """
    # Inject risk_free_rate into the tuning scorer (issue #272)
    if risk_free_rate != 0.0 and tuning_config is not None:
        from dataclasses import replace

        from optimizer.scoring._config import ScorerConfig

        sc = tuning_config.scorer_config
        is_sharpe = sc.ratio_measure == RatioMeasureType.SHARPE_RATIO
        if is_sharpe and sc.risk_free_rate == 0.0:
            new_scorer = ScorerConfig.for_sharpe_with_rf(risk_free_rate)
            tuning_config = replace(tuning_config, scorer_config=new_scorer)

    if isinstance(tuning_config, RandomizedSearchConfig):
        gs = build_randomized_search_cv(pipeline, param_grid, config=tuning_config)
    else:
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


def compute_net_backtest_returns(
    gross_returns: pd.Series,
    weight_changes: pd.DataFrame,
    cost_bps: float = 10.0,
) -> pd.Series:
    """Deduct proportional transaction costs from gross backtest returns.

    For each date with weight changes, the one-way turnover (half the sum
    of absolute weight deltas, consistent with ``compute_turnover()``) is
    multiplied by ``cost_bps / 10_000`` and subtracted from the gross
    return at that date.  A shift of weight *w* from one asset to another
    incurs a cost of ``w * cost_bps / 10_000``, not ``2w``.

    Parameters
    ----------
    gross_returns : pd.Series
        Gross portfolio returns indexed by date.
    weight_changes : pd.DataFrame
        Weight change matrix (dates x assets).  Only dates present
        in this DataFrame incur transaction costs.
    cost_bps : float
        Transaction cost in basis points (default 10 bps).

    Returns
    -------
    pd.Series
        Net returns with costs deducted.
    """
    net = gross_returns.copy()
    cost_frac = cost_bps / 10_000.0

    for date in weight_changes.index:
        if date not in net.index:
            continue
        row = weight_changes.loc[date].to_numpy(dtype=np.float64)
        turnover = float(np.sum(np.abs(row))) / 2.0
        net.at[date] = net.at[date] - turnover * cost_frac

    return net


def run_full_pipeline(
    prices: pd.DataFrame,
    optimizer: Any,
    *,
    pre_selection_config: PreSelectionConfig | None = None,
    sector_mapping: dict[str, str] | None = None,
    cv_config: WalkForwardConfig | None = None,
    previous_weights: npt.NDArray[np.float64] | None = None,
    rebalancing_config: (
        ThresholdRebalancingConfig | HybridRebalancingConfig | None
    ) = None,
    current_date: pd.Timestamp | None = None,
    last_review_date: pd.Timestamp | None = None,
    y_prices: pd.DataFrame | None = None,
    risk_free_rate: float = 0.0,
    delisting_returns: dict[str, float] | None = None,
    fx_config: FxConfig | None = None,
    currency_map: dict[str, str] | None = None,
    fx_rates: pd.DataFrame | None = None,
    benchmark_currency: str | None = None,
    cost_bps: float = 10.0,
    n_jobs: int | None = None,
) -> PortfolioResult:
    """End-to-end: prices → validated weights + backtest + rebalancing.

    This is the single entry point for producing a portfolio from
    raw price data.  It:

    1. Converts prices to linear returns.
    1b. Applies delisting returns (survivorship-bias correction).
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
    rebalancing_config : ThresholdRebalancingConfig or HybridRebalancingConfig or None
        Rebalancing configuration.  Pass a ``ThresholdRebalancingConfig``
        for pure drift-based rebalancing or a ``HybridRebalancingConfig``
        for calendar-gated threshold rebalancing.
    current_date : pd.Timestamp or None
        Evaluation date for hybrid rebalancing.  Defaults to the last
        date in the return series when not provided.
    last_review_date : pd.Timestamp or None
        Date of the last hybrid review.  When ``None`` with a
        ``HybridRebalancingConfig``, the calendar gate is treated as
        already elapsed (threshold alone decides).
    y_prices : pd.DataFrame or None
        Benchmark or factor price series.  Converted to returns
        alongside asset prices.
    delisting_returns : dict[str, float] or None
        Mapping of ticker → terminal delisting return.  When provided,
        each ticker's last valid return is replaced with this value
        after ``prices_to_returns()`` (survivorship-bias correction,
        issue #274).  Tickers not present in the returns columns are
        silently ignored.
    fx_config : FxConfig or None
        Multi-currency FX conversion configuration (issue #283).
        When provided with ``mode != NONE``, prices are converted to
        the base currency before ``prices_to_returns()``.  ``None``
        disables conversion (default, backward-compatible).
    currency_map : dict[str, str] or None
        Ticker → ISO currency code mapping.  Required when
        ``fx_config`` is provided.
    fx_rates : pd.DataFrame or None
        Pre-loaded FX rate DataFrame (dates x currencies).  Each
        column holds units-of-base per one unit-of-foreign.
        Required when ``fx_config`` is provided.
    benchmark_currency : str | None
        ISO currency code for the benchmark in ``y_prices`` (issue #308).
        When provided and FX conversion is active, all columns of
        ``y_prices`` are treated as denominated in this currency and
        converted to ``fx_config.base_currency`` before returns are
        computed.  ``None`` (default) preserves existing behaviour:
        the benchmark is converted only if its ticker already appears
        in ``currency_map``.
    cost_bps : float
        One-way transaction cost in basis points applied to each
        walk-forward rebalancing event.  Subtracted from gross backtest
        returns to produce ``result.net_returns`` and
        ``result.net_sharpe_ratio``.  Default 10 bps.
    n_jobs : int or None
        Number of parallel jobs for backtesting.

    Returns
    -------
    PortfolioResult
        Complete result with weights, portfolio metrics, optional
        backtest, net returns, and rebalancing signals.

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
    # 0.5  FX conversion — convert local-currency prices to base currency
    #       (issue #283).  Must happen before prices_to_returns().
    fx_decomp = None
    fx_currency: str | None = None
    if fx_config is not None and fx_config.mode != FxConversionMode.NONE:
        _missing = [
            name
            for name, val in (
                ("currency_map", currency_map),
                ("fx_rates", fx_rates),
            )
            if val is None
        ]
        if _missing:
            _missing_str = " and ".join(_missing)
            _msg = (
                "FX conversion requested (mode=%s) but %s is None; conversion skipped."
            )
            if fx_config.strict:
                raise ConfigurationError(_msg % (fx_config.mode.value, _missing_str))
            logger.warning(_msg, fx_config.mode.value, _missing_str)
        else:
            # Both are non-None here (guarded by _missing check above)
            _currency_map = cast(dict[str, str], currency_map)
            _fx_rates = cast(pd.DataFrame, fx_rates)
            converter = build_fx_converter(
                fx_config, fx_rates=_fx_rates, currency_map=_currency_map
            )
            local_prices = prices.copy()
            converter.fit(prices)
            prices = converter.transform(prices)
            fx_currency = fx_config.base_currency.value
            logger.info("FX-converted prices to %s.", fx_currency)

            # Also convert benchmark prices (issue #308).
            # Build a dedicated map for y_prices: start from the portfolio
            # currency_map and overlay benchmark_currency for every column
            # in y_prices so foreign benchmarks (e.g. SPY in USD for a
            # EUR-base portfolio) are explicitly converted even when absent
            # from the portfolio currency_map.
            if y_prices is not None:
                _bench_map: dict[str, str] = dict(_currency_map)
                if benchmark_currency is not None:
                    _bench_ccy = benchmark_currency.upper()
                    for _col in y_prices.columns:
                        _bench_map[_col] = _bench_ccy
                    logger.info(
                        "Benchmark FX: treating y_prices columns %s as %s "
                        "(issue #308).",
                        list(y_prices.columns),
                        _bench_ccy,
                    )
                bench_converter = build_fx_converter(
                    fx_config, fx_rates=_fx_rates, currency_map=_bench_map
                )
                bench_converter.fit(y_prices)
                y_prices = bench_converter.transform(y_prices)

            # Decompose returns if requested
            if fx_config.mode == FxConversionMode.DECOMPOSE:
                fx_decomp = decompose_fx_returns(
                    local_prices=local_prices,
                    base_prices=prices,
                    fx_rates_aligned=converter.fx_aligned_,
                    currency_map=_currency_map,
                    base_currency=fx_currency,
                )

    # 1. Prices → returns
    X = cast(pd.DataFrame, prices_to_returns(prices))
    y: pd.DataFrame | None = (
        cast(pd.DataFrame, prices_to_returns(y_prices))
        if y_prices is not None
        else None
    )

    # 1.5  Apply delisting returns (survivorship-bias correction, issue #274)
    if delisting_returns:
        present = {t: r for t, r in delisting_returns.items() if t in X.columns}
        if present:
            X = _apply_delisting_rets(X, present)
            logger.info("Applied delisting returns for %d ticker(s).", len(present))

    # 1.6  Inject risk-free rate into optimizer (issue #272)
    if risk_free_rate != 0.0:
        if hasattr(optimizer, "risk_free_rate"):
            optimizer = deepcopy(optimizer)
            optimizer.risk_free_rate = risk_free_rate
            logger.info(
                "Injected risk_free_rate=%.6f into %s",
                risk_free_rate,
                type(optimizer).__name__,
            )
        else:
            logger.warning(
                "risk_free_rate=%.6f provided but %s has no risk_free_rate "
                "attribute; Sharpe calculation will use rf=0.",
                risk_free_rate,
                type(optimizer).__name__,
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

    # 3b. Net backtest returns (issue #284) + weight history (issue #285)
    if bt is not None and hasattr(bt, "weights_per_observation"):
        wpo = bt.weights_per_observation
        wc = _extract_weight_changes(wpo)
        # Absolute weights at rebalancing dates (for compute_net_alpha)
        result.weight_history = wpo.loc[wc.index]
        # Resilient returns accessor (issue #309): prefer returns_df but fall
        # back to public members if the attribute is absent in future skfolio.
        if hasattr(bt, "returns_df"):
            gross = bt.returns_df
        else:
            logger.warning(
                "bt.returns_df not found on %s; reconstructing from "
                "bt.returns + bt.observations (issue #309).",
                type(bt).__name__,
            )
            gross = pd.Series(index=bt.observations, data=bt.returns, name="returns")
        net = compute_net_backtest_returns(gross, wc, cost_bps=cost_bps)
        result.net_returns = net
        result.net_sharpe_ratio = _compute_net_sharpe(
            net, risk_free_rate=risk_free_rate
        )
        logger.info(
            "Net backtest Sharpe: %.4f (gross: %.4f, cost=%.1f bps)",
            result.net_sharpe_ratio or float("nan"),
            float(bt.sharpe_ratio),
            cost_bps,
        )
    elif bt is not None:
        logger.warning(
            "Net backtest skipped: %s has no weights_per_observation.",
            type(bt).__name__,
        )

    result.risk_free_rate = risk_free_rate
    result.fx_decomposition = fx_decomp
    result.currency = fx_currency

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

        prev_arr = cast(
            npt.NDArray[np.float64], aligned_prev.to_numpy(dtype=np.float64)
        )
        new_arr = cast(
            npt.NDArray[np.float64], result.weights.to_numpy(dtype=np.float64)
        )
        result.turnover = compute_turnover(prev_arr, new_arr)
        if isinstance(rebalancing_config, HybridRebalancingConfig):
            _current = (
                current_date
                if current_date is not None
                else cast(pd.Timestamp, X.index[-1])
            )
            _last_review = (
                last_review_date
                if last_review_date is not None
                else cast(
                    pd.Timestamp,
                    _current
                    - pd.Timedelta(days=rebalancing_config.calendar.trading_days * 2),
                )
            )
            result.rebalance_needed = should_rebalance_hybrid(
                prev_arr, new_arr, rebalancing_config, _current, _last_review
            )
        else:
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
    regime_data: pd.DataFrame | None = None,
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
    rebalancing_config: (
        ThresholdRebalancingConfig | HybridRebalancingConfig | None
    ) = None,
    current_date: pd.Timestamp | None = None,
    last_review_date: pd.Timestamp | None = None,
    y_prices: pd.DataFrame | None = None,
    current_members: pd.Index | None = None,
    ic_history: pd.DataFrame | None = None,
    risk_free_rate: float = 0.0,
    delisting_returns: dict[str, float] | None = None,
    market_returns: pd.Series | None = None,
    fx_config: FxConfig | None = None,
    currency_map: dict[str, str] | None = None,
    fx_rates: pd.DataFrame | None = None,
    benchmark_currency: str | None = None,
    cost_bps: float = 10.0,
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
    regime_data : pd.DataFrame or None
        Merged macro indicators (pmi, spread_2s10s, hy_oas, etc.)
        for composite regime classification.  When provided and
        non-empty, takes precedence over ``macro_data`` for regime
        classification.  Receives the same publication lag filtering.
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
    market_returns : pd.Series or None
        Pre-computed market return series for beta estimation.
        When provided, used as the benchmark instead of the
        equal-weight cross-sectional mean.  Pass a currency-
        consistent broad index (e.g. SPY daily returns) when
        ``prices`` spans multiple currency zones.
    benchmark_currency : str | None
        ISO currency code for the benchmark in ``y_prices``.
        Forwarded verbatim to :func:`run_full_pipeline`; see that
        function's documentation for full semantics (issue #308).
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
        investable_prices = prices[prices.columns.intersection(investable)]

        # 2. Compute factors
        investable_vol = (
            vol[vol.columns.intersection(investable)] if len(vol) > 0 else None
        )
        raw_factors = compute_all_factors(
            fundamentals=investable_fundamentals,
            price_history=investable_prices,
            volume_history=investable_vol,
            analyst_data=analyst_data,
            insider_data=insider_data,
            config=factor_config,
            market_returns=market_returns,
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
        group_weights: dict[str, float] | None = None
        if has_regime:
            if macro_data is None:  # pragma: no cover — guarded by has_regime
                msg = "macro_data is required when regime_config is enabled"
                raise DataError(msg)

            # Apply publication lag to macro data (point-in-time correctness)
            _lag_cfg = (
                factor_config.publication_lag
                if factor_config is not None
                else PublicationLagConfig()
            )
            macro_days = _lag_cfg.macro_days
            as_of = (
                current_date
                if current_date is not None
                else pd.Timestamp(prices.index[-1])
            )
            cutoff = as_of - pd.Timedelta(days=macro_days)
            lagged_macro = macro_data.loc[macro_data.index <= cutoff]
            if len(lagged_macro) == 0:
                lagged_macro = macro_data

            # Prefer merged regime_data when available (enables composite path)
            if regime_data is not None and not regime_data.empty:
                lagged_regime = regime_data.loc[regime_data.index <= cutoff]
                if len(lagged_regime) == 0:
                    lagged_regime = regime_data
                regime = classify_regime(lagged_regime)
            else:
                regime = classify_regime(lagged_macro)

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

            tilted_weights = apply_regime_tilts(base_weights, regime, regime_config)
            group_weights = {k.value: v for k, v in tilted_weights.items()}

        # 5. Composite score
        composite = compute_composite_score(
            standardized,
            coverage,
            config=scoring_config,
            ic_history=ic_history,
            group_weights=group_weights,
        )

        # 6. Select stocks
        selected = select_stocks(
            composite,
            config=selection_config,
            current_members=current_members,
            sector_labels=sector_labels,
            parent_universe=investable,
        )

        n_selected = len(selected)
        n_total = len(composite)
        n_nan = int(composite.isna().sum())
        if n_selected == 0:
            msg = (
                f"select_stocks() returned an empty universe. "
                f"All {n_nan} of {n_total} composite scores are NaN. "
                f"Check factor construction and standardization inputs."
            )
            raise DataError(msg)
        if n_selected < _MIN_SELECTED_STOCKS:
            msg = (
                f"select_stocks() returned only {n_selected} stock(s), "
                f"below the minimum of {_MIN_SELECTED_STOCKS}. "
                f"{n_nan} of {n_total} composite scores were NaN. "
                f"Widen the selection criteria or provide more candidates."
            )
            raise DataError(msg)

        selected_prices = prices[prices.columns.intersection(selected)]

        # 6.5  Factor-to-optimizer integration (alpha bridge)
        if integration_config is not None:
            from optimizer.factors._integration import build_factor_integration

            bl_prior, factor_constraints = build_factor_integration(
                config=integration_config,
                composite_scores=composite,
                standardized_factors=standardized,
                selected_tickers=selected,
            )
            if bl_prior is not None and hasattr(optimizer, "prior_estimator"):
                optimizer = deepcopy(optimizer)
                optimizer.prior_estimator = bl_prior
                # Enforce per-asset diversification when using BL views
                if integration_config.max_weight > 0 and hasattr(
                    optimizer, "max_weights"
                ):
                    optimizer.max_weights = integration_config.max_weight
                logger.info("Injected BL prior from factor scores into optimizer")
            elif bl_prior is not None:
                logger.warning(
                    "Optimizer %s has no prior_estimator; BL views skipped",
                    type(optimizer).__name__,
                )
            if factor_constraints is not None and hasattr(optimizer, "left_inequality"):
                optimizer = deepcopy(optimizer)
                optimizer.left_inequality = factor_constraints.left_inequality
                optimizer.right_inequality = factor_constraints.right_inequality
                logger.info("Injected factor exposure constraints into optimizer")
            elif factor_constraints is not None:
                logger.warning(
                    "Optimizer %s has no left_inequality; constraints skipped",
                    type(optimizer).__name__,
                )

    # 7. Slice currency_map to selected universe to avoid spurious FX log noise
    effective_currency_map = currency_map
    if currency_map is not None and fundamentals is not None:
        selected_cols = set(selected_prices.columns)
        effective_currency_map = {
            t: c for t, c in currency_map.items() if t in selected_cols
        }

    # 8. Delegate to existing pipeline
    return run_full_pipeline(
        prices=selected_prices,
        optimizer=optimizer,
        pre_selection_config=pre_selection_config,
        sector_mapping=sector_mapping,
        cv_config=cv_config,
        previous_weights=previous_weights,
        rebalancing_config=rebalancing_config,
        current_date=current_date,
        last_review_date=last_review_date,
        y_prices=y_prices,
        risk_free_rate=risk_free_rate,
        delisting_returns=delisting_returns,
        fx_config=fx_config,
        currency_map=effective_currency_map,
        fx_rates=fx_rates,
        benchmark_currency=benchmark_currency,
        cost_bps=cost_bps,
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


def _extract_weight_changes(
    weights_per_obs: pd.DataFrame,
    tol: float = 1e-8,
) -> pd.DataFrame:
    """Convert ``weights_per_observation`` to sparse weight-change rows.

    ``MultiPeriodPortfolio.weights_per_observation`` broadcasts constant
    weights across every day in each test window.  This helper diffs
    consecutive rows and keeps only those where at least one weight
    actually changed (rebalancing dates).
    """
    diff = weights_per_obs.diff()
    # First row: initial allocation from zero → full weight
    diff.iloc[0] = weights_per_obs.iloc[0]
    mask = diff.abs().sum(axis=1) > tol
    return diff[mask]


def _compute_net_sharpe(
    net_returns: pd.Series,
    risk_free_rate: float = 0.0,
    trading_days_per_year: int = 252,
) -> float | None:
    """Annualised Sharpe ratio from a daily net-return Series."""
    if len(net_returns) < 2:
        return None
    rf_daily = risk_free_rate / trading_days_per_year
    std = float(net_returns.std(ddof=1))
    if std == 0:
        return None
    return float((net_returns.mean() - rf_daily) / std * sqrt(trading_days_per_year))
