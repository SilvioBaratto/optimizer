"""Factory functions for building scoring functions."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from skfolio.measures import PerfMeasure, RatioMeasure, RiskMeasure
from skfolio.metrics import make_scorer as _skfolio_make_scorer

from optimizer.exceptions import ConfigurationError
from optimizer.optimization._config import RatioMeasureType, RiskMeasureType
from optimizer.scoring._config import PerfMeasureType, ScorerConfig

_RATIO_MEASURE_MAP: dict[RatioMeasureType, RatioMeasure] = {
    RatioMeasureType.SHARPE_RATIO: RatioMeasure.SHARPE_RATIO,
    RatioMeasureType.ANNUALIZED_SHARPE_RATIO: RatioMeasure.ANNUALIZED_SHARPE_RATIO,
    RatioMeasureType.SORTINO_RATIO: RatioMeasure.SORTINO_RATIO,
    RatioMeasureType.ANNUALIZED_SORTINO_RATIO: RatioMeasure.ANNUALIZED_SORTINO_RATIO,
    RatioMeasureType.MEAN_ABSOLUTE_DEVIATION_RATIO: (
        RatioMeasure.MEAN_ABSOLUTE_DEVIATION_RATIO
    ),
    RatioMeasureType.FIRST_LOWER_PARTIAL_MOMENT_RATIO: (
        RatioMeasure.FIRST_LOWER_PARTIAL_MOMENT_RATIO
    ),
    RatioMeasureType.VALUE_AT_RISK_RATIO: RatioMeasure.VALUE_AT_RISK_RATIO,
    RatioMeasureType.CVAR_RATIO: RatioMeasure.CVAR_RATIO,
    RatioMeasureType.ENTROPIC_RISK_MEASURE_RATIO: (
        RatioMeasure.ENTROPIC_RISK_MEASURE_RATIO
    ),
    RatioMeasureType.EVAR_RATIO: RatioMeasure.EVAR_RATIO,
    RatioMeasureType.WORST_REALIZATION_RATIO: RatioMeasure.WORST_REALIZATION_RATIO,
    RatioMeasureType.DRAWDOWN_AT_RISK_RATIO: RatioMeasure.DRAWDOWN_AT_RISK_RATIO,
    RatioMeasureType.CDAR_RATIO: RatioMeasure.CDAR_RATIO,
    RatioMeasureType.CALMAR_RATIO: RatioMeasure.CALMAR_RATIO,
    RatioMeasureType.AVERAGE_DRAWDOWN_RATIO: RatioMeasure.AVERAGE_DRAWDOWN_RATIO,
    RatioMeasureType.EDAR_RATIO: RatioMeasure.EDAR_RATIO,
    RatioMeasureType.ULCER_INDEX_RATIO: RatioMeasure.ULCER_INDEX_RATIO,
    RatioMeasureType.GINI_MEAN_DIFFERENCE_RATIO: (
        RatioMeasure.GINI_MEAN_DIFFERENCE_RATIO
    ),
}

_PERF_MEASURE_MAP: dict[PerfMeasureType, PerfMeasure] = {
    PerfMeasureType.MEAN: PerfMeasure.MEAN,
    PerfMeasureType.ANNUALIZED_MEAN: PerfMeasure.ANNUALIZED_MEAN,
}

# Every ``RiskMeasureType`` member name maps 1:1 onto a
# ``skfolio.measures.RiskMeasure`` member of the same name.
_RISK_MEASURE_MAP: dict[RiskMeasureType, RiskMeasure] = {
    member: RiskMeasure[member.name] for member in RiskMeasureType
}

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Information Ratio implementation
# ---------------------------------------------------------------------------

_ANNUALIZATION_FACTOR = 252  # trading days per year


def _aligned_benchmark(
    benchmark_returns: pd.Series, observations: np.ndarray
) -> np.ndarray:
    """Align a benchmark series onto a portfolio's observation dates.

    Raises a clear error if any observation is missing from the benchmark
    or if the aligned window contains NaNs, rather than silently producing
    ``NaN`` scores that corrupt cross-validation.
    """
    index = pd.DatetimeIndex(observations)
    aligned = benchmark_returns.reindex(index)
    if aligned.isna().any():
        missing = int(aligned.isna().sum())
        msg = (
            f"benchmark_returns is missing {missing} of {len(index)} "
            "portfolio observation dates (or contains NaNs); provide a "
            "benchmark series that fully covers the scored window"
        )
        raise ConfigurationError(msg)
    return aligned.to_numpy(dtype=float)


def _build_ir_scorer(
    benchmark_returns: pd.Series,
    *,
    annualization_factor: float = _ANNUALIZATION_FACTOR,
) -> Callable[..., float]:
    """Build an annualised Information Ratio scorer from benchmark returns.

    IR = annualised active return / annualised tracking error

    where
      active return  = portfolio return - benchmark return (per period)
      tracking error = std(active returns, ddof=1) x sqrt(annualization_factor)

    Parameters
    ----------
    benchmark_returns : pd.Series
        Full benchmark return series indexed by date.  The scorer
        aligns on ``portfolio.observations`` before computing active
        returns.
    annualization_factor : float
        Number of periods per year used to annualise the ratio.

    Returns
    -------
    callable
        A scorer accepting a skfolio ``Portfolio`` and returning the
        annualised IR as a float.
    """
    if not isinstance(benchmark_returns, pd.Series):
        msg = (
            "benchmark_returns must be a pandas Series indexed by date, got "
            f"{type(benchmark_returns).__name__}"
        )
        raise ConfigurationError(msg)

    sqrt_factor = math.sqrt(annualization_factor)

    def _ir(portfolio: Any) -> float:
        bm = _aligned_benchmark(benchmark_returns, portfolio.observations)
        active = np.asarray(portfolio.returns, dtype=float) - bm
        std_active = float(np.std(active, ddof=1))
        if std_active == 0.0 or math.isnan(std_active):
            return 0.0
        mean_active = float(np.mean(active))
        return (mean_active * annualization_factor) / (std_active * sqrt_factor)

    return _skfolio_make_scorer(_ir)


# ---------------------------------------------------------------------------
# Ratio / perf / risk measure scorers
# ---------------------------------------------------------------------------


def _build_measure_scorer(
    measure: RatioMeasure | PerfMeasure | RiskMeasure,
    *,
    risk_free_rate: float,
    annualization_factor: float | None,
) -> Callable[..., float]:
    """Build a scorer from a skfolio measure.

    When neither ``risk_free_rate`` nor ``annualization_factor`` deviates
    from the portfolio defaults the bare measure is handed to skfolio's
    ``make_scorer`` (preserving its auto ``greater_is_better`` detection and
    readable repr).  Otherwise the predicted portfolio's ``risk_free_rate``
    and ``annualization_factor`` are set before the measure is read.
    """
    if risk_free_rate == 0.0 and annualization_factor is None:
        return _skfolio_make_scorer(measure)

    greater_is_better = bool(measure.is_perf or measure.is_ratio)
    attr = measure.value

    def _score(portfolio: Any) -> float:
        if risk_free_rate != 0.0:
            portfolio.risk_free_rate = risk_free_rate
        if annualization_factor is not None:
            portfolio.annualization_factor = annualization_factor
        return float(getattr(portfolio, attr))

    _score.__name__ = repr(measure)
    return _skfolio_make_scorer(_score, greater_is_better=greater_is_better)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_scorer(
    config: ScorerConfig | None = None,
    *,
    score_func: Callable[..., float] | None = None,
    benchmark_returns: pd.Series | None = None,
) -> Callable[..., float]:
    """Build a scoring callable compatible with sklearn cross-validation.

    Parameters
    ----------
    config : ScorerConfig or None
        Scorer configuration.  Defaults to ``ScorerConfig()``
        (Sharpe ratio).
    score_func : callable or None
        Custom scoring function that accepts a portfolio and returns
        a scalar.  Required when no measure family is selected (all of
        ``ratio_measure`` / ``perf_measure`` / ``risk_measure`` are
        ``None``).
    benchmark_returns : pd.Series or None
        Full benchmark return series indexed by date.  Required when
        ``config.ratio_measure`` is
        ``RatioMeasureType.INFORMATION_RATIO``; ignored otherwise.

    Returns
    -------
    callable
        A scorer callable compatible with ``GridSearchCV`` and
        ``RandomizedSearchCV``.

    Raises
    ------
    ConfigurationError
        If no measure family is selected and no ``score_func`` is provided,
        or if the Information Ratio is requested without ``benchmark_returns``.
    """
    if config is None:
        config = ScorerConfig()

    if config.ratio_measure == RatioMeasureType.INFORMATION_RATIO:
        if benchmark_returns is None:
            msg = (
                "benchmark_returns is required when "
                "ratio_measure=RatioMeasureType.INFORMATION_RATIO; "
                "pass a pd.Series of benchmark returns indexed by date"
            )
            raise ConfigurationError(msg)
        factor = config.annualization_factor or _ANNUALIZATION_FACTOR
        return _build_ir_scorer(benchmark_returns, annualization_factor=factor)

    if config.ratio_measure is not None:
        return _build_measure_scorer(
            _RATIO_MEASURE_MAP[config.ratio_measure],
            risk_free_rate=config.risk_free_rate,
            annualization_factor=config.annualization_factor,
        )

    if config.perf_measure is not None:
        return _build_measure_scorer(
            _PERF_MEASURE_MAP[config.perf_measure],
            risk_free_rate=config.risk_free_rate,
            annualization_factor=config.annualization_factor,
        )

    if config.risk_measure is not None:
        return _build_measure_scorer(
            _RISK_MEASURE_MAP[config.risk_measure],
            risk_free_rate=config.risk_free_rate,
            annualization_factor=config.annualization_factor,
        )

    if score_func is None:
        msg = (
            "score_func is required when no measure family is selected; "
            "pass a callable that accepts a portfolio and returns a scalar"
        )
        raise ConfigurationError(msg)

    return _skfolio_make_scorer(
        score_func,
        greater_is_better=config.greater_is_better,
    )


def build_online_measure(
    config: ScorerConfig | None = None,
) -> RatioMeasure | PerfMeasure | RiskMeasure:
    """Return the bare skfolio measure for online model selection.

    ``OnlineGridSearch`` / ``online_score`` score the aggregated
    :class:`~skfolio.portfolio.MultiPeriodPortfolio` and therefore expect a
    :ref:`measure <measures_ref>` passed directly to ``scoring`` -- **not** a
    ``make_scorer`` wrapper (which averages per-fold scores).  This helper
    resolves a :class:`ScorerConfig` to that measure.

    Raises
    ------
    ConfigurationError
        For the custom Information Ratio scorer or a custom ``score_func``
        config -- neither maps onto a native skfolio measure and so cannot be
        used with the online utilities.
    """
    if config is None:
        config = ScorerConfig()

    if config.ratio_measure == RatioMeasureType.INFORMATION_RATIO:
        msg = (
            "the Information Ratio is a custom scorer and is not supported by "
            "the online utilities; use a native ratio/perf/risk measure"
        )
        raise ConfigurationError(msg)
    if config.ratio_measure is not None:
        return _RATIO_MEASURE_MAP[config.ratio_measure]
    if config.perf_measure is not None:
        return _PERF_MEASURE_MAP[config.perf_measure]
    if config.risk_measure is not None:
        return _RISK_MEASURE_MAP[config.risk_measure]

    msg = (
        "a custom score_func config has no native skfolio measure; the online "
        "utilities require a ratio/perf/risk measure"
    )
    raise ConfigurationError(msg)
