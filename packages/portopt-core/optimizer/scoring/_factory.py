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

    Args:
        benchmark_returns: Full benchmark return series indexed by date.
            The scorer aligns on ``portfolio.observations`` before computing
            active returns.
        annualization_factor: Number of periods per year used to annualise
            the ratio.

    Returns:
        A scorer accepting a skfolio ``Portfolio`` and returning the
        annualised IR as a float. The score is ``0.0`` for an exactly zero
        tracking error and ``NaN`` when it is undefined (fewer than two
        observations or non-finite active returns, as for a skfolio
        ``FailedPortfolio``).
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
        # An undefined IR must be NaN so sklearn ranks the candidate last; a
        # finite 0.0 would let a failed fit (all-NaN returns) outrank candidates
        # whose IR is genuinely negative.
        if active.size < 2 or not np.all(np.isfinite(active)):
            return float("nan")
        std_active = float(np.std(active, ddof=1))
        if std_active == 0.0:
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

    The predicted portfolio's ``risk_free_rate`` is always set to the
    configured value, including an explicit ``0.0``. skfolio copies the
    optimizer's own rate onto the predicted portfolio, so leaving the portfolio
    default in place would score candidates that differ in their estimator rate
    against different yardsticks. ``annualization_factor`` is only set when
    given; ``None`` keeps the skfolio default.
    """
    greater_is_better = bool(measure.is_perf or measure.is_ratio)
    attr = measure.value

    def _score(portfolio: Any) -> float:
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

    Args:
        config: Scorer configuration. Defaults to ``ScorerConfig()``
            (Sharpe ratio).
        score_func: Custom scoring function that accepts a portfolio and
            returns a scalar. Required when no measure family is selected
            (all of ``ratio_measure`` / ``perf_measure`` / ``risk_measure``
            are ``None``).
        benchmark_returns: Full benchmark return series indexed by date.
            Required when ``config.ratio_measure`` is
            ``RatioMeasureType.INFORMATION_RATIO``; ignored otherwise.

    Returns:
        A scorer callable compatible with ``GridSearchCV`` and
        ``RandomizedSearchCV``.

    Raises:
        ConfigurationError: If no measure family is selected and no
            ``score_func`` is provided, or if the Information Ratio is
            requested without ``benchmark_returns``.
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


def _resolve_native_measure(
    config: ScorerConfig | None = None,
) -> RatioMeasure | PerfMeasure | RiskMeasure:
    """Map a `ScorerConfig` onto the skfolio measure it names, ignoring its rate.

    Raises:
        ConfigurationError: For the custom Information Ratio scorer or a
            custom ``score_func`` config, neither of which has a native skfolio
            measure.
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


def build_online_measure(
    config: ScorerConfig | None = None,
) -> RatioMeasure | PerfMeasure | RiskMeasure:
    """Return the bare skfolio measure for online model selection.

    ``OnlineGridSearch`` / ``online_score`` score the aggregated
    `MultiPeriodPortfolio` and therefore expect a
    `measure` passed directly to ``scoring`` -- not a
    ``make_scorer`` wrapper (which averages per-fold scores).  This helper
    resolves a `ScorerConfig` to that measure.

    A bare measure cannot carry ``risk_free_rate`` or ``annualization_factor``;
    online search takes them from ``portfolio_params`` instead. Use
    `build_online_portfolio_params` for that dictionary and pass it to
    ``run_online_score`` / ``run_online_predict``.

    Args:
        config: Scorer configuration. Defaults to ``ScorerConfig()``
            (Sharpe ratio).

    Raises:
        ConfigurationError: For the custom Information Ratio scorer or a
            custom ``score_func`` config -- neither maps onto a native skfolio
            measure and so cannot be used with the online utilities -- or when
            the config sets a non-zero ``risk_free_rate`` or an
            ``annualization_factor`` that the returned measure would silently
            drop.
    """
    if config is None:
        config = ScorerConfig()
    measure = _resolve_native_measure(config)

    dropped = []
    if config.risk_free_rate != 0.0:
        dropped.append(f"risk_free_rate={config.risk_free_rate!r}")
    if config.annualization_factor is not None:
        dropped.append(f"annualization_factor={config.annualization_factor!r}")
    if dropped:
        msg = (
            f"a bare skfolio measure cannot carry {', '.join(dropped)}; pass "
            "portfolio_params=build_online_portfolio_params(config) alongside "
            "the measure (accepted by run_online_score and run_online_predict), "
            "or build the search with build_online_grid_search / "
            "build_online_randomized_search, which do so"
        )
        raise ConfigurationError(msg)
    return measure


def build_online_portfolio_params(
    config: ScorerConfig | None = None,
) -> dict[str, float]:
    """Return the ``portfolio_params`` that apply a scorer's rate and annualisation.

    Online model selection scores the aggregated `MultiPeriodPortfolio` with a
    bare measure, so the ``risk_free_rate`` and ``annualization_factor`` of a
    `ScorerConfig` reach the scored portfolio only through the
    ``portfolio_params`` argument of ``OnlineGridSearch`` /
    ``OnlineRandomizedSearch`` / ``online_score``. They take precedence over
    the estimator's own ``risk_free_rate``, so every candidate is scored at the
    same rate.

    Args:
        config: Scorer configuration. Defaults to ``ScorerConfig()``.

    Returns:
        A dictionary with ``risk_free_rate`` and, when the config sets one,
        ``annualization_factor``.
    """
    if config is None:
        config = ScorerConfig()

    params: dict[str, float] = {"risk_free_rate": config.risk_free_rate}
    if config.annualization_factor is not None:
        params["annualization_factor"] = config.annualization_factor
    return params
