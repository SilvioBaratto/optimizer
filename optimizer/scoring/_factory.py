"""Factory functions for building scoring functions."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from skfolio.measures import RatioMeasure
from skfolio.metrics import make_scorer as _skfolio_make_scorer

from optimizer.exceptions import ConfigurationError
from optimizer.optimization._config import RatioMeasureType
from optimizer.scoring._config import ScorerConfig

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

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Information Ratio implementation
# ---------------------------------------------------------------------------

_ANNUALIZED_FACTOR = 252  # trading days per year


def _build_ir_scorer(
    benchmark_returns: pd.Series,
) -> Callable[..., float]:
    """Build an annualised Information Ratio scorer from benchmark returns.

    IR = annualised active return / annualised tracking error

    where
      active return  = portfolio return − benchmark return (per period)
      tracking error = std(active returns, ddof=1) × √252

    Parameters
    ----------
    benchmark_returns : pd.Series
        Full benchmark return series indexed by date.  The scorer
        aligns on ``portfolio.observations`` before computing active
        returns.

    Returns
    -------
    callable
        A scorer accepting a skfolio ``Portfolio`` and returning the
        annualised IR as a float.
    """

    def _ir(portfolio: Any) -> float:
        obs = portfolio.observations  # numpy array of datetime64
        bm = benchmark_returns.loc[pd.DatetimeIndex(obs)].to_numpy()
        active = portfolio.returns - bm
        mean_active = float(np.mean(active))
        std_active = float(np.std(active, ddof=1))
        if std_active == 0.0 or math.isnan(std_active):
            return 0.0
        return (mean_active * _ANNUALIZED_FACTOR) / (
            std_active * math.sqrt(_ANNUALIZED_FACTOR)
        )

    return _skfolio_make_scorer(_ir)


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
        a scalar.  Required when ``config.ratio_measure`` is ``None``.
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
    ValueError
        If ``config.ratio_measure`` is ``None`` and no
        ``score_func`` is provided.
    ValueError
        If ``config.ratio_measure`` is
        ``RatioMeasureType.INFORMATION_RATIO`` and
        ``benchmark_returns`` is ``None``.
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
        return _build_ir_scorer(benchmark_returns)

    if config.ratio_measure is not None:
        measure = _RATIO_MEASURE_MAP[config.ratio_measure]
        return _skfolio_make_scorer(measure)

    if score_func is None:
        msg = (
            "score_func is required when ratio_measure is None; "
            "pass a callable that accepts a portfolio and returns a scalar"
        )
        raise ConfigurationError(msg)

    return _skfolio_make_scorer(
        score_func,
        greater_is_better=config.greater_is_better,
    )
