"""Factory functions for building scoring functions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from skfolio.measures import RatioMeasure
from skfolio.metrics import make_scorer as _skfolio_make_scorer

from optimizer.optimization._config import RatioMeasureType
from optimizer.optimization._factory import _RATIO_MEASURE_MAP
from optimizer.scoring._config import ScorerConfig


def build_scorer(
    config: ScorerConfig | None = None,
    *,
    score_func: Callable[..., float] | None = None,
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
    """
    if config is None:
        config = ScorerConfig()

    if config.ratio_measure is not None:
        measure = _RATIO_MEASURE_MAP[config.ratio_measure]
        return _skfolio_make_scorer(measure)

    if score_func is None:
        msg = (
            "score_func is required when ratio_measure is None; "
            "pass a callable that accepts a portfolio and returns a scalar"
        )
        raise ValueError(msg)

    return _skfolio_make_scorer(
        score_func,
        greater_is_better=config.greater_is_better,
    )
