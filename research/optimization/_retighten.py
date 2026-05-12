"""Cycle-3 §7.3 Top-4 retighten loop and sub-period Sharpe helpers."""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from optimizer.optimization import MeanRiskConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from skfolio.optimization import MeanRisk

logger = logging.getLogger(__name__)

__all__ = [
    "_TOP_N",
    "_TOP4_THRESHOLD",
    "_SHRINK_FACTOR",
    "_TRADING_DAYS_PER_YEAR",
    "_top4_weight",
    "_retighten_error_message",
    "_solve_with_retighten",
    "_annualized_sharpe",
    "_split_into_subperiods",
]

# Cycle-3 §7.3 Top-4 retighten constants
_TOP_N = 4
_TOP4_THRESHOLD = 0.30
_SHRINK_FACTOR = 0.95

_TRADING_DAYS_PER_YEAR = 252


def _top4_weight(weights: np.ndarray) -> float:
    """Sum of the top-4 weights (or all if fewer assets)."""
    return float(np.sum(np.sort(weights)[-_TOP_N:]))


def _retighten_error_message(final_entry: dict[str, Any], max_retries: int) -> str:
    return (
        f"Top-4 retighten loop failed: top4={final_entry['top4']:.4f} "
        f">= {_TOP4_THRESHOLD} after {max_retries} retries "
        f"(final max_weights={final_entry['max_weights']:.6f})"
    )


def _solve_with_retighten(
    optimizer: MeanRisk,
    returns: pd.DataFrame,
    *,
    config: MeanRiskConfig,
    builder: Callable[[MeanRiskConfig], MeanRisk],
    max_retries: int = 5,
) -> tuple[MeanRisk, list[dict[str, Any]]]:
    """Cycle-3 §7.3 Top-4 retighten loop.

    Fit the optimiser; if ``sum(sorted(weights)[-4:]) >= 0.30``, shrink
    ``max_weights`` by ``0.95`` and rebuild + refit.  Cap at
    ``max_retries`` retries (initial fit + ``max_retries`` rebuilds).
    Raise ``RuntimeError`` when the loop fails to converge.
    """
    trace: list[dict[str, Any]] = []
    current = optimizer
    current_config = config
    attempt = 1
    while True:
        current.fit(returns)
        top4 = _top4_weight(np.asarray(current.weights_))
        trace.append(
            {
                "attempt": attempt,
                "top4": top4,
                "max_weights": cast(float, current_config.max_weights),
            }
        )
        logger.debug(
            "retighten attempt=%d top4=%.4f max_weights=%.4f",
            attempt,
            top4,
            cast(float, current_config.max_weights),
        )
        if top4 < _TOP4_THRESHOLD:
            return current, trace
        if attempt > max_retries:
            raise RuntimeError(_retighten_error_message(trace[-1], max_retries))
        new_max = cast(float, current_config.max_weights) * _SHRINK_FACTOR
        current_config = dataclasses.replace(current_config, max_weights=new_max)
        current = builder(current_config)
        attempt += 1


def _annualized_sharpe(returns: np.ndarray) -> float:
    """Annualized Sharpe of a daily-return slice; 0 when std is effectively 0."""
    if returns.size == 0:
        return 0.0
    std = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    if std < 1e-10:
        return 0.0
    return float(np.mean(returns) / std * np.sqrt(_TRADING_DAYS_PER_YEAR))


def _split_into_subperiods(
    returns: pd.Series, n_subperiods: int
) -> list[pd.Series]:
    """Split *returns* into ``n_subperiods`` equal contiguous slices."""
    n = len(returns)
    edges = np.linspace(0, n, n_subperiods + 1, dtype=int)
    return [returns.iloc[edges[i] : edges[i + 1]] for i in range(n_subperiods)]
