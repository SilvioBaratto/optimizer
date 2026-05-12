"""Cycle-3 §8 concentration check and §11 hybrid rebalance decision."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from optimizer.rebalancing import HybridRebalancingConfig, should_rebalance_hybrid
from research.optimization._retighten import (
    _annualized_sharpe,
    _split_into_subperiods,
)

logger = logging.getLogger(__name__)

_HOCKEY_STICK_NEG_THRESHOLD = 0.0
_HOCKEY_STICK_POS_THRESHOLD = 1.5


def _hockey_stick_warn(
    oos_returns: pd.Series | None,
    n_subperiods: int = 3,
) -> None:
    """Cycle-3 §8 hockey-stick concentration check.

    Splits *oos_returns* into ``n_subperiods`` equal slices, computes the
    annualized Sharpe of each, and emits a ``logger.warning`` when the
    minimum Sharpe is below 0 AND the maximum exceeds 1.5 — indicating
    out-performance concentrated in a single sub-period.  When the
    series is shorter than ``n_subperiods`` (or empty / ``None``), the
    check is skipped with a ``logger.debug`` line and no warning fires.
    """
    if oos_returns is None or len(oos_returns) < n_subperiods:
        logger.debug(
            "Concentration check skipped: returns length=%s < %d sub-periods",
            None if oos_returns is None else len(oos_returns),
            n_subperiods,
        )
        return

    sub_returns = _split_into_subperiods(oos_returns, n_subperiods)
    sharpes = [_annualized_sharpe(s.to_numpy()) for s in sub_returns]
    if not (
        min(sharpes) < _HOCKEY_STICK_NEG_THRESHOLD
        and max(sharpes) > _HOCKEY_STICK_POS_THRESHOLD
    ):
        return

    peak_idx = int(np.argmax(sharpes)) + 1  # 1-indexed for human-readable log
    logger.warning(
        "[WARN] hockey-stick — outperformance concentrated in period %d (sharpes=%s)",
        peak_idx,
        [round(s, 3) for s in sharpes],
    )


def _decide_rebalance(
    *,
    prev_weights: np.ndarray | None,
    target_weights: np.ndarray,
    current_date: pd.Timestamp,
    last_review_date: pd.Timestamp,
) -> tuple[bool, str]:
    """Cycle-3 §11 hybrid rebalance decision.

    Returns ``(False, "cold_start")`` when ``prev_weights`` is ``None``.
    Otherwise delegates to
    :func:`optimizer.rebalancing.should_rebalance_hybrid` with
    ``HybridRebalancingConfig.for_quarterly_with_10pct_threshold()``.
    """
    if prev_weights is None:
        logger.info(
            "Hybrid rebalance: cold_start (no previous_weights at %s)",
            current_date.date().isoformat(),
        )
        return False, "cold_start"

    config = HybridRebalancingConfig.for_quarterly_with_10pct_threshold()
    return should_rebalance_hybrid(
        prev_weights, target_weights, config, current_date, last_review_date
    )
