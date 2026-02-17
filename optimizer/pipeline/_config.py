"""Configuration and result types for the portfolio pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class PortfolioResult:
    """Container for the output of a full portfolio optimisation run.

    Attributes
    ----------
    weights : pd.Series
        Final asset weights (ticker → weight).
    portfolio : object
        Skfolio ``Portfolio`` from ``predict()`` on the full dataset.
        Exposes ``.sharpe_ratio``, ``.sortino_ratio``, ``.max_drawdown``,
        ``.composition``, etc.
    backtest : object or None
        Out-of-sample ``MultiPeriodPortfolio`` (walk-forward) or
        ``Population`` (CPCV / MultipleRandomizedCV).  ``None`` when
        backtesting was skipped.
    pipeline : object
        The fitted sklearn ``Pipeline`` (pre-selection + optimiser).
        Can be reused for ``predict()`` on new data.
    summary : dict[str, float]
        Key performance metrics extracted from the in-sample portfolio.
    rebalance_needed : bool or None
        Whether the portfolio exceeds drift thresholds relative to
        ``previous_weights``.  ``None`` when no previous weights were
        provided.
    turnover : float or None
        One-way turnover between ``previous_weights`` and the new
        weights.  ``None`` when no previous weights were provided.
    """

    weights: pd.Series
    portfolio: Any
    backtest: Any | None = None
    pipeline: Any = None
    summary: dict[str, float] = field(default_factory=dict)
    rebalance_needed: bool | None = None
    turnover: float | None = None
