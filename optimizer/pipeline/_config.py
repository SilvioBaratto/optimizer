"""Configuration and result types for the portfolio pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from optimizer.fx._decomposition import FxReturnDecomposition


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
    fx_decomposition : FxReturnDecomposition or None
        FX return decomposition when ``FxConfig.mode == DECOMPOSE``.
    currency : str or None
        Base currency used for FX conversion (e.g. ``"EUR"``).
    net_returns : pd.Series or None
        Net backtest portfolio returns after transaction cost deduction.
        ``None`` when no backtest was run.
    net_sharpe_ratio : float or None
        Annualized Sharpe ratio computed from ``net_returns``.
        ``None`` when no backtest was run.
    weight_history : pd.DataFrame or None
        Absolute portfolio weights at each walk-forward rebalancing date.
        Rows are rebalancing dates; columns are asset tickers.
        Compatible with ``compute_net_alpha(weights_history=...)``.
        ``None`` when no backtest was run.
    """

    weights: pd.Series
    portfolio: Any
    backtest: Any | None = None
    pipeline: Any = None
    summary: dict[str, float] = field(default_factory=dict)
    rebalance_needed: bool | None = None
    turnover: float | None = None
    risk_free_rate: float = 0.0
    fx_decomposition: FxReturnDecomposition | None = None
    currency: str | None = None
    net_returns: pd.Series | None = None
    net_sharpe_ratio: float | None = None
    weight_history: pd.DataFrame | None = None
