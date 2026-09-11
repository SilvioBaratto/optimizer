"""Convenience factory for universe screening."""

from __future__ import annotations

import logging

import pandas as pd

from optimizer.universe._config import InvestabilityScreenConfig
from optimizer.universe._screener import apply_investability_screens
from optimizer.universe._transformer import InvestabilityScreenSelector

logger = logging.getLogger(__name__)


def screen_universe(
    fundamentals: pd.DataFrame,
    price_history: pd.DataFrame,
    volume_history: pd.DataFrame,
    financial_statements: pd.DataFrame | None = None,
    config: InvestabilityScreenConfig | None = None,
    current_members: pd.Index | None = None,
) -> pd.Index:
    """Screen a stock universe for investability.

    Convenience wrapper around :func:`apply_investability_screens`
    that applies default configuration when none is provided.

    Parameters
    ----------
    fundamentals : pd.DataFrame
        Cross-sectional data with one row per ticker.
    price_history : pd.DataFrame
        Price matrix (dates x tickers).
    volume_history : pd.DataFrame
        Volume matrix (dates x tickers).
    financial_statements : pd.DataFrame or None
        Statement-level data.
    config : InvestabilityScreenConfig or None
        Screening configuration.
    current_members : pd.Index or None
        Tickers currently in the universe for hysteresis.

    Returns
    -------
    pd.Index
        Tickers passing all investability screens.
    """
    if config is None:
        config = InvestabilityScreenConfig()

    return apply_investability_screens(
        fundamentals=fundamentals,
        price_history=price_history,
        volume_history=volume_history,
        financial_statements=financial_statements,
        config=config,
        current_members=current_members,
    )


def build_investability_screen(
    fundamentals: pd.DataFrame,
    price_history: pd.DataFrame,
    volume_history: pd.DataFrame,
    financial_statements: pd.DataFrame | None = None,
    config: InvestabilityScreenConfig | None = None,
    current_members: pd.Index | None = None,
) -> InvestabilityScreenSelector:
    """Build a pipeline-composable investability-screen selector.

    Convenience factory that wires the screening data and configuration into an
    :class:`InvestabilityScreenSelector`.  The returned transformer runs the
    fundamental screens at ``fit`` time and, at ``transform`` time, restricts a
    linear-return matrix ``X`` (tickers as columns) to the investable universe —
    so an investability gate can precede skfolio pre-selection / optimisation in
    a single :class:`sklearn.pipeline.Pipeline`.

    Parameters
    ----------
    fundamentals : pd.DataFrame
        Cross-sectional data with one row per ticker.
    price_history : pd.DataFrame
        Price matrix (dates x tickers).
    volume_history : pd.DataFrame
        Volume matrix (dates x tickers).
    financial_statements : pd.DataFrame or None
        Optional statement-level data.
    config : InvestabilityScreenConfig or None
        Screening configuration.  ``None`` uses developed-market defaults.
    current_members : pd.Index or None
        Tickers currently in the universe for hysteresis.

    Returns
    -------
    InvestabilityScreenSelector
        Unfitted selector; call ``fit(X)`` with a return DataFrame.
    """
    return InvestabilityScreenSelector(
        fundamentals=fundamentals,
        price_history=price_history,
        volume_history=volume_history,
        financial_statements=financial_statements,
        config=config,
        current_members=current_members,
    )
