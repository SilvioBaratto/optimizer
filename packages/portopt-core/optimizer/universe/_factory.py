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

    Convenience wrapper around ``apply_investability_screens`` that applies
    default configuration when none is provided.

    Args:
        fundamentals: Cross-sectional data with one row per ticker.
        price_history: Price matrix (dates x tickers).
        volume_history: Volume matrix (dates x tickers).
        financial_statements: Optional statement-level data.
        config: Screening configuration. ``None`` uses developed-market defaults.
        current_members: Tickers currently in the universe for hysteresis.

    Returns:
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

    Wires screening data and configuration into an ``InvestabilityScreenSelector``.
    The returned transformer runs fundamental screens at ``fit`` time and, at
    ``transform`` time, restricts a linear-return matrix ``X`` (tickers as columns)
    to the investable universe — allowing an investability gate to precede skfolio
    pre-selection or optimisation inside a single ``sklearn.pipeline.Pipeline``.

    Args:
        fundamentals: Cross-sectional data with one row per ticker.
        price_history: Price matrix (dates x tickers).
        volume_history: Volume matrix (dates x tickers).
        financial_statements: Optional statement-level data.
        config: Screening configuration. ``None`` uses developed-market defaults.
        current_members: Tickers currently in the universe for hysteresis.

    Returns:
        Unfitted selector; call ``fit(X)`` with a linear-return DataFrame.
    """
    return InvestabilityScreenSelector(
        fundamentals=fundamentals,
        price_history=price_history,
        volume_history=volume_history,
        financial_statements=financial_statements,
        config=config,
        current_members=current_members,
    )
