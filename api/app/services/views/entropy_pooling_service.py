"""Entropy Pooling service: wraps build_entropy_pooling() from the optimizer library.

Single Responsibility: fetch prices, convert to returns, build estimator, fit, extract.

Public API:
  - run_entropy_pooling(returns, **view_kwargs) -> dict
"""

from __future__ import annotations

import logging

import pandas as pd

from app.services._shared import fetch_close_prices
from optimizer.views import EntropyPoolingConfig, build_entropy_pooling

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_entropy_pooling(
    returns: pd.DataFrame,
    *,
    mean_views: tuple[str, ...] | None = None,
    variance_views: tuple[str, ...] | None = None,
    correlation_views: tuple[str, ...] | None = None,
    skew_views: tuple[str, ...] | None = None,
    kurtosis_views: tuple[str, ...] | None = None,
    cvar_views: tuple[str, ...] | None = None,
) -> dict:
    """Fit an Entropy Pooling prior on *returns* and return posterior moments.

    Args:
        returns: Returns DataFrame (dates × tickers) — already converted from prices.
        mean_views: Mean equality view expressions.
        variance_views: Variance view expressions.
        correlation_views: Correlation view expressions in '(A, B) == v' format.
        skew_views: Skewness view expressions.
        kurtosis_views: Kurtosis view expressions.
        cvar_views: CVaR view expressions.

    Returns:
        dict with keys 'mu' (list[float]), 'covariance' (list[list[float]]),
        and 'tickers' (list[str]).

    Raises:
        RuntimeError: When the entropy pooling solver fails to converge.
    """
    config = EntropyPoolingConfig(
        mean_views=mean_views,
        variance_views=variance_views,
        correlation_views=correlation_views,
        skew_views=skew_views,
        kurtosis_views=kurtosis_views,
        cvar_views=cvar_views,
    )
    estimator = build_entropy_pooling(config)
    estimator.fit(returns)
    dist = estimator.return_distribution_
    tickers = list(returns.columns)
    return {
        "mu": dist.mu.tolist(),
        "covariance": dist.covariance.tolist(),
        "tickers": tickers,
    }


# ---------------------------------------------------------------------------
# Price fetching helper — thin wrapper for str | None date inputs
# ---------------------------------------------------------------------------


def fetch_prices_df(
    tickers: list[str],
    start_date: str | None,
    end_date: str | None,
    session: object,
) -> pd.DataFrame:
    """Fetch close prices, accepting ISO date strings (delegates to fetch_close_prices).

    Args:
        tickers: Ticker symbols to fetch.
        start_date: ISO date string or None for no lower bound.
        end_date: ISO date string or None for no upper bound.
        session: SQLAlchemy Session.

    Returns:
        DataFrame of close prices (dates × tickers), sorted ascending by date.
    """
    from datetime import date

    start = date.fromisoformat(start_date) if start_date else None
    end = date.fromisoformat(end_date) if end_date else None
    return fetch_close_prices(tickers, session, start_date=start, end_date=end)  # type: ignore[arg-type]
