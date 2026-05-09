"""Shared helpers and constants for the factor service modules.

Used across factor_compute_service, factor_scoring_service, and
factor_analysis_service. Not intended to be imported directly by routes.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable
from typing import Any, NamedTuple

__all__ = [
    "_PERIOD_DATE_COL",
    "_TICKER_COL",
    "FactorDataError",
    "PriceRow",
    "ProgressCallback",
    "_build_factor_scores_dict",
    "_build_multiindex_factor_df",
    "_build_multiindex_returns",
    "_build_returns_from_price_rows",
    "_build_standardized_scores_df",
    "_fetch_price_rows",
    "_find_instrument",
    "_notify",
]

import pandas as pd
from sqlalchemy.orm import Session

from app.repositories.yfinance_repository import YFinanceRepository

# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------


class FactorDataError(ValueError):
    """Raised when required factor data is missing from the database."""


# ---------------------------------------------------------------------------
# Type alias + constants
# ---------------------------------------------------------------------------

ProgressCallback = Callable[..., None]

_PERIOD_DATE_COL = "period_date"
_TICKER_COL = "ticker"


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


class PriceRow(NamedTuple):
    """Lightweight container for a single price observation."""

    ticker: str
    date: Any  # datetime.date from ORM
    close: float | None


# ---------------------------------------------------------------------------
# Session-level lookups
# ---------------------------------------------------------------------------


def _find_instrument(session: Session, ticker: str) -> Any | None:
    """Return the Instrument ORM row for *ticker*, or None if not found.

    Matches by either ``ticker`` (T212-style, e.g. ``AAPL_US_EQ``) or
    ``yfinance_ticker`` (e.g. ``AAPL``). Callers pass yfinance-style
    symbols in most paths (``compute_all_factors``, factor attribution),
    while the universe screener stores T212 codes.
    """
    from sqlalchemy import or_, select

    from app.models.universe import Instrument

    stmt = (
        select(Instrument)
        .where(or_(Instrument.ticker == ticker, Instrument.yfinance_ticker == ticker))
        .limit(1)
    )
    return session.execute(stmt).scalar_one_or_none()


# ---------------------------------------------------------------------------
# Progress helper
# ---------------------------------------------------------------------------


def _notify(cb: ProgressCallback | None, **kwargs: Any) -> None:
    """Fire progress callback if provided; silently no-op otherwise."""
    if cb is not None:
        cb(**kwargs)


# ---------------------------------------------------------------------------
# Price fetching
# ---------------------------------------------------------------------------


def _fetch_price_rows(
    session: Session,
    tickers: list[str],
    start_date: datetime.date,
    end_date: datetime.date,
) -> list[PriceRow]:
    """Fetch raw price rows from yfinance_repository for the given tickers/dates.

    Returns:
        List of :class:`PriceRow` named tuples (ticker, date, close).
    """
    yf_repo = YFinanceRepository(session)
    rows: list[PriceRow] = []
    for ticker in tickers:
        instrument = _find_instrument(session, ticker)
        if instrument is None:
            continue
        history = yf_repo.get_price_history(
            instrument.id, start_date=start_date, end_date=end_date
        )
        for ph in history:
            rows.append(PriceRow(ticker=ticker, date=ph.date, close=ph.close))
    return rows


def _build_returns_from_price_rows(price_rows: list[PriceRow]) -> pd.DataFrame:
    """Convert price rows to a returns DataFrame (dates x tickers).

    Args:
        price_rows: List of objects with ``ticker``, ``date``, and ``close``
            attributes (typically :class:`PriceRow` NamedTuples or compatible
            mocks).

    Returns:
        Wide DataFrame with DatetimeIndex and ticker columns; empty when input
        is empty.
    """
    if not price_rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        [{"ticker": r.ticker, "date": r.date, "close": r.close} for r in price_rows]
    )
    if df.empty:
        return pd.DataFrame()

    pivot = df.pivot(index="date", columns="ticker", values="close")
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.sort_index()
    return pivot.pct_change().dropna(how="all")


# ---------------------------------------------------------------------------
# Score reshaping helpers (used in >=2 modules)
# ---------------------------------------------------------------------------


def _build_standardized_scores_df(
    scores: list[Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build standardized_factors and coverage DataFrames.

    Args:
        scores: ORM score objects with ``ticker``, ``factor_type``, and
            ``standardized_score`` attributes.

    Returns:
        Tuple of (standardized_factors, coverage) where rows=tickers and
        columns=factor_types.
    """
    rows = [
        {
            "ticker": s.ticker,
            "factor_type": s.factor_type,
            "standardized_score": s.standardized_score,
        }
        for s in scores
        if s.standardized_score is not None
    ]
    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(rows)
    std_df = df.pivot_table(index="ticker", columns="factor_type", values="standardized_score")
    coverage_df = std_df.notna().astype(float)
    return std_df, coverage_df


def _build_factor_scores_dict(scores: list[Any]) -> dict[str, pd.DataFrame]:
    """Pivot score rows to dict[factor_type -> dates x tickers DataFrame].

    Expected by ``run_factor_validation(factor_scores_history=...)``.

    Args:
        scores: ORM score objects with ``score_date``, ``ticker``,
            ``factor_type``, and ``raw_score`` attributes.

    Returns:
        Mapping from factor type string to a wide pivot DataFrame.
    """
    rows = [
        {
            "date": s.score_date,
            "ticker": s.ticker,
            "factor_type": s.factor_type,
            "value": s.raw_score,
        }
        for s in scores
    ]
    if not rows:
        return {}

    df = pd.DataFrame(rows)
    result: dict[str, pd.DataFrame] = {}
    for factor, group in df.groupby("factor_type"):
        pivot = group.pivot(index="date", columns="ticker", values="value")
        pivot.index = pd.to_datetime(pivot.index)
        result[str(factor)] = pivot
    return result


def _build_multiindex_factor_df(scores: list[Any]) -> pd.DataFrame:
    """Build (date, ticker) MultiIndex DataFrame from score rows.

    Expected by ``run_factor_oos_validation(scores=...)``.

    Args:
        scores: ORM score objects with ``score_date``, ``ticker``,
            ``factor_type``, and ``raw_score`` attributes.

    Returns:
        MultiIndex DataFrame grouped by (date, ticker).
    """
    rows = [
        {
            "date": pd.Timestamp(s.score_date),
            "ticker": s.ticker,
            s.factor_type: s.raw_score,
        }
        for s in scores
    ]
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.groupby(["date", "ticker"]).first()
    return df


def _build_multiindex_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Reshape wide returns DataFrame to (date, ticker) MultiIndex.

    Args:
        returns: Wide returns DataFrame with DatetimeIndex.

    Returns:
        Long-form DataFrame with MultiIndex (date, ticker) and column
        ``return``.
    """
    if returns.empty:
        return pd.DataFrame()
    stacked = returns.stack()
    stacked.index.names = ["date", "ticker"]
    return stacked.to_frame(name="return")
