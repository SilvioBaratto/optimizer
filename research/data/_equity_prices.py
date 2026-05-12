"""Equity price and volume assembly functions.

Extracted from ``_equity.py``.  All functions accept a synchronous
SQLAlchemy ``Session`` and return pandas DataFrames.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

# Ensure the api package is importable from the CLI context.
_api_path = Path(__file__).parent.parent.parent / "api"
if str(_api_path) not in sys.path:
    sys.path.insert(0, str(_api_path))

from app.models.market_data.yfinance_data import PriceHistory  # noqa: E402
from app.models.universe.universe import Instrument  # noqa: E402

from ._currency import (  # noqa: E402
    currency_dedup_rank,
    normalize_prices,
)
from ._helpers import (  # noqa: E402
    _pivot_with_dedup,
    _to_float,
)

logger = logging.getLogger(__name__)

__all__ = [
    "_apply_delisting_returns",
    "_build_currency_map_from_instruments",
    "_build_ticker_rank_map",
    "assemble_prices",
    "assemble_volumes",
]


# ---------------------------------------------------------------------------
# Private DB helpers
# ---------------------------------------------------------------------------


def _build_ticker_rank_map(
    session: Session, include_delisted: bool = True
) -> dict[str, tuple[str, int]]:
    """Return ``{instrument_id_hex: (yfinance_ticker, currency_rank)}``.

    Used to deterministically resolve cross-listed instruments sharing
    the same ``yfinance_ticker``.  Lower rank = higher priority (USD=0).
    """
    stmt = (
        select(Instrument.id, Instrument.yfinance_ticker, Instrument.currency_code)
        .where(Instrument.yfinance_ticker.isnot(None))
        .where(Instrument.yfinance_ticker != "")
    )
    if not include_delisted:
        stmt = stmt.where(Instrument.delisted_at.is_(None))
    rows = session.execute(stmt).all()
    return {
        str(r[0]): (str(r[1]), currency_dedup_rank(r[2] if r[2] is not None else None))
        for r in rows
    }


def _build_currency_map_from_instruments(session: Session) -> dict[str, str]:
    """Return {yfinance_ticker: currency_code} from the Instrument table."""
    rows = session.execute(
        select(Instrument.yfinance_ticker, Instrument.currency_code)
        .where(Instrument.yfinance_ticker.isnot(None))
        .where(Instrument.yfinance_ticker != "")
        .where(Instrument.currency_code.isnot(None))
    ).all()
    return {str(t): str(c) for t, c in rows}


def _apply_delisting_returns(
    prices: pd.DataFrame,
    delistings: Sequence[tuple[str, Any, float]],
) -> pd.DataFrame:
    """Append a synthetic delisting-date price row for each delisted instrument.

    For each ``(yf_ticker, delisted_at, delisting_return)`` tuple, finds the
    last known close price on or before ``delisted_at`` and adds a synthetic
    price row at ``delisted_at`` equal to ``last_price * (1 + delisting_return)``.
    """
    if not delistings:
        return prices
    out = prices.copy()
    for yf_ticker, delisted_ts, r in delistings:
        if yf_ticker not in out.columns:
            continue
        col = out[yf_ticker].dropna()
        if col.empty:
            continue
        before = col[col.index <= delisted_ts]
        if before.empty:
            continue
        last_price = float(before.iloc[-1])
        synthetic_price = last_price * (1.0 + r)
        if delisted_ts not in out.index:
            new_row = pd.Series(dict.fromkeys(out.columns, np.nan), name=delisted_ts)
            out = pd.concat([out, new_row.to_frame().T])
            out = out.sort_index()
        if pd.isna(out.loc[delisted_ts, yf_ticker]):
            out.loc[delisted_ts, yf_ticker] = synthetic_price
    return out


# ---------------------------------------------------------------------------
# Public assembly functions
# ---------------------------------------------------------------------------


def assemble_prices(
    session: Session,
    include_delisted: bool = True,
    currency_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build a ``dates x tickers`` close-price DataFrame.

    Parameters
    ----------
    include_delisted : bool, default=True
        When ``True`` (default), delisted instruments are included in the
        price history up to and including their delisting date.  A synthetic
        price row is appended on the delisting date so that
        ``prices_to_returns`` produces the correct final (delisting) return.

        When ``False``, only currently active instruments are included,
        reproducing the original survivorship-biased behaviour.

    Returns
    -------
    pd.DataFrame
        Index = ``pd.DatetimeIndex``, columns = yfinance tickers.
    """
    ticker_rank_map = _build_ticker_rank_map(session, include_delisted=include_delisted)

    price_query = select(
        PriceHistory.instrument_id,
        PriceHistory.date,
        PriceHistory.close,
    ).order_by(PriceHistory.date)

    if not include_delisted:
        price_query = price_query.join(Instrument).where(
            Instrument.delisted_at.is_(None)
        )

    rows = session.execute(price_query).all()

    if not rows:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for instrument_id, row_date, close in rows:
        info = ticker_rank_map.get(str(instrument_id))
        if info is None:
            continue
        ticker, ccy_rank = info
        records.append(
            {
                "date": pd.Timestamp(row_date),
                "ticker": ticker,
                "close": _to_float(close),
                "_ccy_rank": ccy_rank,
            }
        )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Deduplicate cross-listed instruments that share the same yfinance_ticker
    # using a deterministic currency-priority tiebreaker (USD < GBP < EUR < …).
    pivoted = _pivot_with_dedup(df, "date", "ticker", "close", "assemble_prices")
    pivoted.index = pd.DatetimeIndex(pivoted.index)
    pivoted = pivoted.sort_index()

    # Append synthetic delisting-date price rows so that prices_to_returns()
    # produces the correct final return for each delisted instrument.
    if include_delisted and not pivoted.empty:
        delisting_rows = session.execute(
            select(
                Instrument.yfinance_ticker,
                Instrument.delisted_at,
                Instrument.delisting_return,
            )
            .where(Instrument.delisted_at.isnot(None))
            .where(Instrument.yfinance_ticker.isnot(None))
        ).all()

        delistings = [
            (
                yf_ticker,
                pd.Timestamp(delisted_at),
                float(dr) if dr is not None else -0.30,
            )
            for yf_ticker, delisted_at, dr in delisting_rows
            if yf_ticker in pivoted.columns
        ]
        pivoted = _apply_delisting_returns(pivoted, delistings)

    # Normalise minor-unit prices (GBX → GBP, etc.) so that ADDV
    # computation and factor construction use consistent values.
    # Prefer the caller-supplied currency_map (avoids a second DB query
    # when called from assemble_all); fall back to a direct Instrument
    # query for standalone callers.
    effective_map = (
        currency_map
        if currency_map is not None
        else _build_currency_map_from_instruments(session)
    )
    if effective_map:
        pivoted = normalize_prices(pivoted, effective_map)

    return pivoted


def assemble_volumes(
    session: Session,
    include_delisted: bool = True,
) -> pd.DataFrame:
    """Build a ``dates x tickers`` volume DataFrame.

    Parameters
    ----------
    include_delisted : bool, default=True
        When ``False``, volume data for delisted instruments is excluded.

    Returns
    -------
    pd.DataFrame
        Index = ``pd.DatetimeIndex``, columns = yfinance tickers.
    """
    ticker_rank_map = _build_ticker_rank_map(session, include_delisted=include_delisted)

    vol_query = select(
        PriceHistory.instrument_id,
        PriceHistory.date,
        PriceHistory.volume,
    ).order_by(PriceHistory.date)

    if not include_delisted:
        vol_query = vol_query.join(Instrument).where(Instrument.delisted_at.is_(None))

    rows = session.execute(vol_query).all()

    if not rows:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for instrument_id, date, volume in rows:
        info = ticker_rank_map.get(str(instrument_id))
        if info is None:
            continue
        ticker, ccy_rank = info
        records.append(
            {
                "date": pd.Timestamp(date),
                "ticker": ticker,
                "volume": _to_float(volume),
                "_ccy_rank": ccy_rank,
            }
        )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Deduplicate cross-listed instruments that share the same yfinance_ticker
    # using a deterministic currency-priority tiebreaker (USD < GBP < EUR < …).
    pivoted = _pivot_with_dedup(df, "date", "ticker", "volume", "assemble_volumes")
    pivoted.index = pd.DatetimeIndex(pivoted.index)
    pivoted = pivoted.sort_index()
    return pivoted
