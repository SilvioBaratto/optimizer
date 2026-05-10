"""Shared price-fetching helper (issue #382).

Single Responsibility: fetch close prices from price_history table.

Public API:
  - fetch_close_prices(tickers, session, start_date, end_date) -> pd.DataFrame
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy.orm import Session

from app.repositories.market_data.yfinance_repository import YFinanceRepository


def fetch_close_prices(
    tickers: list[str],
    session: Session,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    """Fetch close prices for each ticker from price_history table.

    Args:
        tickers: Ticker symbols to fetch.
        session: SQLAlchemy Session.
        start_date: Inclusive lower bound (no lower bound if None).
        end_date: Inclusive upper bound (no upper bound if None).

    Returns:
        DataFrame of close prices (dates × tickers), sorted ascending by date.
        Tickers absent from the database are silently dropped.
    """
    repo = YFinanceRepository(session)
    data: dict[str, dict] = {}

    for ticker in tickers:
        instrument = repo.get_instrument_by_yfinance_ticker(ticker)
        if instrument is None:
            continue
        rows = repo.get_price_history(
            instrument.id, start_date=start_date, end_date=end_date
        )
        data[ticker] = {r.date: float(r.close) for r in rows if r.close is not None}

    return pd.DataFrame(data).sort_index()
