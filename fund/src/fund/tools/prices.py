"""T3.1 — ``get_prices``: the first node of the allocator critical path.

Reads daily price history out of the shared ``portopt_db`` layer via
:class:`~portopt_db.repositories.market_data.yfinance_repository.YFinanceRepository`
and hands the agent a JSON-serialisable shape/coverage **summary** — never the
raw price matrix (SPEC Fase 3). The tool is a pure function of ``(session, asof,
tickers, field)``: same seeded data + same args ⇒ identical output.

Contract (via :func:`fund.tools._base.tool_envelope`):

* empty ``tickers`` or an unknown price column ⇒ ``{ok: false, error}``;
* a ticker with no instrument row or no priced days ⇒ **flagged** in
  ``data["missing"]``, never raised;
* every other failure (e.g. a malformed ``asof``) is caught by the envelope and
  returned as ``{ok: false, error}``.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import TYPE_CHECKING

import pandas as pd
from portopt_db.repositories.market_data.yfinance_repository import YFinanceRepository

from fund.tools._base import (
    ToolResult,
    coerce_date,
    err,
    ok,
    summarize_frame,
    tool_envelope,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

# OHLCV-and-actions columns that live on ``PriceHistory`` and can be pulled as a
# numeric series. A request for anything else is a caller error, not a crash.
_PRICE_FIELDS: frozenset[str] = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dividends",
        "stock_splits",
        "capital_gains",
    }
)


def load_price_frame(
    session: Session,
    asof: dt.date | str,
    tickers: Sequence[str],
    *,
    field: str = "close",
) -> tuple[pd.DataFrame, list[str]]:
    """Build a wide price frame for ``tickers`` as of ``asof``, plus the missing.

    Shared loader behind :func:`get_prices` and the moment/optimize tools (T3.2),
    so every allocator node re-derives the same panel from the same DB state. The
    frame carries one column per *priced* ticker (in requested order) on a sorted
    ``DatetimeIndex``; ``missing`` lists requested tickers with no instrument row
    or no priced days. The caller validates ``field`` and non-empty ``tickers``.

    Args:
        session: A sync ``portopt_db`` session (D1); the loader does not own it.
        asof: Inclusive upper bound; rows strictly after it are excluded.
        tickers: yfinance tickers, in the order the panel columns should follow.
        field: Which ``PriceHistory`` column to pull (default ``"close"``).

    Returns:
        ``(frame, missing)`` — the wide price frame and the absent tickers.
    """
    end_date = coerce_date(asof)
    repo = YFinanceRepository(session)

    series_by_ticker: dict[str, pd.Series] = {}
    for ticker in tickers:
        instrument = repo.get_instrument_by_yfinance_ticker(ticker)
        if instrument is None:
            continue
        rows = repo.get_price_history(instrument.id, end_date=end_date)
        values = {
            row.date: float(getattr(row, field))
            for row in rows
            if getattr(row, field) is not None
        }
        if values:
            series_by_ticker[ticker] = pd.Series(values)

    frame = pd.DataFrame(series_by_ticker)
    if not frame.empty:
        # A DatetimeIndex lets ``summarize_frame`` report the panel's date span.
        frame.index = pd.to_datetime(frame.index)
        frame = frame.sort_index()
    priced = set(frame.columns)
    missing = [ticker for ticker in tickers if ticker not in priced]
    return frame, missing


@tool_envelope
def get_prices(
    session: Session,
    asof: dt.date | str,
    tickers: Sequence[str],
    *,
    field: str = "close",
) -> ToolResult:
    """Return a price-panel summary for ``tickers`` as of ``asof``.

    Args:
        session: A sync ``portopt_db`` session (D1); the tool does not own it.
        asof: Inclusive upper bound; rows strictly after it are excluded (no
            look-ahead). Accepts a ``date`` or an ISO ``YYYY-MM-DD`` string.
        tickers: yfinance tickers to resolve, in the order the panel columns
            should follow.
        field: Which ``PriceHistory`` column to pull (default ``"close"``).

    Returns:
        ``ok`` with a shape/coverage summary of the wide price frame, plus
        ``asof``, ``field``, ``requested`` and ``missing`` (requested tickers
        absent from the panel). ``err`` on empty ``tickers`` or an unknown
        ``field``.
    """
    if not tickers:
        return err("no tickers requested")
    if field not in _PRICE_FIELDS:
        return err(f"unknown column {field!r}; expected one of {sorted(_PRICE_FIELDS)}")

    frame, missing = load_price_frame(session, asof, tickers, field=field)

    summary = summarize_frame(frame, name="prices")
    summary["asof"] = coerce_date(asof).isoformat()
    summary["field"] = field
    summary["requested"] = list(tickers)
    summary["missing"] = missing
    return ok(summary)


__all__ = ["get_prices", "load_price_frame"]
