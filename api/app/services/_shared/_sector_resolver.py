"""Unified ticker → sector resolver shared by Attribution and Dashboard (issue #427).

Two authoritative sources exist in the codebase:

1. ``PortfolioSnapshot.sector_mapping`` — inline JSON written by the optimiser
   / import pipeline. Keys are the ticker strings used in the snapshot weights
   (yfinance-style on ``trading212``: ``ENGI.PA``, ``ORA.PA`` …).
2. ``TickerProfile.sector`` — yfinance-populated table joined via
   ``Instrument.yfinance_ticker``.

Before #427, Attribution queried ``TickerProfile`` via ``Instrument.ticker``
(the T212-style key ``ENGIp_EQ``), which never matched the ``.PA`` keys in
the weights dict. Dashboard went straight to the snapshot mapping. The two
paths diverged on both data source and lookup key.

This resolver unifies them with **snapshot-first, TickerProfile fallback**
semantics. Missing tickers default to ``Unclassified`` so downstream code can
always look up every input key without raising ``KeyError``.
"""

from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.market_data.yfinance_data import TickerProfile
from app.models.universe.universe import Instrument

UNCLASSIFIED = "Unclassified"


def resolve_sector_map(
    session: Session,
    tickers: Iterable[str],
    snapshot_mapping: dict[str, str] | None = None,
) -> dict[str, str]:
    """Return ``{ticker: sector}`` covering every input ticker.

    Order of precedence:
      1. Non-empty entries from ``snapshot_mapping``.
      2. ``TickerProfile.sector`` via ``Instrument.yfinance_ticker`` join
         for any ticker not covered by (1).
      3. ``Unclassified`` for any ticker still unmatched.
    """
    ticker_list = list(tickers)
    if not ticker_list:
        return {}

    resolved: dict[str, str] = {}
    if snapshot_mapping:
        for ticker in ticker_list:
            sector = snapshot_mapping.get(ticker)
            if sector:
                resolved[ticker] = sector

    remaining = [t for t in ticker_list if t not in resolved]
    if remaining:
        resolved.update(_fetch_from_profiles(session, remaining))

    for ticker in ticker_list:
        resolved.setdefault(ticker, UNCLASSIFIED)
    return resolved


def _fetch_from_profiles(
    session: Session,
    tickers: list[str],
) -> dict[str, str]:
    """Join ``TickerProfile`` via ``Instrument.yfinance_ticker`` — skip NULL sectors."""
    stmt = (
        select(Instrument.yfinance_ticker, TickerProfile.sector)
        .join(TickerProfile, TickerProfile.instrument_id == Instrument.id)
        .where(Instrument.yfinance_ticker.in_(tickers))
    )
    return {
        row.yfinance_ticker: row.sector
        for row in session.execute(stmt).all()
        if row.sector
    }
