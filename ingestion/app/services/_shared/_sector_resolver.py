"""Unified ticker → sector resolver shared by Attribution and Dashboard.

Two authoritative sources exist in the codebase:

1. ``PortfolioSnapshot.sector_mapping`` — inline JSON written by the optimiser
   / import pipeline. Keys are the ticker strings used in the snapshot weights
   (yfinance-style on ``trading212``: ``ENGI.PA``, ``ORA.PA`` …).
2. ``TickerProfile.sector`` — yfinance-populated table joined via
   ``Instrument.yfinance_ticker``.

Snapshot-first, TickerProfile-fallback ordering is required because the
snapshot weights use yfinance-style ``.PA`` keys while ``Instrument.ticker``
carries T212-style keys (``ENGIp_EQ``); a DB-only lookup against the wrong
key column would always miss. Missing tickers default to ``Unclassified`` so
downstream code can always look up every input key without raising
``KeyError``.
"""

from __future__ import annotations

from collections.abc import Iterable

from portopt_db.repositories.market_data.yfinance_repository import YFinanceRepository
from sqlalchemy.orm import Session

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
    return YFinanceRepository(session).get_sectors_by_yfinance_ticker(tickers)
