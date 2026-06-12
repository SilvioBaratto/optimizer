"""Shared T212 → yfinance ticker DB lookup (issue #773).

Extracted from broker_sync_service so both the legacy sync flow and the
Cycle 1 T212 position normalizer share one query path.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.universe.universe import Instrument


def lookup_yf_ticker(t212_ticker: str, session: Session) -> str | None:
    """Look up the yfinance_ticker for a T212 ticker via the instruments table.

    Three-pass lookup to handle T212 API ticker format variations:
    1. Exact match on instruments.ticker (e.g. "SEPLl_EQ" → "SEPL.L")
    2. Direct match on yfinance_ticker (T212 occasionally sends yf format)
    3. Prefix match on yfinance_ticker (T212 sends bare "CCL", DB has "CCL.L")
       — only used when exactly one instrument matches to avoid ambiguity
    """
    # Pass 1: exact T212 internal ticker
    result = session.execute(
        select(Instrument.yfinance_ticker).where(Instrument.ticker == t212_ticker)
    ).scalar_one_or_none()
    if result:
        return result

    # Pass 2: T212 already sent a yfinance-format ticker
    result = session.execute(
        select(Instrument.yfinance_ticker).where(
            Instrument.yfinance_ticker == t212_ticker
        )
    ).scalar_one_or_none()
    if result:
        return result

    # Pass 3: bare symbol — yfinance adds an exchange suffix (e.g. "CCL" → "CCL.L")
    candidates = session.execute(
        select(Instrument.yfinance_ticker).where(
            Instrument.yfinance_ticker.like(t212_ticker + ".%")
        )
    ).scalars().all()
    if len(candidates) == 1:
        return candidates[0]

    return None
