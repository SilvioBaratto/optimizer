"""Shared T212 → yfinance ticker DB lookup (issue #773).

Extracted from broker_sync_service so both the legacy sync flow and the
Cycle 1 T212 position normalizer share one query path.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.universe.universe import Instrument


def lookup_yf_ticker(t212_ticker: str, session: Session) -> str | None:
    """Look up the yfinance_ticker for a T212 ticker via the instruments table."""
    stmt = select(Instrument.yfinance_ticker).where(
        Instrument.ticker == t212_ticker
    )
    return session.execute(stmt).scalar_one_or_none()
