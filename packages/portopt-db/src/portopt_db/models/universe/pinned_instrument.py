"""SQLAlchemy model for instruments the weekly universe build must keep.

Keyed by the Yahoo ticker text, not by an ``instruments`` foreign key: a mandate
may list a ticker before its instrument row exists, and the protection has to
start the moment the row appears.
"""

from __future__ import annotations

from sqlalchemy import Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from portopt_db.base import BaseModel


class PinnedInstrument(BaseModel):
    """One owner's claim that a ticker must stay in the investable universe.

    Several owners may pin the same ticker; each releases only its own row, so
    a ticker stays protected until the last owner lets go.

    Attributes:
        ticker: Yahoo ticker, upper-cased; same width as
            ``instruments.yfinance_ticker``.
        source: Owner of the pin, either ``manual`` or ``mandate:<portfolio_uuid>``.
    """

    __tablename__ = "pinned_instruments"
    __table_args__ = (
        UniqueConstraint(
            "ticker", "source", name="uq_pinned_instruments_ticker_source"
        ),
        Index("ix_pinned_instruments_source", "source"),
    )

    ticker: Mapped[str] = mapped_column(String(100), nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)


__all__ = ["PinnedInstrument"]
