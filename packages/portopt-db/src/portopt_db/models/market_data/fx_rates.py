"""SQLAlchemy model for daily foreign-exchange rates.

Market-wide (not per-ticker) — no Instrument FK; each row keyed on its natural
identity ``(base_currency, quote_currency, date)`` so an at-least-once refetch
converges to one row. Written by the ingestion FX feed and read by the fund
performance tracker, which pivots the rows into the ``optimizer.fx`` rates frame.
"""

from __future__ import annotations

import datetime

from sqlalchemy import Date, Index, Numeric, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from portopt_db.base import BaseModel


class FxRate(BaseModel):
    """One daily FX quote for a currency pair.

    ``rate`` is the amount of ``base_currency`` equal to one unit of
    ``quote_currency`` on ``date`` (e.g. base ``EUR`` / quote ``USD`` with
    ``rate`` 0.92 means 1 USD = 0.92 EUR). Both codes are ISO-4217 *major*
    units (``GBP``, never the ``GBp`` sub-unit) — minor-unit scaling is the
    price reader's job, not the rate feed's.

    Attributes:
        base_currency: ISO-4217 code the rate is expressed in.
        quote_currency: ISO-4217 code of the foreign unit being priced.
        date: Calendar date the rate applies to.
        rate: Base-currency units per one unit of quote currency.
    """

    __tablename__ = "fx_rates"
    __table_args__ = (
        UniqueConstraint(
            "base_currency",
            "quote_currency",
            "date",
            name="uq_fx_rates_pair_date",
        ),
        Index("ix_fx_rates_date", "date"),
        Index("ix_fx_rates_pair", "base_currency", "quote_currency"),
    )

    base_currency: Mapped[str] = mapped_column(String(3), nullable=False)
    quote_currency: Mapped[str] = mapped_column(String(3), nullable=False)
    date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    rate: Mapped[float] = mapped_column(Numeric(20, 10), nullable=False)


__all__ = ["FxRate"]
