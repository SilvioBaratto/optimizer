"""SQLAlchemy model for the persisted marked-to-market NAV series (``nav_history``).

Phase 05 of the ``fund`` bridge: the daily NAV-mark step writes one row per
``(portfolio_id, date)`` holding the base-currency portfolio value, with an
optional benchmark level (base-normalized) cached for cheap reads. Unlike
``positions`` (a current snapshot), this is a genuine time series — the
performance tracker's NAV/return/risk read-outs are built on it.

The model lives here in ``portopt_db`` — pure SQLAlchemy, no
``optimizer``/``deepagents`` import — while the ``NavHistoryRepository`` behavior
lives in ``fund``, mirroring the ``positions`` / ``paper_orders`` split.
"""

from __future__ import annotations

import uuid
from datetime import date
from decimal import Decimal

from sqlalchemy import Date, Index, Numeric, String, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from portopt_db.base import BaseModel


class NavHistory(BaseModel):
    """One marked-to-market NAV point: (portfolio, date) → base-currency value."""

    __tablename__ = "nav_history"
    __table_args__ = (
        UniqueConstraint("portfolio_id", "date", name="uq_nav_history_portfolio_date"),
        Index("ix_nav_history_portfolio_id", "portfolio_id"),
        Index("ix_nav_history_date", "date"),
    )

    # No FK: the `portfolios` table was dropped in the ingestion strip. Uses the
    # portable ``Uuid`` type (native on Postgres, CHAR(32) on SQLite) matching
    # ``Position``/``PaperOrder`` — the pg-specific ``UUID`` result processor
    # mis-binds on a non-PK UUID column alongside ``Numeric`` columns under
    # statement caching on SQLite, and this table carries ``nav``/``benchmark``.
    portfolio_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    nav: Mapped[Decimal] = mapped_column(Numeric(20, 6), nullable=False)
    # Benchmark level FX-normalized to the base currency, date-aligned to the NAV
    # bar; NULL when the mandate carries no benchmark or none aligns. The upsert
    # fully replaces this column on any row it re-marks (not a merge); the daily
    # step re-marks only recent bars, so ``benchmark_ticker`` is what lets readers
    # and the step tell levels of a previous benchmark apart after a mandate edit.
    benchmark: Mapped[Decimal | None] = mapped_column(Numeric(20, 6), nullable=True)
    # Identity of the benchmark ``benchmark`` was taken from; NULL on rows written
    # before the column existed (identity unknown) or when no benchmark was set.
    benchmark_ticker: Mapped[str | None] = mapped_column(String(64), nullable=True)


__all__ = ["NavHistory"]
