"""SQLAlchemy model for current portfolio holdings (``positions``).

Phase 8 of the ``fund`` bridge: the current-snapshot holdings table the state
panel reads. A row is one holding **after the latest approved HITL fill** —
positions are a *current snapshot*, not a time series (history stays derivable
from ``paper_orders``). Written only from ``_finalize_hitl`` on approve, which
**replaces** the whole snapshot for a portfolio; a ``UNIQUE(portfolio_id,
ticker)`` keeps one row per held ticker, and ``paper_order_id`` links back to the
ticket whose fill produced the snapshot.

The model lives here in ``portopt_db`` — pure SQLAlchemy, no
``optimizer``/``deepagents`` import — while the ``PositionRepository`` behavior
lives in ``fund``, mirroring the ``background_jobs`` / ``agent_runs`` /
``paper_orders`` split.
"""

from __future__ import annotations

import uuid
from datetime import date

from sqlalchemy import (
    Date,
    Float,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import Mapped, mapped_column

from portopt_db.base import BaseModel


class Position(BaseModel):
    """One current holding: (portfolio, ticker) → weight/shares/notional snapshot."""

    __tablename__ = "positions"
    __table_args__ = (
        # Current snapshot: one row per (portfolio, ticker).
        UniqueConstraint("portfolio_id", "ticker", name="uq_position_portfolio_ticker"),
        Index("ix_positions_portfolio_id", "portfolio_id"),
    )

    # No FK: the `portfolios` table was dropped in the ingestion strip; holdings
    # may be standalone, so this is a bare, indexed, non-null UUID. Uses the
    # portable ``Uuid`` type (native ``uuid`` on Postgres, ``CHAR(32)`` on SQLite)
    # matching ``PaperOrder``: the pg-specific ``UUID`` result processor mis-binds
    # on non-PK UUID columns alongside ``Float`` columns under statement caching on
    # SQLite, and this table carries ``weight``/``shares``/``notional``.
    portfolio_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    ticker: Mapped[str] = mapped_column(String(32), nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False)
    shares: Mapped[float | None] = mapped_column(Float, nullable=True)
    notional: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Decision/fill bar this snapshot reflects.
    asof: Mapped[date] = mapped_column(Date, nullable=False)
    # The paper ticket whose approved fill produced this snapshot (nullable — a
    # snapshot may be seeded without a linked ticket).
    paper_order_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("paper_orders.id"), nullable=True
    )


__all__ = ["Position"]
