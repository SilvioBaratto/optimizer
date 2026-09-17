"""SQLAlchemy model for simulated paper order tickets (``paper_orders``).

A ``fund`` allocator run ends in ``place_orders``, which turns a target-weight
vector into a **simulated** ticket (SPEC D5) filled at the next close strictly
after the decision bar (D30 — no look-ahead) with a simple slippage + commission
model. The ticket is persisted here so the paper-execution side effect is
**idempotent**: on a HITL ``Command(resume=…)`` the interrupting node re-runs
from the top (D3), so a ``UNIQUE(portfolio_id, asof, weights_hash)`` collapses a
double placement to one row.

The model lives here in ``portopt_db`` — pure SQLAlchemy, no
``optimizer``/``deepagents`` import — while the ``OrderRepository`` behavior
lives in ``fund``, mirroring the ``background_jobs`` / ``agent_runs`` split.
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any

from sqlalchemy import JSON, Date, Float, Index, String, UniqueConstraint, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from portopt_db.base import BaseModel

# Use JSONB on PostgreSQL, plain JSON elsewhere (e.g. SQLite in tests).
_JSON = JSON().with_variant(JSONB, "postgresql")


class PaperOrder(BaseModel):
    """One simulated order ticket: target weights → next-close paper fills."""

    __tablename__ = "paper_orders"
    __table_args__ = (
        # Idempotency key: one ticket per (portfolio, decision bar, weight vector).
        UniqueConstraint(
            "portfolio_id", "asof", "weights_hash", name="uq_paper_order_key"
        ),
        Index("ix_paper_orders_portfolio_id", "portfolio_id"),
        Index("ix_paper_orders_asof", "asof"),
    )

    # No FK: the `portfolios` table was dropped in the ingestion strip; a paper
    # ticket may be standalone, so this is a bare, indexed, non-null UUID. Uses
    # the portable ``Uuid`` type (native ``uuid`` on Postgres, ``CHAR(32)`` on
    # SQLite): the pg-specific ``UUID`` result processor mis-binds on non-PK UUID
    # columns alongside ``Float`` columns under statement caching on SQLite.
    portfolio_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    # Decision bar: the last close the allocator saw. Fills are strictly after it.
    asof: Mapped[date] = mapped_column(Date, nullable=False)
    # sha256 of the canonical weights JSON — the hashable half of the idem key.
    weights_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    weights: Mapped[dict[str, float]] = mapped_column(_JSON, nullable=False)
    # Fill bar: the first close strictly after `asof` (no look-ahead, D30).
    fill_date: Mapped[date] = mapped_column(Date, nullable=False)
    # Per-ticker fill detail: ticker, weight, fill/effective price, notional,
    # commission, slippage cost, fill_date.
    lines: Mapped[list[dict[str, Any]]] = mapped_column(_JSON, nullable=False)
    notional: Mapped[float] = mapped_column(Float, nullable=False)
    total_commission: Mapped[float] = mapped_column(Float, nullable=False)
    total_slippage_cost: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="filled"
    )


__all__ = ["PaperOrder"]
