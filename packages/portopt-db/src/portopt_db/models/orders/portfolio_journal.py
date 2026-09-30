"""SQLAlchemy model for per-portfolio rebalance overlays (``portfolio_journal``).

One row per (portfolio_id, as_of) rebalance bar; written by the fund bridge at
``_finalize_hitl`` on approve.  Carries a denormalized snapshot of trades,
target allocation, and drift so cockpit reads and agent tools are a single indexed
lookup rather than a multi-table join at read time.

No FK to ``agent_runs`` or a ``portfolios`` table — the overlay is keyed by a
portable ``Uuid``, mirroring ``positions`` and ``paper_orders``, so it remains
functional even if those tables are absent (e.g. in unit tests).
"""

from __future__ import annotations

import uuid
from datetime import date
from typing import Any

from sqlalchemy import JSON, Date, Index, Text, UniqueConstraint, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from portopt_db.base import BaseModel

# JSONB on PostgreSQL, plain JSON for SQLite in tests.
_JSON = JSON().with_variant(JSONB, "postgresql")


class PortfolioJournal(BaseModel):
    """One rebalance-bar overlay row per (portfolio_id, as_of).

    Written by ``fund._finalize_hitl`` on an approved run; read by fund history
    tools and the cockpit.  Carries a denormalized digest of trades, weights, and
    drift so history queries require no join.

    Attributes:
        portfolio_id: Opaque portfolio identifier — no FK (mirrors ``positions``).
        as_of: Rebalance-bar date; combined with ``portfolio_id`` forms the
            idempotency key.
        run_id: Soft link to ``agent_runs.id``; ``None`` when written outside a
            fund run (e.g. backfill or manual entry).
        trades: Per-ticker fill detail from the committed ``paper_orders.lines``.
        allocation: Target weight snapshot for this run.
        drift: Drift vs prior holdings, including which bands crossed.
        narrative: Deterministic template string; suitable for agent injection.
    """

    __tablename__ = "portfolio_journal"
    __table_args__ = (
        UniqueConstraint("portfolio_id", "as_of", name="uq_portfolio_journal_key"),
        Index("ix_portfolio_journal_portfolio_id", "portfolio_id"),
        Index("ix_portfolio_journal_as_of", "as_of"),
    )

    # Portable Uuid dodges the SQLite UUID/Float affinity crash that hits the
    # pg-specific UUID type on non-PK columns under statement caching.
    portfolio_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    as_of: Mapped[date] = mapped_column(Date, nullable=False)
    run_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    trades: Mapped[dict[str, Any] | None] = mapped_column(_JSON, nullable=True)
    allocation: Mapped[dict[str, Any] | None] = mapped_column(_JSON, nullable=True)
    drift: Mapped[dict[str, Any] | None] = mapped_column(_JSON, nullable=True)
    narrative: Mapped[str] = mapped_column(Text, nullable=False, default="")


__all__ = ["PortfolioJournal"]
