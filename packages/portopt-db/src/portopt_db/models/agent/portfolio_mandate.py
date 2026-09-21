"""SQLAlchemy model for persisted portfolio mandates (``portfolio_mandates``).

Phase 8 of the ``fund`` bridge: the mandate a run resolves before it starts. One
row per portfolio (``UNIQUE(portfolio_id)``, upserted): the full mandate lives in
the ``mandate`` JSON column — the **source of truth** — with a few scalar
conveniences (``base_currency`` / ``capital`` / ``drift_l1_threshold`` /
``benchmark``) mirrored out and indexed for the state panel and a future Phase-9
drift monitor. ``run_fund``'s signature is unchanged — only the *source* of the
mandate object moves to this table.

The model lives here in ``portopt_db`` — pure SQLAlchemy, no
``optimizer``/``deepagents`` import — while the ``MandateRepository`` behavior
lives in ``fund``. Note the name clash: the pydantic
``fund.schemas.PortfolioMandate`` is imported qualified as
``PortfolioMandateModel`` on the fund side.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import JSON, Float, Index, String, UniqueConstraint, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from portopt_db.base import BaseModel

# Use JSONB on PostgreSQL, plain JSON elsewhere (e.g. SQLite in tests).
_JSON = JSON().with_variant(JSONB, "postgresql")


class PortfolioMandate(BaseModel):
    """One portfolio's mandate: JSON source of truth + indexed scalar mirror."""

    __tablename__ = "portfolio_mandates"
    __table_args__ = (
        # One mandate per portfolio; the upsert conflict target.
        UniqueConstraint("portfolio_id", name="uq_portfolio_mandate_portfolio_id"),
        Index("ix_portfolio_mandates_base_currency", "base_currency"),
    )

    # No FK: the `portfolios` table was dropped in the ingestion strip. Portable
    # ``Uuid`` (matching ``PaperOrder``): the pg ``UUID`` result processor
    # mis-binds on non-PK UUID columns alongside ``Float`` columns on SQLite, and
    # this table carries ``capital`` / ``drift_l1_threshold``.
    portfolio_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False)
    base_currency: Mapped[str] = mapped_column(String(3), nullable=False)
    capital: Mapped[float] = mapped_column(Float, nullable=False)
    drift_l1_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    benchmark: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # Full mandate payload — the source of truth (scalar columns mirror it).
    mandate: Mapped[dict[str, Any]] = mapped_column(_JSON, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="active"
    )


__all__ = ["PortfolioMandate"]
