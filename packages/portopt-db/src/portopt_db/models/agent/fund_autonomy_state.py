"""SQLAlchemy model for the per-scope fund autonomy posture
(``fund_autonomy_state``).

Phase-3 co-pilot: holds one row per portfolio plus a single fund-wide row, each
carrying the ARMED/FROZEN ``state`` and the ``first_round_gated`` flag that forces
every PM gate on a portfolio's first round. FROZEN is an *explicit* set state (the
killswitch writes it), never derived from run history — a portfolio with completed
runs can still be frozen. Fund scope wins over portfolio scope (enforced in the
repository, not here).

The ``state`` / ``scope`` vocabularies are plain ``String`` columns validated in
app code (``AutonomyState`` / ``AutonomyScope`` str enums in ``fund``), never
native DB enums, matching the project's ``status``-as-varchar convention.

The model lives here in ``portopt_db`` — pure SQLAlchemy, no
``optimizer``/``deepagents`` import — while the ``AutonomyRepository`` behavior
lives in ``fund``, mirroring the ``portfolio_mandates`` / ``human_audit_log``
model/repo split.
"""

from __future__ import annotations

import datetime
import uuid

from sqlalchemy import (
    Boolean,
    DateTime,
    Index,
    String,
    UniqueConstraint,
    Uuid,
    text,
    true,
)
from sqlalchemy.orm import Mapped, mapped_column

from portopt_db.base import BaseModel


class FundAutonomyState(BaseModel):
    """One scope's autonomy posture: ARMED/FROZEN + first-round gating."""

    __tablename__ = "fund_autonomy_state"
    __table_args__ = (
        UniqueConstraint(
            "scope", "portfolio_id", name="uq_fund_autonomy_state_scope_portfolio"
        ),
        # Postgres treats NULL != NULL in a plain UNIQUE, so the composite
        # constraint above cannot stop two scope='fund' rows (both NULL
        # portfolio_id). A partial unique index pins the fund row to a singleton
        # on both dialects — SQLite supports partial indexes, so it also fires in
        # the in-memory test harness.
        Index(
            "uq_fund_autonomy_state_fund_singleton",
            "scope",
            unique=True,
            postgresql_where=text("portfolio_id IS NULL"),
            sqlite_where=text("portfolio_id IS NULL"),
        ),
    )

    scope: Mapped[str] = mapped_column(String(20), nullable=False)
    # Portable Uuid (not pg UUID): stores portably on SQLite and carries a
    # genuine NULL for the single fund-scope row.
    portfolio_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    state: Mapped[str] = mapped_column(
        String(20), nullable=False, default="armed", server_default="armed"
    )
    first_round_gated: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True, server_default=true()
    )
    frozen_reason: Mapped[str | None] = mapped_column(String(500), nullable=True)
    frozen_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


__all__ = ["FundAutonomyState"]
