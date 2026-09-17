"""SQLAlchemy model for MiFID II suitability profiles (``mifid_profiles``).

Fase 5 of the ``fund`` bridge: the profiler's *step 0* system of record. A
client questionnaire is mapped to a persisted ``ConstraintSet`` (the risk
profile every later agent reads) and stored **append-only** with a per-portfolio
``version`` — an amended assessment never overwrites the prior one, so a
``UNIQUE(portfolio_id, version)`` guards the versioning. Each row keeps the raw
``questionnaire`` snapshot, the derived ``constraint_set``, and the structured
``suitability`` assessment for MiFID record-keeping, plus the ``store_key`` under
which the active ``ConstraintSet`` is cached in the LangGraph ``PostgresStore``
(so a Phase-4 ``ConstraintSetRef`` resolves).

The model lives here in ``portopt_db`` — pure SQLAlchemy, no
``optimizer``/``deepagents`` import — while the ``MifidProfileRepository``
behavior lives in ``fund``, mirroring the ``background_jobs`` / ``agent_runs`` /
``paper_orders`` split.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import JSON, Index, Integer, String, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from portopt_db.base import BaseModel

# Use JSONB on PostgreSQL, plain JSON elsewhere (e.g. SQLite in tests).
_JSON = JSON().with_variant(JSONB, "postgresql")


class MifidProfile(BaseModel):
    """One MiFID suitability-profile version: questionnaire → ConstraintSet."""

    __tablename__ = "mifid_profiles"
    __table_args__ = (
        # Append-only versioning: one row per (portfolio, version).
        UniqueConstraint(
            "portfolio_id", "version", name="uq_mifid_profile_portfolio_version"
        ),
        Index("ix_mifid_profiles_portfolio_id", "portfolio_id"),
    )

    # No FK: the `portfolios` table was dropped in the ingestion strip; a profile
    # may be standalone, so this is a bare, indexed, non-null UUID.
    portfolio_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    # 1-based version, monotonically incremented per portfolio (append-only): an
    # amended assessment is a new row, never an overwrite.
    version: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("1")
    )
    # Raw questionnaire snapshot (the four MiFID pillars) for record-keeping.
    questionnaire: Mapped[dict[str, Any]] = mapped_column(_JSON, nullable=False)
    # Derived ConstraintSet — the risk profile every later agent reads.
    constraint_set: Mapped[dict[str, Any]] = mapped_column(_JSON, nullable=False)
    # Structured SuitabilityAssessment (per-pillar inputs, flags, rationale).
    suitability: Mapped[dict[str, Any]] = mapped_column(_JSON, nullable=False)
    # Store key under which the active ConstraintSet is cached (resolves a
    # Phase-4 ``ConstraintSetRef``).
    store_key: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="active"
    )


__all__ = ["MifidProfile"]
