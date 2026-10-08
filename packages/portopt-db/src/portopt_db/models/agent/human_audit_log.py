"""SQLAlchemy model for the human-decision audit log (``human_audit_log``).

TUI oversight (Phase 1): an append-only, immutable record of every *human*
action — approve / reject / killswitch / mandate_edit / guardrail_override — that
the shell's unified action path funnels through. Agent decisions live in
``agent_decisions``; this is the human counterpart, and the Audit view unions the
two into one actor-attributed timeline.

Each row carries provenance for the Audit detail pane (``prev_value`` /
``new_value`` / ``source`` / ``idempotency_key``) and the two flags the Audit
view surfaces: ``override_flag`` (⚑ a guardrail edited past its MiFID band) and
``suitability_flag`` (⚠ outside the suitability band). ``idempotency_key`` is
UNIQUE but nullable, so a keyed action de-dups while un-keyed rows may repeat.

The model lives here in ``portopt_db`` — pure SQLAlchemy, no
``optimizer``/``deepagents`` import — while the ``HumanAuditLogRepository``
behavior lives in ``fund``, mirroring the ``agent_runs`` / ``portfolio_mandates``
model/repo split.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Index,
    String,
    Text,
    UniqueConstraint,
    false,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from portopt_db.base import BaseModel

# Use JSONB on PostgreSQL, plain JSON elsewhere (e.g. SQLite in tests).
_JSON = JSON().with_variant(JSONB, "postgresql")


class HumanAuditLog(BaseModel):
    """One human action: actor/action + provenance + override/suitability flags."""

    __tablename__ = "human_audit_log"
    __table_args__ = (
        # Nullable UNIQUE: a keyed action de-dups; un-keyed rows never collide.
        UniqueConstraint("idempotency_key", name="uq_human_audit_log_idempotency_key"),
        Index("ix_human_audit_log_portfolio_id", "portfolio_id"),
        Index("ix_human_audit_log_run_id", "run_id"),
        Index("ix_human_audit_log_action", "action"),
        Index("ix_human_audit_log_created_at", "created_at"),
    )

    # No FK: the `portfolios` table was dropped in the ingestion strip, and a
    # fund-wide action carries no portfolio; both ids are bare nullable UUIDs.
    portfolio_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    actor: Mapped[str] = mapped_column(
        String(50), nullable=False, server_default="human"
    )
    action: Mapped[str] = mapped_column(String(50), nullable=False)
    target: Mapped[str | None] = mapped_column(String(200), nullable=True)
    # Arbitrary JSON (scalar or object): a guardrail edit logs a number, a mandate
    # edit a dict of changed fields.
    prev_value: Mapped[Any | None] = mapped_column(_JSON, nullable=True)
    new_value: Mapped[Any | None] = mapped_column(_JSON, nullable=True)
    source: Mapped[str | None] = mapped_column(String(100), nullable=True)
    override_flag: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )
    suitability_flag: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )
    idempotency_key: Mapped[str | None] = mapped_column(String(200), nullable=True)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)


__all__ = ["HumanAuditLog"]
