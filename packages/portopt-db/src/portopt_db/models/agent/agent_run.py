"""SQLAlchemy models for agent-run audit trail (``agent_runs`` / ``agent_decisions``).

A ``fund`` run is one allocator pass: the LLM chooses, the optimizer computes.
``AgentRun`` is the run-level record (universe in, weights out, config + seed for
determinism); ``AgentDecision`` is the ordered per-step trail (constraint/view
snapshots, the LLM prompt/response-or-hash, any HITL decision). The model lives
here in ``portopt_db`` — pure SQLAlchemy, no ``optimizer``/``deepagents`` import —
while the ``AgentRunRepository`` behavior lives in ``fund``, mirroring the
``background_jobs`` model/repo split.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from portopt_db.base import BaseModel

# Use JSONB on PostgreSQL, plain JSON elsewhere (e.g. SQLite in tests).
_JSON = JSON().with_variant(JSONB, "postgresql")


class AgentRun(BaseModel):
    """One allocator run: input universe/config → output weights, with audit metadata."""

    __tablename__ = "agent_runs"
    __table_args__ = (
        Index("ix_agent_runs_portfolio_id", "portfolio_id"),
        Index("ix_agent_runs_status", "status"),
        Index("ix_agent_runs_asof", "asof"),
    )

    # No FK: the `portfolios` table was dropped in the ingestion strip; a run may
    # be standalone (backtest/paper) so this is a bare, nullable, indexed UUID.
    portfolio_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    asof: Mapped[date] = mapped_column(Date, nullable=False)
    # RNG seed pinned so an identical run reproduces identical weights (D-determinism).
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    universe: Mapped[list[str]] = mapped_column(_JSON, nullable=False)
    optimizer_config: Mapped[dict[str, Any]] = mapped_column(_JSON, nullable=False)
    weights: Mapped[dict[str, float] | None] = mapped_column(_JSON, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="pending"
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # LangGraph checkpointer thread id for this run (Phase 8: per-run threads,
    # defaults to ``str(run_id)``). Nullable — legacy rows predate per-run threading.
    thread_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    decisions: Mapped[list[AgentDecision]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="AgentDecision.decision_index",
    )


class AgentDecision(BaseModel):
    """One ordered step within a run: agent/step + constraint/view/LLM/HITL snapshot."""

    __tablename__ = "agent_decisions"
    __table_args__ = (
        UniqueConstraint(
            "run_id", "decision_index", name="uq_agent_decision_run_index"
        ),
        Index("ix_agent_decisions_run_id", "run_id"),
    )

    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    # 0-based position within the run; unique per run for deterministic ordering.
    decision_index: Mapped[int] = mapped_column(Integer, nullable=False)
    agent: Mapped[str] = mapped_column(String(100), nullable=False)
    step: Mapped[str] = mapped_column(String(100), nullable=False)
    constraint_set: Mapped[dict[str, Any] | None] = mapped_column(_JSON, nullable=True)
    views: Mapped[dict[str, Any] | None] = mapped_column(_JSON, nullable=True)
    llm_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Store the response verbatim, or a hash when the payload is large/sensitive.
    llm_response: Mapped[str | None] = mapped_column(Text, nullable=True)
    llm_response_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    hitl_decision: Mapped[dict[str, Any] | None] = mapped_column(_JSON, nullable=True)

    run: Mapped[AgentRun] = relationship(back_populates="decisions")
