"""Add agent-run audit tables (agent_runs, agent_decisions).

Fase 2 of the ``fund`` bridge: persistence for the allocator audit trail. One
run row (universe in, weights out, config + seed for determinism) with an
ordered per-step decision trail (constraint/view snapshot, LLM prompt/response
-or-hash, HITL decision). Model lives in ``portopt_db``; repo behavior in
``fund`` — the ``background_jobs`` split.

Revision ID: e9f0a1b2c3d4
Revises: d8e9f0a1b2c3
Create Date: 2026-09-16
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision: str = "e9f0a1b2c3d4"
down_revision: str | Sequence[str] | None = "d8e9f0a1b2c3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Dialect-aware JSON: JSONB on PostgreSQL, plain JSON elsewhere (SQLite in tests).
_JSON = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    # 1. agent_runs
    op.create_table(
        "agent_runs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        # No FK: the `portfolios` table was dropped in the ingestion strip.
        sa.Column("portfolio_id", UUID(as_uuid=True), nullable=True),
        sa.Column("asof", sa.Date, nullable=False),
        sa.Column("seed", sa.Integer, nullable=True),
        sa.Column("universe", _JSON, nullable=False),
        sa.Column("optimizer_config", _JSON, nullable=False),
        sa.Column("weights", _JSON, nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_agent_runs_portfolio_id", "agent_runs", ["portfolio_id"])
    op.create_index("ix_agent_runs_status", "agent_runs", ["status"])
    op.create_index("ix_agent_runs_asof", "agent_runs", ["asof"])

    # 2. agent_decisions
    op.create_table(
        "agent_decisions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "run_id",
            UUID(as_uuid=True),
            sa.ForeignKey("agent_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("decision_index", sa.Integer, nullable=False),
        sa.Column("agent", sa.String(100), nullable=False),
        sa.Column("step", sa.String(100), nullable=False),
        sa.Column("constraint_set", _JSON, nullable=True),
        sa.Column("views", _JSON, nullable=True),
        sa.Column("llm_prompt", sa.Text, nullable=True),
        sa.Column("llm_response", sa.Text, nullable=True),
        sa.Column("llm_response_hash", sa.String(64), nullable=True),
        sa.Column("hitl_decision", _JSON, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "run_id", "decision_index", name="uq_agent_decision_run_index"
        ),
    )
    op.create_index("ix_agent_decisions_run_id", "agent_decisions", ["run_id"])


def downgrade() -> None:
    op.drop_index("ix_agent_decisions_run_id", table_name="agent_decisions")
    op.drop_table("agent_decisions")

    op.drop_index("ix_agent_runs_asof", table_name="agent_runs")
    op.drop_index("ix_agent_runs_status", table_name="agent_runs")
    op.drop_index("ix_agent_runs_portfolio_id", table_name="agent_runs")
    op.drop_table("agent_runs")
