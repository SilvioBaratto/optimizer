"""Add the human-decision audit log table (``human_audit_log``).

TUI oversight (Phase 1): append-only, immutable record of every human action
(approve / reject / killswitch / mandate_edit / guardrail_override) that the
shell's unified action path funnels through, with provenance (prev/new value,
source, idempotency key) and the override / suitability flags the Audit view
surfaces. Model lives in ``portopt_db``; repo behavior in ``fund``.

Revision ID: f1a2b3c4d5e6
Revises: d5e6f7a8b9c0
Create Date: 2026-10-08
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision: str = "f1a2b3c4d5e6"
down_revision: str | Sequence[str] | None = "d5e6f7a8b9c0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Dialect-aware JSON: JSONB on PostgreSQL, plain JSON elsewhere (SQLite in tests).
_JSON = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    """Create the ``human_audit_log`` table with its indexes."""
    op.create_table(
        "human_audit_log",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        # No FK: the `portfolios` table was dropped in the ingestion strip, and a
        # fund-wide action carries no portfolio.
        sa.Column("portfolio_id", UUID(as_uuid=True), nullable=True),
        sa.Column("run_id", UUID(as_uuid=True), nullable=True),
        sa.Column("actor", sa.String(50), nullable=False, server_default="human"),
        sa.Column("action", sa.String(50), nullable=False),
        sa.Column("target", sa.String(200), nullable=True),
        sa.Column("prev_value", _JSON, nullable=True),
        sa.Column("new_value", _JSON, nullable=True),
        sa.Column("source", sa.String(100), nullable=True),
        sa.Column(
            "override_flag", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column(
            "suitability_flag", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column("idempotency_key", sa.String(200), nullable=True),
        sa.Column("rationale", sa.Text, nullable=True),
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
            "idempotency_key", name="uq_human_audit_log_idempotency_key"
        ),
    )
    op.create_index(
        "ix_human_audit_log_portfolio_id", "human_audit_log", ["portfolio_id"]
    )
    op.create_index("ix_human_audit_log_run_id", "human_audit_log", ["run_id"])
    op.create_index("ix_human_audit_log_action", "human_audit_log", ["action"])
    op.create_index("ix_human_audit_log_created_at", "human_audit_log", ["created_at"])


def downgrade() -> None:
    """Drop the ``human_audit_log`` table and its indexes."""
    op.drop_index("ix_human_audit_log_created_at", table_name="human_audit_log")
    op.drop_index("ix_human_audit_log_action", table_name="human_audit_log")
    op.drop_index("ix_human_audit_log_run_id", table_name="human_audit_log")
    op.drop_index("ix_human_audit_log_portfolio_id", table_name="human_audit_log")
    op.drop_table("human_audit_log")
