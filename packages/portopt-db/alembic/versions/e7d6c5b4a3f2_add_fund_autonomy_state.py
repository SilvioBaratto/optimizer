"""Add the per-scope fund autonomy-state table (``fund_autonomy_state``).

Phase-3 co-pilot: one row per portfolio plus a single fund-wide row, each holding
the ARMED/FROZEN posture and the ``first_round_gated`` flag that forces every PM
gate on a portfolio's first round. A partial unique index pins the fund row (NULL
portfolio_id) to a singleton — a plain UNIQUE cannot, since Postgres treats
NULL != NULL. ``scope`` / ``state`` stay plain varchars (app-side str enums), not
native DB enums. Model lives in ``portopt_db``; the ``AutonomyRepository``
behavior in ``fund``.

Revision ID: e7d6c5b4a3f2
Revises: f1a2b3c4d5e6
Create Date: 2026-10-09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

revision: str = "e7d6c5b4a3f2"
down_revision: str | Sequence[str] | None = "f1a2b3c4d5e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the ``fund_autonomy_state`` table with its uniqueness guards."""
    op.create_table(
        "fund_autonomy_state",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("scope", sa.String(20), nullable=False),
        # Portable Uuid: the fund row keeps a genuine NULL portfolio_id.
        sa.Column("portfolio_id", sa.Uuid(), nullable=True),
        sa.Column("state", sa.String(20), nullable=False, server_default="armed"),
        sa.Column(
            "first_round_gated",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
        sa.Column("frozen_reason", sa.String(500), nullable=True),
        sa.Column("frozen_at", sa.DateTime(timezone=True), nullable=True),
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
            "scope", "portfolio_id", name="uq_fund_autonomy_state_scope_portfolio"
        ),
    )
    # Partial unique index so only one scope='fund' row (NULL portfolio_id) can
    # exist — NULL != NULL defeats a plain UNIQUE. Cross-dialect via both the
    # postgresql_ and sqlite_ where clauses.
    op.create_index(
        "uq_fund_autonomy_state_fund_singleton",
        "fund_autonomy_state",
        ["scope"],
        unique=True,
        postgresql_where=text("portfolio_id IS NULL"),
        sqlite_where=text("portfolio_id IS NULL"),
    )


def downgrade() -> None:
    """Drop the ``fund_autonomy_state`` table and its partial unique index."""
    op.drop_index(
        "uq_fund_autonomy_state_fund_singleton", table_name="fund_autonomy_state"
    )
    op.drop_table("fund_autonomy_state")
