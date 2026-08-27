"""Add market_summaries table.

Task B3 (Gap B): regional index/quote summaries from yf.Market(id).summary.

Revision ID: d3e4f5a6b7c8
Revises: c2d3e4f5a6b7
Create Date: 2026-08-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "d3e4f5a6b7c8"
down_revision: str | Sequence[str] | None = "c2d3e4f5a6b7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "market_summaries",
        sa.Column("market", sa.String(20), nullable=False),
        sa.Column("symbol", sa.String(40), nullable=False),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("short_name", sa.String(255), nullable=True),
        sa.Column("price", sa.Numeric(24, 6), nullable=True),
        sa.Column("change", sa.Numeric(24, 6), nullable=True),
        sa.Column("change_percent", sa.Numeric(20, 6), nullable=True),
        sa.Column("previous_close", sa.Numeric(24, 6), nullable=True),
        sa.Column("market_state", sa.String(20), nullable=True),
        sa.Column(
            "id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("market", "symbol", "as_of", name="uq_market_summary"),
    )
    op.create_index("ix_market_summaries_market", "market_summaries", ["market"])


def downgrade() -> None:
    op.drop_index("ix_market_summaries_market", table_name="market_summaries")
    op.drop_table("market_summaries")
