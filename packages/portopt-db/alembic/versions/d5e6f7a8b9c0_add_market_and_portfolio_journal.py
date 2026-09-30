"""Add market_journal and portfolio_journal tables.

``market_journal`` — one global macro/market/news digest row per trading day,
written by the ingestion ``daily_events`` step and read by fund history tools.

``portfolio_journal`` — one per-portfolio rebalance overlay per
(portfolio_id, as_of), written by ``fund._finalize_hitl`` on approve.

Both use the ``_JSON = JSON().with_variant(JSONB, "postgresql")`` pattern so
SQLite migration tests can create and tear down the tables without a live PG.

Revision ID: d5e6f7a8b9c0
Revises: c4d5e6f7a8b9
Create Date: 2026-09-30
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

revision: str = "d5e6f7a8b9c0"
down_revision: str | Sequence[str] | None = "c4d5e6f7a8b9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# JSONB on PostgreSQL, plain JSON elsewhere (SQLite in migration tests).
_JSON = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    """Create ``market_journal`` and ``portfolio_journal`` tables with their indexes."""
    # 1. market_journal — global daily digest
    op.create_table(
        "market_journal",
        sa.Column("id", sa.Uuid, primary_key=True),
        sa.Column("as_of", sa.Date, nullable=False),
        sa.Column("region", sa.String(8), nullable=False, server_default="US"),
        sa.Column("macro_deltas", _JSON, nullable=False, server_default="{}"),
        sa.Column("market_moves", _JSON, nullable=False, server_default="{}"),
        sa.Column("news_themes", _JSON, nullable=False, server_default="{}"),
        sa.Column("narrative", sa.Text, nullable=False, server_default=""),
        sa.Column("source_counts", _JSON, nullable=False, server_default="{}"),
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
        sa.UniqueConstraint("as_of", name="uq_market_journal_as_of"),
    )
    op.create_index("ix_market_journal_as_of", "market_journal", ["as_of"])

    # 2. portfolio_journal — per-portfolio rebalance overlay
    op.create_table(
        "portfolio_journal",
        sa.Column("id", sa.Uuid, primary_key=True),
        # Portable Uuid avoids the SQLite UUID/Float affinity crash on non-PK
        # UUID columns under statement caching.
        sa.Column("portfolio_id", sa.Uuid, nullable=False),
        sa.Column("as_of", sa.Date, nullable=False),
        sa.Column("run_id", sa.Uuid, nullable=True),
        sa.Column("trades", _JSON, nullable=True),
        sa.Column("allocation", _JSON, nullable=True),
        sa.Column("drift", _JSON, nullable=True),
        sa.Column("narrative", sa.Text, nullable=False, server_default=""),
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
            "portfolio_id", "as_of", name="uq_portfolio_journal_key"
        ),
    )
    op.create_index(
        "ix_portfolio_journal_portfolio_id", "portfolio_journal", ["portfolio_id"]
    )
    op.create_index("ix_portfolio_journal_as_of", "portfolio_journal", ["as_of"])


def downgrade() -> None:
    """Drop ``portfolio_journal`` then ``market_journal`` and all their indexes."""
    op.drop_index("ix_portfolio_journal_as_of", table_name="portfolio_journal")
    op.drop_index(
        "ix_portfolio_journal_portfolio_id", table_name="portfolio_journal"
    )
    op.drop_table("portfolio_journal")
    op.drop_index("ix_market_journal_as_of", table_name="market_journal")
    op.drop_table("market_journal")
