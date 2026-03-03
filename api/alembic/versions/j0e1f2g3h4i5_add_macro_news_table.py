"""add_macro_news_table

Revision ID: j0e1f2g3h4i5
Revises: i9d0e1f2g3h4
Create Date: 2026-03-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "j0e1f2g3h4i5"
down_revision: str | Sequence[str] | None = "i9d0e1f2g3h4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "macro_news",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("news_id", sa.String(200), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("publisher", sa.String(500), nullable=True),
        sa.Column("link", sa.Text(), nullable=True),
        sa.Column("publish_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_ticker", sa.String(50), nullable=True),
        sa.Column("source_query", sa.String(200), nullable=True),
        sa.Column("themes", sa.Text(), nullable=True),
        sa.Column("snippet", sa.Text(), nullable=True),
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
        sa.UniqueConstraint("news_id", name="uq_macro_news_id"),
    )
    op.create_index("ix_macro_news_publish_time", "macro_news", ["publish_time"])
    op.create_index("ix_macro_news_themes", "macro_news", ["themes"])


def downgrade() -> None:
    op.drop_index("ix_macro_news_themes", table_name="macro_news")
    op.drop_index("ix_macro_news_publish_time", table_name="macro_news")
    op.drop_table("macro_news")
