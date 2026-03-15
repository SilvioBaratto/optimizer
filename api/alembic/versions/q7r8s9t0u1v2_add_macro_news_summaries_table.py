"""Add macro_news_summaries table for daily country-level news summaries.

Revision ID: q7r8s9t0u1v2
Revises: p6q7r8s9t0u1
Create Date: 2026-03-15
"""

from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

revision = "q7r8s9t0u1v2"
down_revision = "p6q7r8s9t0u1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "macro_news_summaries",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid4),
        sa.Column("country", sa.String(100), nullable=False),
        sa.Column("summary_date", sa.Date, nullable=False),
        sa.Column("summary", sa.Text, nullable=True),
        sa.Column("sentiment", sa.String(50), nullable=True),
        sa.Column("sentiment_score", sa.Float, nullable=True),
        sa.Column("article_count", sa.Integer, nullable=True),
        sa.Column("news_summary", sa.Text, nullable=True),
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
            "country", "summary_date",
            name="uq_macro_news_summary_country_date",
        ),
    )
    op.create_index(
        "ix_macro_news_summaries_country", "macro_news_summaries", ["country"]
    )
    op.create_index(
        "ix_macro_news_summaries_summary_date",
        "macro_news_summaries",
        ["summary_date"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_macro_news_summaries_summary_date",
        table_name="macro_news_summaries",
    )
    op.drop_index(
        "ix_macro_news_summaries_country",
        table_name="macro_news_summaries",
    )
    op.drop_table("macro_news_summaries")
