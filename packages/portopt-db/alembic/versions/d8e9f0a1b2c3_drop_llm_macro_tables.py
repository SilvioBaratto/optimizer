"""Drop LLM-output macro tables (macro_calibrations, macro_news_summaries).

The LLM layer is leaving ingestion to become a future standalone package, so
its two output tables are removed here. Everything scraper-populated
(``macro_news``, ``macro_news_themes``) and every other macro table is
untouched — these two held only LLM-generated regime calibrations and daily
news summaries.

``downgrade()`` faithfully recreates both tables (columns, constraints, and
indexes) as they stood at head ``b6c7d8e9f0a1``, i.e. the union of
``p6q7r8s9t0u1`` (create macro_calibrations), ``b1c2d3e4f5g6`` (add
regime_classification), and ``q7r8s9t0u1v2`` (create macro_news_summaries).
It does not restore row data — the tables come back empty.

Revision ID: d8e9f0a1b2c3
Revises: b6c7d8e9f0a1
Create Date: 2026-09-10
"""

from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

revision: str = "d8e9f0a1b2c3"
down_revision: str | None = "b6c7d8e9f0a1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index(
        "ix_macro_news_summaries_summary_date",
        table_name="macro_news_summaries",
    )
    op.drop_index(
        "ix_macro_news_summaries_country",
        table_name="macro_news_summaries",
    )
    op.drop_table("macro_news_summaries")

    op.drop_index("ix_macro_calibrations_country", table_name="macro_calibrations")
    op.drop_table("macro_calibrations")


def downgrade() -> None:
    op.create_table(
        "macro_calibrations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid4),
        sa.Column("country", sa.String(100), nullable=False),
        sa.Column("phase", sa.String(50), nullable=False),
        sa.Column("delta", sa.Float, nullable=False),
        sa.Column("tau", sa.Float, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("rationale", sa.Text, nullable=True),
        sa.Column("macro_summary", sa.Text, nullable=True),
        sa.Column("regime_classification", sa.String(50), nullable=True),
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
        sa.UniqueConstraint("country", name="uq_macro_calibration_country"),
    )
    op.create_index("ix_macro_calibrations_country", "macro_calibrations", ["country"])

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
            "country",
            "summary_date",
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
