"""Add sec_filings table.

Task A5 (Gap A): persist SEC filings from yf.Ticker.sec_filings.

Revision ID: b5c6d7e8f9a0
Revises: a4b5c6d7e8f9
Create Date: 2026-08-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "b5c6d7e8f9a0"
down_revision: str | Sequence[str] | None = "a4b5c6d7e8f9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "sec_filings",
        sa.Column(
            "instrument_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("instruments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("filing_date", sa.Date(), nullable=False),
        sa.Column("form_type", sa.String(50), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("url", sa.Text(), nullable=True),
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
        sa.UniqueConstraint(
            "instrument_id", "filing_date", "form_type", "title", name="uq_sec_filing"
        ),
    )
    op.create_index("ix_sec_filings_instrument_id", "sec_filings", ["instrument_id"])


def downgrade() -> None:
    op.drop_index("ix_sec_filings_instrument_id", table_name="sec_filings")
    op.drop_table("sec_filings")
