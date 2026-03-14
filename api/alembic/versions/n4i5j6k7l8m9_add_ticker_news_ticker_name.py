"""add_ticker_news_ticker_name

Revision ID: n4i5j6k7l8m9
Revises: m3h4i5j6k7l8
Create Date: 2026-03-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "n4i5j6k7l8m9"
down_revision: str | Sequence[str] | None = "m3h4i5j6k7l8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("ticker_news", sa.Column("ticker_name", sa.String(500), nullable=True))
    # Backfill from instruments.name
    op.execute(
        """
        UPDATE ticker_news tn
        SET ticker_name = i.name
        FROM instruments i
        WHERE tn.instrument_id = i.id
        """
    )


def downgrade() -> None:
    op.drop_column("ticker_news", "ticker_name")
