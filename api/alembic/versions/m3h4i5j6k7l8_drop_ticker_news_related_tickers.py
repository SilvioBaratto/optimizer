"""drop_ticker_news_related_tickers

Revision ID: m3h4i5j6k7l8
Revises: l2g3h4i5j6k7
Create Date: 2026-03-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "m3h4i5j6k7l8"
down_revision: str | Sequence[str] | None = "l2g3h4i5j6k7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_column("ticker_news", "related_tickers")


def downgrade() -> None:
    op.add_column("ticker_news", sa.Column("related_tickers", sa.Text(), nullable=True))
