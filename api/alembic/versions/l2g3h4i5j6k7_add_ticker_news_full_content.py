"""add_ticker_news_full_content

Revision ID: l2g3h4i5j6k7
Revises: k1f2g3h4i5j6
Create Date: 2026-03-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "l2g3h4i5j6k7"
down_revision: str | Sequence[str] | None = "k1f2g3h4i5j6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("ticker_news", sa.Column("full_content", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("ticker_news", "full_content")
