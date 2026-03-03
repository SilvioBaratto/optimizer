"""add_macro_news_full_content

Revision ID: k1f2g3h4i5j6
Revises: j0e1f2g3h4i5
Create Date: 2026-03-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "k1f2g3h4i5j6"
down_revision: str | Sequence[str] | None = "j0e1f2g3h4i5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("macro_news", sa.Column("full_content", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("macro_news", "full_content")
