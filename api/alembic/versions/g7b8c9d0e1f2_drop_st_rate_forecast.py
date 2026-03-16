"""drop_st_rate_forecast

Revision ID: g7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-03-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "g7b8c9d0e1f2"
down_revision: str | Sequence[str] | None = "f6a7b8c9d0e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    conn = op.get_bind()
    cols = {c["name"] for c in sa.inspect(conn).get_columns("economic_indicators")}
    if "st_rate_forecast" in cols:
        op.drop_column("economic_indicators", "st_rate_forecast")


def downgrade() -> None:
    op.add_column(
        "economic_indicators",
        sa.Column("st_rate_forecast", sa.Float(), nullable=True),
    )
