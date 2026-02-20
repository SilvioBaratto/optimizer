"""Add delisting columns to instruments table for survivorship-bias correction.

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-02-20
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, Sequence[str], None] = "c3d4e5f6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "instruments",
        sa.Column("delisted_at", sa.Date(), nullable=True),
    )
    op.add_column(
        "instruments",
        sa.Column("delisting_return", sa.Float(), nullable=True),
    )
    op.create_index(
        "ix_instrument_delisted_at",
        "instruments",
        ["delisted_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_instrument_delisted_at", table_name="instruments")
    op.drop_column("instruments", "delisting_return")
    op.drop_column("instruments", "delisted_at")
