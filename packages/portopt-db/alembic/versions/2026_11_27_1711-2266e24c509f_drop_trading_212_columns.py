"""Drop the Trading 212 columns of ``exchanges`` and ``instruments``.

The Trading 212 integration is gone, so nothing reads or writes
``exchanges.t212_id``, ``instruments.t212_ticker`` or
``instruments.is_t212_tradeable`` any more. The upgrade deletes the data in them.
The downgrade re-adds the three columns nullable and empty: the values came from the
broker's catalogue and are not restored. Take a database dump before upgrading if the
data matters.

Revision ID: 2266e24c509f
Revises: b8d0f2a4c6e1
Create Date: 2026-11-27
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "2266e24c509f"
down_revision: str | Sequence[str] | None = "b8d0f2a4c6e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Drop the three columns; their data is deleted."""
    op.drop_column("instruments", "is_t212_tradeable")
    op.drop_column("instruments", "t212_ticker")
    op.drop_column("exchanges", "t212_id")


def downgrade() -> None:
    """Re-add the three columns as nullable and empty; the dropped data is not restored."""
    op.add_column("exchanges", sa.Column("t212_id", sa.Integer(), nullable=True))
    op.add_column(
        "instruments", sa.Column("t212_ticker", sa.String(length=100), nullable=True)
    )
    op.add_column(
        "instruments", sa.Column("is_t212_tradeable", sa.Boolean(), nullable=True)
    )
