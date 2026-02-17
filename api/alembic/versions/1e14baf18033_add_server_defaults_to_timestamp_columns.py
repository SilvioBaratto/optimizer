"""add server defaults to timestamp columns

Revision ID: 1e14baf18033
Revises: c51c70973022
Create Date: 2026-01-30 15:34:51.517959

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1e14baf18033'
down_revision: Union[str, Sequence[str], None] = 'c51c70973022'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add server_default=now() to all timestamp columns."""
    for table in ("exchanges", "instruments"):
        for column in ("created_at", "updated_at"):
            op.alter_column(
                table,
                column,
                server_default=sa.func.now(),
            )


def downgrade() -> None:
    """Remove server defaults from timestamp columns."""
    for table in ("exchanges", "instruments"):
        for column in ("created_at", "updated_at"):
            op.alter_column(
                table,
                column,
                server_default=None,
            )
