"""first commit

Revision ID: cdf65b4bcc4a
Revises: 583bf3da4b28
Create Date: 2025-10-13 12:58:22.985710

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'cdf65b4bcc4a'
down_revision: Union[str, Sequence[str], None] = '583bf3da4b28'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # NOTE: This migration is a no-op because the tables were already created
    # in the previous migration (583bf3da4b28). This was a duplicate auto-generated
    # migration that has been cleaned up.
    pass


def downgrade() -> None:
    """Downgrade schema."""
    # NOTE: No-op - tables are managed by migration 583bf3da4b28
    pass
