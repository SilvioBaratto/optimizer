"""Add currency_code to financial_statements for audit and normalization.

Revision ID: u1v2w3x4y5z6
Revises: t0u1v2w3x4y5
Create Date: 2026-03-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "u1v2w3x4y5z6"
down_revision: str | Sequence[str] | None = "t0u1v2w3x4y5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "financial_statements",
        sa.Column("currency_code", sa.String(10), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("financial_statements", "currency_code")
