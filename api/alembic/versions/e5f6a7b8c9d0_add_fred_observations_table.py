"""add_fred_observations_table

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-03-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision: str = "e5f6a7b8c9d0"
down_revision: str | Sequence[str] | None = "d4e5f6a7b8c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "fred_observations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("series_id", sa.String(50), nullable=False),
        sa.Column("date", sa.Date, nullable=False),
        sa.Column("value", sa.Float, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "series_id", "date", name="uq_fred_observation_series_date"
        ),
    )
    op.create_index(
        "ix_fred_observations_series_id", "fred_observations", ["series_id"]
    )
    op.create_index(
        "ix_fred_observations_date", "fred_observations", ["date"]
    )


def downgrade() -> None:
    op.drop_index("ix_fred_observations_date", table_name="fred_observations")
    op.drop_index(
        "ix_fred_observations_series_id", table_name="fred_observations"
    )
    op.drop_table("fred_observations")
