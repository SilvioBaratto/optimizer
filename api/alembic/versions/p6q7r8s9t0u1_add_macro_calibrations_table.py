"""Add macro_calibrations table for cached LLM regime classifications.

Revision ID: p6q7r8s9t0u1
Revises: o5j6k7l8m9n0
Create Date: 2026-03-15
"""

from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

from alembic import op

revision = "p6q7r8s9t0u1"
down_revision = "o5j6k7l8m9n0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "macro_calibrations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid4),
        sa.Column("country", sa.String(100), nullable=False),
        sa.Column("phase", sa.String(50), nullable=False),
        sa.Column("delta", sa.Float, nullable=False),
        sa.Column("tau", sa.Float, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("rationale", sa.Text, nullable=True),
        sa.Column("macro_summary", sa.Text, nullable=True),
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
        sa.UniqueConstraint("country", name="uq_macro_calibration_country"),
    )
    op.create_index(
        "ix_macro_calibrations_country", "macro_calibrations", ["country"]
    )


def downgrade() -> None:
    op.drop_index("ix_macro_calibrations_country", table_name="macro_calibrations")
    op.drop_table("macro_calibrations")
