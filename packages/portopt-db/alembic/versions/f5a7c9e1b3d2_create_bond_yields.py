"""Create the government bond yield snapshot table (``bond_yields``).

One row per ``(country, maturity)`` with the latest yield and its period changes.
No earlier revision ever created it (``ecf0a9a2bfdc`` is an empty stub and
``a1b2c3d4e5f6`` only rebuilt the other macro tables), so it existed solely
where ingestion's ``create_all`` had run, and an Alembic-only database lacked
the table the bond-yield upsert targets. The upgrade is guarded so databases
that ``create_all`` already built keep their rows.

Revision ID: f5a7c9e1b3d2
Revises: d8f0b2c4e6a1
Create Date: 2026-10-23
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "f5a7c9e1b3d2"
down_revision: str | Sequence[str] | None = "d8f0b2c4e6a1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create ``bond_yields`` unless ``create_all`` already built it."""
    if sa.inspect(op.get_bind()).has_table("bond_yields"):
        return
    op.create_table(
        "bond_yields",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("country", sa.String(100), nullable=False),
        sa.Column("maturity", sa.String(10), nullable=False),
        sa.Column("yield_value", sa.Float(), nullable=True),
        sa.Column("day_change", sa.Float(), nullable=True),
        sa.Column("month_change", sa.Float(), nullable=True),
        sa.Column("year_change", sa.Float(), nullable=True),
        sa.Column("reference_date", sa.Date(), nullable=True),
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
            "country", "maturity", name="uq_bond_yield_country_maturity"
        ),
    )
    op.create_index("ix_bond_yields_country", "bond_yields", ["country"])


def downgrade() -> None:
    """Drop the ``bond_yields`` table and its index."""
    op.drop_index("ix_bond_yields_country", table_name="bond_yields")
    op.drop_table("bond_yields")
