"""Align exchanges table columns with ORM model.

DB has exchange_name/exchange_id(int)/is_active/last_updated;
ORM expects name/t212_id(int, nullable). Rename columns, drop extras,
fix indexes to match.

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-02-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c3d4e5f6a7b8"
down_revision: str | Sequence[str] | None = "b2c3d4e5f6a7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Drop redundant indexes (keep one of each after rename)
    op.drop_index("idx_exchange_id", table_name="exchanges")
    op.drop_index("ix_exchanges_exchange_id", table_name="exchanges")
    op.drop_index("idx_exchange_name", table_name="exchanges")
    op.drop_index("ix_exchanges_exchange_name", table_name="exchanges")

    # 2. Rename columns to match ORM
    op.alter_column(
        "exchanges",
        "exchange_name",
        new_column_name="name",
        existing_type=sa.String(255),
        existing_nullable=False,
    )
    op.alter_column(
        "exchanges",
        "exchange_id",
        new_column_name="t212_id",
        existing_type=sa.Integer(),
        existing_nullable=False,
    )

    # 3. Make t212_id nullable (ORM: Optional[int])
    op.alter_column(
        "exchanges",
        "t212_id",
        existing_type=sa.Integer(),
        nullable=True,
    )

    # 4. Drop columns not in ORM (BaseModel already provides updated_at)
    op.drop_column("exchanges", "is_active")
    op.drop_column("exchanges", "last_updated")

    # 5. Recreate unique constraint and index on name
    op.create_unique_constraint("uq_exchanges_name", "exchanges", ["name"])


def downgrade() -> None:
    # Reverse: restore old column names, constraints, and columns
    op.drop_constraint("uq_exchanges_name", "exchanges", type_="unique")

    op.add_column(
        "exchanges",
        sa.Column("last_updated", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "exchanges",
        sa.Column(
            "is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")
        ),
    )

    op.alter_column(
        "exchanges",
        "t212_id",
        existing_type=sa.Integer(),
        nullable=False,
    )
    op.alter_column(
        "exchanges",
        "t212_id",
        new_column_name="exchange_id",
        existing_type=sa.Integer(),
        existing_nullable=False,
    )
    op.alter_column(
        "exchanges",
        "name",
        new_column_name="exchange_name",
        existing_type=sa.String(255),
        existing_nullable=False,
    )

    op.create_index("ix_exchanges_exchange_name", "exchanges", ["exchange_name"])
    op.create_index("idx_exchange_name", "exchanges", ["exchange_name"])
    op.create_index(
        "ix_exchanges_exchange_id", "exchanges", ["exchange_id"], unique=True
    )
    op.create_index("idx_exchange_id", "exchanges", ["exchange_id"])
