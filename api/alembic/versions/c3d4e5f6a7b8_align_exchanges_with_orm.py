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
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    cols = {c["name"] for c in inspector.get_columns("exchanges")}
    indexes = {idx["name"] for idx in inspector.get_indexes("exchanges")}

    # If exchanges already has ORM schema (name, t212_id), skip everything.
    if "name" in cols and "exchange_name" not in cols:
        return

    # 1. Drop redundant indexes (only if they exist)
    for idx in (
        "idx_exchange_id",
        "ix_exchanges_exchange_id",
        "idx_exchange_name",
        "ix_exchanges_exchange_name",
    ):
        if idx in indexes:
            op.drop_index(idx, table_name="exchanges")

    # 2. Rename columns to match ORM
    if "exchange_name" in cols:
        op.alter_column(
            "exchanges",
            "exchange_name",
            new_column_name="name",
            existing_type=sa.String(255),
            existing_nullable=False,
        )
    if "exchange_id" in cols:
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

    # 4. Drop columns not in ORM
    for col in ("is_active", "last_updated"):
        if col in cols:
            op.drop_column("exchanges", col)

    # 5. Recreate unique constraint on name (skip if already present)
    uqs = {
        c["name"]
        for c in inspector.get_unique_constraints("exchanges")
        if c["name"]
    }
    if "uq_exchanges_name" not in uqs:
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
