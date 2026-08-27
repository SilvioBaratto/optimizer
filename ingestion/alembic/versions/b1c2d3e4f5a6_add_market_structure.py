"""Add market-structure tables (sector snapshots / industries / top companies).

Task B1 (Gap B / OQ4): sector & industry rollups from yf.Sector, per region.

Revision ID: b1c2d3e4f5a6
Revises: a0b1c2d3e4f5
Create Date: 2026-08-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "b1c2d3e4f5a6"
down_revision: str | Sequence[str] | None = "a0b1c2d3e4f5"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _id_and_timestamps() -> list[sa.Column]:
    return [
        sa.Column(
            "id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    ]


def upgrade() -> None:
    op.create_table(
        "sector_snapshots",
        sa.Column("sector_key", sa.String(50), nullable=False),
        sa.Column("region", sa.String(8), nullable=False),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("name", sa.String(100), nullable=True),
        sa.Column("symbol", sa.String(30), nullable=True),
        sa.Column("market_cap", sa.Numeric(28, 2), nullable=True),
        sa.Column("market_weight", sa.Numeric(12, 8), nullable=True),
        sa.Column("companies_count", sa.Integer(), nullable=True),
        sa.Column("industries_count", sa.Integer(), nullable=True),
        sa.Column("employee_count", sa.BigInteger(), nullable=True),
        *_id_and_timestamps(),
        sa.UniqueConstraint("sector_key", "region", "as_of", name="uq_sector_snapshot"),
    )
    op.create_index(
        "ix_sector_snapshots_key_region",
        "sector_snapshots",
        ["sector_key", "region"],
    )

    op.create_table(
        "sector_industries",
        sa.Column("sector_key", sa.String(50), nullable=False),
        sa.Column("region", sa.String(8), nullable=False),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("industry_key", sa.String(80), nullable=False),
        sa.Column("industry_name", sa.String(150), nullable=True),
        *_id_and_timestamps(),
        sa.UniqueConstraint(
            "sector_key",
            "region",
            "as_of",
            "industry_key",
            name="uq_sector_industry",
        ),
    )
    op.create_index(
        "ix_sector_industries_key_region",
        "sector_industries",
        ["sector_key", "region"],
    )

    op.create_table(
        "sector_top_companies",
        sa.Column("sector_key", sa.String(50), nullable=False),
        sa.Column("region", sa.String(8), nullable=False),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("symbol", sa.String(30), nullable=False),
        sa.Column("name", sa.String(255), nullable=True),
        sa.Column("weight", sa.Numeric(12, 8), nullable=True),
        sa.Column("rating", sa.String(50), nullable=True),
        *_id_and_timestamps(),
        sa.UniqueConstraint(
            "sector_key",
            "region",
            "as_of",
            "symbol",
            name="uq_sector_top_company",
        ),
    )
    op.create_index(
        "ix_sector_top_companies_key_region",
        "sector_top_companies",
        ["sector_key", "region"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_sector_top_companies_key_region", table_name="sector_top_companies"
    )
    op.drop_table("sector_top_companies")
    op.drop_index("ix_sector_industries_key_region", table_name="sector_industries")
    op.drop_table("sector_industries")
    op.drop_index("ix_sector_snapshots_key_region", table_name="sector_snapshots")
    op.drop_table("sector_snapshots")
