"""Stock + bond investable universe: asset-class tagging + ETF fund metadata.

Adds the asset-class taxonomy to ``instruments`` (every existing row backfilled
to ``equity`` via the column server_default) and the four point-in-time ETF
metadata tables. Additive and non-destructive; downgrade drops the new tables
and columns.

Revision ID: f2a3b4c5d6e7
Revises: e1f2a3b4c5d6
Create Date: 2026-08-25
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "f2a3b4c5d6e7"
down_revision: str | Sequence[str] | None = "e1f2a3b4c5d6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _timestamps() -> list[sa.Column]:
    return [
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    ]


def upgrade() -> None:
    # --- instruments: asset-class taxonomy -------------------------------------
    # server_default 'equity' backfills every existing row in one shot; the
    # column stays NOT NULL for all future inserts.
    op.add_column(
        "instruments",
        sa.Column(
            "asset_class",
            sa.String(length=20),
            nullable=False,
            server_default="equity",
        ),
    )
    op.add_column(
        "instruments",
        sa.Column("fi_subclass", sa.String(length=20), nullable=True),
    )
    op.add_column(
        "instruments",
        sa.Column("duration_bucket", sa.String(length=20), nullable=True),
    )

    # --- etf_metadata (one row per instrument) --------------------------------
    op.create_table(
        "etf_metadata",
        sa.Column("id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("instrument_id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("aum", sa.Numeric(24, 2), nullable=True),
        sa.Column("nav", sa.Numeric(20, 6), nullable=True),
        sa.Column("fund_family", sa.String(length=255), nullable=True),
        sa.Column("legal_type", sa.String(length=100), nullable=True),
        sa.Column("expense_ratio", sa.Numeric(10, 6), nullable=True),
        sa.Column("base_currency", sa.String(length=10), nullable=True),
        sa.Column("as_of", sa.Date(), nullable=True),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["instrument_id"], ["instruments.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("instrument_id", name="uq_etf_metadata_instrument"),
    )
    op.create_index("ix_etf_metadata_instrument_id", "etf_metadata", ["instrument_id"])

    # --- etf_asset_classes (point-in-time stock/bond/cash/other %) -------------
    op.create_table(
        "etf_asset_classes",
        sa.Column("id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("instrument_id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("stock_pct", sa.Numeric(10, 6), nullable=True),
        sa.Column("bond_pct", sa.Numeric(10, 6), nullable=True),
        sa.Column("cash_pct", sa.Numeric(10, 6), nullable=True),
        sa.Column("other_pct", sa.Numeric(10, 6), nullable=True),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["instrument_id"], ["instruments.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "instrument_id", "as_of", name="uq_etf_asset_classes_instrument_asof"
        ),
    )
    op.create_index(
        "ix_etf_asset_classes_instrument_id",
        "etf_asset_classes",
        ["instrument_id"],
    )

    # --- etf_holdings (top-N constituents) ------------------------------------
    op.create_table(
        "etf_holdings",
        sa.Column("id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("instrument_id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("holding_symbol", sa.String(length=50), nullable=False),
        sa.Column("holding_name", sa.String(length=255), nullable=True),
        sa.Column("weight", sa.Numeric(10, 6), nullable=True),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["instrument_id"], ["instruments.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "instrument_id",
            "as_of",
            "holding_symbol",
            name="uq_etf_holdings_instrument_asof_symbol",
        ),
    )
    op.create_index("ix_etf_holdings_instrument_id", "etf_holdings", ["instrument_id"])

    # --- etf_sector_weights ----------------------------------------------------
    op.create_table(
        "etf_sector_weights",
        sa.Column("id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("instrument_id", sa.UUID(as_uuid=True), nullable=False),
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("sector", sa.String(length=100), nullable=False),
        sa.Column("weight", sa.Numeric(10, 6), nullable=True),
        *_timestamps(),
        sa.ForeignKeyConstraint(
            ["instrument_id"], ["instruments.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "instrument_id",
            "as_of",
            "sector",
            name="uq_etf_sector_weights_instrument_asof_sector",
        ),
    )
    op.create_index(
        "ix_etf_sector_weights_instrument_id",
        "etf_sector_weights",
        ["instrument_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_etf_sector_weights_instrument_id", "etf_sector_weights")
    op.drop_table("etf_sector_weights")
    op.drop_index("ix_etf_holdings_instrument_id", "etf_holdings")
    op.drop_table("etf_holdings")
    op.drop_index("ix_etf_asset_classes_instrument_id", "etf_asset_classes")
    op.drop_table("etf_asset_classes")
    op.drop_index("ix_etf_metadata_instrument_id", "etf_metadata")
    op.drop_table("etf_metadata")
    op.drop_column("instruments", "duration_bucket")
    op.drop_column("instruments", "fi_subclass")
    op.drop_column("instruments", "asset_class")
