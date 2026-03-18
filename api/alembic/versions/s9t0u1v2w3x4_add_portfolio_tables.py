"""Add portfolio persistence tables.

Revision ID: s9t0u1v2w3x4
Revises: r8s9t0u1v2w3
Create Date: 2026-03-18
"""

from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision = "s9t0u1v2w3x4"
down_revision = "r8s9t0u1v2w3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. portfolios
    op.create_table(
        "portfolios",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid4),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("currency", sa.String(10), nullable=False, server_default="EUR"),
        sa.Column("benchmark_ticker", sa.String(50), nullable=False, server_default="SPY"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("name", name="uq_portfolio_name"),
    )

    # 2. portfolio_snapshots
    op.create_table(
        "portfolio_snapshots",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid4),
        sa.Column("portfolio_id", UUID(as_uuid=True), sa.ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False),
        sa.Column("snapshot_date", sa.Date, nullable=False),
        sa.Column("snapshot_type", sa.String(30), nullable=False),
        sa.Column("weights", JSONB, nullable=False),
        sa.Column("sector_mapping", JSONB, nullable=True),
        sa.Column("summary", JSONB, nullable=True),
        sa.Column("optimizer_config", JSONB, nullable=True),
        sa.Column("turnover", sa.Float, nullable=True),
        sa.Column("holding_count", sa.Integer, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("portfolio_id", "snapshot_date", "snapshot_type", name="uq_snapshot_portfolio_date_type"),
    )
    op.create_index("ix_snapshots_portfolio_id", "portfolio_snapshots", ["portfolio_id"])
    op.create_index("ix_snapshots_date", "portfolio_snapshots", ["snapshot_date"])

    # 3. broker_positions
    op.create_table(
        "broker_positions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid4),
        sa.Column("portfolio_id", UUID(as_uuid=True), sa.ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False),
        sa.Column("ticker", sa.String(100), nullable=False),
        sa.Column("yfinance_ticker", sa.String(100), nullable=True),
        sa.Column("name", sa.String(500), nullable=True),
        sa.Column("quantity", sa.Float, nullable=False),
        sa.Column("average_price", sa.Float, nullable=False),
        sa.Column("current_price", sa.Float, nullable=True),
        sa.Column("ppl", sa.Float, nullable=True),
        sa.Column("fx_ppl", sa.Float, nullable=True),
        sa.Column("initial_fill_date", sa.Date, nullable=True),
        sa.Column("synced_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("portfolio_id", "ticker", name="uq_broker_position_portfolio_ticker"),
    )
    op.create_index("ix_broker_positions_portfolio_id", "broker_positions", ["portfolio_id"])

    # 4. broker_account_snapshots
    op.create_table(
        "broker_account_snapshots",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid4),
        sa.Column("portfolio_id", UUID(as_uuid=True), sa.ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False),
        sa.Column("total", sa.Float, nullable=False),
        sa.Column("free", sa.Float, nullable=False),
        sa.Column("invested", sa.Float, nullable=False),
        sa.Column("blocked", sa.Float, nullable=True),
        sa.Column("result", sa.Float, nullable=True),
        sa.Column("currency", sa.String(10), nullable=False),
        sa.Column("synced_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_broker_account_portfolio_id", "broker_account_snapshots", ["portfolio_id"])

    # 5. activity_events
    op.create_table(
        "activity_events",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid4),
        sa.Column("portfolio_id", UUID(as_uuid=True), sa.ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=True),
        sa.Column("event_type", sa.String(30), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("metadata", JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_activity_events_portfolio_id", "activity_events", ["portfolio_id"])
    op.create_index("ix_activity_events_event_type", "activity_events", ["event_type"])
    op.create_index("ix_activity_events_created_at", "activity_events", ["created_at"])

    # 6. regime_states
    op.create_table(
        "regime_states",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid4),
        sa.Column("state_date", sa.Date, nullable=False),
        sa.Column("regime", sa.String(20), nullable=False),
        sa.Column("probabilities", JSONB, nullable=False),
        sa.Column("model_type", sa.String(30), nullable=False, server_default="hmm"),
        sa.Column("metadata", JSONB, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("state_date", "model_type", name="uq_regime_state_date_model"),
    )


def downgrade() -> None:
    op.drop_table("regime_states")
    op.drop_index("ix_activity_events_created_at", table_name="activity_events")
    op.drop_index("ix_activity_events_event_type", table_name="activity_events")
    op.drop_index("ix_activity_events_portfolio_id", table_name="activity_events")
    op.drop_table("activity_events")
    op.drop_index("ix_broker_account_portfolio_id", table_name="broker_account_snapshots")
    op.drop_table("broker_account_snapshots")
    op.drop_index("ix_broker_positions_portfolio_id", table_name="broker_positions")
    op.drop_table("broker_positions")
    op.drop_index("ix_snapshots_date", table_name="portfolio_snapshots")
    op.drop_index("ix_snapshots_portfolio_id", table_name="portfolio_snapshots")
    op.drop_table("portfolio_snapshots")
    op.drop_table("portfolios")
