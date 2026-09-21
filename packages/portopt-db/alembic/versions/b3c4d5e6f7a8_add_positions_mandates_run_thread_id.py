"""Add positions + portfolio_mandates tables and agent_runs.thread_id.

Phase 8 of the ``fund`` bridge: human observability & control. This single
revision lands the three schema changes the CLI/TUI read model and the
rebuild-to-resume path need:

* ``positions`` — the current-holdings snapshot (one row per
  ``(portfolio_id, ticker)``) written on an approved HITL fill; ``paper_order_id``
  links back to the ticket. Positions are a *current snapshot*, not a time series
  (history stays derivable from ``paper_orders``).
* ``portfolio_mandates`` — one row per portfolio (``UNIQUE(portfolio_id)``,
  upserted): the full mandate lives in the ``mandate`` JSON column (source of
  truth) with a few indexed scalar mirrors for the state panel / future drift
  monitor.
* ``agent_runs.thread_id`` — nullable per-run LangGraph checkpointer thread id
  (Phase 8 defaults it to ``str(run_id)``); nullable because legacy rows predate
  per-run threading.

UUID PKs use the pg-specific ``UUID(as_uuid=True)`` (matching ``BaseModel``);
the non-PK ``portfolio_id`` / ``paper_order_id`` columns use the portable
``sa.Uuid`` (matching the ``Position`` / ``PortfolioMandate`` models). The
``mandate`` payload uses the dialect-aware ``_JSON`` variant so the SQLite replay
test can create the table.

Revision ID: b3c4d5e6f7a8
Revises: a2b3c4d5e6f7
Create Date: 2026-09-21
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision: str = "b3c4d5e6f7a8"
down_revision: str | Sequence[str] | None = "a2b3c4d5e6f7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Dialect-aware JSON: JSONB on PostgreSQL, plain JSON elsewhere (SQLite in tests).
_JSON = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    # 1. positions — current holdings snapshot (one row per portfolio+ticker).
    op.create_table(
        "positions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        # No FK: the `portfolios` table was dropped in the ingestion strip.
        sa.Column("portfolio_id", sa.Uuid(), nullable=False),
        sa.Column("ticker", sa.String(32), nullable=False),
        sa.Column("weight", sa.Float, nullable=False),
        sa.Column("shares", sa.Float, nullable=True),
        sa.Column("notional", sa.Float, nullable=True),
        sa.Column("asof", sa.Date, nullable=False),
        # The paper ticket whose approved fill produced this snapshot (nullable).
        sa.Column(
            "paper_order_id",
            sa.Uuid(),
            sa.ForeignKey("paper_orders.id"),
            nullable=True,
        ),
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
            "portfolio_id", "ticker", name="uq_position_portfolio_ticker"
        ),
    )
    op.create_index("ix_positions_portfolio_id", "positions", ["portfolio_id"])

    # 2. portfolio_mandates — JSON source of truth + indexed scalar mirror.
    op.create_table(
        "portfolio_mandates",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        # No FK: the `portfolios` table was dropped in the ingestion strip.
        sa.Column("portfolio_id", sa.Uuid(), nullable=False),
        sa.Column("base_currency", sa.String(3), nullable=False),
        sa.Column("capital", sa.Float, nullable=False),
        sa.Column("drift_l1_threshold", sa.Float, nullable=False),
        sa.Column("benchmark", sa.String(32), nullable=True),
        sa.Column("mandate", _JSON, nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="active"),
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
        sa.UniqueConstraint("portfolio_id", name="uq_portfolio_mandate_portfolio_id"),
    )
    op.create_index(
        "ix_portfolio_mandates_base_currency",
        "portfolio_mandates",
        ["base_currency"],
    )

    # 3. agent_runs.thread_id — per-run LangGraph checkpointer thread id.
    op.add_column(
        "agent_runs",
        sa.Column("thread_id", sa.String(64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("agent_runs", "thread_id")

    op.drop_index(
        "ix_portfolio_mandates_base_currency", table_name="portfolio_mandates"
    )
    op.drop_table("portfolio_mandates")

    op.drop_index("ix_positions_portfolio_id", table_name="positions")
    op.drop_table("positions")
