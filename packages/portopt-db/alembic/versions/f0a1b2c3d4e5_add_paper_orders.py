"""Add paper-order tickets table (paper_orders).

Fase 3 of the ``fund`` bridge: idempotent paper execution. ``place_orders``
turns target weights into a simulated ticket filled at the next close strictly
after the decision bar (SPEC D5/D30). A ``UNIQUE(portfolio_id, asof,
weights_hash)`` makes the side effect idempotent under the HITL re-run gotcha
(D3). Model lives in ``portopt_db``; repo behavior in ``fund`` — the
``agent_runs`` split.

Revision ID: f0a1b2c3d4e5
Revises: e9f0a1b2c3d4
Create Date: 2026-09-16
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

revision: str = "f0a1b2c3d4e5"
down_revision: str | Sequence[str] | None = "e9f0a1b2c3d4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Dialect-aware JSON: JSONB on PostgreSQL, plain JSON elsewhere (SQLite in tests).
_JSON = sa.JSON().with_variant(JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "paper_orders",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        # No FK: the `portfolios` table was dropped in the ingestion strip.
        sa.Column("portfolio_id", UUID(as_uuid=True), nullable=False),
        sa.Column("asof", sa.Date, nullable=False),
        sa.Column("weights_hash", sa.String(64), nullable=False),
        sa.Column("weights", _JSON, nullable=False),
        sa.Column("fill_date", sa.Date, nullable=False),
        sa.Column("lines", _JSON, nullable=False),
        sa.Column("notional", sa.Float, nullable=False),
        sa.Column("total_commission", sa.Float, nullable=False),
        sa.Column("total_slippage_cost", sa.Float, nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="filled"),
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
            "portfolio_id", "asof", "weights_hash", name="uq_paper_order_key"
        ),
    )
    op.create_index("ix_paper_orders_portfolio_id", "paper_orders", ["portfolio_id"])
    op.create_index("ix_paper_orders_asof", "paper_orders", ["asof"])


def downgrade() -> None:
    op.drop_index("ix_paper_orders_asof", table_name="paper_orders")
    op.drop_index("ix_paper_orders_portfolio_id", table_name="paper_orders")
    op.drop_table("paper_orders")
