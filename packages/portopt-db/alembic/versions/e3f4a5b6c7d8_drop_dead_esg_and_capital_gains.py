"""Drop the dead esg_scores and capital_gains tables.

Both tables can never be populated by the ingestion pipeline as it stands:

* ``esg_scores`` is fed by ``yf.Ticker.sustainability`` (quoteSummary module
  ``esgScores``), which Yahoo now returns **404 Not Found** for every symbol —
  ESG moved behind the Sustainalytics paywall. Verified against the live API.
* ``capital_gains`` (fund distributions) is fed by ``yf.Ticker.capital_gains``,
  which reads the price-history ``capitalGains`` events. Yahoo stopped emitting
  those events (the chart ``events`` object carries only dividends/splits), so
  the series is empty for every ticker.

Both tables held 0 rows, so the drop is non-destructive. The
``price_history.capital_gains`` *column* is a separate per-bar field and is
left untouched. ``downgrade`` recreates both tables (empty) for reversibility.

Revision ID: e3f4a5b6c7d8
Revises: d3e4f5a6b7c8
Create Date: 2026-09-01
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "e3f4a5b6c7d8"
down_revision: str | Sequence[str] | None = "d3e4f5a6b7c8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_index("ix_esg_scores_instrument_id", table_name="esg_scores")
    op.drop_table("esg_scores")
    op.drop_index("ix_capital_gains_instrument_id", table_name="capital_gains")
    op.drop_table("capital_gains")


def downgrade() -> None:
    op.create_table(
        "esg_scores",
        sa.Column(
            "instrument_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("instruments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("total_esg", sa.Numeric(20, 6), nullable=True),
        sa.Column("environment_score", sa.Numeric(20, 6), nullable=True),
        sa.Column("social_score", sa.Numeric(20, 6), nullable=True),
        sa.Column("governance_score", sa.Numeric(20, 6), nullable=True),
        sa.Column("highest_controversy", sa.Numeric(20, 6), nullable=True),
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
        sa.UniqueConstraint("instrument_id", name="uq_esg_score_instrument"),
    )
    op.create_index("ix_esg_scores_instrument_id", "esg_scores", ["instrument_id"])

    op.create_table(
        "capital_gains",
        sa.Column(
            "instrument_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("instruments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("amount", sa.Numeric(20, 6), nullable=False),
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
        sa.UniqueConstraint(
            "instrument_id", "date", name="uq_capital_gain_instrument_date"
        ),
    )
    op.create_index(
        "ix_capital_gains_instrument_id", "capital_gains", ["instrument_id"]
    )
