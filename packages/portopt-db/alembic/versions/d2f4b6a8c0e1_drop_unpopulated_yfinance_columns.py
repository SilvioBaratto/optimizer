"""Drop yfinance columns Yahoo never populates.

A full-DB audit found four columns NULL for every row across the whole universe
and every ``as_of`` snapshot, with nothing reading them. Yahoo returns the row
label but an ``<NA>`` value for the three fund metrics, and omits the
economic-calendar "Expected" column entirely, so the repo mappings were already
correct — the emptiness is a source gap, not a write bug. The columns are dropped
rather than kept as permanently-empty schema; downgrade re-adds them nullable if
Yahoo ever restores the data.

Revision ID: d2f4b6a8c0e1
Revises: a7c9e1b3d5f2
Create Date: 2026-10-26
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "d2f4b6a8c0e1"
down_revision: str | Sequence[str] | None = "a7c9e1b3d5f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Drop the four never-populated columns."""
    # Literal table/column names: the chain-replay guard reads them from the AST.
    op.drop_column("etf_equity_holdings", "median_market_cap")
    op.drop_column("etf_equity_holdings", "three_year_earnings_growth")
    op.drop_column("etf_bond_holdings", "credit_quality")
    op.drop_column("economic_event_calendar", "forecast")


def downgrade() -> None:
    """Re-add the columns as nullable, matching their original types."""
    op.add_column(
        "economic_event_calendar",
        sa.Column("forecast", sa.String(length=50), nullable=True),
    )
    op.add_column(
        "etf_bond_holdings",
        sa.Column("credit_quality", sa.Numeric(precision=20, scale=6), nullable=True),
    )
    op.add_column(
        "etf_equity_holdings",
        sa.Column(
            "three_year_earnings_growth",
            sa.Numeric(precision=20, scale=6),
            nullable=True,
        ),
    )
    op.add_column(
        "etf_equity_holdings",
        sa.Column(
            "median_market_cap", sa.Numeric(precision=24, scale=2), nullable=True
        ),
    )
