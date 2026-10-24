"""Record the currency of the forward-estimate tables.

Yahoo quotes each estimate period in its own currency, which can differ from the
listing currency (a USD-listed ADR whose estimates are in TWD or DKK). The tables
kept only the numbers, so a forward P/E built from them and a price history could
be off by the exchange rate with no way to detect it. ``currency`` is nullable:
rows written before this revision stay NULL (unknown) and are labelled by the
next fetch.

Revision ID: a7c9e1b3d5f2
Revises: 70393a606218
Create Date: 2026-10-24
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "a7c9e1b3d5f2"
down_revision: str | Sequence[str] | None = "70393a606218"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the nullable ``currency`` column to the four estimate tables."""
    # Literal table names: the chain-replay test reads names from the AST.
    op.add_column(
        "earnings_estimate", sa.Column("currency", sa.String(10), nullable=True)
    )
    op.add_column(
        "revenue_estimate", sa.Column("currency", sa.String(10), nullable=True)
    )
    op.add_column("eps_trend", sa.Column("currency", sa.String(10), nullable=True))
    op.add_column("eps_revisions", sa.Column("currency", sa.String(10), nullable=True))


def downgrade() -> None:
    """Drop the ``currency`` column from the four estimate tables."""
    op.drop_column("eps_revisions", "currency")
    op.drop_column("eps_trend", "currency")
    op.drop_column("revenue_estimate", "currency")
    op.drop_column("earnings_estimate", "currency")
