"""Add the Trading 212 tradability flag (``instruments.is_t212_tradeable``).

Trading 212 is a tradability overlay on the yfinance-built universe, not its
source. The ``t212_annotate`` step sets this flag from the broker's catalogue so a
run that must be executable at the broker can be restricted to it, while research
keeps the full multi-region universe.

Nullable with no default on purpose: NULL means "not evaluated" (Trading 212
unconfigured, or no ISIN to match), which is distinct from False ("matched
against the catalogue and absent"). Existing rows therefore come out NULL and a
deployment without Trading 212 keeps its whole universe.

Revision ID: f4b6d8a0c2e3
Revises: e3a5c7d9b1f4
Create Date: 2026-11-03
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "f4b6d8a0c2e3"
down_revision: str | Sequence[str] | None = "e3a5c7d9b1f4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the nullable ``is_t212_tradeable`` column."""
    op.add_column(
        "instruments",
        sa.Column("is_t212_tradeable", sa.Boolean(), nullable=True),
    )


def downgrade() -> None:
    """Drop the ``is_t212_tradeable`` column."""
    op.drop_column("instruments", "is_t212_tradeable")
