"""Add the pinned-instruments table (``pinned_instruments``).

One row per ``(ticker, source)`` naming an owner that wants a ticker kept in the
investable universe. The weekly universe build treats "absent from the Screener
snapshot" as "delisted", which would drop every hand-added instrument (the ETFs
of a real portfolio are not in the Screener). Keyed by the Yahoo ticker text, not
an ``instruments`` FK, so a pin can precede its instrument row.

Revision ID: e3a5c7d9b1f4
Revises: d2f4b6a8c0e1
Create Date: 2026-11-03
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "e3a5c7d9b1f4"
down_revision: str | Sequence[str] | None = "d2f4b6a8c0e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create the ``pinned_instruments`` table, its uniqueness guard and index."""
    op.create_table(
        "pinned_instruments",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("ticker", sa.String(100), nullable=False),
        sa.Column("source", sa.String(64), nullable=False),
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
            "ticker",
            "source",
            name="uq_pinned_instruments_ticker_source",
        ),
    )
    op.create_index("ix_pinned_instruments_source", "pinned_instruments", ["source"])


def downgrade() -> None:
    """Drop the ``pinned_instruments`` table and its index."""
    op.drop_index("ix_pinned_instruments_source", table_name="pinned_instruments")
    op.drop_table("pinned_instruments")
