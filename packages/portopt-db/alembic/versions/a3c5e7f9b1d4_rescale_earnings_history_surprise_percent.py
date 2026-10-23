"""Rescale ``earnings_history.surprise_percent`` from a fraction to percent points.

Yahoo's ``earnings_history`` reports ``surprisePercent`` as a fraction (0.04 for
a 4% beat) and the writer stored it raw, while ``earnings_dates`` and
``earnings_calendar`` hold percent points (4.0). The same quarter therefore read
100x apart across the tables. The writer now multiplies by 100; this revision
brings the rows already stored in line. It runs in the container entrypoint
before the worker starts, so no row is written in the new unit before it runs.

Revision ID: a3c5e7f9b1d4
Revises: e9a1c3d5f7b2
Create Date: 2026-10-23
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "a3c5e7f9b1d4"
down_revision: str | Sequence[str] | None = "e9a1c3d5f7b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Multiply every stored surprise by 100 (fraction -> percent points)."""
    op.execute(
        sa.text(
            "UPDATE earnings_history "
            "SET surprise_percent = surprise_percent * 100 "
            "WHERE surprise_percent IS NOT NULL"
        )
    )


def downgrade() -> None:
    """Divide every stored surprise by 100 (percent points -> fraction)."""
    op.execute(
        sa.text(
            "UPDATE earnings_history "
            "SET surprise_percent = surprise_percent / 100.0 "
            "WHERE surprise_percent IS NOT NULL"
        )
    )
