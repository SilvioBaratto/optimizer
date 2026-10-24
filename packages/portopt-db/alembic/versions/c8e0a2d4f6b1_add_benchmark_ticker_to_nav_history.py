"""Record which benchmark each ``nav_history`` level was taken from.

``nav_history.benchmark`` stores only a level, so after a mandate benchmark change
the older rows (re-marked only inside the daily step's lookback window) kept the
previous benchmark's level and the stitched series produced a phantom benchmark
return. ``benchmark_ticker`` tags each row with its benchmark so readers and the
NAV-mark step can detect the mismatch. Nullable: rows written before this revision
stay NULL (identity unknown) and are re-tagged by the next full re-mark.

Revision ID: c8e0a2d4f6b1
Revises: d8f0b2c4e6a1
Create Date: 2026-10-24
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "c8e0a2d4f6b1"
down_revision: str | Sequence[str] | None = "d8f0b2c4e6a1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add the nullable ``benchmark_ticker`` column."""
    op.add_column(
        "nav_history",
        sa.Column("benchmark_ticker", sa.String(64), nullable=True),
    )


def downgrade() -> None:
    """Drop the ``benchmark_ticker`` column."""
    op.drop_column("nav_history", "benchmark_ticker")
