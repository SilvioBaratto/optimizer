"""Add snapshot_date to broker_account_snapshots for daily deduplication.

Fixes unbounded table growth (issue #327): upsert_account_snapshot now keys
on (portfolio_id, snapshot_date) so repeated syncs within a day update the
existing row rather than inserting a new one.

Steps:
  1. Add nullable snapshot_date column.
  2. Backfill from the existing synced_at timestamp (cast to date).
  3. Delete duplicate rows, keeping the one with the latest synced_at per
     (portfolio_id, snapshot_date).
  4. Set snapshot_date NOT NULL.
  5. Add the unique constraint.

Revision ID: w3x4y5z6a7b8
Revises: v2w3x4y5z6a7
Create Date: 2026-03-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "w3x4y5z6a7b8"
down_revision: str | Sequence[str] | None = "v2w3x4y5z6a7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Add nullable snapshot_date column
    op.add_column(
        "broker_account_snapshots",
        sa.Column("snapshot_date", sa.Date, nullable=True),
    )

    # 2. Backfill snapshot_date from synced_at for all existing rows
    op.execute("""
        UPDATE broker_account_snapshots
        SET snapshot_date = DATE(synced_at)
    """)

    # 3. Remove duplicate rows created by repeated daily syncs, keeping the
    #    row with the latest synced_at per (portfolio_id, snapshot_date)
    op.execute("""
        DELETE FROM broker_account_snapshots
        WHERE id IN (
            SELECT id FROM (
                SELECT id,
                       ROW_NUMBER() OVER (
                           PARTITION BY portfolio_id, snapshot_date
                           ORDER BY synced_at DESC
                       ) AS rn
                FROM broker_account_snapshots
            ) ranked
            WHERE rn > 1
        )
    """)

    # 4. Enforce NOT NULL now that all rows are populated
    op.alter_column(
        "broker_account_snapshots",
        "snapshot_date",
        nullable=False,
    )

    # 5. Add the unique constraint
    op.create_unique_constraint(
        "uq_broker_account_snapshot_portfolio_date",
        "broker_account_snapshots",
        ["portfolio_id", "snapshot_date"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_broker_account_snapshot_portfolio_date",
        "broker_account_snapshots",
        type_="unique",
    )
    op.drop_column("broker_account_snapshots", "snapshot_date")
