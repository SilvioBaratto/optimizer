"""Add external_id to activity_events for broker order deduplication.

Revision ID: v2w3x4y5z6a7
Revises: u1v2w3x4y5z6
Create Date: 2026-03-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "v2w3x4y5z6a7"
down_revision: str | Sequence[str] | None = "u1v2w3x4y5z6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Add nullable external_id column
    op.add_column(
        "activity_events",
        sa.Column("external_id", sa.String(200), nullable=True),
    )

    # 2. Backfill external_id from the EAV metadata table for trade events
    #    that have an order_id key. value_text is JSON-encoded (e.g. '"abc123"'),
    #    so strip the surrounding quotes.
    op.execute("""
        UPDATE activity_events ae
        SET external_id = trim('"' FROM aem.value_text)
        FROM activity_event_metadata aem
        WHERE aem.event_id = ae.id
          AND aem.key = 'order_id'
          AND ae.event_type = 'trade'
    """)

    # 3. Remove duplicate rows created by repeated syncs, keeping the oldest
    #    (lowest created_at) row for each external_id value.
    op.execute("""
        DELETE FROM activity_events
        WHERE id IN (
            SELECT id FROM (
                SELECT id,
                       ROW_NUMBER() OVER (
                           PARTITION BY external_id
                           ORDER BY created_at ASC
                       ) AS rn
                FROM activity_events
                WHERE external_id IS NOT NULL
            ) ranked
            WHERE rn > 1
        )
    """)

    # 4. Create unique constraint on external_id
    op.create_unique_constraint(
        "uq_activity_event_external_id",
        "activity_events",
        ["external_id"],
    )

    # 5. Create supporting index for lookup by external_id
    op.create_index(
        "ix_activity_events_external_id",
        "activity_events",
        ["external_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_activity_events_external_id", table_name="activity_events")
    op.drop_constraint(
        "uq_activity_event_external_id",
        "activity_events",
        type_="unique",
    )
    op.drop_column("activity_events", "external_id")
