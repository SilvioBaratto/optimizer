"""Add attempt counter to background_jobs.

R3 / ARCHITECTURE.md §5.3: the orphan reaper's RECLAIM strategy re-dispatches a
dead-worker job. ``attempt`` records how many reclaim retries a job carries so
the scheduler can cap re-dispatch (``SCHEDULER_ORPHAN_MAX_RECLAIM_ATTEMPTS``)
and not loop forever on a job whose worker keeps dying. Non-null, defaults 0;
pre-migration rows and normal (fail-strategy) jobs stay at 0.

Revision ID: e1f2a3b4c5d6
Revises: d1e2f3a4b5c6
Create Date: 2026-08-24
"""

import sqlalchemy as sa

from alembic import op

revision: str = "e1f2a3b4c5d6"
down_revision: str | None = "d1e2f3a4b5c6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "background_jobs",
        sa.Column(
            "attempt",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade() -> None:
    op.drop_column("background_jobs", "attempt")
