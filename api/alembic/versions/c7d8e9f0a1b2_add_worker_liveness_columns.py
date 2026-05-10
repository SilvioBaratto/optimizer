"""Add worker liveness columns to background_jobs.

Issue #585: substrate for the liveness-aware orphan reaper. Adds
``worker_pid``, ``worker_host``, ``last_heartbeat_at`` plus a composite
``(status, last_heartbeat_at)`` index. All columns are nullable; pre-migration
rows leave them NULL and the reaper treats NULL host/heartbeat as orphan.

Revision ID: c7d8e9f0a1b2
Revises: b1c2d3e4f5g6
Create Date: 2026-05-10
"""

import sqlalchemy as sa

from alembic import op

revision: str = "c7d8e9f0a1b2"
down_revision: str | None = "b1c2d3e4f5g6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "background_jobs",
        sa.Column("worker_pid", sa.Integer(), nullable=True),
    )
    op.add_column(
        "background_jobs",
        sa.Column("worker_host", sa.Text(), nullable=True),
    )
    op.add_column(
        "background_jobs",
        sa.Column(
            "last_heartbeat_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_background_jobs_status_heartbeat",
        "background_jobs",
        ["status", "last_heartbeat_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_background_jobs_status_heartbeat",
        table_name="background_jobs",
    )
    op.drop_column("background_jobs", "last_heartbeat_at")
    op.drop_column("background_jobs", "worker_host")
    op.drop_column("background_jobs", "worker_pid")
