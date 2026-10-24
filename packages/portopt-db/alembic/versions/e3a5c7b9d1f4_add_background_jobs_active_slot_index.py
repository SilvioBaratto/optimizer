"""Add the active-slot partial unique index to ``background_jobs``.

The job-slot claim is ``INSERT ... SELECT ... WHERE NOT EXISTS (active row)``.
Under PostgreSQL's READ COMMITTED isolation two concurrent claimers (a manual CLI
run racing the cron, or a reclaim re-dispatch) both pass the check and both
insert, running one job type twice at once. A partial unique index over the
active statuses lets the database refuse the second row.

A past race may already have left duplicate active rows, which would make the
index creation fail, so every active row of a job type except its newest is
first failed with an explanatory error.

Revision ID: e3a5c7b9d1f4
Revises: f5a7c9e1b3d2
Create Date: 2026-10-23
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "e3a5c7b9d1f4"
down_revision: str | Sequence[str] | None = "f5a7c9e1b3d2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# The index name stays a literal in create_index/drop_index: the revision-chain
# replay test reads names from the AST and cannot resolve module constants.
_ACTIVE = sa.text("status IN ('pending', 'running')")


def upgrade() -> None:
    """Fail superseded duplicate active rows, then create the unique index."""
    op.execute(
        sa.text(
            "UPDATE background_jobs "
            "SET status = 'failed', "
            "error = 'superseded - duplicate active row demoted by migration', "
            "finished_at = CURRENT_TIMESTAMP "
            "WHERE status IN ('pending', 'running') "
            "AND EXISTS ("
            "SELECT 1 FROM background_jobs newer "
            "WHERE newer.job_type = background_jobs.job_type "
            "AND newer.status IN ('pending', 'running') "
            "AND (newer.started_at > background_jobs.started_at "
            "OR (newer.started_at = background_jobs.started_at "
            "AND newer.id > background_jobs.id)))"
        )
    )
    op.create_index(
        "uq_background_jobs_active_type",
        "background_jobs",
        ["job_type"],
        unique=True,
        postgresql_where=_ACTIVE,
        sqlite_where=_ACTIVE,
    )


def downgrade() -> None:
    """Drop the active-slot unique index (the demoted rows stay failed)."""
    op.drop_index("uq_background_jobs_active_type", table_name="background_jobs")
