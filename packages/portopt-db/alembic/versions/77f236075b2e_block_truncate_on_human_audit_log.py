"""Block ``TRUNCATE`` on the append-only ``human_audit_log``.

``f1a2b3c4d5e6`` rejects UPDATE and DELETE with a row-level trigger. PostgreSQL
never fires row-level triggers for ``TRUNCATE``, so a single
``TRUNCATE human_audit_log`` still emptied the immutable approval and override
trail without any error. A statement-level ``BEFORE TRUNCATE`` trigger on the
same exception function closes that gap. The guard is PostgreSQL-only: SQLite
has no ``TRUNCATE`` and its DELETE trigger already covers ``DELETE FROM``.

Revision ID: 77f236075b2e
Revises: c8d0bb63d826
Create Date: 2026-10-24
"""

from collections.abc import Sequence

from alembic import op

revision: str = "77f236075b2e"
down_revision: str | Sequence[str] | None = "c8d0bb63d826"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_DROP_TRIGGER = (
    "DROP TRIGGER IF EXISTS human_audit_log_block_truncate ON human_audit_log"
)

# CREATE TRIGGER has no IF NOT EXISTS, so upgrade drops first to stay replayable.
_CREATE_TRIGGER = (
    "CREATE TRIGGER human_audit_log_block_truncate "
    "BEFORE TRUNCATE ON human_audit_log "
    "FOR EACH STATEMENT EXECUTE FUNCTION human_audit_log_append_only()"
)


def upgrade() -> None:
    """Install the statement-level ``BEFORE TRUNCATE`` guard on PostgreSQL."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute(_DROP_TRIGGER)
    op.execute(_CREATE_TRIGGER)


def downgrade() -> None:
    """Remove the ``TRUNCATE`` guard; the row-level guard stays with its revision."""
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute(_DROP_TRIGGER)
