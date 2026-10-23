"""Cross-process serialisation of Alembic runs against one PostgreSQL database."""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection

# ASCII "portoptd". Any constant bigint works; it only has to be the same in
# every process that migrates this database and unlikely to collide with
# another application's advisory locks on a shared server.
MIGRATION_LOCK_KEY = 0x706F72746F707464


def acquire_migration_lock(connection: Connection) -> None:
    """Block until no other process is migrating this database.

    Must be called inside the migration transaction and before Alembic reads
    ``alembic_version``. The lock is transaction-scoped, so it is released by
    the commit or rollback that ends the migration and can never outlive a
    crashed runner. A second runner waits here, then reads the already-advanced
    revision and finds nothing left to do, instead of replaying the same DDL
    and failing on a duplicate relation or type.

    Args:
        connection: Connection that Alembic is migrating through. Only the
            PostgreSQL dialect has advisory locks; on any other dialect (the
            SQLite test databases) this is a no-op.
    """
    if connection.dialect.name != "postgresql":
        return
    connection.execute(
        text("SELECT pg_advisory_xact_lock(:key)"), {"key": MIGRATION_LOCK_KEY}
    )
