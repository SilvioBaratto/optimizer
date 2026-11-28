"""The compose database as the host-side ``portopt`` commands reach it.

``portopt setup`` and ``portopt migrate`` run Alembic on the host against the
database's published port, and ``portopt start`` reads its migration state there
before it starts or recreates a container. The URL, the migration tree and the
start guard live here so the three commands agree on which database and which
tree, and so ``docker_bootstrap`` and ``lifecycle`` share them without importing
each other.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from portopt_db.migration_state import (
    MigrationState,
    MigrationStatus,
    auto_migrate_enabled,
    inspect_migration_state,
    pending_refusal_message,
)

# The well-known compose credentials (docker-compose.yml) on the loopback-only
# published port; never printed, since every message renders it password-hidden.
HOST_DB_URL = "postgresql://postgres:postgres@localhost:54320/optimizer_db"
DB_PACKAGE_DIR = Path("packages", "portopt-db")
ALEMBIC_DIRNAME = "alembic"


def db_package_dir(base: Path) -> Path:
    """Return the directory holding ``alembic.ini`` and the tree under ``base``.

    Args:
        base: A checkout root.
    """
    return base / DB_PACKAGE_DIR


def read_state(db_dir: Path) -> MigrationStatus:
    """Read the compose database's migration state against a checkout's tree.

    The checkout's tree is the one the next image build ships, so this answers what
    the containers would find once rebuilt.

    Args:
        db_dir: The ``packages/portopt-db`` directory of the checkout.

    Returns:
        The database's migration status.

    Raises:
        MigrationStateError: The database cannot be read or placed in the tree.
    """
    return inspect_migration_state(HOST_DB_URL, db_dir / ALEMBIC_DIRNAME)


def pending_refusal(db_dir: Path, *, environ: Mapping[str, str]) -> str | None:
    """Return why a migrate-on-start must not run, or ``None`` when it may.

    Args:
        db_dir: The ``packages/portopt-db`` directory of the checkout.
        environ: The environment holding ``PORTOPT_AUTO_MIGRATE``; when it is on,
            nothing is read and ``None`` is returned, as before the guard existed.

    Returns:
        The refusal message when revisions are pending on a database that holds
        data, ``None`` for an empty or current database.

    Raises:
        MigrationStateError: The state could not be read; each caller decides
            whether that stops it.
    """
    if auto_migrate_enabled(environ):
        return None
    status = read_state(db_dir)
    if status.state is MigrationState.PENDING:
        return pending_refusal_message(status)
    return None
