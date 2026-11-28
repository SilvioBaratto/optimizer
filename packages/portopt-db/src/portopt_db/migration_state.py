"""Tell a brand-new database from one whose Alembic revisions are pending.

Some revisions delete data, so a container must not apply them on its own to a
database that already holds data: the operator takes a backup first
(``portopt migrate``). This module answers the question the container
entrypoints and the ``portopt`` CLI ask before migrating: is the database empty,
at the head of this checkout's migration tree, or behind it, and by which
revisions.

Run as ``python -m portopt_db.migration_state [--alembic-dir DIR]`` with
``DATABASE_URL`` exported. The exit code is what the entrypoints branch on:
``EXIT_PROCEED`` (empty, at head, or ``PORTOPT_AUTO_MIGRATE`` is on),
``EXIT_PENDING`` (revisions are pending on a database with data; the refusal is
printed to stderr) and ``EXIT_ERROR`` (the state could not be read).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from alembic.script.revision import RevisionError
from alembic.util import CommandError
from sqlalchemy import create_engine, inspect, make_url
from sqlalchemy.engine import Connection
from sqlalchemy.exc import DBAPIError, SQLAlchemyError
from sqlalchemy.pool import NullPool

from portopt_db.config import pin_postgres_driver

EXIT_PROCEED = 0
EXIT_ERROR = 1
EXIT_PENDING = 3

AUTO_MIGRATE_ENV_VAR = "PORTOPT_AUTO_MIGRATE"
DATABASE_URL_ENV_VAR = "DATABASE_URL"
DEFAULT_ALEMBIC_DIR = "alembic"

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_VERSION_TABLE = "alembic_version"
_SYSTEM_SCHEMA = "information_schema"
_SYSTEM_SCHEMA_PREFIX = "pg_"
_BASE_LABEL = "base"
# A container waits for pg_isready before it asks, so a connect that takes longer
# than this is a network fault, not a slow start.
_CONNECT_TIMEOUT_SECONDS = 10


class MigrationState(str, Enum):
    """Where a database stands relative to the migration tree's head.

    Attributes:
        EMPTY: No user table: a brand-new database, safe to migrate on start.
        AT_HEAD: Every revision of the tree is applied.
        PENDING: User tables exist and at least one revision is not applied.
    """

    EMPTY = "empty"
    AT_HEAD = "at_head"
    PENDING = "pending"


@dataclass(frozen=True)
class PendingRevision:
    """One revision an upgrade to head would run.

    Attributes:
        revision: The Alembic revision id.
        message: The first paragraph of its docstring, on one line.
    """

    revision: str
    message: str


@dataclass(frozen=True)
class MigrationStatus:
    """The migration state of one database and the facts it was judged on.

    Attributes:
        state: The verdict.
        head: The single head revision of the migration tree.
        current: The revisions stamped in ``alembic_version``; empty when the
            table is missing or holds no row.
        pending: The revisions an upgrade to head would run, oldest first.
        tables: The schema-qualified user tables found (``alembic_version``
            excluded), sorted; a backup is checked against this list.
    """

    state: MigrationState
    head: str
    current: tuple[str, ...]
    pending: tuple[PendingRevision, ...]
    tables: tuple[str, ...]

    @property
    def count(self) -> int:
        """Return how many revisions an upgrade to head would run."""
        return len(self.pending)

    @property
    def current_label(self) -> str:
        """Return the stamped revisions for display, ``base`` when there are none."""
        return ", ".join(self.current) or _BASE_LABEL


class MigrationStateError(RuntimeError):
    """Raised when the state cannot be read or the tree cannot place the database."""


class _Revision(Protocol):
    @property
    def revision(self) -> str: ...

    @property
    def doc(self) -> str: ...


class RevisionSource(Protocol):
    """The part of Alembic's ``ScriptDirectory`` the classification needs."""

    def get_heads(self) -> list[str]:
        """Return the head revisions of the tree."""
        ...

    def iterate_revisions(
        self, upper: str, lower: tuple[str, ...]
    ) -> Iterable[_Revision]:
        """Yield the revisions above ``lower`` up to ``upper``, newest first."""
        ...


class CurrentHeadsSource(Protocol):
    """The part of Alembic's ``MigrationContext`` the classification needs."""

    def get_current_heads(self) -> tuple[str, ...]:
        """Return the revisions stamped in the database."""
        ...


def classify(
    *,
    user_tables: Sequence[str],
    context: CurrentHeadsSource,
    script: RevisionSource,
) -> MigrationStatus:
    """Decide the migration state from the database's tables and stamp.

    A database with no user table is ``EMPTY`` whatever it is stamped with: it
    holds nothing a migration could destroy. One with user tables but no stamp is
    ``PENDING`` from the base of the tree, so the start stops instead of replaying
    every revision over existing data.

    Args:
        user_tables: The schema-qualified user tables, ``alembic_version`` excluded.
        context: Reads the revisions stamped in the database.
        script: The migration tree of the code that is about to run.

    Returns:
        The verdict with the head, the stamp and the pending revisions.

    Raises:
        MigrationStateError: The tree does not have exactly one head, or the
            database is stamped with a revision the tree does not hold (a newer
            version migrated it).
    """
    head = _single_head(script)
    current = tuple(context.get_current_heads())
    tables = tuple(sorted(user_tables))
    if not tables:
        return MigrationStatus(MigrationState.EMPTY, head, current, (), tables)
    pending = _pending_revisions(script, head, current)
    state = MigrationState.PENDING if pending else MigrationState.AT_HEAD
    return MigrationStatus(state, head, current, pending, tables)


def _single_head(script: RevisionSource) -> str:
    heads = script.get_heads()
    if len(heads) != 1:
        raise MigrationStateError(
            f"the migration tree has {len(heads)} heads ({', '.join(heads)}); "
            "an upgrade needs exactly one head"
        )
    return heads[0]


def _pending_revisions(
    script: RevisionSource, head: str, current: tuple[str, ...]
) -> tuple[PendingRevision, ...]:
    try:
        newest_first = list(script.iterate_revisions(head, current))
    except (RevisionError, CommandError) as exc:
        raise MigrationStateError(
            f"the database is stamped with revision {', '.join(current)}, which this "
            "version's migration tree does not contain; it was migrated by a newer "
            f"version ({exc})"
        ) from exc
    return tuple(
        PendingRevision(step.revision, " ".join(step.doc.split()))
        for step in reversed(newest_first)
    )


def inspect_migration_state(database_url: str, alembic_dir: Path) -> MigrationStatus:
    """Read a database's migration state against a migration tree on disk.

    The connection is read-only on PostgreSQL: the check must never be the thing
    that changes the database it is guarding.

    Args:
        database_url: SQLAlchemy URL of the database; a bare ``postgresql://`` is
            bound to psycopg2 like every other engine of this package.
        alembic_dir: The Alembic script directory (the one holding ``versions/``).

    Returns:
        The database's :class:`MigrationStatus`.

    Raises:
        MigrationStateError: The tree cannot be loaded, the database cannot be
            read, or :func:`classify` cannot place it. The message never carries
            the URL's password.
    """
    script = _load_script_directory(alembic_dir)
    engine = create_engine(
        pin_postgres_driver(database_url),
        poolclass=NullPool,
        connect_args=_connect_args(database_url),
    )
    try:
        with engine.connect() as connection:
            connection.execution_options(postgresql_readonly=True)
            return classify(
                user_tables=_user_tables(connection),
                context=MigrationContext.configure(connection),
                script=script,
            )
    except SQLAlchemyError as exc:
        raise MigrationStateError(
            f"could not read the database at {_safe_url(database_url)}: "
            f"{_describe_error(exc)}"
        ) from exc
    finally:
        engine.dispose()


def _load_script_directory(alembic_dir: Path) -> ScriptDirectory:
    try:
        return ScriptDirectory(str(alembic_dir))
    except CommandError as exc:
        raise MigrationStateError(
            f"no migration tree at {alembic_dir} ({exc})"
        ) from exc


def _connect_args(database_url: str) -> dict[str, Any]:
    if make_url(database_url).get_backend_name() == "postgresql":
        return {"connect_timeout": _CONNECT_TIMEOUT_SECONDS}
    return {}


def _user_tables(connection: Connection) -> list[str]:
    inspector = inspect(connection)
    default_schema = inspector.default_schema_name
    tables: list[str] = []
    for schema in inspector.get_schema_names():
        if schema == _SYSTEM_SCHEMA or schema.startswith(_SYSTEM_SCHEMA_PREFIX):
            continue
        tables.extend(
            f"{schema}.{table}"
            for table in inspector.get_table_names(schema=schema)
            if (schema, table) != (default_schema, _VERSION_TABLE)
        )
    return tables


def _safe_url(database_url: str) -> str:
    return make_url(database_url).render_as_string(hide_password=True)


def _describe_error(exc: SQLAlchemyError) -> str:
    # The driver's own message, without SQLAlchemy's statement echo and doc link.
    error: BaseException = exc
    if isinstance(exc, DBAPIError) and exc.orig is not None:
        error = exc.orig
    text = str(error).strip()
    return text.splitlines()[0] if text else type(error).__name__


def auto_migrate_enabled(environ: Mapping[str, str]) -> bool:
    """Return whether ``PORTOPT_AUTO_MIGRATE`` asks for the old migrate-on-start.

    Args:
        environ: The environment to read, usually ``os.environ``.

    Returns:
        ``True`` only for an explicit yes (``1``, ``true``, ``yes``, ``on``, any
        case); unset, empty or anything else keeps the pending check on.
    """
    return environ.get(AUTO_MIGRATE_ENV_VAR, "").strip().lower() in _TRUTHY


def _revisions(count: int) -> str:
    return f"{count} revision{'' if count == 1 else 's'}"


def pending_refusal_message(status: MigrationStatus) -> str:
    """Return the message that stops a start on a database with pending revisions.

    Args:
        status: A ``PENDING`` status.

    Returns:
        Several lines naming the stamped and the head revision, how far behind the
        database is, the command that migrates with a backup, and the opt-out.
    """
    return (
        f"Database migrations are pending: the database is at revision "
        f"{status.current_label} and this version's head is {status.head} "
        f"({_revisions(status.count)} behind).\n"
        "Some migrations delete data, so they are not applied automatically to a "
        "database that already holds data.\n"
        "Run `portopt migrate` (it takes a backup first), then start again.\n"
        f"Unattended setups can set {AUTO_MIGRATE_ENV_VAR}=true to migrate on every "
        "start, without a backup, as before."
    )


def describe(status: MigrationStatus) -> str:
    """Return a one-line summary of a state that lets a start proceed.

    Args:
        status: An ``EMPTY`` or ``AT_HEAD`` status (a ``PENDING`` one is described
            by :func:`pending_refusal_message`).

    Returns:
        The summary, naming the head revision.
    """
    if status.state is MigrationState.EMPTY:
        return (
            f"The database is empty: migrating it to {status.head} creates its schema."
        )
    return f"The database is at the head revision {status.head}: nothing to migrate."


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m portopt_db.migration_state",
        description="Exit 0 to migrate on start, 3 when revisions are pending on a "
        "database that holds data, 1 when the state cannot be read.",
    )
    parser.add_argument(
        "--alembic-dir",
        default=DEFAULT_ALEMBIC_DIR,
        help="Alembic script directory (default: ./alembic).",
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None, environ: Mapping[str, str] | None = None
) -> int:
    """Check the database in ``DATABASE_URL`` and return the entrypoint's exit code.

    Args:
        argv: Command-line arguments; ``sys.argv[1:]`` when omitted.
        environ: The environment; ``os.environ`` when omitted.

    Returns:
        ``EXIT_PROCEED`` for an empty or current database, or when
        ``PORTOPT_AUTO_MIGRATE`` is on (then no check runs, as before the check
        existed); ``EXIT_PENDING`` with the refusal on stderr; ``EXIT_ERROR`` when
        ``DATABASE_URL`` is unset or the state cannot be read.
    """
    env = os.environ if environ is None else environ
    args = _parse_args(argv)
    if auto_migrate_enabled(env):
        print(f"{AUTO_MIGRATE_ENV_VAR} is on: migrating without the pending check.")
        return EXIT_PROCEED
    database_url = env.get(DATABASE_URL_ENV_VAR)
    if not database_url:
        print(f"{DATABASE_URL_ENV_VAR} is not set; cannot check.", file=sys.stderr)
        return EXIT_ERROR
    try:
        status = inspect_migration_state(database_url, Path(args.alembic_dir))
    except MigrationStateError as exc:
        print(f"Could not check for pending migrations: {exc}", file=sys.stderr)
        return EXIT_ERROR
    if status.state is MigrationState.PENDING:
        print(pending_refusal_message(status), file=sys.stderr)
        return EXIT_PENDING
    print(describe(status))
    return EXIT_PROCEED


if __name__ == "__main__":
    raise SystemExit(main())
