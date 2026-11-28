"""``portopt migrate``: back up the compose database, verify the backup, then migrate.

The containers and ``portopt start`` refuse to apply pending migrations to a
database that holds data, because some migrations delete data. This is the command
they send the operator to. It shows the revisions that will run and asks, stops the
scheduler and fund daemons so nothing writes during the dump or holds a lock the
DDL needs, takes and verifies a ``pg_dump`` (:mod:`app.setup.db_backup`), runs the
Alembic upgrade to head on the host, and reads the state back. A failed backup
means no migration; a failed backup or upgrade restarts the daemons it stopped, on
their previous code, which still fits the unchanged database.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from portopt_db.migration_state import (
    MigrationState,
    MigrationStateError,
    MigrationStatus,
    describe,
)
from sqlalchemy import make_url

from app.setup import config_file, db_backup, docker_bootstrap, host_database

PROCEED_PROMPT = "Proceed?"
APP_SERVICES = frozenset({"scheduler", "fund"})
NO_BACKUP_NEEDS_YES = (
    "--no-backup migrates without a safety net; add --yes to confirm you want that."
)
_UPGRADE_TAIL_LINES = 10


class MigrateError(RuntimeError):
    """Raised when ``portopt migrate`` refuses or fails; the message says what was left."""


@dataclass(frozen=True)
class MigrateOptions:
    """How ``portopt migrate`` was asked to run.

    Attributes:
        backup_dir: Where the dump goes; the ``backup_dir`` config key or
            portopt's own ``backups`` folder when ``None``.
        no_backup: Skip the dump (only together with ``assume_yes``).
        assume_yes: Do not ask for confirmation.
    """

    backup_dir: Path | None = None
    no_backup: bool = False
    assume_yes: bool = False


@dataclass(frozen=True)
class MigrateResult:
    """What ``portopt migrate`` did.

    Attributes:
        migrated: Whether an upgrade ran (``False`` for an empty or current database).
        head: The head revision of the checkout's migration tree.
        backup: The verified dump taken first, ``None`` when none was taken.
    """

    migrated: bool
    head: str
    backup: db_backup.Backup | None


def run_migrate(
    options: MigrateOptions,
    *,
    confirm: Callable[[str], bool],
    now: Callable[[], datetime] = datetime.now,
) -> MigrateResult:
    """Back up and migrate the compose database when revisions are pending.

    Args:
        options: The command-line choices.
        confirm: Asks the operator a yes/no question; not called with ``assume_yes``.
        now: The clock that names the dump.

    Returns:
        What was done; an empty or current database is left alone, without a dump.

    Raises:
        MigrateError: ``no_backup`` without ``assume_yes``, the state cannot be read,
            a fund round is running, the operator declined, or the backup, the
            upgrade or the final check failed.
        DockerError: Docker or the db service is unavailable.
    """
    if options.no_backup and not options.assume_yes:
        raise MigrateError(NO_BACKUP_NEEDS_YES)
    docker_bootstrap.ensure_docker()
    repo = docker_bootstrap.compose_dir()
    status = _read_state(repo)
    if status.state is not MigrationState.PENDING:
        print(describe(status))
        return MigrateResult(migrated=False, head=status.head, backup=None)
    directory = _backup_dir(options)
    _refuse_while_a_fund_round_runs(repo)
    print(_plan(status, directory))
    if not options.assume_yes and not confirm(PROCEED_PROMPT):
        raise MigrateError("Not confirmed; nothing was changed.")
    stopped = _stop_app_services(repo)
    try:
        backup = _back_up(repo, status, directory, now())
        _upgrade(repo, backup)
    except MigrateError:
        _restart(repo, stopped)
        raise
    _confirm_at_head(repo)
    _report(status.head, backup)
    return MigrateResult(migrated=True, head=status.head, backup=backup)


def _read_state(repo: Path) -> MigrationStatus:
    docker_bootstrap.bring_up_db(repo=repo)
    try:
        return host_database.read_state(host_database.db_package_dir(repo))
    except MigrationStateError as exc:
        raise MigrateError(f"Could not read the migration state: {exc}") from exc


def _backup_dir(options: MigrateOptions) -> Path | None:
    if options.no_backup:
        return None
    if options.backup_dir is not None:
        return db_backup.resolve_backup_dir(options.backup_dir, {})
    return db_backup.resolve_backup_dir(None, config_file.load_config())


def _refuse_while_a_fund_round_runs(repo: Path) -> None:
    drivers = docker_bootstrap.fund_drivers(repo=repo)
    if drivers:
        running = "\n  ".join(drivers)
        raise MigrateError(
            "the fund container is running these, and migrating stops it, which "
            f"would kill them mid-flight:\n  {running}\n"
            "Let them finish, then rerun `portopt migrate`."
        )


def _plan(status: MigrationStatus, directory: Path | None) -> str:
    steps = "\n".join(f"  {step.revision}  {step.message}" for step in status.pending)
    backup = (
        f"A verified pg_dump is written to {directory} before anything changes."
        if directory is not None
        else "WARNING: --no-backup: nothing is backed up, and revisions that delete "
        "data cannot be undone."
    )
    return (
        f"The database is at revision {status.current_label}; this version's head "
        f"is {status.head}. The upgrade runs {status.count} revision(s):\n"
        f"{steps}\n"
        "The scheduler and fund containers are stopped first, so nothing writes to "
        "the database while it is backed up and migrated.\n"
        f"{backup}"
    )


def _stop_app_services(repo: Path) -> list[str]:
    running = sorted(docker_bootstrap.running_services(repo=repo) & APP_SERVICES)
    if running:
        print(f"Stopping {', '.join(running)}...")
        docker_bootstrap.stop_services(running, repo=repo)
    return running


def _back_up(
    repo: Path, status: MigrationStatus, directory: Path | None, started: datetime
) -> db_backup.Backup | None:
    if directory is None:
        print("warning: migrating without a backup (--no-backup).")
        return None
    target = make_url(host_database.HOST_DB_URL)
    if not target.database or not target.username:
        raise MigrateError("the host database URL names no database or user")
    print(f"Backing up {target.database} to {directory} (this can take a while)...")
    try:
        return db_backup.take_verified_backup(
            repo=repo,
            directory=db_backup.prepare_backup_dir(directory),
            database=target.database,
            user=target.username,
            expected_tables=status.tables,
            now=started,
        )
    except (db_backup.BackupError, OSError) as exc:
        raise MigrateError(
            f"The backup failed, so the database was not migrated: {exc}"
        ) from exc


def _upgrade(repo: Path, backup: db_backup.Backup | None) -> None:
    print("Applying the migrations...")
    try:
        result = docker_bootstrap.run_alembic_upgrade(
            host_database.db_package_dir(repo)
        )
    except (OSError, docker_bootstrap.DockerError) as exc:
        raise MigrateError(_upgrade_failed(str(exc), backup)) from exc
    if result.returncode != 0:
        output = (result.stderr or result.stdout or "").strip().splitlines()
        raise MigrateError(
            _upgrade_failed("\n".join(output[-_UPGRADE_TAIL_LINES:]), backup)
        )


def _upgrade_failed(detail: str, backup: db_backup.Backup | None) -> str:
    kept = f"\nThe backup taken before it is {backup.path}." if backup else ""
    return (
        "The upgrade failed. Alembic applies it in one transaction, so the database "
        f"is still at its earlier revision.\n{detail}{kept}"
    )


def _restart(repo: Path, stopped: list[str]) -> None:
    if not stopped:
        return
    print(f"Restarting {', '.join(stopped)} on their previous code.")
    try:
        docker_bootstrap.restart_services(stopped, repo=repo)
    except docker_bootstrap.DockerError as exc:
        print(f"warning: could not restart {', '.join(stopped)}: {exc}")


def _confirm_at_head(repo: Path) -> None:
    try:
        status = host_database.read_state(host_database.db_package_dir(repo))
    except MigrationStateError as exc:
        raise MigrateError(
            f"The upgrade finished, but the state could not be read back: {exc}"
        ) from exc
    if status.state is not MigrationState.AT_HEAD:
        raise MigrateError(
            "The upgrade finished, but the database is not at head (at "
            f"{status.current_label}, head {status.head})."
        )


def _report(head: str, backup: db_backup.Backup | None) -> None:
    print(f"Migrated to {head}.")
    if backup is not None:
        print(
            f"Backup: {backup.path} ({db_backup.format_size(backup.size_bytes)}). "
            "pg_restore it into a scratch database to read the data the migrations "
            "removed."
        )
    print("Run `portopt start` to rebuild and start the stack on the new code.")
