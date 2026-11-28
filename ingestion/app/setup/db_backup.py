"""The verified database dump ``portopt migrate`` takes before it migrates.

Some migrations delete data and cannot be undone by a downgrade, so the dump is the
only way back. It is taken with the db container's own ``pg_dump`` (the app images
carry no PostgreSQL client tools), streamed to a file on the host, and accepted only
after ``pg_restore`` has listed every table of the database in it and read it back in
full. A dump that fails any step is deleted, so nothing that looks like a backup is
left behind; a finished backup is never overwritten or deleted.

The dumps go to a ``backups`` folder in portopt's own directory (next to
``config.toml``), never into the checkout: a checkout is often synced between
machines, and a dump of the whole database is large and private.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.setup import config_file, docker_bootstrap

BACKUP_DIRNAME = "backups"
BACKUP_DIR_CONFIG_KEY = "backup_dir"
BACKUP_DIR_MODE = 0o700
DUMP_FILE_MODE = 0o600
DUMP_STAMP_FORMAT = "%Y%m%d-%H%M%S"
DUMP_NAME_TEMPLATE = "{database}-{stamp}-pre-migrate.dump"
PARTIAL_SUFFIX = ".partial"
DB_SERVICE = "db"
VERSION_TABLE = "public.alembic_version"

# A dump of tens of GB streams for minutes to hours; past this the db container is
# taken to be hung rather than slow.
_DUMP_TIMEOUT_SECONDS = 6 * 60 * 60
_STDERR_TAIL_LINES = 5
_SIZE_STEP = 1024
_SIZE_UNITS = ("B", "KiB", "MiB", "GiB", "TiB")
# `pg_restore --list` prints one `<id>; <oid> <oid> TABLE DATA <schema> <name> <owner>`
# line per table whose rows the archive holds, empty tables included.
_TABLE_DATA_ENTRY = re.compile(
    r"^\d+; \d+ \d+ TABLE DATA (?P<schema>\S+) (?P<name>\S+) ", re.MULTILINE
)


class BackupError(RuntimeError):
    """Raised when the dump cannot be written or does not verify; nothing is kept."""


@dataclass(frozen=True)
class Backup:
    """A dump that passed verification.

    Attributes:
        path: Where the dump is.
        size_bytes: Its size on disk.
    """

    path: Path
    size_bytes: int


def default_backup_dir() -> Path:
    """Return the ``backups`` folder in portopt's own directory, outside any checkout."""
    return config_file.DEFAULT_CONFIG_PATH.parent / BACKUP_DIRNAME


def resolve_backup_dir(override: Path | None, config: Mapping[str, Any]) -> Path:
    """Return where the dump goes.

    Args:
        override: The ``--backup-dir`` the operator passed, if any.
        config: The persisted ``config.toml``; its ``backup_dir`` key is used when no
            flag is given.

    Returns:
        The flag, else the config key, else :func:`default_backup_dir`, with ``~``
        expanded.
    """
    if override is not None:
        return override.expanduser()
    configured = config.get(BACKUP_DIR_CONFIG_KEY)
    if isinstance(configured, str) and configured.strip():
        return Path(configured.strip()).expanduser()
    return default_backup_dir()


def prepare_backup_dir(directory: Path) -> Path:
    """Create the backup directory, private to the user when this call creates it.

    A directory that already exists keeps the permissions its owner gave it; each
    dump is written ``0600`` regardless.

    Args:
        directory: The resolved backup directory.

    Returns:
        ``directory``.
    """
    if not directory.exists():
        directory.mkdir(mode=BACKUP_DIR_MODE, parents=True)
        # mkdir's mode is filtered through the umask; the dump holds the database.
        directory.chmod(BACKUP_DIR_MODE)
    return directory


def dump_path(directory: Path, database: str, now: datetime) -> Path:
    """Return the dump file name for ``database`` taken at ``now``.

    Args:
        directory: The backup directory.
        database: The database name.
        now: When the dump starts.
    """
    stamp = now.strftime(DUMP_STAMP_FORMAT)
    return directory / DUMP_NAME_TEMPLATE.format(database=database, stamp=stamp)


def parse_listed_tables(listing: str) -> set[str]:
    """Return the ``schema.table`` names whose rows a ``pg_restore --list`` holds.

    Args:
        listing: The text ``pg_restore --list`` printed.
    """
    return {
        f"{match['schema']}.{match['name']}"
        for match in _TABLE_DATA_ENTRY.finditer(listing)
    }


def take_verified_backup(
    *,
    repo: Path,
    directory: Path,
    database: str,
    user: str,
    expected_tables: Sequence[str],
    now: datetime,
) -> Backup:
    """Dump the database, verify the dump, and keep it under its final name.

    The dump is written to a ``.partial`` file and renamed only after it verified,
    so an interrupted run never leaves a file named like a backup.

    Args:
        repo: The checkout whose compose project runs the db service.
        directory: An existing backup directory.
        database: The database to dump.
        user: The database role to dump as, over the container's local socket (no
            password is sent anywhere).
        expected_tables: The ``schema.table`` user tables the database holds; each,
            and the Alembic stamp, must have its data in the dump.
        now: When the dump starts, for its name.

    Returns:
        The verified backup.

    Raises:
        BackupError: A dump of that name already exists, or the dump or one of
            the verification steps failed; nothing new is left on disk.
    """
    target = dump_path(directory, database, now)
    partial = target.with_name(target.name + PARTIAL_SUFFIX)
    for existing in (target, partial):
        if existing.exists():
            raise BackupError(f"{existing} already exists; a dump is never overwritten")
    try:
        _dump(repo, partial, database=database, user=user)
        _verify(repo, partial, expected=(*expected_tables, VERSION_TABLE))
    except (BackupError, docker_bootstrap.DockerError, OSError) as exc:
        partial.unlink(missing_ok=True)
        if isinstance(exc, BackupError):
            raise
        raise BackupError(f"the dump could not be completed: {exc}") from exc
    partial.rename(target)
    return Backup(target, target.stat().st_size)


def _dump(repo: Path, partial: Path, *, database: str, user: str) -> None:
    descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, DUMP_FILE_MODE)
    with os.fdopen(descriptor, "wb") as archive:
        result = docker_bootstrap.exec_in_service(
            DB_SERVICE,
            ["pg_dump", "--format=custom", "--username", user, "--dbname", database],
            repo=repo,
            stdout=archive,
            timeout=_DUMP_TIMEOUT_SECONDS,
        )
    if result.returncode != 0:
        raise BackupError(
            f"pg_dump failed (exit {result.returncode}): {_tail(result.stderr)}"
        )
    if partial.stat().st_size == 0:
        raise BackupError("pg_dump wrote an empty file")


def _verify(repo: Path, partial: Path, *, expected: Sequence[str]) -> None:
    listing = _restore(repo, partial, "--list")
    if listing.returncode != 0:
        raise BackupError(
            f"pg_restore could not list the dump: {_tail(listing.stderr)}"
        )
    missing = sorted(set(expected) - parse_listed_tables(listing.stdout.decode()))
    if missing:
        raise BackupError(f"the dump holds no data for {', '.join(missing)}")
    # The listing reads only the table of contents; a dump cut short after it still
    # lists cleanly, so every data block is read back too.
    full_read = _restore(repo, partial, "--file=/dev/null")
    if full_read.returncode != 0:
        raise BackupError(
            f"pg_restore could not read back the dump: {_tail(full_read.stderr)}"
        )


def _restore(
    repo: Path, partial: Path, mode: str
) -> subprocess.CompletedProcess[bytes]:
    with partial.open("rb") as archive:
        return docker_bootstrap.exec_in_service(
            DB_SERVICE,
            ["pg_restore", mode],
            repo=repo,
            stdin=archive,
            timeout=_DUMP_TIMEOUT_SECONDS,
        )


def _tail(stderr: bytes | None) -> str:
    lines = (stderr or b"").decode(errors="replace").strip().splitlines()
    return " | ".join(lines[-_STDERR_TAIL_LINES:]) or "no error output"


def format_size(size_bytes: int) -> str:
    """Return ``size_bytes`` in the largest binary unit that keeps it at least 1.

    Args:
        size_bytes: A file size.

    Returns:
        For example ``512 B``, ``2.0 KiB`` or ``5.5 GiB``.
    """
    if size_bytes < _SIZE_STEP:
        return f"{size_bytes} {_SIZE_UNITS[0]}"
    value = size_bytes / _SIZE_STEP
    for unit in _SIZE_UNITS[1:-1]:
        if value < _SIZE_STEP:
            return f"{value:.1f} {unit}"
        value /= _SIZE_STEP
    return f"{value:.1f} {_SIZE_UNITS[-1]}"
