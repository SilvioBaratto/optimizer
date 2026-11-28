"""``app.setup.db_backup``: the verified ``pg_dump`` that ``portopt migrate`` takes first.

``docker_bootstrap.exec_in_service`` is replaced by a fake that plays ``pg_dump`` and
``pg_restore`` inside the db container, so no Docker or database is touched. What is
pinned: where the dump goes and with which permissions, that a failed dump or a
failed verification leaves no file that looks like a backup, and that no command line
carries a password.
"""

from __future__ import annotations

import stat
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

import pytest

from app.setup import config_file, db_backup

# unit -> setup -> tests -> ingestion -> repo root.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_NOW = datetime(2026, 11, 28, 13, 5, 1)
_TABLES = ("langgraph.checkpoints", "public.instruments", "public.price_history")
_DUMP_BYTES = b"PGDMP\x01\x0f\x00fake archive"


def _listing(*tables: str) -> bytes:
    lines = [";", "; Archive created at 2026-11-28 12:00:34 UTC", ";"]
    for index, qualified in enumerate(tables, start=4000):
        schema, name = qualified.split(".")
        lines.append(f"{index}; 1259 16385 TABLE {schema} {name} postgres")
        lines.append(f"{index + 500}; 0 16385 TABLE DATA {schema} {name} postgres")
    return "\n".join(lines).encode()


@dataclass
class _FakeDbContainer:
    """Plays the db container: ``pg_dump`` writes an archive, ``pg_restore`` reads it."""

    dump_exit: int = 0
    list_exit: int = 0
    full_read_exit: int = 0
    listed: tuple[str, ...] = (*_TABLES, "public.alembic_version")
    calls: list[list[str]] = field(default_factory=list)

    def __call__(
        self,
        service: str,
        argv: Sequence[str],
        *,
        repo: Path,
        stdin: BinaryIO | None = None,
        stdout: BinaryIO | None = None,
        timeout: float,
    ) -> subprocess.CompletedProcess[bytes]:
        assert service == "db"
        self.calls.append(list(argv))
        if argv[0] == "pg_dump":
            assert stdout is not None
            stdout.write(_DUMP_BYTES)
            return self._done(argv, self.dump_exit, b"")
        assert stdin is not None
        assert stdin.read() == _DUMP_BYTES
        if "--list" in argv:
            return self._done(argv, self.list_exit, _listing(*self.listed))
        return self._done(argv, self.full_read_exit, b"")

    @staticmethod
    def _done(
        argv: Sequence[str], code: int, out: bytes
    ) -> subprocess.CompletedProcess[bytes]:
        err = b"" if code == 0 else b"pg_restore: error: could not read from input"
        return subprocess.CompletedProcess(list(argv), code, out, err)


@pytest.fixture
def container(monkeypatch: pytest.MonkeyPatch) -> _FakeDbContainer:
    """Route every db-container command to a fresh fake."""
    fake = _FakeDbContainer()
    monkeypatch.setattr(db_backup.docker_bootstrap, "exec_in_service", fake)
    return fake


def _backup(directory: Path, repo: Path) -> db_backup.Backup:
    return db_backup.take_verified_backup(
        repo=repo,
        directory=directory,
        database="optimizer_db",
        user="postgres",
        expected_tables=_TABLES,
        now=_NOW,
    )


def _files(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir())


@pytest.mark.criterion("REQ-ING-030")
def test_the_default_backup_dir_is_portopts_own_dir_outside_the_checkout() -> None:
    """Validates app.setup.db_backup.default_backup_dir: a multi-GB dump never lands
    in a checkout that Syncthing copies to other machines."""
    default = db_backup.default_backup_dir()

    assert default == config_file.DEFAULT_CONFIG_PATH.parent / "backups"
    assert _REPO_ROOT not in default.parents


@pytest.mark.criterion("REQ-ING-030")
def test_the_flag_wins_over_the_config_key_which_wins_over_the_default(
    tmp_path: Path,
) -> None:
    """Validates app.setup.db_backup.resolve_backup_dir precedence and ``~``."""
    config = {"backup_dir": "~/dumps"}

    assert db_backup.resolve_backup_dir(tmp_path, config) == tmp_path
    assert db_backup.resolve_backup_dir(None, config) == Path.home() / "dumps"
    assert db_backup.resolve_backup_dir(None, {}) == db_backup.default_backup_dir()
    assert db_backup.resolve_backup_dir(None, {"backup_dir": " "}) == (
        db_backup.default_backup_dir()
    )


@pytest.mark.criterion("REQ-ING-030")
def test_a_new_backup_dir_is_private_to_the_user(tmp_path: Path) -> None:
    """Validates app.setup.db_backup.prepare_backup_dir: the dump holds the whole
    database, so the directory it creates is 0700."""
    directory = db_backup.prepare_backup_dir(tmp_path / "home" / ".portopt" / "backups")

    assert directory.is_dir()
    assert stat.S_IMODE(directory.stat().st_mode) == db_backup.BACKUP_DIR_MODE == 0o700


@pytest.mark.criterion("REQ-ING-030")
def test_an_existing_backup_dir_is_left_as_the_user_set_it(tmp_path: Path) -> None:
    """Validates app.setup.db_backup.prepare_backup_dir on a directory the user chose."""
    chosen = tmp_path / "shared"
    chosen.mkdir(mode=0o750)
    chosen.chmod(0o750)

    db_backup.prepare_backup_dir(chosen)

    assert stat.S_IMODE(chosen.stat().st_mode) == 0o750


@pytest.mark.criterion("REQ-ING-030")
def test_the_dump_is_named_after_the_database_and_the_time(tmp_path: Path) -> None:
    """Validates app.setup.db_backup.dump_path."""
    path = db_backup.dump_path(tmp_path, "optimizer_db", _NOW)

    assert path == tmp_path / "optimizer_db-20261128-130501-pre-migrate.dump"


@pytest.mark.criterion("REQ-ING-030")
def test_the_listing_parser_reads_only_table_data_entries() -> None:
    """Validates app.setup.db_backup.parse_listed_tables."""
    assert db_backup.parse_listed_tables(_listing(*_TABLES).decode()) == set(_TABLES)


@pytest.mark.criterion("REQ-ING-030")
def test_a_verified_backup_is_kept_private_under_its_final_name(
    container: _FakeDbContainer, tmp_path: Path
) -> None:
    """Validates app.setup.db_backup.take_verified_backup: dump, TOC check against
    the database's tables, a full read, then the rename that makes it a backup."""
    backup = _backup(tmp_path, tmp_path / "repo")

    assert backup.path == tmp_path / "optimizer_db-20261128-130501-pre-migrate.dump"
    assert backup.path.read_bytes() == _DUMP_BYTES
    assert backup.size_bytes == len(_DUMP_BYTES)
    assert stat.S_IMODE(backup.path.stat().st_mode) == db_backup.DUMP_FILE_MODE == 0o600
    assert _files(tmp_path) == [backup.path.name]
    assert [call[0] for call in container.calls] == [
        "pg_dump",
        "pg_restore",
        "pg_restore",
    ]


@pytest.mark.criterion("REQ-ING-030")
def test_the_dump_command_carries_no_password(
    container: _FakeDbContainer, tmp_path: Path
) -> None:
    """Validates app.setup.db_backup.take_verified_backup: pg_dump runs inside the db
    container over its local socket, so no argv names a password or a URL."""
    _backup(tmp_path, tmp_path / "repo")

    dump = container.calls[0]
    assert dump == [
        "pg_dump",
        "--format=custom",
        "--username",
        "postgres",
        "--dbname",
        "optimizer_db",
    ]
    flat = " ".join(" ".join(call) for call in container.calls)
    assert "@" not in flat
    assert "PGPASSWORD" not in flat
    assert "--password" not in flat


@pytest.mark.criterion("REQ-ING-030")
@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ("dump_exit", "pg_dump"),
        ("list_exit", "list"),
        ("full_read_exit", "read back"),
    ],
)
def test_a_failed_dump_or_verification_leaves_nothing_that_looks_like_a_backup(
    container: _FakeDbContainer, tmp_path: Path, fault: str, reason: str
) -> None:
    """Validates app.setup.db_backup.take_verified_backup on each failing step."""
    setattr(container, fault, 1)

    with pytest.raises(db_backup.BackupError, match=reason):
        _backup(tmp_path, tmp_path / "repo")

    assert _files(tmp_path) == []


@pytest.mark.criterion("REQ-ING-030")
def test_a_dump_missing_a_table_fails_verification(
    container: _FakeDbContainer, tmp_path: Path
) -> None:
    """Validates app.setup.db_backup.take_verified_backup: every table the database
    holds, the Alembic stamp included, must have its data in the archive."""
    container.listed = ("public.instruments", "public.alembic_version")

    with pytest.raises(db_backup.BackupError, match="public.price_history"):
        _backup(tmp_path, tmp_path / "repo")

    assert _files(tmp_path) == []


@pytest.mark.criterion("REQ-ING-030")
def test_a_dump_without_the_alembic_stamp_fails_verification(
    container: _FakeDbContainer, tmp_path: Path
) -> None:
    """Validates app.setup.db_backup.take_verified_backup: a restore needs the stamp
    to know which revision the data belongs to."""
    container.listed = _TABLES

    with pytest.raises(db_backup.BackupError, match="alembic_version"):
        _backup(tmp_path, tmp_path / "repo")


@pytest.mark.criterion("REQ-ING-030")
def test_an_existing_backup_is_never_overwritten(
    container: _FakeDbContainer, tmp_path: Path
) -> None:
    """Validates app.setup.db_backup.take_verified_backup: old dumps are never
    replaced or deleted."""
    existing = db_backup.dump_path(tmp_path, "optimizer_db", _NOW)
    existing.write_bytes(b"an earlier backup")

    with pytest.raises(db_backup.BackupError, match="already exists"):
        _backup(tmp_path, tmp_path / "repo")

    assert existing.read_bytes() == b"an earlier backup"
    assert container.calls == []


@pytest.mark.criterion("REQ-ING-030")
@pytest.mark.parametrize(
    ("size", "text"),
    [(512, "512 B"), (2048, "2.0 KiB"), (5 * 1024**3 + 1024**3 // 2, "5.5 GiB")],
)
def test_sizes_are_printed_in_binary_units(size: int, text: str) -> None:
    """Validates app.setup.db_backup.format_size."""
    assert db_backup.format_size(size) == text
