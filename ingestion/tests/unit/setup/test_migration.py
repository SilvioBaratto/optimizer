"""``app.setup.migration.run_migrate``: back up, verify, then migrate, or do nothing.

Most tests replace the collaborators at the module seams (Docker, the state read, the
backup, the upgrade) and record the order they run in. The last ones fake only
``subprocess.run``, so every command the real code builds (compose, ``pg_dump``,
``pg_restore``, Alembic) is visible, and pin that a failed dump or verification never
reaches Alembic and that no command line carries the database password.
"""

from __future__ import annotations

import stat
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from portopt_db.migration_state import (
    MigrationState,
    MigrationStateError,
    MigrationStatus,
    PendingRevision,
)

from app.setup import db_backup, docker_bootstrap, host_database, migration

_CURRENT = "b8d0f2a4c6e1"
_HEAD = "8ed6504aee6a"
_PENDING_STEPS = (
    PendingRevision("2266e24c509f", "Drop the broker columns."),
    PendingRevision(_HEAD, "Drop the six scraper tables."),
)
_TABLES = ("public.instruments", "public.price_history")
_NOW = datetime(2026, 11, 28, 13, 5, 1)


def _status(state: MigrationState) -> MigrationStatus:
    if state is MigrationState.PENDING:
        return MigrationStatus(state, _HEAD, (_CURRENT,), _PENDING_STEPS, _TABLES)
    if state is MigrationState.EMPTY:
        return MigrationStatus(state, _HEAD, (), (), ())
    return MigrationStatus(state, _HEAD, (_HEAD,), (), _TABLES)


@dataclass
class _Stack:
    """What the fakes saw, in order, and what they answer."""

    events: list[str] = field(default_factory=list)
    states: list[MigrationStatus | Exception] = field(default_factory=list)
    running: set[str] = field(default_factory=lambda: {"scheduler", "fund"})
    drivers: list[str] = field(default_factory=list)
    backup_error: Exception | None = None
    upgrade_exit: int = 0
    prompts: list[str] = field(default_factory=list)
    answer: bool = True
    backup_kwargs: dict[str, Any] = field(default_factory=dict)

    def confirm(self, message: str) -> bool:
        """Record the prompt and give the configured answer."""
        self.prompts.append(message)
        return self.answer


@pytest.fixture
def stack(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Stack:
    """Replace every collaborator of ``run_migrate`` with a recording fake."""
    seen = _Stack(
        states=[_status(MigrationState.PENDING), _status(MigrationState.AT_HEAD)]
    )
    bootstrap = migration.docker_bootstrap

    def _read_state(db_dir: Path) -> MigrationStatus:
        seen.events.append("read")
        answer = seen.states.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def _backup(**kwargs: Any) -> db_backup.Backup:
        seen.events.append("backup")
        seen.backup_kwargs = kwargs
        if seen.backup_error is not None:
            raise seen.backup_error
        return db_backup.Backup(kwargs["directory"] / "optimizer_db.dump", 3 * 1024**2)

    def _upgrade(db_dir: Path) -> subprocess.CompletedProcess[str]:
        seen.events.append("upgrade")
        return subprocess.CompletedProcess([], seen.upgrade_exit, "", "boom")

    monkeypatch.setattr(
        bootstrap, "ensure_docker", lambda: seen.events.append("docker")
    )
    monkeypatch.setattr(bootstrap, "bring_up_db", lambda **kw: seen.events.append("db"))
    monkeypatch.setattr(bootstrap, "running_services", lambda **kw: set(seen.running))
    monkeypatch.setattr(bootstrap, "fund_drivers", lambda **kw: list(seen.drivers))
    monkeypatch.setattr(
        bootstrap,
        "stop_services",
        lambda names, **kw: seen.events.append(f"stop:{','.join(names)}"),
    )
    monkeypatch.setattr(
        bootstrap,
        "restart_services",
        lambda names, **kw: seen.events.append(f"restart:{','.join(names)}"),
    )
    monkeypatch.setattr(bootstrap, "run_alembic_upgrade", _upgrade)
    monkeypatch.setattr(migration.host_database, "read_state", _read_state)
    monkeypatch.setattr(migration.db_backup, "take_verified_backup", _backup)
    monkeypatch.setattr(migration.config_file, "load_config", lambda **kw: {})
    monkeypatch.setattr(
        migration.config_file, "DEFAULT_CONFIG_PATH", tmp_path / "home" / "config.toml"
    )
    return seen


def _run(stack: _Stack, **options: Any) -> migration.MigrateResult:
    return migration.run_migrate(
        migration.MigrateOptions(**options), confirm=stack.confirm, now=lambda: _NOW
    )


@pytest.mark.criterion("REQ-ING-030")
def test_no_backup_without_yes_is_refused_before_docker_is_touched(
    stack: _Stack,
) -> None:
    """Validates app.setup.migration.run_migrate: skipping the backup is explicit."""
    with pytest.raises(migration.MigrateError, match="--yes"):
        _run(stack, no_backup=True)

    assert stack.events == []


@pytest.mark.criterion("REQ-ING-030")
@pytest.mark.parametrize("state", [MigrationState.AT_HEAD, MigrationState.EMPTY])
def test_a_database_with_nothing_to_migrate_is_left_alone(
    stack: _Stack, state: MigrationState, capsys: pytest.CaptureFixture[str]
) -> None:
    """Validates app.setup.migration.run_migrate on an empty or current database: no
    dump of an empty database, no stop, no upgrade, no prompt."""
    stack.states = [_status(state)]

    result = _run(stack)

    assert result == migration.MigrateResult(migrated=False, head=_HEAD, backup=None)
    assert stack.events == ["docker", "db", "read"]
    assert stack.prompts == []
    assert _HEAD in capsys.readouterr().out


@pytest.mark.criterion("REQ-ING-030")
def test_a_pending_database_is_backed_up_then_migrated_with_the_daemons_stopped(
    stack: _Stack, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Validates app.setup.migration.run_migrate: confirm, stop the daemons so nothing
    writes during the dump or holds locks the DDL needs, back up, upgrade, re-check."""
    result = _run(stack, backup_dir=tmp_path / "dumps")

    assert stack.events == [
        "docker",
        "db",
        "read",
        "stop:fund,scheduler",
        "backup",
        "upgrade",
        "read",
    ]
    assert result.migrated is True
    assert result.backup is not None
    assert result.backup.path.parent == tmp_path / "dumps"
    assert stack.prompts == ["Proceed?"]
    assert stack.backup_kwargs["expected_tables"] == _TABLES
    assert stack.backup_kwargs["database"] == "optimizer_db"
    assert stack.backup_kwargs["user"] == "postgres"
    out = capsys.readouterr().out
    assert "2266e24c509f" in out and "Drop the six scraper tables." in out
    assert str(result.backup.path) in out
    assert "3.0 MiB" in out
    assert "portopt start" in out


@pytest.mark.criterion("REQ-ING-030")
def test_the_default_backup_dir_is_created_private_outside_the_checkout(
    stack: _Stack, tmp_path: Path
) -> None:
    """Validates app.setup.migration.run_migrate without --backup-dir: the dump goes
    to portopt's own directory, created 0700."""
    result = _run(stack, assume_yes=True)

    assert result.backup is not None
    directory = result.backup.path.parent
    assert directory == tmp_path / "home" / "backups"
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    repo = docker_bootstrap.compose_dir()
    assert repo not in directory.parents


@pytest.mark.criterion("REQ-ING-030")
def test_the_config_key_chooses_the_backup_dir_when_no_flag_does(
    stack: _Stack, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validates app.setup.migration.run_migrate with ``backup_dir`` in config.toml."""
    monkeypatch.setattr(
        migration.config_file,
        "load_config",
        lambda **kw: {"backup_dir": str(tmp_path / "from-config")},
    )

    result = _run(stack, assume_yes=True)

    assert result.backup is not None
    assert result.backup.path.parent == tmp_path / "from-config"


@pytest.mark.criterion("REQ-ING-030")
def test_a_declined_prompt_changes_nothing(stack: _Stack) -> None:
    """Validates app.setup.migration.run_migrate when the operator answers no."""
    stack.answer = False

    with pytest.raises(migration.MigrateError, match="nothing was changed"):
        _run(stack)

    assert stack.events == ["docker", "db", "read"]


@pytest.mark.criterion("REQ-ING-030")
def test_yes_skips_the_prompt(stack: _Stack) -> None:
    """Validates app.setup.migration.run_migrate with --yes."""
    _run(stack, assume_yes=True)

    assert stack.prompts == []
    assert "upgrade" in stack.events


@pytest.mark.criterion("REQ-ING-030")
def test_a_failed_backup_never_migrates_and_restarts_what_it_stopped(
    stack: _Stack,
) -> None:
    """Validates app.setup.migration.run_migrate when the dump or its verification
    fails: the database stays untouched and the old daemons come back."""
    stack.backup_error = db_backup.BackupError("pg_dump failed (exit 1)")

    with pytest.raises(migration.MigrateError, match="not migrated"):
        _run(stack, assume_yes=True)

    assert "upgrade" not in stack.events
    assert stack.events[-1] == "restart:fund,scheduler"


@pytest.mark.criterion("REQ-ING-030")
def test_no_backup_with_yes_warns_and_migrates(
    stack: _Stack, capsys: pytest.CaptureFixture[str]
) -> None:
    """Validates app.setup.migration.run_migrate with --no-backup --yes."""
    result = _run(stack, no_backup=True, assume_yes=True)

    assert "backup" not in stack.events
    assert "upgrade" in stack.events
    assert result.backup is None
    assert "without a backup" in capsys.readouterr().out


@pytest.mark.criterion("REQ-ING-030")
def test_a_running_fund_round_stops_the_migrate_before_the_prompt(
    stack: _Stack,
) -> None:
    """Validates app.setup.migration.run_migrate: stopping the fund would kill the
    round mid-flight, as `portopt stop` refuses to."""
    stack.drivers = ["/venv/bin/python /venv/bin/fund run f47ac10b"]

    with pytest.raises(migration.MigrateError, match="fund run f47ac10b"):
        _run(stack)

    assert stack.prompts == []
    assert stack.events == ["docker", "db", "read"]


@pytest.mark.criterion("REQ-ING-030")
def test_a_failed_upgrade_names_the_backup_and_restarts_the_daemons(
    stack: _Stack,
) -> None:
    """Validates app.setup.migration.run_migrate when Alembic fails: it runs in one
    transaction, so the database is at the old revision and the old daemons fit it."""
    stack.upgrade_exit = 1

    with pytest.raises(migration.MigrateError, match="optimizer_db.dump"):
        _run(stack, assume_yes=True)

    assert stack.events[-1] == "restart:fund,scheduler"


@pytest.mark.criterion("REQ-ING-030")
def test_an_upgrade_that_does_not_reach_head_is_reported(stack: _Stack) -> None:
    """Validates app.setup.migration.run_migrate's re-read after the upgrade."""
    stack.states = [_status(MigrationState.PENDING), _status(MigrationState.PENDING)]

    with pytest.raises(migration.MigrateError, match="not at head"):
        _run(stack, assume_yes=True)


@pytest.mark.criterion("REQ-ING-030")
def test_an_unreadable_state_is_a_failure(stack: _Stack) -> None:
    """Validates app.setup.migration.run_migrate: it never migrates what it cannot
    read."""
    stack.states = [MigrationStateError("connection refused")]

    with pytest.raises(migration.MigrateError, match="connection refused"):
        _run(stack)

    assert stack.events == ["docker", "db", "read"]


@pytest.mark.criterion("REQ-ING-030")
def test_only_the_running_daemons_are_stopped(stack: _Stack) -> None:
    """Validates app.setup.migration.run_migrate with only the scheduler up."""
    stack.running = {"scheduler", "db", "adminer"}

    _run(stack, assume_yes=True)

    assert "stop:scheduler" in stack.events


_URL_CREDENTIAL = "s3cret-pw"
_DUMP = b"PGDMP fake archive"


def _listing(*tables: str) -> bytes:
    rows = [
        f"{4000 + i}; 0 1 TABLE DATA {t.split('.')[0]} {t.split('.')[1]} postgres"
        for i, t in enumerate(tables)
    ]
    return "\n".join(rows).encode()


@dataclass
class _FakeSubprocess:
    """Answers each command ``run_migrate`` builds; ``fail`` names one to fail."""

    fail: str = ""
    argvs: list[list[str]] = field(default_factory=list)
    envs: list[dict[str, str] | None] = field(default_factory=list)

    def __call__(self, argv: Sequence[str], **kwargs: Any) -> Any:
        self.argvs.append(list(argv))
        self.envs.append(kwargs.get("env"))
        if "pg_dump" in argv:
            kwargs["stdout"].write(_DUMP)
            return self._done(argv, "dump", b"")
        if "pg_restore" in argv:
            kwargs["stdin"].read()
            listing = _listing(*_TABLES, "public.alembic_version")
            if "--list" in argv:
                return self._done(argv, "list", listing)
            return self._done(argv, "read", b"")
        if "ps" in argv:
            return subprocess.CompletedProcess(argv, 0, "scheduler\n", "")
        if "fund" in argv and "exec" in argv:
            return subprocess.CompletedProcess(argv, 1, "", "no such service")
        return subprocess.CompletedProcess(argv, 0, "", "")

    def _done(
        self, argv: Sequence[str], step: str, out: bytes
    ) -> subprocess.CompletedProcess[bytes]:
        code = 1 if self.fail == step else 0
        return subprocess.CompletedProcess(list(argv), code, out, b"failed")


@pytest.fixture
def subprocess_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[[str], _FakeSubprocess]:
    """Leave the real command builders in place behind a fake ``subprocess.run``."""

    def _install(fail: str) -> _FakeSubprocess:
        fake = _FakeSubprocess(fail=fail)
        states = [_status(MigrationState.PENDING), _status(MigrationState.AT_HEAD)]
        monkeypatch.setattr(subprocess, "run", fake)
        monkeypatch.setattr(docker_bootstrap, "ensure_docker", lambda: None)
        monkeypatch.setattr(
            host_database,
            "HOST_DB_URL",
            f"postgresql://postgres:{_URL_CREDENTIAL}@localhost:54320/optimizer_db",
        )
        monkeypatch.setattr(
            host_database, "inspect_migration_state", lambda *a: states.pop(0)
        )
        monkeypatch.setattr(
            migration.config_file, "DEFAULT_CONFIG_PATH", tmp_path / "config.toml"
        )
        monkeypatch.setattr(migration.config_file, "load_config", lambda **kw: {})
        return fake

    return _install


def _migrate_quietly() -> None:
    migration.run_migrate(
        migration.MigrateOptions(assume_yes=True),
        confirm=lambda message: True,
        now=lambda: _NOW,
    )


@pytest.mark.criterion("REQ-ING-030")
def test_no_command_line_carries_the_database_password(
    subprocess_only: Callable[[str], _FakeSubprocess],
) -> None:
    """Validates app.setup.migration.run_migrate end to end at the subprocess
    boundary: the dump runs over the db container's socket, and Alembic gets the
    URL through its environment, never its argv."""
    fake = subprocess_only("")

    _migrate_quietly()

    assert fake.argvs, "nothing ran"
    assert not [
        argv for argv in fake.argvs if any(_URL_CREDENTIAL in arg for arg in argv)
    ]
    alembic = [i for i, argv in enumerate(fake.argvs) if "alembic" in argv]
    assert len(alembic) == 1
    env = fake.envs[alembic[0]]
    assert env is not None
    assert _URL_CREDENTIAL in env["DATABASE_URL"]


@pytest.mark.criterion("REQ-ING-030")
@pytest.mark.parametrize("fail", ["dump", "list", "read"])
def test_a_failed_dump_or_verification_never_reaches_alembic(
    subprocess_only: Callable[[str], _FakeSubprocess], fail: str
) -> None:
    """Validates app.setup.migration.run_migrate end to end at the subprocess
    boundary for each failing backup step."""
    fake = subprocess_only(fail)

    with pytest.raises(migration.MigrateError):
        _migrate_quietly()

    assert not [argv for argv in fake.argvs if "alembic" in argv]
