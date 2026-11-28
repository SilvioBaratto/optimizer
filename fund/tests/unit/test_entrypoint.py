"""``fund/entrypoint.sh``: bounded DB wait, pending check, prompt shutdown, ``exec``.

The script is PID 1 of the fund container until ``exec``. A PID-namespace init drops
any signal it has no handler for, so without a trap ``docker stop`` during the DB wait
or the migrations hangs for the whole grace period and ends in SIGKILL. Before
migrating it asks ``python -m portopt_db.migration_state`` and stops with that exit
code when revisions are pending on a database that holds data. The tests run the real
script against fake ``pg_isready`` / ``python`` / ``alembic`` executables.
"""

from __future__ import annotations

import os
import signal
import stat
import subprocess
import time
from pathlib import Path

import pytest

_ENTRYPOINT = Path(__file__).resolve().parents[2] / "entrypoint.sh"
_SIGTERM_EXIT = 143


def _executable(path: Path, body: str) -> None:
    path.write_text(f"#!/bin/bash\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


_PENDING_EXIT = 3


@pytest.fixture
def stubs(tmp_path: Path) -> Path:
    """A bin dir whose ``pg_isready`` fails and ``alembic`` blocks, plus a DB dir.

    The ``python`` stub stands in for ``python -m portopt_db.migration_state``: it
    records where and how it ran, then exits with ``$CHECK_EXIT`` (0 by default,
    the empty or current database that migrates on start as before).
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (tmp_path / "db").mkdir()
    _executable(bin_dir / "pg_isready", "exit 1")
    _executable(
        bin_dir / "alembic",
        f'echo $$ > "{tmp_path}/alembic.pid"\nexec sleep 30',
    )
    _executable(
        bin_dir / "python",
        f'printf "%s\\n" "$(pwd -P)" "$*" >> "{tmp_path}/check.log"\n'
        'exit "${CHECK_EXIT:-0}"',
    )
    return tmp_path


def _start(
    stubs: Path, *command: str, env: dict[str, str] | None = None
) -> subprocess.Popen[str]:
    environment = {
        **os.environ,
        "PATH": f"{stubs / 'bin'}{os.pathsep}{os.environ['PATH']}",
        "MIGRATIONS_DIR": str(stubs / "db"),
        **(env or {}),
    }
    return subprocess.Popen(  # noqa: S603 (fixed argv: bash + the repo's own script)
        ["/bin/bash", str(_ENTRYPOINT), *(command or ("true",))],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _await_output(proc: subprocess.Popen[str], needle: str) -> None:
    assert proc.stdout is not None
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if needle in line:
            return
        if not line and proc.poll() is not None:
            break
    proc.kill()
    raise AssertionError(f"never saw {needle!r}")


def test_a_database_that_never_becomes_ready_fails_the_start(stubs: Path) -> None:
    proc = _start(stubs, env={"DB_WAIT_TIMEOUT": "2"})
    try:
        output, _ = proc.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail("the DB wait loop is unbounded")

    assert proc.returncode == 1
    assert "not ready" in output


def test_sigterm_during_the_db_wait_exits_promptly(stubs: Path) -> None:
    proc = _start(stubs)
    _await_output(proc, "Waiting for database")

    proc.send_signal(signal.SIGTERM)
    proc.communicate(timeout=5)

    assert proc.returncode == _SIGTERM_EXIT


def test_sigterm_during_the_migrations_stops_them_and_exits(stubs: Path) -> None:
    _executable(stubs / "bin" / "pg_isready", "exit 0")
    # Alembic answers SIGTERM by rolling back, which takes a moment. The entrypoint
    # must outlast that: once PID 1 exits, the kernel SIGKILLs the rest of the
    # namespace mid-rollback. Off stdout so ``communicate`` cannot wait on it instead.
    _executable(
        stubs / "bin" / "alembic",
        "exec >/dev/null 2>&1\n"
        "trap 'kill $sleeper; sleep 0.5; exit 0' TERM\n"
        "sleep 30 & sleeper=$!\n"
        f'echo $$ > "{stubs}/alembic.tmp" && mv "{stubs}/alembic.tmp" '
        f'"{stubs}/alembic.pid"\n'
        "wait",
    )
    proc = _start(stubs)
    _await_output(proc, "Running Alembic")
    pid_file = stubs / "alembic.pid"
    deadline = time.monotonic() + 5
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    alembic_pid = int(pid_file.read_text())

    started = time.monotonic()
    proc.send_signal(signal.SIGTERM)
    proc.communicate(timeout=5)

    assert proc.returncode == _SIGTERM_EXIT
    assert time.monotonic() - started < 3
    with pytest.raises(ProcessLookupError):
        os.kill(alembic_pid, 0)


def test_a_ready_database_migrates_then_runs_the_command(stubs: Path) -> None:
    _executable(stubs / "bin" / "pg_isready", "exit 0")
    _executable(stubs / "bin" / "alembic", "exit 0")
    proc = _start(stubs, "echo", "daemon-started")

    output, _ = proc.communicate(timeout=15)

    assert proc.returncode == 0
    assert "Migrations complete" in output
    assert output.rstrip().endswith("daemon-started")


def _ready_and_recording(stubs: Path) -> None:
    _executable(stubs / "bin" / "pg_isready", "exit 0")
    _executable(stubs / "bin" / "alembic", f'touch "{stubs}/alembic.ran"')


@pytest.mark.criterion("REQ-ING-027")
def test_the_pending_check_runs_in_the_migrations_dir_before_alembic(
    stubs: Path,
) -> None:
    """Validates fund/entrypoint.sh: the check reads the same tree alembic upgrades,
    and an empty or current database still migrates on start."""
    _ready_and_recording(stubs)
    proc = _start(stubs, "echo", "daemon-started")

    output, _ = proc.communicate(timeout=15)

    assert proc.returncode == 0
    check = (stubs / "check.log").read_text().splitlines()
    assert check == [str((stubs / "db").resolve()), "-m portopt_db.migration_state"]
    assert (stubs / "alembic.ran").exists()
    assert output.rstrip().endswith("daemon-started")


@pytest.mark.criterion("REQ-ING-027")
def test_pending_migrations_stop_the_start_with_exit_three(stubs: Path) -> None:
    """Validates fund/entrypoint.sh: a database with data and pending revisions is
    left as it is and the worker never starts."""
    _ready_and_recording(stubs)
    proc = _start(stubs, "echo", "daemon-started", env={"CHECK_EXIT": "3"})

    output, _ = proc.communicate(timeout=15)

    assert proc.returncode == _PENDING_EXIT
    assert not (stubs / "alembic.ran").exists()
    assert "daemon-started" not in output


@pytest.mark.criterion("REQ-ING-027")
def test_a_failed_check_stops_the_start_without_migrating(stubs: Path) -> None:
    """Validates fund/entrypoint.sh when the state cannot be read: fail closed."""
    _ready_and_recording(stubs)
    proc = _start(stubs, "echo", "daemon-started", env={"CHECK_EXIT": "1"})

    output, _ = proc.communicate(timeout=15)

    assert proc.returncode == 1
    assert not (stubs / "alembic.ran").exists()
    assert "daemon-started" not in output


@pytest.mark.criterion("REQ-ING-027")
def test_sigterm_during_the_pending_check_exits_promptly(stubs: Path) -> None:
    """Validates fund/entrypoint.sh: the check runs under the signal trap too."""
    _executable(stubs / "bin" / "pg_isready", "exit 0")
    _executable(
        stubs / "bin" / "python",
        "exec >/dev/null 2>&1\n"
        f'echo $$ > "{stubs}/check.tmp" && mv "{stubs}/check.tmp" "{stubs}/check.pid"\n'
        "exec sleep 30",
    )
    proc = _start(stubs)
    _await_output(proc, "Checking for pending migrations")
    pid_file = stubs / "check.pid"
    deadline = time.monotonic() + 5
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    check_pid = int(pid_file.read_text())

    started = time.monotonic()
    proc.send_signal(signal.SIGTERM)
    proc.communicate(timeout=5)

    assert proc.returncode == _SIGTERM_EXIT
    assert time.monotonic() - started < 3
    with pytest.raises(ProcessLookupError):
        os.kill(check_pid, 0)
