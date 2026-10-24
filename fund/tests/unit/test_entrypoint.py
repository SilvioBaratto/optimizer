"""``fund/entrypoint.sh``: bounded DB wait, prompt shutdown, then ``exec`` the command.

The script is PID 1 of the fund container until ``exec``. A PID-namespace init drops
any signal it has no handler for, so without a trap ``docker stop`` during the DB wait
or the migrations hangs for the whole grace period and ends in SIGKILL. The tests run
the real script against fake ``pg_isready`` / ``alembic`` executables.
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


@pytest.fixture
def stubs(tmp_path: Path) -> Path:
    """A bin dir whose ``pg_isready`` fails and ``alembic`` blocks, plus a DB dir."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (tmp_path / "db").mkdir()
    _executable(bin_dir / "pg_isready", "exit 1")
    _executable(
        bin_dir / "alembic",
        f'echo $$ > "{tmp_path}/alembic.pid"\nexec sleep 30',
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
