"""Docker/DB bootstrap contract (SPEC D4/D11/D12, task T6).

`check_docker` verifies the daemon + compose plugin cross-platform and aborts
with a hint; `bring_up_db` runs `docker compose up -d --wait db`; `migrate` runs
`alembic upgrade head` (migrate-only). All shell-outs are patched — no real
Docker in the unit suite.
"""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from app.setup import docker_bootstrap as db


def _cp(returncode: int, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout="", stderr=stderr
    )


@patch("app.setup.docker_bootstrap.shutil.which", return_value="/usr/bin/docker")
@patch("app.setup.docker_bootstrap.subprocess.run")
def test_check_docker_ok(mock_run: MagicMock, _which: MagicMock) -> None:
    mock_run.return_value = _cp(0)
    db.check_docker()  # no raise


@patch("app.setup.docker_bootstrap.shutil.which", return_value=None)
def test_check_docker_missing_binary_raises(_which: MagicMock) -> None:
    with pytest.raises(db.DockerError):
        db.check_docker()


@patch("app.setup.docker_bootstrap.shutil.which", return_value="/usr/bin/docker")
@patch("app.setup.docker_bootstrap.subprocess.run")
def test_check_docker_daemon_down_raises(
    mock_run: MagicMock, _which: MagicMock
) -> None:
    mock_run.return_value = _cp(1, "Cannot connect to the Docker daemon")
    with pytest.raises(db.DockerError):
        db.check_docker()


@patch("app.setup.docker_bootstrap.shutil.which", return_value="/usr/bin/docker")
@patch("app.setup.docker_bootstrap.subprocess.run")
def test_check_docker_compose_missing_raises(
    mock_run: MagicMock, _which: MagicMock
) -> None:
    mock_run.side_effect = [_cp(0), _cp(1, "no compose")]  # info ok, compose fails
    with pytest.raises(db.DockerError):
        db.check_docker()


@patch("app.setup.docker_bootstrap.subprocess.run")
def test_bring_up_db_runs_compose_wait(mock_run: MagicMock) -> None:
    mock_run.return_value = _cp(0)
    db.bring_up_db()
    argv = mock_run.call_args[0][0]
    assert argv[:3] == ["docker", "compose", "up"]
    assert "--wait" in argv and argv[-1] == "db"


@patch("app.setup.docker_bootstrap.subprocess.run")
def test_bring_up_db_failure_raises(mock_run: MagicMock) -> None:
    mock_run.return_value = _cp(1, "boom")
    with pytest.raises(db.DockerError):
        db.bring_up_db()


@patch("app.setup.docker_bootstrap.subprocess.run")
def test_migrate_runs_alembic_upgrade_head(mock_run: MagicMock) -> None:
    mock_run.return_value = _cp(0)
    db.migrate()
    assert mock_run.call_args[0][0] == ["alembic", "upgrade", "head"]


@patch("app.setup.docker_bootstrap.subprocess.run")
def test_migrate_failure_raises(mock_run: MagicMock) -> None:
    mock_run.return_value = _cp(1, "bad migration")
    with pytest.raises(db.DockerError):
        db.migrate()
