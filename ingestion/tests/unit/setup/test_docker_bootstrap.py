"""Docker/DB bootstrap contract (SPEC D4/D11/D12).

`check_docker` verifies the daemon + compose plugin cross-platform and aborts
with a hint; `bring_up_db` runs `docker compose up -d --wait db`; `migrate` runs
`alembic upgrade head` (migrate-only). All shell-outs are patched — no real
Docker in the unit suite.
"""

import subprocess
import sys
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
    mock_run.side_effect = [_cp(0), _cp(1, "no compose")]
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


@patch("app.setup.docker_bootstrap._find_db_package")
@patch("app.setup.docker_bootstrap.subprocess.run")
def test_migrate_runs_alembic_from_db_package(
    mock_run: MagicMock, mock_find: MagicMock, tmp_path
) -> None:
    mock_find.return_value = tmp_path
    mock_run.return_value = _cp(0)
    db.migrate()
    assert mock_run.call_args[0][0] == [
        sys.executable,
        "-m",
        "alembic",
        "upgrade",
        "head",
    ]
    assert mock_run.call_args.kwargs["cwd"] == str(tmp_path)


@patch("app.setup.docker_bootstrap._find_db_package")
@patch("app.setup.docker_bootstrap.subprocess.run")
def test_migrate_failure_is_best_effort(
    mock_run: MagicMock, mock_find: MagicMock, tmp_path, capsys
) -> None:
    mock_find.return_value = tmp_path
    mock_run.return_value = _cp(1, "bad migration")
    db.migrate()  # must not raise — the fund container migrates on start
    assert "did not complete" in capsys.readouterr().out


@patch("app.setup.docker_bootstrap._find_db_package", return_value=None)
def test_migrate_skips_when_db_package_missing(mock_find: MagicMock, capsys) -> None:
    db.migrate()  # must not raise — warns and defers to the container
    assert "not found" in capsys.readouterr().out


@patch("app.setup.docker_bootstrap._find_db_package")
@patch(
    "app.setup.docker_bootstrap.subprocess.run",
    side_effect=FileNotFoundError("alembic missing"),
)
def test_migrate_missing_alembic_is_best_effort(
    mock_run: MagicMock, mock_find: MagicMock, tmp_path, capsys
) -> None:
    mock_find.return_value = tmp_path
    db.migrate()  # must not raise even when alembic cannot be launched
    assert "could not run host-side migration" in capsys.readouterr().out


@patch("app.setup.docker_bootstrap.subprocess.run")
def test_stop_services_stops_named_services(mock_run: MagicMock) -> None:
    mock_run.return_value = _cp(0)
    db.stop_services(["fund"])
    assert mock_run.call_args[0][0] == ["docker", "compose", "stop", "fund"]


@patch("app.setup.docker_bootstrap.subprocess.run")
def test_stop_services_failure_raises(mock_run: MagicMock) -> None:
    mock_run.return_value = _cp(1, "stop boom")
    with pytest.raises(db.DockerError):
        db.stop_services(["fund"])


@patch("app.setup.docker_bootstrap.subprocess.run")
def test_running_services_parses_lines(mock_run: MagicMock) -> None:
    result = _cp(0)
    result.stdout = "db\nscheduler\n"
    mock_run.return_value = result
    assert db.running_services() == {"db", "scheduler"}


@patch("app.setup.docker_bootstrap.subprocess.run")
def test_running_services_empty_on_error(mock_run: MagicMock) -> None:
    mock_run.return_value = _cp(1, "boom")
    assert db.running_services() == set()


@patch("app.setup.docker_bootstrap.shutil.which", return_value="/usr/bin/docker")
@patch("app.setup.docker_bootstrap.subprocess.run")
def test_docker_available_true(mock_run: MagicMock, _which: MagicMock) -> None:
    mock_run.return_value = _cp(0)
    assert db.docker_available() is True


@patch("app.setup.docker_bootstrap.shutil.which", return_value=None)
def test_docker_available_false(_which: MagicMock) -> None:
    assert db.docker_available() is False


def _cp_stdout(stdout: str) -> subprocess.CompletedProcess:
    """A succeeded run carrying `stdout` (for the version probes)."""
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


class TestVersionGates:
    """`warn_stale_versions` is advisory: below-minimum tooling warns, never raises.

    The two probes run in order: Docker Engine (server) then Compose.
    """

    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_current_versions_produce_no_warnings(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = [_cp_stdout("24.0.7"), _cp_stdout("2.29.1")]
        assert db.warn_stale_versions() == []

    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_old_engine_warns(
        self, mock_run: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DOCKER_BUILDKIT", raising=False)
        mock_run.side_effect = [_cp_stdout("20.10.0"), _cp_stdout("2.29.1")]
        assert any("Engine" in w for w in db.warn_stale_versions())

    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_buildkit_escape_hatch_suppresses_engine_warning(
        self, mock_run: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOCKER_BUILDKIT", "1")
        mock_run.side_effect = [_cp_stdout("20.10.0"), _cp_stdout("2.29.1")]
        assert db.warn_stale_versions() == []

    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_old_compose_warns(
        self, mock_run: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DOCKER_BUILDKIT", raising=False)
        mock_run.side_effect = [_cp_stdout("24.0.7"), _cp_stdout("1.29.2")]
        assert any("Compose" in w for w in db.warn_stale_versions())

    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_compose_just_below_2_24_warns(
        self, mock_run: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The floor is the compose-file parse floor: 2.23.x can't read the
        long-form `env_file` `required:` syntax, so it must warn (not merely a
        soft `up --wait` degradation)."""
        monkeypatch.delenv("DOCKER_BUILDKIT", raising=False)
        mock_run.side_effect = [_cp_stdout("24.0.7"), _cp_stdout("2.23.0")]
        # Pin the HARD-failure wording, not just presence: a revert to the old
        # soft `up --wait` degradation message must fail this test.
        assert any("parse" in w for w in db.warn_stale_versions())

    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_compose_at_floor_produces_no_warning(self, mock_run: MagicMock) -> None:
        """2.24.0 is the inclusive floor — exactly at it, no warning fires."""
        mock_run.side_effect = [_cp_stdout("24.0.7"), _cp_stdout("2.24.0")]
        assert db.warn_stale_versions() == []

    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_undetectable_versions_produce_no_warnings(
        self, mock_run: MagicMock
    ) -> None:
        mock_run.side_effect = [_cp(1, "boom"), _cp(1, "boom")]
        assert db.warn_stale_versions() == []

    def test_parse_version_returns_none_on_unparseable_output(self) -> None:
        assert db._parse_version("Docker version unknown") is None


class TestBuildAndUp:
    """`build_and_up` builds only when an image is absent, then `up -d --wait`."""

    @patch("app.setup.docker_bootstrap.warn_stale_versions", return_value=[])
    @patch("app.setup.docker_bootstrap._image_exists", return_value=False)
    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_builds_when_image_absent_then_up_waits(
        self, mock_run: MagicMock, _img: MagicMock, _warn: MagicMock
    ) -> None:
        mock_run.return_value = _cp(0)
        db.build_and_up(profiles=("fund",))
        calls = [c[0][0] for c in mock_run.call_args_list]
        build = next(c for c in calls if "build" in c)
        up = next(c for c in calls if "up" in c)
        assert {"--profile", "fund", "--pull", "--progress"} <= set(build)
        assert {"--profile", "fund", "up", "-d", "--wait"} <= set(up)
        assert "--wait-timeout" in up and "300" in up

    @patch("app.setup.docker_bootstrap.warn_stale_versions", return_value=[])
    @patch("app.setup.docker_bootstrap._image_exists", return_value=True)
    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_skips_build_when_image_present(
        self, mock_run: MagicMock, _img: MagicMock, _warn: MagicMock
    ) -> None:
        mock_run.return_value = _cp(0)
        db.build_and_up(profiles=("fund",))
        calls = [c[0][0] for c in mock_run.call_args_list]
        assert not any("build" in c for c in calls)
        assert any("up" in c for c in calls)

    @patch("app.setup.docker_bootstrap.warn_stale_versions", return_value=[])
    @patch("app.setup.docker_bootstrap._image_exists", return_value=False)
    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_prints_slow_once_note_when_building(
        self,
        mock_run: MagicMock,
        _img: MagicMock,
        _warn: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_run.return_value = _cp(0)
        db.build_and_up(profiles=("fund",))
        assert "slow" in capsys.readouterr().out.lower()

    @patch("app.setup.docker_bootstrap.warn_stale_versions", return_value=[])
    @patch("app.setup.docker_bootstrap._image_exists", return_value=False)
    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_build_failure_raises(
        self, mock_run: MagicMock, _img: MagicMock, _warn: MagicMock
    ) -> None:
        mock_run.return_value = _cp(1, "build boom")
        with pytest.raises(db.DockerError):
            db.build_and_up(profiles=("fund",))

    @patch("app.setup.docker_bootstrap.warn_stale_versions", return_value=[])
    @patch("app.setup.docker_bootstrap._image_exists", return_value=True)
    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_up_failure_raises(
        self, mock_run: MagicMock, _img: MagicMock, _warn: MagicMock
    ) -> None:
        mock_run.return_value = _cp(1, "up boom")
        with pytest.raises(db.DockerError):
            db.build_and_up(profiles=("fund",))

    @patch("app.setup.docker_bootstrap.warn_stale_versions", return_value=["stale!"])
    @patch("app.setup.docker_bootstrap._image_exists", return_value=True)
    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_surfaces_version_warnings(
        self,
        mock_run: MagicMock,
        _img: MagicMock,
        _warn: MagicMock,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_run.return_value = _cp(0)
        assert db.build_and_up(profiles=("fund",)) == ["stale!"]
        assert "stale!" in capsys.readouterr().out

    @patch("app.setup.docker_bootstrap.warn_stale_versions", return_value=[])
    @patch("app.setup.docker_bootstrap._image_exists", return_value=True)
    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_brings_up_multiple_profiles(
        self, mock_run: MagicMock, _img: MagicMock, _warn: MagicMock
    ) -> None:
        """`portopt start` passes both profiles → both --profile flags reach `up`."""
        mock_run.return_value = _cp(0)
        db.build_and_up(profiles=("fund", "ingestion"))
        up = next(c[0][0] for c in mock_run.call_args_list if "up" in c[0][0])
        assert up.count("--profile") == 2
        assert {"fund", "ingestion"} <= set(up)


class TestImageExists:
    """`_image_exists` reports whether a local image is present."""

    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_true_when_inspect_succeeds(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _cp(0)
        assert db._image_exists("optimizer-fund:latest") is True
        assert mock_run.call_args[0][0][:3] == ["docker", "image", "inspect"]

    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_false_when_inspect_fails(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _cp(1, "No such image")
        assert db._image_exists("absent:latest") is False
