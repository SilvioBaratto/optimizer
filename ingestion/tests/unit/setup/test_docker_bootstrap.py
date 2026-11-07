"""Docker/DB bootstrap contract (SPEC D4/D11/D12).

`check_docker` verifies the daemon + compose plugin cross-platform and aborts
with a hint; `bring_up_db` runs `docker compose up -d --wait db`; `migrate` runs
`alembic upgrade head` (migrate-only). All shell-outs are patched — no real
Docker in the unit suite.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.setup import ca_bundle
from app.setup import docker_bootstrap as db

# unit -> setup -> tests -> ingestion -> repo root (docker-compose.yml owner).
_REPO_ROOT = Path(__file__).resolve().parents[4]


# What the patched fingerprint reports, so a "present" image is current when its
# stamp equals it.
_CURRENT = "current-hash"


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
def test_migrate_targets_the_compose_db_despite_an_ambient_database_url(
    mock_run: MagicMock,
    mock_find: MagicMock,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wizard just started the compose db for exactly this migration; a
    DATABASE_URL exported for another project (or production) must not redirect
    `alembic upgrade head` onto that database."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://prod-user:secret@prod.example/app")
    mock_find.return_value = tmp_path
    mock_run.return_value = _cp(0)
    db.migrate()
    assert mock_run.call_args.kwargs["env"]["DATABASE_URL"] == db._HOST_DB_URL


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
def test_restart_services_restarts_named_services(mock_run: MagicMock) -> None:
    mock_run.return_value = _cp(0)
    db.restart_services(["fund", "scheduler"])
    assert mock_run.call_args[0][0] == [
        "docker",
        "compose",
        "restart",
        "fund",
        "scheduler",
    ]


@patch("app.setup.docker_bootstrap.subprocess.run")
def test_restart_services_failure_raises(mock_run: MagicMock) -> None:
    mock_run.return_value = _cp(1, "restart boom")
    with pytest.raises(db.DockerError):
        db.restart_services(["fund"])


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


@patch("app.setup.docker_bootstrap.subprocess.run")
def test_fund_drivers_lists_only_fund_cli_and_tui_processes(
    mock_run: MagicMock,
) -> None:
    """The worker, the probe itself and anything else in the container are not
    rounds a stop would cut short; the fund CLI and TUI are."""
    mock_run.return_value = _cp_stdout(
        "/app/.venv/bin/python /app/.venv/bin/fund run f47ac10b --asof 2026-01-30\n"
        "/app/.venv/bin/python /app/.venv/bin/fund-tui\n"
        "python -m fund.cli approve 1b2c3d4e\n"
        "/app/.venv/bin/python /app/.venv/bin/fund-worker\n"
        "python -c import os\n"
        "\n"
    )

    drivers = db.fund_drivers()

    assert drivers == [
        "/app/.venv/bin/python /app/.venv/bin/fund run f47ac10b --asof 2026-01-30",
        "/app/.venv/bin/python /app/.venv/bin/fund-tui",
        "python -m fund.cli approve 1b2c3d4e",
    ]
    argv = mock_run.call_args.args[0]
    assert argv[argv.index("exec") : argv.index("exec") + 3] == ["exec", "-T", "fund"]


@patch("app.setup.docker_bootstrap.subprocess.run")
def test_fund_drivers_is_empty_when_the_fund_container_is_down(
    mock_run: MagicMock,
) -> None:
    mock_run.return_value = _cp(1, 'service "fund" is not running')
    assert db.fund_drivers() == []


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

    @pytest.fixture(autouse=True)
    def _checkout(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        """Pin the checkout to a tmp dir so the real repo is never written to."""
        monkeypatch.setattr(db, "compose_dir", lambda: tmp_path)
        monkeypatch.setattr(
            db.image_source, "source_hash", lambda repo, image: _CURRENT
        )
        return tmp_path

    @patch("app.setup.docker_bootstrap.warn_stale_versions", return_value=[])
    @patch("app.setup.docker_bootstrap._image_source_hash", return_value=None)
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
    @patch("app.setup.docker_bootstrap._image_source_hash", return_value=_CURRENT)
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
    @patch("app.setup.docker_bootstrap._image_source_hash", return_value=_CURRENT)
    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_force_build_rebuilds_even_when_image_present(
        self, mock_run: MagicMock, _img: MagicMock, _warn: MagicMock
    ) -> None:
        """`force_build=True` rebuilds a present image — the escape hatch for a start
        after a code or migration change, which otherwise reuses the stale image."""
        mock_run.return_value = _cp(0)
        db.build_and_up(profiles=("fund",), force_build=True)
        calls = [c[0][0] for c in mock_run.call_args_list]
        assert any("build" in c for c in calls)

    @patch("app.setup.docker_bootstrap.warn_stale_versions", return_value=[])
    @patch("app.setup.docker_bootstrap._image_source_hash", return_value=None)
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
    @patch("app.setup.docker_bootstrap._image_source_hash", return_value=None)
    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_build_failure_raises(
        self, mock_run: MagicMock, _img: MagicMock, _warn: MagicMock
    ) -> None:
        mock_run.return_value = _cp(1, "build boom")
        with pytest.raises(db.DockerError):
            db.build_and_up(profiles=("fund",))

    @patch("app.setup.docker_bootstrap.warn_stale_versions", return_value=[])
    @patch("app.setup.docker_bootstrap._image_source_hash", return_value=_CURRENT)
    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_up_failure_raises(
        self, mock_run: MagicMock, _img: MagicMock, _warn: MagicMock
    ) -> None:
        mock_run.return_value = _cp(1, "up boom")
        with pytest.raises(db.DockerError):
            db.build_and_up(profiles=("fund",))

    @patch("app.setup.docker_bootstrap.warn_stale_versions", return_value=["stale!"])
    @patch("app.setup.docker_bootstrap._image_source_hash", return_value=_CURRENT)
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
    @patch("app.setup.docker_bootstrap._image_source_hash", return_value=_CURRENT)
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


class TestDockerCallsAreBounded:
    """A frozen Docker VM (Colima after an external-volume drop) never answers; an
    unbounded call would hang `start`, `stop` and `status` forever."""

    @staticmethod
    def _timeout(mock_run: MagicMock) -> float | None:
        return mock_run.call_args.kwargs.get("timeout")

    @patch("app.setup.docker_bootstrap.shutil.which", return_value="/usr/bin/docker")
    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_the_daemon_probe_has_a_timeout(
        self, mock_run: MagicMock, _which: MagicMock
    ) -> None:
        mock_run.return_value = _cp(0)
        db.check_docker()
        assert self._timeout(mock_run) is not None

    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_status_queries_have_a_timeout(self, mock_run: MagicMock, tmp_path) -> None:
        mock_run.return_value = _cp(0)
        db.running_services(repo=tmp_path)
        assert self._timeout(mock_run) is not None
        db.service_states(repo=tmp_path)
        assert self._timeout(mock_run) is not None
        db.fund_drivers(repo=tmp_path)
        assert self._timeout(mock_run) is not None
        db._image_source_hash("optimizer-fund:latest")
        assert self._timeout(mock_run) is not None

    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_stop_waits_out_the_longest_grace_period(
        self, mock_run: MagicMock, tmp_path
    ) -> None:
        """The scheduler gets 60s to drain before Docker kills it."""
        mock_run.return_value = _cp(0)
        db.stop_services(["fund", "scheduler"], repo=tmp_path)
        assert self._timeout(mock_run) > 60

    @patch("app.setup.docker_bootstrap.warn_stale_versions", return_value=[])
    @patch("app.setup.docker_bootstrap._image_source_hash", return_value=None)
    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_build_and_up_outlast_their_own_waits(
        self, mock_run: MagicMock, _img: MagicMock, _warn: MagicMock, tmp_path
    ) -> None:
        mock_run.return_value = _cp(0)
        db.build_and_up(profiles=("fund",), wait_timeout=300, repo=tmp_path)
        build, up = (c.kwargs["timeout"] for c in mock_run.call_args_list)
        assert build >= 1800
        assert up > 300

    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_a_hung_call_becomes_a_docker_error_naming_the_command(
        self, mock_run: MagicMock, tmp_path
    ) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker", timeout=60)
        with pytest.raises(db.DockerError, match="docker compose .*stop"):
            db.stop_services(["fund"], repo=tmp_path)

    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_status_reads_a_hung_daemon_as_nothing_running(
        self, mock_run: MagicMock, tmp_path
    ) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker", timeout=60)
        assert db.running_services(repo=tmp_path) == set()
        assert db.service_states(repo=tmp_path) == {}
        assert db.fund_drivers(repo=tmp_path) == []

    @patch("app.setup.docker_bootstrap.shutil.which", return_value="/usr/bin/docker")
    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_a_hung_daemon_is_not_available(
        self, mock_run: MagicMock, _which: MagicMock
    ) -> None:
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="docker", timeout=30)
        assert db.docker_available() is False


def _ps_json(*rows: dict) -> str:
    import json

    return "\n".join(json.dumps(row) for row in rows) + "\n"


class TestServiceStates:
    """`service_states` reads state and health, so a crash-looping container is not
    reported like a healthy one."""

    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_parses_one_json_object_per_line(
        self, mock_run: MagicMock, tmp_path
    ) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            [],
            0,
            stdout=_ps_json(
                {"Service": "db", "State": "running", "Health": "healthy"},
                {"Service": "fund", "State": "restarting", "Health": ""},
            ),
            stderr="",
        )
        assert db.service_states(repo=tmp_path) == {
            "db": db.ServiceState("running", "healthy"),
            "fund": db.ServiceState("restarting", ""),
        }
        argv = mock_run.call_args[0][0]
        assert "--all" in argv
        assert argv[argv.index("--format") + 1] == "json"

    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_parses_a_json_array(self, mock_run: MagicMock, tmp_path) -> None:
        """Older Compose prints one array instead of one object per line."""
        mock_run.return_value = subprocess.CompletedProcess(
            [],
            0,
            stdout='[{"Service": "db", "State": "exited", "Health": ""}]',
            stderr="",
        )
        assert db.service_states(repo=tmp_path) == {"db": db.ServiceState("exited", "")}

    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_empty_on_error_or_unparseable_output(
        self, mock_run: MagicMock, tmp_path
    ) -> None:
        mock_run.return_value = _cp(1, "boom")
        assert db.service_states(repo=tmp_path) == {}
        mock_run.return_value = subprocess.CompletedProcess(
            [], 0, stdout="not json\n", stderr=""
        )
        assert db.service_states(repo=tmp_path) == {}


class TestStaleImages:
    """An image built from older sources than the checkout is rebuilt on start.

    `portopt setup` migrates the database with the checkout's migrations, so a start
    that reused the images built before an upgrade would run old code against the new
    schema, and the old entrypoint's `alembic upgrade head` would fail on a revision
    it has never heard of.
    """

    @pytest.fixture(autouse=True)
    def _checkout(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(db, "compose_dir", lambda: tmp_path)
        monkeypatch.setattr(
            db.image_source, "source_hash", lambda repo, image: f"{image}-{_CURRENT}"
        )
        return tmp_path

    @staticmethod
    def _stamps(**by_ref: str | None):
        return lambda ref: by_ref[ref.split(":")[0].removeprefix("optimizer-")]

    @patch("app.setup.docker_bootstrap.warn_stale_versions", return_value=[])
    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_rebuilds_an_image_whose_stamp_differs_from_the_checkout(
        self, mock_run: MagicMock, _warn: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(db, "_image_source_hash", self._stamps(fund="fund-old"))
        mock_run.return_value = _cp(0)

        db.build_and_up(profiles=("fund",))

        calls = [c[0][0] for c in mock_run.call_args_list]
        assert any("build" in c for c in calls)

    @patch("app.setup.docker_bootstrap.warn_stale_versions", return_value=[])
    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_rebuilds_an_unstamped_image(
        self, mock_run: MagicMock, _warn: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Built by a plain `docker compose build`: no way to know what it holds."""
        monkeypatch.setattr(db, "_image_source_hash", self._stamps(scheduler=""))
        mock_run.return_value = _cp(0)

        db.build_and_up(profiles=("ingestion",))

        assert any("build" in c[0][0] for c in mock_run.call_args_list)

    @patch("app.setup.docker_bootstrap.warn_stale_versions", return_value=[])
    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_reuses_images_stamped_with_the_current_hash(
        self, mock_run: MagicMock, _warn: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            db,
            "_image_source_hash",
            self._stamps(fund=f"fund-{_CURRENT}", scheduler=f"scheduler-{_CURRENT}"),
        )
        mock_run.return_value = _cp(0)

        db.build_and_up(profiles=("fund", "ingestion"))

        assert not any("build" in c[0][0] for c in mock_run.call_args_list)

    @patch("app.setup.docker_bootstrap.warn_stale_versions", return_value=[])
    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_only_the_profiles_started_are_compared(
        self, mock_run: MagicMock, _warn: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stale fund image must not slow a start that leaves the fund off."""
        monkeypatch.setattr(
            db,
            "_image_source_hash",
            self._stamps(fund="fund-old", scheduler=f"scheduler-{_CURRENT}"),
        )
        mock_run.return_value = _cp(0)

        db.build_and_up(profiles=("ingestion",))

        assert not any("build" in c[0][0] for c in mock_run.call_args_list)

    @patch("app.setup.docker_bootstrap.warn_stale_versions", return_value=[])
    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_the_build_stamps_the_hash_it_compared(
        self, mock_run: MagicMock, _warn: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Compose reads the hash from the environment into the image label."""
        monkeypatch.setattr(db, "_image_source_hash", lambda ref: None)
        mock_run.return_value = _cp(0)

        db.build_and_up(profiles=("fund", "ingestion"))

        for call in mock_run.call_args_list:
            env = call.kwargs["env"]
            assert env["PORTOPT_FUND_SOURCE_HASH"] == f"fund-{_CURRENT}"
            assert env["PORTOPT_SCHEDULER_SOURCE_HASH"] == f"scheduler-{_CURRENT}"

    @patch("app.setup.docker_bootstrap.warn_stale_versions", return_value=[])
    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_a_rebuild_is_announced_as_a_checkout_change(
        self,
        mock_run: MagicMock,
        _warn: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(db, "_image_source_hash", self._stamps(fund="fund-old"))
        mock_run.return_value = _cp(0)

        db.build_and_up(profiles=("fund",))

        assert "changed" in capsys.readouterr().out

    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_other_compose_calls_do_not_pay_for_a_fingerprint(
        self, mock_run: MagicMock
    ) -> None:
        mock_run.return_value = _cp(0)

        db.stop_services(["fund"])

        env = mock_run.call_args.kwargs["env"]
        assert env is None or "PORTOPT_FUND_SOURCE_HASH" not in env


class TestComposeRunsFromTheRepo:
    """With a checkout given, every compose call is anchored to it, so the files
    `portopt start` rendered there are the ones compose mounts."""

    @patch("app.setup.docker_bootstrap.warn_stale_versions", return_value=[])
    @patch("app.setup.docker_bootstrap._image_source_hash", return_value=None)
    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_build_and_up_pins_build_and_up_to_the_repo(
        self, mock_run: MagicMock, _img: MagicMock, _warn: MagicMock, tmp_path
    ) -> None:
        mock_run.return_value = _cp(0)
        db.build_and_up(profiles=("fund",), repo=tmp_path)
        assert len(mock_run.call_args_list) == 2  # build, then up
        for call in mock_run.call_args_list:
            argv = call[0][0]
            assert argv[argv.index("--project-directory") + 1] == str(tmp_path)
            assert call.kwargs["cwd"] == str(tmp_path)

    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_stop_services_runs_from_the_repo(
        self, mock_run: MagicMock, tmp_path
    ) -> None:
        mock_run.return_value = _cp(0)
        db.stop_services(["fund"], repo=tmp_path)
        argv = mock_run.call_args[0][0]
        assert argv[argv.index("--project-directory") + 1] == str(tmp_path)
        assert argv[-2:] == ["stop", "fund"]
        assert mock_run.call_args.kwargs["cwd"] == str(tmp_path)

    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_running_services_runs_from_the_repo(
        self, mock_run: MagicMock, tmp_path
    ) -> None:
        mock_run.return_value = _cp(0)
        db.running_services(repo=tmp_path)
        argv = mock_run.call_args[0][0]
        assert argv[argv.index("--project-directory") + 1] == str(tmp_path)
        assert mock_run.call_args.kwargs["cwd"] == str(tmp_path)


class TestImageSourceHash:
    """`_image_source_hash` reads the stamp `portopt start` left on a built image."""

    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_returns_the_stamped_hash(self, mock_run: MagicMock) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            [], 0, stdout='{"io.portopt.source-hash": "abc123"}\n', stderr=""
        )
        assert db._image_source_hash("optimizer-fund:latest") == "abc123"
        assert mock_run.call_args[0][0][:3] == ["docker", "image", "inspect"]

    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_none_when_the_image_is_absent(self, mock_run: MagicMock) -> None:
        mock_run.return_value = _cp(1, "No such image")
        assert db._image_source_hash("absent:latest") is None

    @pytest.mark.parametrize("labels", ["null", "{}", '{"other": "x"}', "not json"])
    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_empty_when_the_image_carries_no_stamp(
        self, mock_run: MagicMock, labels: str
    ) -> None:
        """An image built by a plain `docker compose build` has no stamp."""
        mock_run.return_value = subprocess.CompletedProcess(
            [], 0, stdout=labels + "\n", stderr=""
        )
        assert db._image_source_hash("optimizer-fund:latest") == ""


class TestComposeRunsFromTheCheckout:
    """Compose resolves `docker-compose.yml` and `./secrets/*` relative to the cwd, so
    every compose call must be pinned to the checkout, never to wherever the operator
    happens to be (a `uv tool install` user runs `portopt` from `~`)."""

    @pytest.fixture
    def checkout(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        repo = tmp_path / "checkout"
        repo.mkdir()
        (repo / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.setenv("OPTIMIZER_REPO", str(repo))
        monkeypatch.chdir(elsewhere)
        return repo

    @patch("app.setup.docker_bootstrap.warn_stale_versions", return_value=[])
    @patch("app.setup.docker_bootstrap._image_source_hash", return_value=None)
    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_every_compose_command_runs_in_the_checkout(
        self, mock_run: MagicMock, _img: MagicMock, _warn: MagicMock, checkout
    ) -> None:
        mock_run.return_value = _cp(0)
        db.bring_up_db()
        db.stop_services(["fund"])
        db.running_services()
        db.build_and_up(profiles=("fund",))
        compose_calls = [
            c for c in mock_run.call_args_list if c[0][0][:2] == ["docker", "compose"]
        ]
        assert len(compose_calls) >= 5  # up db, stop, ps, build, up
        assert {c.kwargs.get("cwd") for c in compose_calls} == {str(checkout)}

    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_no_checkout_fails_before_any_compose_call(
        self,
        mock_run: MagicMock,
        tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A `uv tool install` with no clone has no compose file anywhere: fail with
        guidance instead of letting compose answer 'no configuration file'."""
        from app.setup import path_install

        bare = tmp_path / "bare"
        bare.mkdir()
        monkeypatch.delenv("OPTIMIZER_REPO", raising=False)
        monkeypatch.chdir(bare)
        monkeypatch.setattr(path_install.config_file, "load_config", lambda **kw: {})
        monkeypatch.setattr(path_install, "_repo_root", lambda: bare)
        with pytest.raises(db.DockerError, match="OPTIMIZER_REPO"):
            db.bring_up_db()
        mock_run.assert_not_called()

    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_running_services_is_empty_without_a_checkout(
        self, mock_run: MagicMock, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.setup import path_install

        bare = tmp_path / "bare"
        bare.mkdir()
        monkeypatch.delenv("OPTIMIZER_REPO", raising=False)
        monkeypatch.chdir(bare)
        monkeypatch.setattr(path_install.config_file, "load_config", lambda **kw: {})
        monkeypatch.setattr(path_install, "_repo_root", lambda: bare)
        assert db.running_services() == set()
        mock_run.assert_not_called()


class TestCaBundleBuildSecret:
    """The image builds read an optional `ca_bundle` secret. Compose stats the secret
    file before building, so the file it names must exist on a fresh clone; the
    generated corporate bundle is swapped in through `CA_BUNDLE_FILE` instead of
    being created at a path the committed compose file hard-codes."""

    @pytest.fixture
    def checkout(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(db, "compose_dir", lambda: tmp_path)
        monkeypatch.delenv("CA_BUNDLE_FILE", raising=False)
        return tmp_path

    @staticmethod
    def _compose_env(mock_run: MagicMock) -> dict[str, str] | None:
        """Env passed to the last compose call (None means 'inherit')."""
        return mock_run.call_args.kwargs["env"]

    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_a_generated_bundle_is_handed_to_compose(
        self, mock_run: MagicMock, checkout: Path
    ) -> None:
        bundle = checkout / ca_bundle.DEFAULT_BUNDLE_PATH
        bundle.parent.mkdir()
        bundle.write_text("CORP BUNDLE", encoding="utf-8")
        mock_run.return_value = _cp(0)
        db.bring_up_db()
        env = self._compose_env(mock_run)
        assert env is not None
        assert env["CA_BUNDLE_FILE"] == str(bundle)

    @patch("app.setup.docker_bootstrap.subprocess.run")
    @pytest.mark.parametrize("content", [None, ""])
    def test_without_a_real_bundle_compose_keeps_its_committed_default(
        self, mock_run: MagicMock, checkout: Path, content: str | None
    ) -> None:
        if content is not None:
            bundle = checkout / ca_bundle.DEFAULT_BUNDLE_PATH
            bundle.parent.mkdir()
            bundle.write_text(content, encoding="utf-8")
        mock_run.return_value = _cp(0)
        db.bring_up_db()
        env = self._compose_env(mock_run)
        assert env is None or "CA_BUNDLE_FILE" not in env

    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_an_operator_supplied_bundle_wins(
        self, mock_run: MagicMock, checkout: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bundle = checkout / ca_bundle.DEFAULT_BUNDLE_PATH
        bundle.parent.mkdir()
        bundle.write_text("CORP BUNDLE", encoding="utf-8")
        monkeypatch.setenv("CA_BUNDLE_FILE", "/operator/bundle.pem")
        mock_run.return_value = _cp(0)
        db.bring_up_db()
        env = self._compose_env(mock_run)
        assert env is None or env["CA_BUNDLE_FILE"] == "/operator/bundle.pem"

    @patch("app.setup.docker_bootstrap.warn_stale_versions", return_value=[])
    @patch("app.setup.docker_bootstrap._image_source_hash", return_value=None)
    @patch("app.setup.docker_bootstrap.subprocess.run")
    def test_build_and_up_writes_nothing_into_the_checkout(
        self, mock_run: MagicMock, _img: MagicMock, _warn: MagicMock, checkout: Path
    ) -> None:
        """The committed default makes a placeholder unnecessary, and a stray
        `.certs/ca-bundle.pem` would shadow nothing but still litter the clone."""
        mock_run.return_value = _cp(0)
        db.build_and_up(profiles=("fund",))
        assert list(checkout.iterdir()) == []


def _compose_doc() -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load((_REPO_ROOT / "docker-compose.yml").read_text("utf-8"))


@pytest.mark.parametrize("service", ["scheduler", "fund"])
def test_compose_passes_the_ca_bundle_secret_to_each_image_build(
    service: str,
) -> None:
    """Without `build.secrets` the Dockerfiles' optional `--mount=type=secret` is
    always empty, so `uv sync` still cannot verify PyPI behind a TLS-inspecting
    proxy even after `portopt setup --corp-ca` produced a bundle."""
    build = _compose_doc()["services"][service]["build"]
    assert "ca_bundle" in build["secrets"]


_SECRET_FILE = re.compile(r"^\$\{CA_BUNDLE_FILE:-(?P<default>[^}]+)\}$")


def _ca_bundle_default() -> Path:
    declared = _compose_doc()["secrets"]["ca_bundle"]["file"]
    match = _SECRET_FILE.match(declared)
    assert match, f"ca_bundle.file must default through CA_BUNDLE_FILE: {declared!r}"
    return _REPO_ROOT / match["default"]


def test_ca_bundle_secret_default_is_not_the_generated_bundle() -> None:
    """The generated path is git-ignored, so a default pointing at it would be
    missing on a fresh clone."""
    assert _ca_bundle_default() != _REPO_ROOT / ca_bundle.DEFAULT_BUNDLE_PATH


def test_ca_bundle_secret_default_exists_on_a_fresh_clone() -> None:
    """Compose aborts `build` and `up` when a `file:` secret is missing, and the
    generated `.certs/` is git-ignored: the default must be a committed, empty
    file so a plain `docker compose up` works before `portopt setup` ever ran."""
    default = _ca_bundle_default()
    assert default.is_file()
    assert default.stat().st_size == 0
    git = shutil.which("git")
    if git is None or not (_REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    tracked = subprocess.run(  # noqa: S603 - resolved git path, static argv
        [git, "ls-files", "--error-unmatch", str(default.relative_to(_REPO_ROOT))],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert tracked.returncode == 0, tracked.stderr


class _FakeDocker:
    """Scripted ``subprocess.run`` for a host whose Docker daemon starts down.

    ``docker info`` fails ``down_polls`` times and then answers; ``docker context
    show`` prints ``context``; the runtime launch command exits ``start_rc``.
    """

    def __init__(
        self,
        *,
        context: str = "colima",
        down_polls: int = 1,
        start_rc: int = 0,
        start_stderr: str = "",
    ) -> None:
        self.context = context
        self.down_polls = down_polls
        self.start_rc = start_rc
        self.start_stderr = start_stderr
        self.calls: list[list[str]] = []
        self._info_polls = 0

    def __call__(self, cmd: list[str], **_kw: object) -> subprocess.CompletedProcess:
        self.calls.append(cmd)
        if cmd[:2] == ["docker", "info"]:
            self._info_polls += 1
            return _cp(0 if self._info_polls > self.down_polls else 1)
        if cmd[:3] == ["docker", "context", "show"]:
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=f"{self.context}\n", stderr=""
            )
        if cmd[:2] == ["docker", "compose"]:
            return _cp(0)
        return _cp(self.start_rc, self.start_stderr)

    @property
    def launches(self) -> list[list[str]]:
        return [cmd for cmd in self.calls if cmd[0] != "docker"]


class TestEnsureDocker:
    """`ensure_docker` boots a stopped local runtime so `portopt start` works cold."""

    @pytest.fixture(autouse=True)
    def _host(self, monkeypatch: pytest.MonkeyPatch) -> list[float]:
        installed = {"docker", "colima", "orb"}
        monkeypatch.setattr(
            db.shutil,
            "which",
            lambda name: f"/usr/local/bin/{name}" if name in installed else None,
        )
        monkeypatch.setattr(db.sys, "platform", "darwin")
        sleeps: list[float] = []
        monkeypatch.setattr(db.time, "sleep", sleeps.append)
        return sleeps

    def _install(self, monkeypatch: pytest.MonkeyPatch, fake: _FakeDocker) -> None:
        monkeypatch.setattr(db.subprocess, "run", fake)

    def test_running_daemon_launches_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeDocker(down_polls=0)
        self._install(monkeypatch, fake)
        db.ensure_docker()
        assert fake.launches == []

    def test_colima_context_starts_colima(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeDocker(context="colima")
        self._install(monkeypatch, fake)
        db.ensure_docker()
        assert fake.launches == [["colima", "start", "--profile", "default"]]

    def test_colima_profile_context_starts_that_profile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeDocker(context="colima-work")
        self._install(monkeypatch, fake)
        db.ensure_docker()
        assert fake.launches == [["colima", "start", "--profile", "work"]]

    def test_orbstack_context_starts_orbstack(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeDocker(context="orbstack")
        self._install(monkeypatch, fake)
        db.ensure_docker()
        assert fake.launches == [["orb", "start"]]

    def test_docker_desktop_is_polled_until_its_daemon_answers(
        self, monkeypatch: pytest.MonkeyPatch, _host: list[float]
    ) -> None:
        """`open -a Docker` returns before the daemon is up, so it must be polled."""
        fake = _FakeDocker(context="desktop-linux", down_polls=3)
        self._install(monkeypatch, fake)
        db.ensure_docker()
        assert fake.launches == [["open", "-a", "Docker"]]
        assert len(_host) == 2

    def test_linux_engine_is_not_started_and_hints_systemctl(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(db.sys, "platform", "linux")
        fake = _FakeDocker(context="default", down_polls=99)
        self._install(monkeypatch, fake)
        with pytest.raises(db.DockerError, match="systemctl"):
            db.ensure_docker()
        assert fake.launches == []

    def test_failed_launch_raises_with_its_stderr(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeDocker(start_rc=1, start_stderr="disk /Volumes/Colima missing")
        self._install(monkeypatch, fake)
        with pytest.raises(db.DockerError, match="/Volumes/Colima missing"):
            db.ensure_docker()

    def test_daemon_that_never_answers_times_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeDocker(context="desktop-linux", down_polls=99)
        self._install(monkeypatch, fake)
        with pytest.raises(db.DockerError, match="unreachable"):
            db.ensure_docker(timeout=0)

    def test_missing_cli_raises_the_install_hint_without_launching(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(db.shutil, "which", lambda name: None)
        fake = _FakeDocker(down_polls=99)
        self._install(monkeypatch, fake)
        with pytest.raises(db.DockerError, match="not found"):
            db.ensure_docker()
        assert fake.calls == []
