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

    @pytest.fixture(autouse=True)
    def _checkout(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        """Pin the checkout to a tmp dir so the real repo is never written to."""
        monkeypatch.setattr(db, "compose_dir", lambda: tmp_path)
        return tmp_path

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
    @patch("app.setup.docker_bootstrap._image_exists", return_value=True)
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


class TestComposeRunsFromTheRepo:
    """With a checkout given, every compose call is anchored to it, so the files
    `portopt start` rendered there are the ones compose mounts."""

    @patch("app.setup.docker_bootstrap.warn_stale_versions", return_value=[])
    @patch("app.setup.docker_bootstrap._image_exists", return_value=False)
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
    @patch("app.setup.docker_bootstrap._image_exists", return_value=False)
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
    @patch("app.setup.docker_bootstrap._image_exists", return_value=False)
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
