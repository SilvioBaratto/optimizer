"""Shared fixtures for deployment-verification tests (issue #4).

Source-blind by construction: fixtures here drive real collaborators named
by the acceptance criteria — a fresh virtualenv's ``pip``/``python``, and the
real ``docker compose`` CLI against ``docker-compose.yml`` — never a mock of
the daemon or of Docker. No implementation module under ``app/`` is imported
by the test process itself; ``app.worker`` is imported only inside the
*subprocess* running in the fresh venv, exactly as the criterion states.

Opt-in (issue #12): ``pytest.ini`` already excludes this package from the
default ``pytest tests/`` run via ``--ignore``, but that only protects the
*default* invocation — a developer or CI job can still target this directory
(or a single module in it) directly, bypassing ``--ignore``. Both
session fixtures below re-check the opt-in themselves as a second line of
defence, so pointing pytest straight at this package without opting in still
skips rather than building a venv or shelling out to Docker.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import venv
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_INGESTION_ROOT = _REPO_ROOT / "ingestion"
_REQUIREMENTS = _INGESTION_ROOT / "requirements.txt"

_HEALTH_POLL_INTERVAL_SECONDS = 3
_HEALTH_POLL_TIMEOUT_SECONDS = 180

_RUN_DEPLOYMENT_ENV_VAR = "RUN_DEPLOYMENT_TESTS"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-deployment",
        action="store_true",
        default=False,
        help=(
            "Opt in to deployment-verification tests: builds a fresh "
            "virtualenv and drives real docker compose/scheduler "
            "containers. Skipped by default. Equivalent to setting "
            f"{_RUN_DEPLOYMENT_ENV_VAR}=1."
        ),
    )


def _deployment_opt_in(request: pytest.FixtureRequest) -> bool:
    return (
        bool(request.config.getoption("--run-deployment"))
        or os.environ.get(_RUN_DEPLOYMENT_ENV_VAR) == "1"
    )


def _skip_unless_opted_in(request: pytest.FixtureRequest) -> None:
    if not _deployment_opt_in(request):
        pytest.skip(
            "deployment-verification tests are opt-in: pass --run-deployment "
            f"or set {_RUN_DEPLOYMENT_ENV_VAR}=1"
        )


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


@pytest.fixture(scope="session")
def fresh_ingestion_venv(request: pytest.FixtureRequest, tmp_path_factory):
    """Build one virtualenv with only ``ingestion/requirements.txt`` installed.

    Session-scoped: scope-1 ("pip install succeeds, fastapi absent") and
    scope-2 ("that same venv imports app.worker") both name the *same*
    fresh venv, so the expensive build + install runs once and is shared.

    Skips (never fails) when the opt-in is absent or venv creation itself
    fails — e.g. no ``venv``/``ensurepip`` support in this Python build, or
    no disk space in ``tmp_path_factory``'s base dir. A venv-creation
    failure is an environment precondition, not a defect this suite proves.
    """
    _skip_unless_opted_in(request)

    venv_dir = tmp_path_factory.mktemp("fastapi-free-venv") / "venv"
    try:
        venv.EnvBuilder(with_pip=True, clear=True).create(venv_dir)
    except Exception as exc:  # any venv-creation failure is environmental, not a defect
        pytest.skip(f"could not build a virtualenv: {exc}")
    python_bin = _venv_python(venv_dir)

    install_result = subprocess.run(  # noqa: S603
        [str(python_bin), "-m", "pip", "install", "-r", str(_REQUIREMENTS)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    return {"python_bin": python_bin, "install_result": install_result}


def _docker_daemon_unreachable(docker: str) -> bool:
    info = subprocess.run(  # noqa: S603
        [docker, "info"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return info.returncode != 0


@pytest.fixture(scope="session")
def running_scheduler_container(request: pytest.FixtureRequest):
    """Build and start the ``scheduler`` service, wait for it to go healthy.

    Session-scoped and torn down with ``docker compose down`` so scope-3
    (in ``test_scheduler_container_health.py``) and scope-4 (in
    ``test_metrics_endpoint_exposure.py``) share one build/up cycle instead
    of each test module racing an independent one against the same
    container name.

    Skips (never fails, never reaches the ``docker compose down`` teardown)
    when the opt-in is absent or Docker is unavailable — binary missing from
    PATH, or the daemon unreachable (present binary, no running daemon, the
    common case on a CI runner with no Docker service configured). Once
    those preconditions are confirmed, a ``build``/``up`` failure is treated
    as a real defect and still fails the test.
    """
    _skip_unless_opted_in(request)

    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("docker not available on PATH")
    if _docker_daemon_unreachable(docker):
        pytest.skip("docker daemon not reachable")

    build = subprocess.run(  # noqa: S603
        [docker, "compose", "build", "scheduler"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if build.returncode != 0:
        pytest.fail(f"docker compose build scheduler failed:\n{build.stderr}")

    up = subprocess.run(  # noqa: S603
        [docker, "compose", "up", "-d"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if up.returncode != 0:
        pytest.fail(f"docker compose up -d failed:\n{up.stderr}")

    try:
        deadline = time.monotonic() + _HEALTH_POLL_TIMEOUT_SECONDS
        status = None
        while time.monotonic() < deadline:
            status = _scheduler_status(docker)
            if status == "healthy":
                break
            time.sleep(_HEALTH_POLL_INTERVAL_SECONDS)
        yield {"docker": docker, "final_status": status}
    finally:
        subprocess.run(  # noqa: S603
            [docker, "compose", "down"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )


def _compose_ps_record(docker: str, cwd: Path) -> dict | None:
    """Run ``docker compose ps`` for the ``scheduler`` service and parse it.

    The single place that builds the ``compose ps`` argv — both
    ``_scheduler_status`` (below) and the metrics-endpoint test's port
    assertion call this instead of each hand-rolling their own
    ``subprocess.run``. ``cwd`` is required rather than defaulted so a caller
    cannot forget it and fall back to the process's working directory, which
    is what let this call silently fail when pytest ran from ``ingestion/``.

    Returns:
        The service's parsed JSON record, or ``None`` if the command failed
        or produced no output (e.g. the service is not up yet).
    """
    ps = subprocess.run(  # noqa: S603
        [docker, "compose", "ps", "--format", "json", "scheduler"],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if ps.returncode != 0 or not ps.stdout.strip():
        return None
    lines = [line for line in ps.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    return json.loads(lines[0])


def _scheduler_status(docker: str) -> str | None:
    record = _compose_ps_record(docker, cwd=_REPO_ROOT)
    if record is None:
        return None
    return record.get("Health") or record.get("State")
