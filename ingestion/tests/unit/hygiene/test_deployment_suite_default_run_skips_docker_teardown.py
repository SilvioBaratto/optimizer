"""
Pins acceptance criterion scope-3 (no-teardown half): neither `fresh_ingestion_venv`
nor `running_scheduler_container` ever calls `docker compose down` during a default
(no opt-in) run.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

INGESTION_ROOT = Path(__file__).resolve().parents[3]
DEPLOYMENT_TESTS = INGESTION_ROOT / "tests" / "integration" / "deployment"

_FAKE_DOCKER_SCRIPT = """#!/bin/sh
echo "$@" >> "$DOCKER_INVOCATION_LOG"
exit 0
"""


@pytest.mark.criterion("scope-3")
def test_when_deployment_suite_runs_without_opt_in_then_docker_compose_down_is_never_invoked():
    """
    Assumption: verified by shimming the `docker` executable on PATH to append every
    invocation's arguments to a log file, then asserting the log never records a `down`
    subcommand after a default run of the deployment suite. This is a mechanism-
    independent, black-box check of the shelled-out command line, not of internals.
    """
    with (
        tempfile.TemporaryDirectory() as shim_dir,
        tempfile.TemporaryDirectory() as log_dir,
    ):
        docker_shim = Path(shim_dir) / "docker"
        docker_shim.write_text(_FAKE_DOCKER_SCRIPT)
        docker_shim.chmod(0o755)
        log_file = Path(log_dir) / "docker_invocations.log"
        log_file.write_text("")

        env = os.environ.copy()
        env["PATH"] = f"{shim_dir}{os.pathsep}{env.get('PATH', '')}"
        env["DOCKER_INVOCATION_LOG"] = str(log_file)

        subprocess.run(  # noqa: S603
            [sys.executable, "-m", "pytest", str(DEPLOYMENT_TESTS), "-q"],
            cwd=INGESTION_ROOT,
            capture_output=True,
            text=True,
            timeout=180,
            env=env,
        )

        invocations = log_file.read_text()

    assert "down" not in invocations
