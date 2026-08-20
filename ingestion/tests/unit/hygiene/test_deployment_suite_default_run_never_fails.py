"""
Pins acceptance criterion scope-3 (skip-not-fail half): `fresh_ingestion_venv` skips
(not fails) when it cannot build a venv, and `running_scheduler_container` skips when
Docker is unavailable *or* when the opt-in is absent.
"""

import subprocess
import sys
from pathlib import Path

import pytest

INGESTION_ROOT = Path(__file__).resolve().parents[3]
DEPLOYMENT_TESTS = INGESTION_ROOT / "tests" / "integration" / "deployment"


@pytest.mark.criterion("scope-3")
def test_when_deployment_suite_runs_without_opt_in_then_no_test_is_reported_failed():
    """
    Assumption: the exact opt-in mechanism (flag/marker/env var) is unspecified by the
    acceptance criteria (scope-2 is marked not-verifiable for that reason), so this test
    checks the observable, mechanism-independent contract at the process boundary:
    pointed directly at the deployment suite with no opt-in configured, neither fixture
    may surface as FAILED or ERROR — degrading to a skip either because the opt-in is
    absent or because venv/docker provisioning is impossible in this environment.
    """
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pytest", str(DEPLOYMENT_TESTS), "-q"],
        cwd=INGESTION_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert "FAILED" not in result.stdout
    assert "ERROR" not in result.stdout
