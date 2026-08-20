"""Example tests for scope-1 and scope-2: the ``docker compose ps`` call in
``test_metrics_endpoint_exposure.py`` must resolve the repository root the
same way ``conftest.py``'s ``_scheduler_status`` does — via the
``Path(__file__).resolve().parents[...]`` anchor baked into the ``cwd=``
kwarg passed to ``subprocess.run`` — rather than depending on the pytest
process's current working directory. That is what makes the test pass
whether pytest is invoked from the repository root or from ``ingestion/``.

Source-blind by construction: both tests treat
``test_metrics_endpoint_exposure.py`` as data (its own source text, or its
process-boundary behaviour when run as a subprocess) — neither test imports
or inspects any module under ``app/``.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

_DEPLOYMENT_DIR = Path(__file__).resolve().parent
_INGESTION_ROOT = _DEPLOYMENT_DIR.parents[1]
_TARGET_MODULE = _DEPLOYMENT_DIR / "test_metrics_endpoint_exposure.py"


@pytest.mark.criterion("scope-1")
def test_when_the_ports_are_inspected_test_builds_its_compose_ps_call_then_cwd_is_repo_root():
    """
    Assumption: "matching deployment/conftest.py:112-118" was read as "uses the
    same repo-root anchor as the module's other subprocess.run calls", since
    the line numbers named in the criterion point at ``conftest.py``'s
    session-scoped fixtures (which already pass ``cwd=_REPO_ROOT`` to every
    ``docker compose`` invocation), not at the file under test.
    """
    source = _TARGET_MODULE.read_text()

    match = re.search(
        r"def test_when_scheduler_service_ports_are_inspected_.*?(?=\ndef |\Z)",
        source,
        flags=re.DOTALL,
    )
    assert match, "expected the ports-inspection test to still exist in this module"

    ps_call_body = match.group(0)
    assert "compose" in ps_call_body
    assert "ps" in ps_call_body
    assert "cwd=_REPO_ROOT" in ps_call_body, (
        "the docker compose ps subprocess.run call must pass cwd=_REPO_ROOT, "
        "not rely on the process's current working directory"
    )


@pytest.mark.criterion("scope-2")
def test_when_pytest_is_invoked_with_ingestion_as_the_working_directory_then_the_metrics_module_collects_cleanly():
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pytest", str(_TARGET_MODULE), "-q"],
        cwd=_INGESTION_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert "FAILED" not in result.stdout
    assert "ERROR" not in result.stdout
