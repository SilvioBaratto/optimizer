"""Integration test proving the migrations-reversible hygiene guard runs to
green using only the packages the daemon declares in
``ingestion/pyproject.toml`` (its runtime deps + the ``[test]`` extra) — no
dev-only extra may be silently relied upon from the outer/dev environment
(issue #11, scope-1).

Source-blind by construction: drives a real subprocess (the fresh venv's
own ``pytest``) against the real guard file. No implementation module is
imported by this test *process* itself — only by the subprocess, inside the
fresh venv, exactly as the criterion states. Reuses the session-scoped
``fresh_ingestion_venv`` fixture (``conftest.py`` in this directory) so the
expensive venv build is shared with the sibling deployment-verification
tests rather than rebuilt per module.

Ambiguity resolution: the criterion says a Python environment built from
the daemon's declared dependencies alone can run ... pytest ... to green.
``pytest`` itself lives in the ``[test]`` extra, so it is installable from
``pyproject.toml`` alone. The assertion below checks the concrete subprocess
return code: if ``pytest`` is absent from the venv, invoking ``python -m
pytest`` exits non-zero ("No module named pytest"), so the test cannot pass
vacuously either way — it fails exactly when the environment is incomplete,
and only turns green once the guard is genuinely runnable from
``pyproject.toml`` alone.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_INGESTION_ROOT = _REPO_ROOT / "ingestion"
_GUARD_FILE = "tests/unit/hygiene/test_migrations_reversible.py"


@pytest.mark.criterion("scope-1")
@pytest.mark.integration
def test_when_migrations_guard_runs_in_a_requirements_only_venv_then_it_exits_zero(
    fresh_ingestion_venv,
):
    python_bin = fresh_ingestion_venv["python_bin"]

    result = subprocess.run(  # noqa: S603
        [str(python_bin), "-m", "pytest", _GUARD_FILE],
        cwd=_INGESTION_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, (
        f"pytest exit code: {result.returncode}\n"
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )
