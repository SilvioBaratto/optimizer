"""
Pins acceptance criterion scope-1: `cd ingestion && pytest tests/` collects zero tests
from `tests/integration/deployment` by default, verified with `--collect-only -q`.
"""

import subprocess
import sys
from pathlib import Path

import pytest

INGESTION_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.criterion("scope-1")
def test_when_default_test_run_collects_then_deployment_dir_is_absent():
    """
    Assumption: "collects zero tests ... by default" is read as a default-collection
    exclusion (config- or conftest-level), not a `-k`/`-m` selector layered on top of an
    otherwise-full collection — the simplest reading consistent with "by default".
    `--collect-only` never executes a test body, so this subprocess call cannot recurse
    into itself even though it targets the whole `tests/` tree.

    `pytest.ini` bakes a single `-v` into `addopts`; passing one `-q` here would net to
    plain (tree-diagram) verbosity, which prints bare `deployment` without the
    `tests/integration/deployment` path segment and would pass vacuously. `-qq` forces
    the flat `path::test` listing regardless of `addopts`, matching what the criterion's
    `--collect-only -q` is actually checking for.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-qq"],
        cwd=INGESTION_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert "tests/integration/deployment" not in result.stdout
    assert "tests\\integration\\deployment" not in result.stdout
