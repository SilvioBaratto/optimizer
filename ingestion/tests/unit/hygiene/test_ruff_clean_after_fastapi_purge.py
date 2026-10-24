"""Guard: ``ruff check ingestion/`` and ``ruff format --check ingestion/`` are
both clean.

Source-blind by construction: runs the project's ``ruff`` (the one installed
beside the interpreter, not whichever ``ruff`` is first on PATH) on the tracked
tree, no implementation module is imported or exercised. Skips if ruff is not
installed rather than failing the environment check.

Anchored on ``Path(__file__).resolve().parents[4]`` for the repository root:
from ``ingestion/tests/unit/hygiene/test_ruff_clean_after_fastapi_purge.py``,
parents[0]=hygiene, [1]=unit, [2]=tests, [3]=ingestion, [4]=repository root.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.criterion("scope-4")
def test_when_ingestion_tree_is_linted_then_ruff_check_and_format_are_clean():
    if importlib.util.find_spec("ruff") is None:
        pytest.skip("ruff is not installed in this environment")
    ruff = [sys.executable, "-m", "ruff"]

    check = subprocess.run(  # noqa: S603
        [*ruff, "check", "ingestion/"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    fmt = subprocess.run(  # noqa: S603
        [*ruff, "format", "--check", "ingestion/"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert check.returncode == 0, check.stdout + check.stderr
    assert fmt.returncode == 0, fmt.stdout + fmt.stderr


@pytest.mark.criterion("scope-4")
def test_ruff_guard_does_not_depend_on_which_ruff_is_first_on_path(
    monkeypatch, tmp_path
):
    """A foreign ``ruff`` earlier on PATH must not decide the guard's verdict.

    Different ruff versions disagree about the tree (a newer one accepts what an
    older one flags), so the guard has to use the ruff the project pins, i.e.
    the one installed next to the interpreter running the suite.
    """
    stub = tmp_path / "ruff"
    stub.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    test_when_ingestion_tree_is_linted_then_ruff_check_and_format_are_clean()
