"""Example test for issue #3, criterion scope-4.

After purging the residual FastAPI references from ``ingestion/pytest.ini``
and ``ingestion/ruff.toml`` and deleting the dead smoke fixture, both
``ruff check ingestion/`` and ``ruff format --check ingestion/`` must be
clean.

Source-blind by construction: shells out to the ``ruff`` binary on the
tracked tree, no implementation module is imported or exercised. Skips if
``ruff`` is not on PATH rather than failing the environment check.

Anchored on ``Path(__file__).resolve().parents[4]`` for the repository root:
from ``ingestion/tests/unit/hygiene/test_ruff_clean_after_fastapi_purge.py``,
parents[0]=hygiene, [1]=unit, [2]=tests, [3]=ingestion, [4]=repository root.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.criterion("scope-4")
def test_when_ingestion_tree_is_linted_then_ruff_check_and_format_are_clean():
    ruff = shutil.which("ruff")
    if ruff is None:
        pytest.skip("ruff executable not found on PATH")

    check = subprocess.run(  # noqa: S603
        [ruff, "check", "ingestion/"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    fmt = subprocess.run(  # noqa: S603
        [ruff, "format", "--check", "ingestion/"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert check.returncode == 0, check.stdout + check.stderr
    assert fmt.returncode == 0, fmt.stdout + fmt.stderr
