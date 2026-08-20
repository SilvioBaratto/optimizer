"""Example test for scope-2: ``app.worker`` imports cleanly in the same
fastapi-free venv, with no ``ImportError``.

Source-blind by construction: the import happens inside a *subprocess*
running the fresh venv's interpreter — the test process itself never
imports ``app.worker`` or any other ``ingestion/app`` module.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_INGESTION_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.criterion("scope-2")
@pytest.mark.integration
def test_when_worker_module_is_imported_in_the_fastapi_free_venv_then_it_exits_zero_with_no_import_error(
    fresh_ingestion_venv,
):
    python_bin = fresh_ingestion_venv["python_bin"]

    result = subprocess.run(  # noqa: S603
        [str(python_bin), "-c", "import app.worker"],
        cwd=str(_INGESTION_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "ImportError" not in result.stderr
