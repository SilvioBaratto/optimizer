"""Example test for scope-1: a fresh venv installs ``requirements.txt``
without pulling in ``fastapi``.

Source-blind by construction: exercises the real ``pip`` CLI inside a real,
disposable virtualenv against the real ``ingestion/requirements.txt`` — no
mock of pip, no assumption about what the file currently contains.
"""

from __future__ import annotations

import subprocess

import pytest


@pytest.mark.criterion("scope-1")
@pytest.mark.integration
def test_when_requirements_are_installed_in_a_fresh_venv_then_pip_show_reports_fastapi_absent(
    fresh_ingestion_venv,
):
    install_result = fresh_ingestion_venv["install_result"]
    assert install_result.returncode == 0, install_result.stderr

    python_bin = fresh_ingestion_venv["python_bin"]
    show_result = subprocess.run(  # noqa: S603
        [str(python_bin), "-m", "pip", "show", "fastapi"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert show_result.returncode != 0
