"""Example test for issue #15, [scope-6]: ``tests/unit/hygiene/conftest.py``
is deleted, and the suite still collects cleanly under ``--strict-markers``.

Source-blind by construction: checks for file absence and scans the raw text
of ``ingestion/pytest.ini`` for the marker registrations that used to live in
the deleted ``conftest.py``. Never spawns a nested pytest process to prove
collection — that would violate [scope-2] — so "collects cleanly under
--strict-markers" is proven indirectly: the ``criterion`` and ``integration``
markers this suite decorates its tests with must be declared in the ini file
that ``--strict-markers`` consults, once the per-directory ``conftest.py``
that used to declare them is gone.

Ambiguity resolution: "markers registered elsewhere" is read as the project
``pytest.ini`` (the file already carrying the ``markers =`` section for this
suite), not a new conftest at a different level.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_HYGIENE_DIR = Path(__file__).resolve().parent
_INGESTION_ROOT = _HYGIENE_DIR.parents[2]
_PYTEST_INI = _INGESTION_ROOT / "pytest.ini"


@pytest.mark.criterion("scope-6")
def test_when_hygiene_directory_is_listed_then_conftest_is_absent():
    assert not (_HYGIENE_DIR / "conftest.py").exists()


@pytest.mark.criterion("scope-6")
def test_when_pytest_ini_is_read_then_criterion_marker_is_registered():
    text = _PYTEST_INI.read_text(encoding="utf-8")
    assert "criterion(id)" in text


@pytest.mark.criterion("scope-6")
def test_when_pytest_ini_is_read_then_integration_marker_is_registered():
    text = _PYTEST_INI.read_text(encoding="utf-8")
    assert "integration" in text
