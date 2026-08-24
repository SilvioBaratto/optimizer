"""Example tests for issue #2: drop the leftover ``fastapi`` dependency
declaration from ``ingestion/pyproject.toml``. (``ingestion/requirements.txt``
has since been removed — ``pyproject.toml`` is the daemon's single source of
truth for dependencies.)

Source-blind by construction: every assertion reads the tracked file's raw
text. No implementation module is imported or exercised.

Anchored on ``Path(__file__).resolve().parents[4]`` for the repository root
(see ``test_no_http_surface.py`` for the parents-index derivation).
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _read(relative_path: str) -> str:
    return (_REPO_ROOT / relative_path).read_text(encoding="utf-8")


@pytest.mark.criterion("scope-1")
def test_when_ingestion_pyproject_is_read_then_fastapi_dependency_is_absent():
    content = _read("ingestion/pyproject.toml")
    assert "fastapi" not in content.lower()


@pytest.mark.criterion("scope-1")
def test_when_ingestion_pyproject_is_read_then_httpx_dependency_remains():
    content = _read("ingestion/pyproject.toml")
    assert "httpx==0.28.1" in content
