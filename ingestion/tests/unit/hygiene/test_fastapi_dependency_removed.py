"""Guards ``fastapi`` is absent from ``ingestion/pyproject.toml``.
``pyproject.toml`` is the daemon's single source of truth for dependencies.

Source-blind by construction: every assertion reads the tracked file's raw
text. No implementation module is imported or exercised.

Anchored on ``Path(__file__).resolve().parents[4]`` for the repository root
(see ``test_no_http_surface.py`` for the parents-index derivation).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import tomllib

_REPO_ROOT = Path(__file__).resolve().parents[4]
_REQUIREMENT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _read(relative_path: str) -> str:
    return (_REPO_ROOT / relative_path).read_text(encoding="utf-8")


def declared_runtime_dependencies(manifest_text: str) -> set[str]:
    """Return the lower-cased distribution names in ``[project].dependencies``.

    Compares names only, so a version bump or an added extra/marker never
    changes the answer, and a lookalike such as ``httpx-sse`` is a different
    name rather than a substring hit.

    Args:
        manifest_text: Raw contents of a ``pyproject.toml``.

    Returns:
        The declared names, empty when the manifest has no runtime dependencies.
    """
    requirements = (
        tomllib.loads(manifest_text).get("project", {}).get("dependencies", [])
    )
    names: set[str] = set()
    for requirement in requirements:
        match = _REQUIREMENT_NAME.match(requirement)
        if match:
            names.add(match.group(0).lower())
    return names


@pytest.mark.criterion("scope-1")
def test_when_ingestion_pyproject_is_read_then_fastapi_dependency_is_absent():
    content = _read("ingestion/pyproject.toml")
    assert "fastapi" not in content.lower()


@pytest.mark.criterion("scope-1")
def test_when_ingestion_pyproject_is_read_then_httpx_dependency_remains():
    content = _read("ingestion/pyproject.toml")
    assert "httpx" in declared_runtime_dependencies(content)


@pytest.mark.parametrize(
    "requirement", ["httpx==0.28.2", "httpx>=0.28", "httpx[http2]==0.28.1", "HTTPX"]
)
def test_when_httpx_is_declared_in_any_form_then_it_is_detected(requirement):
    manifest = f'[project]\ndependencies = ["{requirement}"]\n'
    assert "httpx" in declared_runtime_dependencies(manifest)


def test_when_only_a_lookalike_distribution_is_declared_then_httpx_is_not_detected():
    manifest = '[project]\ndependencies = ["httpx-sse==0.4.0", "python-httpx"]\n'
    assert "httpx" not in declared_runtime_dependencies(manifest)
