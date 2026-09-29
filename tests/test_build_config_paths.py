"""Guard: Makefile + CI must target the src-layout optimizer path.

After the src-layout move the library lives at ``packages/portopt-core/optimizer``,
not the repo-root ``optimizer/``. A bare ``optimizer/`` lint/typecheck target
rots silently — ``ruff check optimizer/`` errors with E902 (file not found) and
so never actually lints. This guard fails if either the Makefile or the CI
workflow passes a bare ``optimizer/`` path to ruff or mypy, and asserts both
reference the real src-layout path.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MAKEFILE = _REPO_ROOT / "Makefile"
_CI_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
_SRC_LAYOUT_PATH = "packages/portopt-core/optimizer/"

# A ruff/mypy invocation whose target is the (nonexistent) bare optimizer/ path.
# ``--cov=optimizer`` (a package name, still importable) is deliberately not matched.
_STALE_TARGET = re.compile(r"\b(ruff check|ruff format(?: --check)?|mypy)\s+optimizer/")


def _stale_targets(text: str) -> list[str]:
    return _STALE_TARGET.findall(text)


@pytest.mark.parametrize("path", [_MAKEFILE, _CI_WORKFLOW], ids=["makefile", "ci"])
def test_no_bare_optimizer_lint_target(path: Path) -> None:
    """Assert the build config passes no ruff/mypy command a bare ``optimizer/``."""
    assert _stale_targets(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("path", [_MAKEFILE, _CI_WORKFLOW], ids=["makefile", "ci"])
def test_references_src_layout_optimizer_path(path: Path) -> None:
    """Assert the build config references the real src-layout optimizer path."""
    assert _SRC_LAYOUT_PATH in path.read_text(encoding="utf-8")


def test_guard_catches_a_bare_optimizer_target() -> None:
    """The matcher flags a bare ``optimizer/`` ruff target (guards the guard)."""
    assert _stale_targets("uv run ruff check optimizer/ tests/")


def test_guard_ignores_the_src_layout_path() -> None:
    """The matcher does not flag the correct src-layout mypy target."""
    assert _stale_targets("uv run mypy packages/portopt-core/optimizer/") == []
