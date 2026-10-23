"""Guard: Makefile, CI and pre-commit must target the src-layout source trees.

After the src-layout move the library lives at ``packages/portopt-core/optimizer``,
not the repo-root ``optimizer/``. A bare ``optimizer/`` lint/typecheck target
rots silently — ``ruff check optimizer/`` errors with E902 (file not found) and
so never actually lints. This guard fails if either the Makefile or the CI
workflow passes a bare ``optimizer/`` path to ruff or mypy, and asserts both
reference the real src-layout path. The pre-commit ruff hooks must match files in
every workspace source tree at the same ruff version the lock (and CI) uses, and CI
must lint every member, not only the library.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import tomllib
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MAKEFILE = _REPO_ROOT / "Makefile"
_PRE_COMMIT = _REPO_ROOT / ".pre-commit-config.yaml"
_UV_LOCK = _REPO_ROOT / "uv.lock"
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


_SOURCE_TREES = [
    "packages/portopt-core/optimizer",
    "packages/portopt-db/src/portopt_db",
    "ingestion/app",
    "fund/src/fund",
    "tests",
]
_NON_LIBRARY_MEMBERS = ["packages/portopt-db/", "ingestion/", "fund/"]


def _ruff_hooks() -> list[dict]:
    config = yaml.safe_load(_PRE_COMMIT.read_text(encoding="utf-8"))
    repo = next(r for r in config["repos"] if "ruff-pre-commit" in r["repo"])
    return [repo | {"hook": h} for h in repo["hooks"]]


@pytest.mark.parametrize("tree", _SOURCE_TREES)
def test_pre_commit_ruff_hooks_match_each_source_tree(tree: str) -> None:
    """A hook whose ``files`` regex matches nothing real lints nothing."""
    sample = next((_REPO_ROOT / tree).rglob("*.py")).relative_to(_REPO_ROOT)
    for entry in _ruff_hooks():
        pattern = entry["hook"]["files"]
        assert re.search(pattern, sample.as_posix()), (
            f"{entry['hook']['id']} files={pattern!r} skips {sample}"
        )


def test_pre_commit_ruff_rev_matches_the_locked_ruff() -> None:
    """The hook and CI/``uv run ruff`` must format identically."""
    lock = tomllib.loads(_UV_LOCK.read_text(encoding="utf-8"))
    locked = next(p["version"] for p in lock["package"] if p["name"] == "ruff")
    assert {entry["rev"] for entry in _ruff_hooks()} == {f"v{locked}"}


@pytest.mark.parametrize("member", _NON_LIBRARY_MEMBERS)
@pytest.mark.parametrize("command", ["ruff check", "ruff format --check"])
def test_ci_lints_every_workspace_member(member: str, command: str) -> None:
    """Violations in the daemon, fund bridge and DB layer must fail CI too."""
    workflow = yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8"))
    runs = [
        step["run"]
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if command in step.get("run", "")
    ]
    assert any(member in run for run in runs)
