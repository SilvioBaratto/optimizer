"""Guard: dependency automation must see the whole uv workspace and its lock.

The four workspace members share one ``uv.lock`` and repeat several exact ``==``
pins. A Dependabot ecosystem that reads only the root ``pyproject.toml`` (``pip``)
cannot refresh the lock, never scans the members, and bumps a repeated pin in one
file only, leaving a workspace that no longer resolves. CI must also refuse to
re-resolve silently, otherwise a stale lock passes CI while the ``--frozen``
Docker builds keep shipping the old versions.
"""

from __future__ import annotations

import collections
import re
from pathlib import Path

import pytest
import tomllib

yaml = pytest.importorskip("yaml")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEPENDABOT = _REPO_ROOT / ".github" / "dependabot.yml"
_CI_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
_ROOT_PYPROJECT = _REPO_ROOT / "pyproject.toml"

_EXACT_PIN = re.compile(r"^\s*([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?\s*==")


def _updates() -> list[dict]:
    return yaml.safe_load(_DEPENDABOT.read_text(encoding="utf-8"))["updates"]


def _directories(update: dict) -> set[str]:
    dirs = set(update.get("directories", []))
    if "directory" in update:
        dirs.add(update["directory"])
    return dirs


def _covered(ecosystem: str) -> set[str]:
    return {
        d
        for u in _updates()
        if u["package-ecosystem"] == ecosystem
        for d in _directories(u)
    }


def _member_dirs() -> list[str]:
    """Workspace members as Dependabot paths, root first."""
    doc = tomllib.loads(_ROOT_PYPROJECT.read_text(encoding="utf-8"))
    return ["/"] + ["/" + m for m in doc["tool"]["uv"]["workspace"]["members"]]


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _repeated_exact_pins() -> set[str]:
    """Package names pinned with ``==`` in more than one workspace member."""
    owners: dict[str, set[str]] = collections.defaultdict(set)
    for member in _member_dirs():
        path = _REPO_ROOT / member.lstrip("/") / "pyproject.toml"
        project = tomllib.loads(path.read_text(encoding="utf-8"))["project"]
        requirements = list(project.get("dependencies", []))
        for extra in project.get("optional-dependencies", {}).values():
            requirements.extend(extra)
        for req in requirements:
            if match := _EXACT_PIN.match(req):
                owners[_normalise(match.group(1))].add(member)
    return {name for name, members in owners.items() if len(members) > 1}


def test_uv_ecosystem_covers_every_workspace_member() -> None:
    """The uv ecosystem updates pyproject pins and ``uv.lock`` for each member."""
    assert set(_member_dirs()) <= _covered("uv")


def test_pip_ecosystem_is_not_used_for_the_workspace() -> None:
    """``pip`` only edits the root pyproject and cannot touch ``uv.lock``."""
    assert _covered("pip") == set()


def test_docker_ecosystem_covers_every_service_dockerfile() -> None:
    """Pinned base images and the uv binary are bumped like any other dependency."""
    dockerfile_dirs = {
        "/" + p.parent.relative_to(_REPO_ROOT).as_posix()
        for p in _REPO_ROOT.glob("*/Dockerfile")
    }
    assert dockerfile_dirs
    assert dockerfile_dirs <= _covered("docker")


def test_repeated_exact_pins_are_bumped_together() -> None:
    """A shared ``==`` pin must move in every member in the same pull request."""
    assert _repeated_exact_pins()
    grouped: set[str] = set()
    for update in _updates():
        if update["package-ecosystem"] != "uv":
            continue
        for group in update.get("groups", {}).values():
            grouped.update(_normalise(p) for p in group.get("patterns", []))
    assert _repeated_exact_pins() <= grouped


def test_ci_never_re_resolves_the_lock() -> None:
    """Every ``uv sync`` in CI asserts the committed lock is current."""
    workflow = yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8"))
    syncs = [
        step["run"]
        for job in workflow["jobs"].values()
        for step in job["steps"]
        if "uv sync" in step.get("run", "")
    ]
    assert syncs
    assert [s for s in syncs if "--locked" not in s] == []
