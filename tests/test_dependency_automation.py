"""Guard: dependency automation must see the whole uv workspace and its lock.

The four workspace members share one ``uv.lock`` and repeat several exact ``==``
pins. A Dependabot ecosystem that reads only the root ``pyproject.toml`` (``pip``)
cannot refresh the lock, never scans the members, and bumps a repeated pin in one
file only, leaving a workspace that no longer resolves. CI must also refuse to
re-resolve silently, otherwise a stale lock passes CI while the ``--frozen``
Docker builds keep shipping the old versions.

Workspace members resolve each other only through ``[tool.uv.sources]``, which pip
ignores: a bare ``pip install`` of a member that depends on an unpublished sibling
would look that name up on PyPI. The docs must not advertise that path.
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


_DOCS = ["CLAUDE.md", "README.md", "CONTRIBUTING.md"]


def _members_resolving_siblings_via_uv_sources() -> list[str]:
    """Member directories whose internal dependencies exist only as uv sources."""
    dependents = []
    for member in _member_dirs():
        path = _REPO_ROOT / member.lstrip("/") / "pyproject.toml"
        project = tomllib.loads(path.read_text(encoding="utf-8"))
        sources = project.get("tool", {}).get("uv", {}).get("sources", {})
        if any(source.get("workspace") for source in sources.values()):
            dependents.append(member.strip("/"))
    return dependents


@pytest.mark.parametrize("doc", _DOCS)
def test_docs_never_advertise_bare_pip_installs_of_uv_only_members(doc: str) -> None:
    """Sibling dependencies are unpublished names: only uv maps them to local paths."""
    dependents = _members_resolving_siblings_via_uv_sources()
    assert dependents
    text = (_REPO_ROOT / doc).read_text(encoding="utf-8")

    assert not re.search(r"pip install[^\n]*works\s+per-package", text)
    for command in re.findall(r"pip install[^\n`]*", text):
        for member in dependents:
            assert not re.search(rf"(?<![\w-]){re.escape(member)}\b", command), command


_PIP_EDITABLE_TARGET = re.compile(
    r"""pip install[^\n`]*?\s(?:-e|--editable)\s+["']?([^\s"'`\[]+)"""
)
# A bare member directory (no deeper path) named on a line about pip installs.
_MEMBER_DIR = re.compile(r"packages/[\w-]+(?![\w/-])")


def _pip_install_targets(text: str) -> set[str]:
    """Directories a doc tells the reader to pip-install, as written in the doc."""
    targets = set(_PIP_EDITABLE_TARGET.findall(text))
    for line in text.splitlines():
        if re.search(r"pip[ -]install", line):
            targets.update(_MEMBER_DIR.findall(line))
    return targets


@pytest.mark.parametrize("doc", _DOCS)
def test_pip_install_targets_in_docs_hold_a_pyproject(doc: str) -> None:
    """pip needs a ``pyproject.toml`` there; the root one declares portopt-core."""
    text = (_REPO_ROOT / doc).read_text(encoding="utf-8")

    for target in sorted(_pip_install_targets(text)):
        assert (_REPO_ROOT / target / "pyproject.toml").is_file(), (
            f"{doc} says to pip install {target!r}, which has no pyproject.toml"
        )
