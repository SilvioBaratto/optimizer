"""Example test for the cross-cycle guard: no stale intra-repository
``api/`` path survives, outside the recorded allowlist.

Source-blind by construction: every assertion reads tracked files' raw text
and checks for the absence of a dead path string. No implementation module
is imported or exercised.

Anchored on ``Path(__file__).resolve().parents[4]`` for the repository root:
from ``ingestion/tests/unit/hygiene/test_no_legacy_api_references.py``,
parents[0]=hygiene, [1]=unit, [2]=tests, [3]=ingestion, [4]=repository root.
Never anchored on ``Path.cwd()`` — CI may invoke pytest from either the repo
root or ``ingestion/``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.unit.hygiene._shared_scan import _iter_git_tracked_relpaths

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SELF = Path(__file__).resolve()

# Matches only paths that can only be an internal repository path
# (`api/v1/...`, `api/app/...`, etc.), never a third-party URL segment like
# `/api/v0/...` or `discord.com/api/webhooks`.
_LEGACY_API_PATTERN = re.compile(
    r"(?<![\w./-])api/(v1/|app/|scripts/|tests/|requirements\.txt)"
)

# (path, line-substring, reason). Matched by exact path + substring-in-line,
# so an allowlist entry only silences the one documented occurrence, not the
# whole file. The last three are documented as *currently non-matching* —
# each is `/api/v1/...` with a leading slash the lookbehind blocks — kept
# anyway as defence in depth against a future regex loosening.
_ALLOWLIST: tuple[tuple[str, str, str], ...] = (
    ("CHANGELOG.md", "api/requirements.txt", "historical record"),
    (
        "CLAUDE.md",
        "/api/v1/",
        "the sentence that teaches how to find leftovers "
        "(currently non-matching: leading slash blocks the lookbehind)",
    ),
    (
        "tests/scheduler/test_shell_wrappers.py",
        "/api/v1",
        "itself a guard asserting absence "
        "(currently non-matching: leading slash blocks the lookbehind)",
    ),
    (
        "README.md",
        "api/v1/",
        "CI/PyPI badge link "
        "(currently non-matching: leading slash blocks the lookbehind)",
    ),
)

_EXCLUDE_DIR_PARTS = {"todo", ".code-generator"}
# The whole hygiene package asserts *absence* of dead patterns by
# construction — its own reason strings and assertions quote the very
# strings this guard looks for, so it must not scan itself or its siblings.
_EXCLUDE_DIR_NAMES = {"hygiene"}


def _is_allowlisted(rel: str, line: str) -> bool:
    return any(rel == path and substring in line for path, substring, _ in _ALLOWLIST)


def _iter_all_relpaths(root: Path) -> list[str]:
    return [
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    ]


def _iter_repo_files(root: Path) -> Iterator[str]:
    # Only tracked files when root is a git checkout — generated/untracked
    # artefacts (build output, scratch directories) are not "repository
    # paths" for this guard. A root with no git repository (a synthetic
    # test tree) scans every file instead.
    relpaths = _iter_git_tracked_relpaths(root)
    if relpaths is None:
        relpaths = _iter_all_relpaths(root)
    for rel in relpaths:
        if (root / rel).resolve() == _SELF:
            continue
        parts = Path(rel).parts
        if any(part in _EXCLUDE_DIR_PARTS for part in parts):
            continue
        if any(part in _EXCLUDE_DIR_NAMES for part in parts[:-1]):
            continue
        yield rel


def find_legacy_api_violations(root: Path) -> list[str]:
    """Scan ``root`` for stale intra-repository ``api/`` path references.

    Injectable-root sibling of the module-level real-repository scan (issue
    #7): callers outside this module — the cross-guard checkpoint — point
    the scan at a synthetic tree instead of the real repository root.

    Args:
        root: Directory tree to scan.

    Returns:
        One relative path string per offending file, empty when clean.
    """
    offending: list[str] = []
    for rel in _iter_repo_files(root):
        path = root / rel
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line in text.splitlines():
            if _LEGACY_API_PATTERN.search(line) and not _is_allowlisted(rel, line):
                offending.append(rel)
                break
    return offending


@pytest.mark.criterion("global-9")
def test_when_repository_is_scanned_then_no_unallowlisted_legacy_api_reference_remains():
    assert find_legacy_api_violations(_REPO_ROOT) == []


# Folded from the deleted test_legacy_api_reference_allowlist_covers_known_files.py
# (issue #15): the unique assertion it added over the guard above was that the
# three known residual matches are the *reason* the whole-repo scan stays clean,
# not an accident of them being absent from the tree.
_KNOWN_ALLOWLISTED_FILES = (
    "CHANGELOG.md",
    "CLAUDE.md",
    "tests/scheduler/test_shell_wrappers.py",
)


@pytest.mark.criterion("scope-3")
def test_when_the_known_allowlisted_files_are_scanned_then_none_are_reported():
    for relative_path in _KNOWN_ALLOWLISTED_FILES:
        assert (_REPO_ROOT / relative_path).is_file(), (
            f"expected {relative_path} to exist so the allowlist has a "
            "known residual match to cover"
        )

    violations = find_legacy_api_violations(_REPO_ROOT)

    for relative_path in _KNOWN_ALLOWLISTED_FILES:
        assert relative_path not in violations
