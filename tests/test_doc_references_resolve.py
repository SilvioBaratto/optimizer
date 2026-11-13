"""Guard: tracked files do not point at design documents that are not in the repo.

The architecture and decision records used to live in the git-ignored ``todo/``
folder, so every pointer to them led to a file that exists only on one machine, and
three further names (``SPEC.md``, ``deep_agent.md``, ``tasks/plan.md``) were cited
for years although they were never committed. The records now live in ``docs/``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

_DESIGN_DOCS = (
    "ARCHITECTURE.md",
    "DECISIONS.md",
    "SPEC-RECONCILIATION.md",
    "VALIDATION.md",
    "WRITE-PATH-AUDIT.md",
)

_NEVER_COMMITTED = re.compile(r"SPEC\.md|deep_agent\.md|tasks/plan\.md")
_MOVED_FROM_TODO = re.compile(
    "todo/(?:" + "|".join(re.escape(name) for name in _DESIGN_DOCS) + ")"
)

# This file names the patterns; .gitignore lists the ignored SPEC.md; CHANGELOG.md
# records the removal of these very names; the two skill files are read by the
# profiler agent and ``test_profiler_prompt.py`` asserts that they cite
# ``deep_agent`` as the source of their theory line references, so rewording them
# would change a prompt, not documentation.
_EXEMPT_FILES = {
    ".gitignore",
    "CHANGELOG.md",
    "tests/test_doc_references_resolve.py",
    "fund/src/fund/skills/mifid-profiling/SKILL.md",
    "fund/src/fund/skills/mifid-profiling/reference.md",
}
_EXEMPT_PREFIXES = ("optimizerwiki/",)


def _tracked_files() -> list[str]:
    git = shutil.which("git")
    assert git, "git is required to list the tracked files"
    listing = subprocess.run(  # noqa: S603 - fixed argv, no user input
        [git, "ls-files"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return listing.stdout.splitlines()


def _offending_lines(pattern: re.Pattern[str]) -> list[str]:
    offenders: list[str] = []
    for name in _tracked_files():
        if name in _EXEMPT_FILES or name.startswith(_EXEMPT_PREFIXES):
            continue
        try:
            text = (_REPO_ROOT / name).read_text(encoding="utf-8")
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"{name}:{number}: {line.strip()[:100]}")
    return offenders


def test_design_documents_are_tracked_under_docs() -> None:
    """The architecture and decision records are versioned, not machine-local."""
    tracked = set(_tracked_files())

    missing = [name for name in _DESIGN_DOCS if f"docs/{name}" not in tracked]

    assert missing == []


def test_no_tracked_file_cites_a_document_that_was_never_committed() -> None:
    """``SPEC.md``, ``deep_agent.md`` and ``tasks/plan.md`` do not exist."""
    assert _offending_lines(_NEVER_COMMITTED) == []


def test_no_tracked_file_uses_the_old_todo_path_of_a_moved_document() -> None:
    """Moved records are cited as ``docs/...``, never ``todo/...``."""
    assert _offending_lines(_MOVED_FROM_TODO) == []
