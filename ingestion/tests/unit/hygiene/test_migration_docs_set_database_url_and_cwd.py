"""Guard: every documented ``alembic`` command is runnable as written.

``packages/portopt-db/alembic/env.py`` reads only the ``DATABASE_URL`` process
variable (no dotenv) and raises before touching a database when it is unset, and
``alembic.ini`` lives in ``packages/portopt-db`` so alembic only finds its
``script_location`` when run from there. A snippet that omits either fails on a
fresh checkout, so each fenced block that invokes alembic must set
``DATABASE_URL`` and change into ``packages/portopt-db`` on or before that line.
An inline-code mention in prose has no preceding lines to lean on, so each such
span must carry both on its own.

Source-blind by construction: reads the raw text of tracked Markdown files.

Anchored on ``Path(__file__).resolve().parents[4]`` for the repository root:
from ``ingestion/tests/unit/hygiene/<this file>``, parents[0]=hygiene, [1]=unit,
[2]=tests, [3]=ingestion, [4]=repository root.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]

_DOCS = (
    "README.md",
    "CLAUDE.md",
    "ingestion/README.md",
    "packages/portopt-db/README.md",
    ".claude/skills/db-context/SKILL.md",
)

_ALEMBIC_COMMAND = re.compile(
    r"\balembic\s+(upgrade|downgrade|current|revision|history|heads|stamp)\b"
)
_FENCE = re.compile(r"^\s*```")
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_SETS_DATABASE_URL = re.compile(r"\bDATABASE_URL=\S+")
_ENTERS_DB_PACKAGE = re.compile(r"\bcd\s+\S*packages/portopt-db\b")


def _fenced_blocks(text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in text.splitlines():
        if _FENCE.match(line):
            if current is None:
                current = []
            else:
                blocks.append(current)
                current = None
        elif current is not None:
            current.append(line)
    return blocks


def _alembic_commands() -> list[tuple[str, list[str], str]]:
    """Return ``(doc, block lines up to and including the command, command)``."""
    found: list[tuple[str, list[str], str]] = []
    for doc in _DOCS:
        text = (_REPO_ROOT / doc).read_text(encoding="utf-8")
        for block in _fenced_blocks(text):
            for idx, line in enumerate(block):
                if _ALEMBIC_COMMAND.search(line):
                    found.append((doc, block[: idx + 1], line))
    return found


def _inline_alembic_commands() -> list[tuple[str, str]]:
    """Return ``(doc, span)`` for alembic commands quoted as inline code in prose."""
    found: list[tuple[str, str]] = []
    for doc in _DOCS:
        in_fence = False
        for line in (_REPO_ROOT / doc).read_text(encoding="utf-8").splitlines():
            if _FENCE.match(line):
                in_fence = not in_fence
            elif not in_fence:
                found.extend(
                    (doc, span)
                    for span in _INLINE_CODE.findall(line)
                    if _ALEMBIC_COMMAND.search(span)
                )
    return found


_PARAMS = [
    pytest.param(doc, context, command, id=f"{doc}::{command.strip()[:48]}")
    for doc, context, command in _alembic_commands()
]

_INLINE_PARAMS = [
    pytest.param(doc, span, id=f"{doc}::inline::{span.strip()[:48]}")
    for doc, span in _inline_alembic_commands()
]


def test_every_listed_doc_documents_an_alembic_command() -> None:
    """Fails when a doc stops showing alembic, so the list cannot go stale."""
    docs_with_commands = {doc for doc, _, _ in _alembic_commands()}

    assert docs_with_commands == set(_DOCS)


@pytest.mark.parametrize(("doc", "context", "command"), _PARAMS)
def test_documented_alembic_command_sets_database_url(
    doc: str, context: list[str], command: str
) -> None:
    """The snippet exports DATABASE_URL, the only variable env.py reads."""
    prelude = "\n".join(context)

    assert _SETS_DATABASE_URL.search(prelude), (
        f"{doc}: `{command.strip()}` never sets DATABASE_URL, so env.py "
        "refuses to run the migration"
    )


@pytest.mark.parametrize(("doc", "context", "command"), _PARAMS)
def test_documented_alembic_command_runs_from_the_db_package(
    doc: str, context: list[str], command: str
) -> None:
    """The snippet changes into packages/portopt-db before invoking alembic."""
    prelude = "\n".join(context)

    assert _ENTERS_DB_PACKAGE.search(prelude), (
        f"{doc}: `{command.strip()}` is not run from packages/portopt-db, "
        "where alembic.ini lives"
    )


@pytest.mark.parametrize(("doc", "span"), _INLINE_PARAMS)
def test_inline_alembic_command_is_self_contained(doc: str, span: str) -> None:
    """A prose mention is copy-pasted alone, so it sets the URL and the cwd itself."""
    assert _SETS_DATABASE_URL.search(span) and _ENTERS_DB_PACKAGE.search(span), (
        f"{doc}: inline `{span.strip()}` must set DATABASE_URL and cd into "
        "packages/portopt-db, or a copy-paste hits env.py's RuntimeError"
    )
