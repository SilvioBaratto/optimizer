"""Chain-level guards for the portopt-db Alembic tree.

The tree is the single migration owner, so it must keep exactly one head and
CLAUDE.md must name that head.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from alembic.script import ScriptDirectory

_PKG_ROOT = Path(__file__).resolve().parents[1]
_REPO_CLAUDE_MD = _PKG_ROOT.parents[1] / "CLAUDE.md"


def test_alembic_chain_has_exactly_one_head() -> None:
    """A second head makes ``alembic upgrade head`` ambiguous and fails."""
    script = ScriptDirectory(str(_PKG_ROOT / "alembic"))

    assert len(script.get_heads()) == 1


def test_claude_md_head_notes_name_the_real_head() -> None:
    """CLAUDE.md quotes the head twice; a stale quote misleads every new session."""
    if not _REPO_CLAUDE_MD.is_file():
        pytest.skip("CLAUDE.md is not part of this checkout")
    (head,) = ScriptDirectory(str(_PKG_ROOT / "alembic")).get_heads()

    quoted = re.findall(
        r"\bhead:?\s+\**`?([0-9a-f]{12})\b",
        _REPO_CLAUDE_MD.read_text(encoding="utf-8"),
    )

    assert quoted, "CLAUDE.md no longer states the alembic head"
    assert set(quoted) == {head}
