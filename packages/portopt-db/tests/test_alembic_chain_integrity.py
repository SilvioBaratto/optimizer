"""Chain-level guards for the portopt-db Alembic tree.

The tree is the single migration owner, so a table that only the ORM
(``Base.metadata.create_all``) knows about is a drift bug: a database built by
``alembic upgrade head`` alone would be missing it.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from alembic.script import ScriptDirectory

_PKG_ROOT = Path(__file__).resolve().parents[1]
_VERSIONS_DIR = _PKG_ROOT / "alembic" / "versions"
_REPO_CLAUDE_MD = _PKG_ROOT.parents[1] / "CLAUDE.md"


def _created_tables() -> set[str]:
    """Names passed as the first argument of any ``op.create_table`` call."""
    created: set[str] = set()
    for path in sorted(_VERSIONS_DIR.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "create_table"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                created.add(node.args[0].value)
    return created


def test_alembic_chain_has_exactly_one_head() -> None:
    """A second head makes ``alembic upgrade head`` ambiguous and fails."""
    script = ScriptDirectory(str(_PKG_ROOT / "alembic"))

    assert len(script.get_heads()) == 1


def test_alembic_chain_creates_bond_yields_table() -> None:
    """``bond_yields`` is a live write target and needs a creating revision."""
    assert "bond_yields" in _created_tables()


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
