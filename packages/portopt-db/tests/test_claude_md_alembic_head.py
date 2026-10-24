"""Guard: the Alembic head quoted in the root ``CLAUDE.md`` is the real head.

``CLAUDE.md`` states the migration head as part of the repo contract, so a
migration that lands without updating it leaves agents and reviewers working
from a stale revision id. The check reads the single head from the alembic
script tree and asserts every ``head ...`` mention in the doc names it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory

# tests/ -> portopt-db/ -> packages/ -> repo root
_PKG_ROOT = Path(__file__).resolve().parents[1]
_CLAUDE_MD = Path(__file__).resolve().parents[3] / "CLAUDE.md"

# Matches "head: <rev>" and "head `<rev>`" (optionally bold), the two spellings
# used in CLAUDE.md.
_HEAD_MENTION = re.compile(r"head(?::\s*|\s+)\*{0,2}`?([0-9a-f]{12})`?", re.IGNORECASE)


def _alembic_head() -> str:
    config = Config()
    config.set_main_option("script_location", str(_PKG_ROOT / "alembic"))
    heads = ScriptDirectory.from_config(config).get_heads()
    assert len(heads) == 1, f"expected a single alembic head, got {heads}"
    return heads[0]


@pytest.mark.skipif(not _CLAUDE_MD.is_file(), reason="repo-root CLAUDE.md not present")
def test_claude_md_head_mentions_match_alembic_head() -> None:
    """Assert every alembic head quoted in CLAUDE.md equals the real head."""
    head = _alembic_head()
    mentions = _HEAD_MENTION.findall(_CLAUDE_MD.read_text(encoding="utf-8"))

    assert mentions, "CLAUDE.md no longer states the alembic head; update this guard"
    stale = sorted({m for m in mentions if m != head})
    assert not stale, (
        f"CLAUDE.md names stale alembic head(s) {stale}; real head is {head}"
    )
