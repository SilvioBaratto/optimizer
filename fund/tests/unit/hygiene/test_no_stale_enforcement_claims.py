"""Guard: docs and skills neither call the mandate unenforced nor misstate nu tiers.

The ESG exclusions, K&E instrument limits, position cap and drawdown tiers are
enforced in code by the bound toolsets. A docstring, skill or reference that still
says they are "stored only" or "enforced in Fase 7" misleads the next reader (and
the model, for skill text), and two files of one skill disagreeing on what
``nu1/nu2/nu3`` mean sends the risk agent two contradictory instructions.

Source-blind: scans raw tracked-file text, imports no implementation module.
``Path(__file__).resolve().parents[3]`` is the ``fund/`` directory.
"""

from __future__ import annotations

import re
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "fund"
_RISK_SKILL = _SRC_ROOT / "skills" / "risk-limits-check"

_STALE_CLAIM = re.compile(
    r"Fase[- ]7|display-only this phase|enforced by the ``?risk_check``? step|"
    r"stored/validated only here",
)


def _text_files() -> list[Path]:
    return [
        path
        for pattern in ("*.py", "*.md")
        for path in _SRC_ROOT.rglob(pattern)
        if "__pycache__" not in path.parts
    ]


def test_no_source_or_skill_claims_enforcement_is_still_deferred() -> None:
    offenders = [
        f"{path.relative_to(_SRC_ROOT)}: {match.group(0)!r}"
        for path in _text_files()
        for match in _STALE_CLAIM.finditer(path.read_text(encoding="utf-8"))
    ]

    assert offenders == []


def test_risk_limits_skill_files_agree_on_the_drawdown_tier_semantics() -> None:
    """``SKILL.md`` and ``reference.md`` both describe the tiers ``backtest`` gives."""
    skill = (_RISK_SKILL / "SKILL.md").read_text(encoding="utf-8")
    reference = (_RISK_SKILL / "reference.md").read_text(encoding="utf-8")

    for text in (skill, reference):
        assert "drawdown_tiers" in text
        assert "soft warning" in text
        assert "hard ceiling" in text
        assert "absolute stop" in text
