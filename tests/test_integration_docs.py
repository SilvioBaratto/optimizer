"""Guard: every external system is documented the way standard 2.10 asks.

A third-party integration needs its API, its authentication, what depends on it, a
risk analysis and diagrams of the interaction. The document also cites the source
files that implement each integration, and a citation of a renamed file is worse than
none.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DOC = _REPO_ROOT / "docs" / "integrations.md"

_SYSTEMS = (
    "Yahoo Finance",
    "FRED",
    "Trading 212",
    "Trading Economics",
    "Il Sole 24 Ore",
    "Business Insider",
    "LLM providers",
)
_LABELS = ("API", "Authentication", "Impact if down", "Risk and mitigation", "Code")
_SOURCE_PATH = re.compile(r"`((?:ingestion|fund|packages)/[A-Za-z0-9_./-]+\.py)`")


def _text() -> str:
    return _DOC.read_text(encoding="utf-8")


def _section(name: str) -> str:
    text = _text()
    start = text.index(f"\n## {name}\n")
    following = text.find("\n## ", start + 1)
    return text[start : following if following != -1 else len(text)]


def test_every_external_system_has_a_section() -> None:
    """Each system the platform talks to is named as a section heading."""
    missing = [name for name in _SYSTEMS if f"\n## {name}\n" not in _text()]

    assert missing == []


def test_every_section_states_api_auth_impact_risk_and_code() -> None:
    """The five facts a reader needs are present under each system."""
    problems = [
        f"{name}: missing '**{label}**'"
        for name in _SYSTEMS
        for label in _LABELS
        if f"**{label}**" not in _section(name)
    ]

    assert problems == []


def test_the_document_has_sequence_and_communication_diagrams() -> None:
    """Standard 2.10 asks for at least one of each, as diagrams in the repository."""
    text = _text()

    assert text.count("sequenceDiagram") >= 2
    assert "flowchart" in text


def test_every_cited_source_file_exists() -> None:
    """Paths in backticks point at files that are in the repository."""
    cited = sorted(set(_SOURCE_PATH.findall(_text())))

    missing = [path for path in cited if not (_REPO_ROOT / path).is_file()]

    assert cited, "the document should cite its source files"
    assert missing == []
