"""Guard: the architecture decision log keeps one shape and an index.

Standard 2.14 asks for Architectural Decision Records whose application is
monitored. A log nobody can parse or find a record in stops being read, so each
record carries the same sections, a status and a date, and is listed in the index.
"""

from __future__ import annotations

import re
from pathlib import Path

_ADR_DIR = Path(__file__).resolve().parents[1] / "docs" / "adr"
_INDEX = _ADR_DIR / "README.md"

_SECTIONS = ("Context", "Decision", "Alternatives considered", "Consequences")
_STATUS = re.compile(r"^- Status: (Proposed|Accepted|Superseded by \d{4})$", re.M)
_DATE = re.compile(r"^- Date: (\d{4}-\d{2}-\d{2})\b", re.M)
_NAME = re.compile(r"^(\d{4})-[a-z0-9-]+\.md$")


def _number(path: Path) -> int:
    match = _NAME.match(path.name)
    assert match, path.name
    return int(match.group(1))


def _records() -> list[Path]:
    return sorted(p for p in _ADR_DIR.glob("*.md") if _NAME.match(p.name))


def test_the_log_has_an_index_that_explains_the_template() -> None:
    """The README names every section and the allowed statuses."""
    text = _INDEX.read_text(encoding="utf-8")

    for section in _SECTIONS:
        assert f"## {section}" in text, section
    assert "Superseded by" in text


def test_the_log_holds_the_foundational_decisions() -> None:
    """The decisions that only lived in CLAUDE.md gotchas are recorded."""
    assert len(_records()) >= 7


def test_record_numbers_start_at_one_and_have_no_gaps() -> None:
    """Numbers are the stable identity of a record, so a gap means a lost one."""
    numbers = [_number(p) for p in _records()]

    assert numbers == list(range(1, len(numbers) + 1))


def test_every_record_has_the_same_sections_a_status_and_a_date() -> None:
    """Each record is parseable: title, status, ISO date and the four sections."""
    problems: list[str] = []
    for path in _records():
        text = path.read_text(encoding="utf-8")
        number = f"{_number(path):04d}"
        if not text.startswith(f"# ADR-{number}: "):
            problems.append(f"{path.name}: title must start with '# ADR-{number}: '")
        if not _STATUS.search(text):
            problems.append(f"{path.name}: missing or invalid '- Status:' line")
        if not _DATE.search(text):
            problems.append(f"{path.name}: missing '- Date: YYYY-MM-DD' line")
        problems.extend(
            f"{path.name}: missing '## {section}'"
            for section in _SECTIONS
            if f"\n## {section}\n" not in text
        )

    assert problems == []


def test_every_record_is_listed_in_the_index() -> None:
    """A record that the index does not link cannot be found."""
    index = _INDEX.read_text(encoding="utf-8")

    unlisted = [p.name for p in _records() if p.name not in index]

    assert unlisted == []
