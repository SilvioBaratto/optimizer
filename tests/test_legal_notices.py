"""Guard: the README carries the legal notices and every package declares one licence.

The project is source-available under PolyForm Noncommercial 1.0.0, which the Open
Source Definition does not accept, and it is research software run by a private
individual: the README has to say so, and it must not promise anything the code does
not do (a broker connection, an advice service). A wheel built from ``portopt``,
``portopt-fund`` or ``portopt-db`` carries no licence terms unless its own
``pyproject.toml`` declares them.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path

import pytest
import tomllib

_REPO_ROOT = Path(__file__).resolve().parents[1]
_README_PATH = _REPO_ROOT / "README.md"
_PYPROJECTS = {
    "portopt-core": _REPO_ROOT / "pyproject.toml",
    "portopt": _REPO_ROOT / "ingestion" / "pyproject.toml",
    "portopt-fund": _REPO_ROOT / "fund" / "pyproject.toml",
    "portopt-db": _REPO_ROOT / "packages" / "portopt-db" / "pyproject.toml",
}
_LICENCE_CLASSIFIER = "License :: Other/Proprietary License"

# Short, stable phrases of the Notices section, each a promise the README must keep.
_NOTICE_PINS = {
    "llm provider": "to the LLM provider you configure",
    "third-party terms": "You, not the project, are the party to their terms",
    "yahoo acknowledgement": "asks you to acknowledge them",
    "no affiliation": "not affiliated with, endorsed by or sponsored by Yahoo",
    "news publishers": "news publishers' own sites",
    "publisher terms": "each publisher's terms apply to what you fetch",
    "no hosted service": "No hosted service, no fees",
}


@cache
def _readme() -> str:
    """Return the README text; a missing file fails the calling test, not collection."""
    assert _README_PATH.is_file(), f"{_README_PATH} does not exist"
    return _README_PATH.read_text(encoding="utf-8")


def _flat(text: str) -> str:
    """Collapse whitespace so a phrase wrapped over two README lines still matches."""
    return " ".join(text.split())


def _section(heading: str) -> str:
    """Return the flattened body of the ``## <heading>`` section, up to the next one."""
    pattern = rf"^## {re.escape(heading)}\s*$(.*?)(?=^## |\Z)"
    match = re.search(pattern, _readme(), re.M | re.S)
    assert match, f"README.md has no '## {heading}' section"
    return _flat(match.group(1))


@pytest.mark.criterion("REQ-OPS-003")
def test_readme_has_notices_section_with_not_advice_statement() -> None:
    """Validates README.md: the notices say the output is not advice, up front."""
    assert "not investment advice" in _section("Notices").lower()


@pytest.mark.criterion("REQ-OPS-003")
def test_readme_states_no_broker_execution() -> None:
    """Validates README.md: ``fund`` keeps paper orders only and the notices say so."""
    notices = _section("Notices")
    assert "places no orders with any broker" in notices
    assert "paper" in notices


@pytest.mark.criterion("REQ-OPS-003")
def test_readme_describes_licence_as_source_available_not_open_source() -> None:
    """Validates README.md: PolyForm Noncommercial is not an OSI licence."""
    for heading in ("Notices", "License"):
        assert "source-available" in _section(heading).lower(), heading
    assert not re.search(r"open\s*-?\s*source", _flat(_readme()), re.I)


def _missing_pins(notices: str) -> list[str]:
    """Return the labels of the pinned phrases ``notices`` no longer contains."""
    return sorted(
        label for label, phrase in _NOTICE_PINS.items() if phrase not in notices
    )


@pytest.mark.criterion("REQ-OPS-003")
def test_readme_notices_keep_every_pinned_statement() -> None:
    """Validates README.md: the notices still say where your data goes, whose terms
    apply to the data the software fetches, and that there is no hosted service and
    no fee."""
    assert _missing_pins(_section("Notices")) == []


@pytest.mark.criterion("REQ-OPS-003")
@pytest.mark.parametrize("label", sorted(_NOTICE_PINS))
def test_a_notice_deleted_from_a_copy_is_reported(label: str) -> None:
    """Validates ``_missing_pins`` on a copy of the README notices: deleting one
    pinned statement reports exactly that statement."""
    notices = _section("Notices").replace(_NOTICE_PINS[label], "")

    assert _missing_pins(notices) == [label]


@pytest.mark.criterion("REQ-OPS-004")
def test_member_pyprojects_declare_the_same_licence_as_root() -> None:
    """Validates the four pyproject.toml files: one licence text and classifier each."""
    declared = {}
    for name, path in _PYPROJECTS.items():
        project = tomllib.loads(path.read_text(encoding="utf-8"))["project"]
        declared[name] = (
            project.get("license", {}).get("text"),
            project.get("classifiers", []),
        )

    root_text, _ = declared["portopt-core"]
    assert root_text == "PolyForm-Noncommercial-1.0.0"
    for name, (text, classifiers) in declared.items():
        assert text == root_text, name
        assert _LICENCE_CLASSIFIER in classifiers, name
