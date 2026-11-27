"""Guard: the README carries the legal notices and every package declares one licence.

The project is source-available under PolyForm Noncommercial 1.0.0, which the Open
Source Definition does not accept, and it is research software run by a private
individual: the README has to say so, and it must not promise anything the code does
not do (a broker connection, an advice service). The member ``pyproject.toml`` files
once carried no licence at all, so a built wheel of ``portopt``, ``portopt-fund`` or
``portopt-db`` was published without terms.
"""

from __future__ import annotations

import re
from pathlib import Path

import tomllib

_REPO_ROOT = Path(__file__).resolve().parents[1]
_README = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
_PYPROJECTS = {
    "portopt-core": _REPO_ROOT / "pyproject.toml",
    "portopt": _REPO_ROOT / "ingestion" / "pyproject.toml",
    "portopt-fund": _REPO_ROOT / "fund" / "pyproject.toml",
    "portopt-db": _REPO_ROOT / "packages" / "portopt-db" / "pyproject.toml",
}
_LICENCE_CLASSIFIER = "License :: Other/Proprietary License"


def _flat(text: str) -> str:
    """Collapse whitespace so a phrase wrapped over two README lines still matches."""
    return " ".join(text.split())


def _notices_section() -> str:
    """Return the body of the ``## Notices`` section, up to the next ``## `` heading."""
    match = re.search(r"^## Notices\s*$(.*?)(?=^## |\Z)", _README, re.M | re.S)
    assert match, "README.md has no '## Notices' section"
    return _flat(match.group(1))


def test_readme_has_notices_section_with_not_advice_statement() -> None:
    """The notices say the output is not advice, so the reader cannot miss it."""
    assert "not investment advice" in _notices_section().lower()


def test_readme_states_no_broker_execution() -> None:
    """The ``fund`` component keeps paper orders only; the README must say so."""
    notices = _notices_section()
    assert "places no orders with any broker" in notices
    assert "paper" in notices


def test_readme_describes_licence_as_source_available_not_open_source() -> None:
    """PolyForm Noncommercial is not an OSI licence, so 'open source' would mislead."""
    assert "source-available" in _README.lower()
    assert not re.search(r"open[ -]source", _README, re.I)


def test_member_pyprojects_declare_the_same_licence_as_root() -> None:
    """Every member wheel ships the root package's licence text and classifier."""
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
