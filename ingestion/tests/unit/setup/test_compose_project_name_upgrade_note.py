"""Upgrade note for the pinned compose project name.

Pinning `name: optimizer` only matches the volume of installs whose checkout was
named `optimizer`. An install under another basename (a GitHub zip such as
`optimizer-development`) keeps its data in `<basename>_postgres_data`, so after
the pin its next `portopt start` would hit the old project's fixed
`container_name`s or come up on an empty database. The CHANGELOG has to tell
those users how to keep their data.
"""

from __future__ import annotations

from pathlib import Path

# unit -> setup -> tests -> ingestion -> repo root (CHANGELOG.md owner).
_CHANGELOG = Path(__file__).resolve().parents[4] / "CHANGELOG.md"


def _unreleased() -> str:
    text = _CHANGELOG.read_text(encoding="utf-8")
    head = text.split("## [Unreleased]", 1)[1]
    return head.split("\n## [", 1)[0]


def test_changelog_records_the_project_name_pin() -> None:
    """The pin and the volume it fixes are both named, so the entry is findable."""
    section = _unreleased()
    assert "name: optimizer" in section
    assert "optimizer_postgres_data" in section


def test_changelog_documents_upgrade_for_non_default_checkout_names() -> None:
    """Both migration routes are spelled out, with the data-destroying flag called out."""
    section = _unreleased()
    # Free the old project's fixed container names without deleting its volume.
    assert "docker compose -p" in section
    # Or keep using the old project (and so its volume) without moving data.
    assert "COMPOSE_PROJECT_NAME" in section
    # `down -v` would destroy exactly the data the note is trying to preserve.
    assert "down -v" in section
