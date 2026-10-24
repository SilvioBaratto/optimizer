"""Compose project-name contract.

Compose derives the project name from the checkout directory unless the file
sets `name:`, and the project name prefixes the volume and network. The
containers use fixed `container_name`s, so a checkout renamed or re-cloned
under another basename would get a fresh, empty `<project>_postgres_data`
volume (the database appears wiped) while the old containers still own the
fixed names. Pinning the name keeps the volume stable across checkouts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

# unit -> setup -> tests -> ingestion -> repo root (docker-compose.yml owner).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPOSE = _REPO_ROOT / "docker-compose.yml"


def _doc() -> dict:
    return yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))


def test_project_name_is_pinned_to_optimizer() -> None:
    """Without `name:` the data volume follows the checkout directory name."""
    assert _doc().get("name") == "optimizer"


def test_fixed_container_names_share_the_project_prefix() -> None:
    """The fixed container names and the pinned project name must agree."""
    doc = _doc()
    prefix = f"{doc.get('name')}_"
    stray = {
        name: svc["container_name"]
        for name, svc in doc["services"].items()
        if "container_name" in svc and not svc["container_name"].startswith(prefix)
    }
    assert not stray, f"container_name outside the {prefix!r} prefix: {stray}"
