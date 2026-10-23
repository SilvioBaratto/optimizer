"""Compose registry-image pinning contract.

Images pulled from a registry (as opposed to built from this repo) must carry an
explicit version tag. A floating `latest`/untagged reference silently swaps the
code behind a database-superuser UI on the next `docker compose pull`, and makes
two installs of the same checkout run different software.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

# unit -> setup -> tests -> ingestion -> repo root (docker-compose.yml owner).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPOSE = _REPO_ROOT / "docker-compose.yml"

_FLOATING_TAGS = {"", "latest", "stable"}


def _services() -> dict[str, dict]:
    doc = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))
    return doc["services"]


def _tag(image: str) -> str:
    """Tag of an image reference, or "" when it names no tag (implicit `latest`)."""
    name = image.split("@", 1)[0]
    last = name.rsplit("/", 1)[-1]
    return last.split(":", 1)[1] if ":" in last else ""


def test_registry_images_are_pinned_to_a_version() -> None:
    """A pulled (non-`build:`) image on a floating tag is not reproducible."""
    floating = {
        name: svc["image"]
        for name, svc in _services().items()
        if "image" in svc
        and "build" not in svc
        and "@sha256:" not in svc["image"]
        and _tag(svc["image"]) in _FLOATING_TAGS
    }
    assert not floating, f"registry images on a floating tag: {floating}"
