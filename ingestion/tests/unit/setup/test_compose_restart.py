"""Compose `restart:` policy contract.

`depends_on: condition: service_healthy` only orders the initial `up`; after a
host reboot or a Docker daemon restart Docker revives each container on its own
restart policy. A daemon with `restart: unless-stopped` whose database has the
default policy (`no`) comes back alone and waits forever in the entrypoint's
`pg_isready` loop, silently skipping every scheduled job.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

# unit -> setup -> tests -> ingestion -> repo root (docker-compose.yml owner).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPOSE = _REPO_ROOT / "docker-compose.yml"


def _services() -> dict[str, dict]:
    doc = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))
    return doc["services"]


def _restarts(service: dict) -> bool:
    """Whether Docker revives the service on its own (compose default is `no`)."""
    return str(service.get("restart", "no")) != "no"


def _dependencies(service: dict) -> set[str]:
    dep = service.get("depends_on", {})
    return set(dep) if isinstance(dep, (dict, list)) else set()


def test_database_restarts_with_the_daemons_that_depend_on_it() -> None:
    """Every dependency of an auto-restarting service must itself auto-restart."""
    services = _services()
    stranded = sorted(
        f"{name} -> {dep}"
        for name, svc in services.items()
        if _restarts(svc)
        for dep in _dependencies(svc)
        if not _restarts(services[dep])
    )
    assert not stranded, f"restart policy leaves dependencies down: {stranded}"


def test_database_survives_a_daemon_restart_unless_stopped_by_the_user() -> None:
    """`unless-stopped` (not `always`) keeps `portopt stop` authoritative."""
    assert _services()["db"].get("restart") == "unless-stopped"
