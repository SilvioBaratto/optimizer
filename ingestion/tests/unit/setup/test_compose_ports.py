"""Compose host-port exposure contract.

The stack ships a well-known superuser password (``postgres:postgres``) and an
unauthenticated Adminer + metrics endpoint, so none of their published ports may
bind a non-loopback host interface: a bare ``"54320:5432"`` binds ``0.0.0.0`` and
hands the database to everyone on the same network.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

# unit -> setup -> tests -> ingestion -> repo root (docker-compose.yml owner).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPOSE = _REPO_ROOT / "docker-compose.yml"
_LOOPBACK = {"127.0.0.1", "[::1]"}


def _published_ports() -> list[tuple[str, str | None]]:
    """(service, host IP or None) for every published port mapping."""
    doc = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))
    found: list[tuple[str, str | None]] = []
    for name, svc in doc["services"].items():
        for port in svc.get("ports", []):
            if isinstance(port, dict):
                found.append((name, port.get("host_ip")))
                continue
            # Resolve ``${VAR:-default}`` first: its ``:-`` would otherwise be split.
            resolved = re.sub(r"\$\{\w+:-([^}]*)\}", r"\1", str(port))
            parts = resolved.rsplit(":", 2)
            found.append((name, parts[0] if len(parts) == 3 else None))
    return found


def test_every_published_port_binds_loopback_only() -> None:
    exposed = [name for name, host_ip in _published_ports() if host_ip not in _LOOPBACK]
    assert not exposed, f"services publishing on all interfaces: {exposed}"


def test_the_stack_still_publishes_db_adminer_and_metrics() -> None:
    """Guard against 'fixing' exposure by silently dropping a port."""
    assert {name for name, _ in _published_ports()} == {"db", "adminer", "scheduler"}
