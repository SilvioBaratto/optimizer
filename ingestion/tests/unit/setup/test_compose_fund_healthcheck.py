"""Compose `fund` healthcheck contract (task T7).

The fund service reports healthy only once the single portopt-db Alembic tree is
at head, so `docker compose --profile fund up -d --wait` gates on migrations
rather than merely on the process being alive. A `start_period` grace keeps the
pre-migration window from flipping the container to unhealthy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

# unit -> setup -> tests -> ingestion -> repo root (docker-compose.yml owner).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPOSE = _REPO_ROOT / "docker-compose.yml"


def _fund_healthcheck() -> dict:
    doc = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))
    return doc["services"]["fund"].get("healthcheck", {})


def _test_command() -> str:
    """The healthcheck `test` flattened to a single string for substring checks."""
    test = _fund_healthcheck().get("test", [])
    return " ".join(test) if isinstance(test, list) else str(test)


def test_fund_service_declares_a_healthcheck() -> None:
    """Without a healthcheck, `up --wait` returns as soon as the process starts."""
    assert _fund_healthcheck().get("test")


def test_healthcheck_encodes_alembic_at_head() -> None:
    """Readiness = Alembic at head: `alembic current` annotates the revision
    `(head)` only once migrations completed."""
    cmd = _test_command()
    assert "alembic current" in cmd
    assert "(head)" in cmd


def test_healthcheck_runs_from_the_migration_owner_dir() -> None:
    """alembic.ini lives in packages/portopt-db; the probe must cd there (same as
    the entrypoint) or alembic finds no config."""
    assert "packages/portopt-db" in _test_command()


def test_healthcheck_has_a_start_period_grace() -> None:
    """The grace window prevents the pre-migration state from counting as a
    failure and marking the container unhealthy."""
    assert _fund_healthcheck().get("start_period")


def test_healthcheck_bounds_its_retries() -> None:
    """A finite retry count lets a genuinely stuck fund eventually go unhealthy."""
    assert _fund_healthcheck().get("retries")
