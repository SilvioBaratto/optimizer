"""Compose service `profiles:` contract (task T6).

Every service carries a profile so a bare `docker compose up` starts nothing;
`--profile fund` brings up db+fund and `--profile ingestion` brings up
db+scheduler+adminer. `db` shares a profile with each service that depends on it,
so `depends_on: db` stays satisfiable under either operational profile.
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


def _profiles_of(service: dict) -> set[str]:
    return set(service.get("profiles", []))


def _started(enabled: set[str]) -> set[str]:
    """Names of services `docker compose` would start with `enabled` profiles on.

    A service with no profiles is always started; one with profiles starts only
    when it shares at least one with the enabled set.
    """
    services = _services()
    return {
        name
        for name, svc in services.items()
        if not _profiles_of(svc) or _profiles_of(svc) & enabled
    }


def _depends_on(service: dict) -> set[str]:
    dep = service.get("depends_on", {})
    return set(dep) if isinstance(dep, (dict, list)) else set()


def test_bare_up_starts_nothing() -> None:
    """No enabled profile ⇒ nothing starts, so every service must carry a profile."""
    assert _started(set()) == set()


def test_fund_profile_starts_db_and_fund() -> None:
    """`--profile fund` brings up only the database and the fund bridge."""
    assert _started({"fund"}) == {"db", "fund"}


def test_ingestion_profile_starts_db_scheduler_adminer() -> None:
    """`--profile ingestion` brings up the database, scheduler, and Adminer."""
    assert _started({"ingestion"}) == {"db", "scheduler", "adminer"}


def test_adminer_reachable_via_tools_alias() -> None:
    """adminer is a tools-only convenience surface as well as part of ingestion."""
    assert {"ingestion", "tools"} <= _profiles_of(_services()["adminer"])


def test_depends_on_valid_under_each_operational_profile() -> None:
    """Under fund and under ingestion, every started service's `depends_on`
    targets are themselves started (db shares the profile)."""
    services = _services()
    for profile in ("fund", "ingestion"):
        started = _started({profile})
        for name in started:
            missing = _depends_on(services[name]) - started
            assert not missing, f"{name} under --profile {profile} misses {missing}"
