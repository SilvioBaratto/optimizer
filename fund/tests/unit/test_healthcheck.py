"""Container liveness probe: healthy only while the worker keeps its heartbeat fresh.

The compose healthcheck used to ask only whether Alembic is at head, which stays true
whether or not the scheduler ever fires again. The worker now touches a file on every
monitor tick, and the probe fails once that file is missing or stale.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from fund import healthcheck

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_a_freshly_touched_heartbeat_is_healthy(tmp_path: Path) -> None:
    beat = tmp_path / "beat"

    healthcheck.touch_heartbeat(beat)

    assert healthcheck.heartbeat_is_fresh(beat, max_age_seconds=60) is True


def test_a_stale_heartbeat_is_unhealthy(tmp_path: Path) -> None:
    beat = tmp_path / "beat"
    healthcheck.touch_heartbeat(beat)
    old = beat.stat().st_mtime - 3600
    os.utime(beat, (old, old))

    assert healthcheck.heartbeat_is_fresh(beat, max_age_seconds=60) is False


def test_a_missing_heartbeat_is_unhealthy(tmp_path: Path) -> None:
    absent = tmp_path / "absent"

    assert healthcheck.heartbeat_is_fresh(absent, max_age_seconds=60) is False


def test_touching_an_unwritable_path_does_not_raise(tmp_path: Path) -> None:
    healthcheck.touch_heartbeat(tmp_path / "no-such-dir" / "beat")


def test_main_reports_the_verdict_as_an_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    beat = tmp_path / "beat"
    monkeypatch.setenv("FUND_HEARTBEAT_FILE", str(beat))
    assert healthcheck.main() == 1

    healthcheck.touch_heartbeat()

    assert healthcheck.main() == 0


def test_the_compose_healthcheck_requires_the_worker_heartbeat_and_migrations() -> None:
    compose = yaml.safe_load((_REPO_ROOT / "docker-compose.yml").read_text())
    test = compose["services"]["fund"]["healthcheck"]["test"]
    command = test if isinstance(test, str) else " ".join(test)

    assert "fund.healthcheck" in command
    assert "alembic current" in command
