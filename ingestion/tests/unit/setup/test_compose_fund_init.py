"""Compose `fund` init contract.

The fund container `exec`s its worker, so Python is PID 1 and nothing reaps orphans.
Every healthcheck probe that times out leaves its `alembic current` / `grep` children
reparented to the worker as zombies, and a SIGTERM that arrives during the module
imports (before `main()` installs its handlers) is dropped by a PID 1 with no handler.
`init: true` puts Docker's tini in front, which reaps and forwards signals.
"""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

# unit -> setup -> tests -> ingestion -> repo root (docker-compose.yml owner).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_COMPOSE = _REPO_ROOT / "docker-compose.yml"


def test_fund_runs_under_an_init_process() -> None:
    """The fund worker is a long-lived PID 1 that spawns healthcheck children."""
    fund = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))["services"]["fund"]
    assert fund.get("init") is True
