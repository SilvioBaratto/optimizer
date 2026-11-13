"""Guard: the ``postgres-integration`` CI job validates migrations, not only applies.

``alembic upgrade head`` alone cannot tell that a model drifted from the migrations or
that a recent migration cannot be reversed, and SQLite-backed suites never see
PostgreSQL-only behaviour. These steps are the only place those are checked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CI_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


def _job_steps() -> list[dict[str, Any]]:
    workflow = yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8"))
    return workflow["jobs"]["postgres-integration"]["steps"]


def _commands() -> list[str]:
    return [step["run"] for step in _job_steps() if "run" in step]


def test_job_checks_that_models_match_the_migrations() -> None:
    """``alembic check`` fails the build when a model changed without a migration."""
    assert any("alembic check" in command for command in _commands())


def test_job_downgrades_the_recent_window_and_upgrades_back() -> None:
    """The five revisions after the merge point must stay reversible."""
    assert any(
        "alembic downgrade 70393a606218" in command
        and "alembic upgrade head" in command
        for command in _commands()
    )


def test_job_runs_the_portopt_db_integration_suite_on_a_scratch_database() -> None:
    """The suite gets ``PORTOPT_TEST_DATABASE_URL`` naming a ``*_test`` database."""
    steps = [
        step
        for step in _job_steps()
        if "tests/integration -m integration" in step.get("run", "")
    ]
    portopt_db_steps = [step for step in steps if "packages/portopt-db" in step["run"]]
    assert portopt_db_steps, "no step runs the portopt-db integration suite"
    url = portopt_db_steps[0].get("env", {}).get("PORTOPT_TEST_DATABASE_URL", "")
    assert url.endswith("_test")
