"""Structural tests for the GitHub Actions smoke workflow.

Covers three artifacts:
  - ``.github/workflows/smoke.yml`` — the workflow file
  - ``scheduler/smoke.sh`` — the local-runnable smoke script invoked by the workflow
  - ``api/tests/fixtures/smoke_prices.sql`` — deterministic synthetic prices
    loaded into the DB before hitting /optimize and /backtest

Tests verify the shape of each file rather than running the full workflow,
because executing the workflow requires docker-in-docker and real GHA runners.
The goal is to catch the most common breakage modes up front: wrong endpoint
paths, hardcoded ports, missing teardown steps, forgotten seed step, etc.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

SMOKE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "smoke.yml"
SMOKE_SCRIPT = REPO_ROOT / "scheduler" / "smoke.sh"
SMOKE_FIXTURE = REPO_ROOT / "api" / "tests" / "fixtures" / "smoke_prices.sql"


# ===========================================================================
# smoke.yml — workflow
# ===========================================================================


@pytest.fixture(scope="module")
def workflow_doc() -> dict:
    return yaml.safe_load(SMOKE_WORKFLOW.read_text())


class TestSmokeWorkflowFileExists:
    def test_workflow_file_is_present(self) -> None:
        assert SMOKE_WORKFLOW.exists(), f"expected workflow at {SMOKE_WORKFLOW}"

    def test_workflow_parses_as_yaml(self, workflow_doc: dict) -> None:
        assert isinstance(workflow_doc, dict)


class TestSmokeWorkflowTriggers:
    def test_triggers_on_pull_request_to_main(self, workflow_doc: dict) -> None:
        # PyYAML parses the reserved word `on` as the Python bool True; handle both.
        on_block = workflow_doc.get("on") or workflow_doc.get(True)
        assert on_block is not None
        pr = on_block.get("pull_request")
        assert pr is not None, "expected pull_request trigger"
        branches = pr.get("branches") or []
        assert "main" in branches

    def test_has_timeout_under_twenty_minutes(self, workflow_doc: dict) -> None:
        jobs = workflow_doc["jobs"]
        smoke_job = next(iter(jobs.values()))
        timeout = smoke_job.get("timeout-minutes")
        assert timeout is not None, "smoke job must declare timeout-minutes"
        assert timeout <= 20


class TestSmokeWorkflowSteps:
    @pytest.fixture
    def job_steps(self, workflow_doc: dict) -> list[dict]:
        smoke_job = next(iter(workflow_doc["jobs"].values()))
        return smoke_job["steps"]

    def test_uses_host_port_8005_not_8000(self, job_steps: list[dict]) -> None:
        # Search the combined `run:` blocks across all steps.
        combined = _combined_run(job_steps)
        assert (
            ":8005" in combined
            or "8005/" in combined
            or '8005"' in combined
            or "localhost:8005" in combined
        ), "workflow must target host port 8005"
        # 8000 should NOT appear in smoke calls (it is the container-internal port).
        # Grep for 8000 that is not part of 18080/18081 or documentation.
        # The DB/admin port is 18081, so 8000 should not appear at all in runs.
        assert not re.search(r"localhost:8000(?![0-9])", combined), (
            "workflow must not hit localhost:8000"
        )

    def test_creates_env_file_from_secrets_before_compose_up(
        self, job_steps: list[dict]
    ) -> None:
        combined = _combined_run(job_steps)
        assert ".env" in combined, "workflow must create a .env for docker-compose"
        # Must be a secret-backed write, not a static literal placeholder.
        assert "secrets" in combined.lower() or "secrets." in combined, (
            "workflow must source .env from GH secrets"
        )

    def test_runs_docker_compose_up(self, job_steps: list[dict]) -> None:
        combined = _combined_run(job_steps)
        assert re.search(r"docker compose up", combined)

    def test_waits_for_api_health(self, job_steps: list[dict]) -> None:
        combined = _combined_run(job_steps)
        assert "/health" in combined

    def test_runs_alembic_upgrade_head(self, job_steps: list[dict]) -> None:
        combined = _combined_run(job_steps)
        assert "alembic upgrade head" in combined

    def test_seeds_smoke_price_fixture(self, job_steps: list[dict]) -> None:
        combined = _combined_run(job_steps)
        assert "smoke_prices.sql" in combined, (
            "workflow must seed api/tests/fixtures/smoke_prices.sql before smoke calls"
        )

    def test_invokes_smoke_script(self, job_steps: list[dict]) -> None:
        combined = _combined_run(job_steps)
        assert "scheduler/smoke.sh" in combined, (
            "workflow must invoke scheduler/smoke.sh to run the smoke calls"
        )

    def test_tears_down_stack_in_always_step(self, job_steps: list[dict]) -> None:
        teardown_steps = [
            s for s in job_steps if s.get("if", "").strip().startswith("always()")
        ]
        assert teardown_steps, "expected at least one if: always() teardown step"
        combined_teardown = "\n".join(s.get("run", "") for s in teardown_steps)
        assert "docker compose down" in combined_teardown

    def test_dumps_api_logs_on_failure(self, job_steps: list[dict]) -> None:
        combined = _combined_run(job_steps)
        assert "docker compose logs" in combined


# ===========================================================================
# smoke.sh — shell script
# ===========================================================================


@pytest.fixture(scope="module")
def smoke_script_source() -> str:
    return SMOKE_SCRIPT.read_text()


class TestSmokeScriptExists:
    def test_script_is_present(self) -> None:
        assert SMOKE_SCRIPT.exists(), f"expected script at {SMOKE_SCRIPT}"

    def test_sh_n_parses_without_errors(self) -> None:
        result = subprocess.run(
            ["sh", "-n", str(SMOKE_SCRIPT)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


class TestSmokeScriptContent:
    def test_targets_health_endpoint(self, smoke_script_source: str) -> None:
        assert "/health" in smoke_script_source

    def test_targets_optimize_endpoint(self, smoke_script_source: str) -> None:
        assert "/api/v1/optimize" in smoke_script_source

    def test_targets_backtest_endpoint(self, smoke_script_source: str) -> None:
        assert "/api/v1/backtest" in smoke_script_source

    def test_polls_jobs_or_backtest_progress(self, smoke_script_source: str) -> None:
        assert "jobs/" in smoke_script_source or "/backtest/" in smoke_script_source

    def test_asserts_non_empty_weights(self, smoke_script_source: str) -> None:
        # The script must verify /optimize response contains a populated weights dict.
        assert re.search(r"weights", smoke_script_source, re.IGNORECASE)

    def test_asserts_non_empty_equity_curve(self, smoke_script_source: str) -> None:
        # equityCurve (camelCase JSON field) or equity_curve in poll response
        assert re.search(r"equity.?[cC]urve", smoke_script_source)

    def test_honours_smoke_api_url_env_var(self, smoke_script_source: str) -> None:
        assert (
            "SMOKE_API_URL" in smoke_script_source or "API_URL" in smoke_script_source
        )


# ===========================================================================
# smoke_prices.sql — fixture
# ===========================================================================


@pytest.fixture(scope="module")
def fixture_source() -> str:
    return SMOKE_FIXTURE.read_text()


class TestFixtureSqlExists:
    def test_fixture_is_present(self) -> None:
        assert SMOKE_FIXTURE.exists(), f"expected fixture at {SMOKE_FIXTURE}"


class TestFixtureSqlContent:
    def test_inserts_an_exchange_row(self, fixture_source: str) -> None:
        assert re.search(r"INSERT INTO exchanges", fixture_source, re.IGNORECASE)

    def test_inserts_instrument_rows(self, fixture_source: str) -> None:
        assert re.search(r"INSERT INTO instruments", fixture_source, re.IGNORECASE)

    def test_inserts_price_history_rows(self, fixture_source: str) -> None:
        assert re.search(r"INSERT INTO price_history", fixture_source, re.IGNORECASE)

    def test_uses_on_conflict_for_idempotency(self, fixture_source: str) -> None:
        # Multiple CI retries must not error on duplicate inserts.
        assert re.search(r"ON CONFLICT", fixture_source, re.IGNORECASE)

    def test_seeds_smoke_tickers(self, fixture_source: str) -> None:
        # Must seed at least three tickers to exercise /optimize.
        tickers_found = len(re.findall(r"SM[0-9]+", fixture_source))
        assert tickers_found >= 3


# ===========================================================================
# Helpers
# ===========================================================================


def _combined_run(steps: list[dict]) -> str:
    """Concatenate the `run:` field across every step in a job."""
    return "\n".join(s.get("run", "") for s in steps if s.get("run"))
