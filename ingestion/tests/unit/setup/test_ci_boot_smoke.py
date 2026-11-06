"""CI boots the shipped images, not just builds them.

Building proves a Dockerfile parses; only starting the container exercises the
entrypoint, the Alembic chain on an empty database and the worker's boot and shutdown
paths. Without a boot step a broken entrypoint reaches a user's `portopt start`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml")

# unit -> setup -> tests -> ingestion -> repo root.
_CI = Path(__file__).resolve().parents[4] / ".github" / "workflows" / "ci.yml"


def _steps() -> list[dict[str, Any]]:
    doc = yaml.safe_load(_CI.read_text(encoding="utf-8"))
    return doc["jobs"]["docker-images"]["steps"]


def _index(needle: str) -> int:
    for i, step in enumerate(_steps()):
        if needle in step.get("run", ""):
            return i
    pytest.fail(f"no docker-images step runs {needle!r}")


def test_the_ingestion_stack_is_booted_and_waited_on_after_the_build() -> None:
    """`up --wait` fails the job when a service never turns healthy."""
    boot = _index("up -d --wait")
    cmd = _steps()[boot]["run"]
    assert "--profile ingestion" in cmd
    assert "--wait-timeout" in cmd
    assert boot > _index("build")


def test_the_fund_worker_is_booted_too_on_a_placeholder_provider_key() -> None:
    """The worker builds its LLM client before its first heartbeat; building a client
    needs a key but makes no call, so a non-empty placeholder lets CI boot the fund
    container (an empty secret file reads as no key and the worker exits)."""
    assert "--profile fund" in _steps()[_index("up -d --wait")]["run"]
    secrets = _steps()[_index("mkdir -p secrets")]["run"]
    assert "secrets/ollama_api_key" in secrets
    assert "echo" in secrets or "printf" in secrets


def test_a_stop_must_leave_both_workers_with_a_clean_exit_code() -> None:
    """The entrypoints and workers handle SIGTERM; a 137 means a stop was ignored."""
    stop = _steps()[_index("compose --profile ingestion --profile fund stop")]["run"]
    assert "ExitCode" in stop
    assert "optimizer_scheduler" in stop
    assert "optimizer_fund" in stop
    assert _index("compose --profile ingestion --profile fund stop") > _index(
        "up -d --wait"
    )


def test_the_stack_is_torn_down_even_when_a_step_failed() -> None:
    teardown = _steps()[_index("down")]
    assert teardown.get("if") == "always()"
    assert "--volumes" in teardown["run"]


def test_container_logs_are_printed_when_the_boot_fails() -> None:
    logs = _steps()[_index("logs")]
    assert logs.get("if") == "failure()"
