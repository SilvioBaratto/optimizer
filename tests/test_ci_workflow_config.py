"""Guards on the GitHub Actions workflows.

The workflows are the only gate that runs the lint, type-check, coverage and
import-boundary suites, so a trigger that never fires or a pin that floats
silently removes that gate. These tests parse the YAML instead of grepping it so
a reformat cannot hide a regression.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOWS = _REPO_ROOT / ".github" / "workflows"


def _load(name: str) -> dict[Any, Any]:
    return yaml.safe_load((_WORKFLOWS / name).read_text(encoding="utf-8"))


def _triggers(workflow: dict[Any, Any]) -> dict[str, Any]:
    # YAML 1.1 (PyYAML) reads the bare key ``on`` as the boolean True.
    return workflow["on"] if "on" in workflow else workflow[True]


@pytest.mark.parametrize("event", ["push", "pull_request"])
def test_ci_runs_for_the_integration_branch(event: str) -> None:
    """Pushes and PRs on ``development`` must trigger CI, not only ``main``."""
    branches = _triggers(_load("ci.yml"))[event]["branches"]

    assert "main" in branches
    assert "development" in branches


def _jobs(name: str = "ci.yml") -> dict[str, dict[str, Any]]:
    return _load(name)["jobs"]


def _run_commands(job: dict[str, Any]) -> list[str]:
    return [step["run"] for step in job["steps"] if "run" in step]


def _postgres_jobs() -> dict[str, dict[str, Any]]:
    return {
        name: job
        for name, job in _jobs().items()
        if any(
            "postgres" in service.get("image", "")
            for service in job.get("services", {}).values()
        )
    }


def test_ci_applies_the_alembic_chain_to_a_real_postgres() -> None:
    """A migration that only works step by step on SQLite must fail CI.

    The per-migration tests run on in-memory SQLite with hand-built prerequisite
    tables; only ``alembic upgrade head`` on an empty PostgreSQL proves the whole
    chain (JSONB, ``op.execute`` DDL, duplicate indexes) applies.
    """
    jobs = _postgres_jobs()

    assert jobs, "no CI job starts a postgres service container"
    for job in jobs.values():
        assert job["env"]["DATABASE_URL"].startswith("postgresql://")
        assert any("alembic upgrade head" in cmd for cmd in _run_commands(job))


def test_ci_runs_the_postgres_integration_tests() -> None:
    """The ``integration`` marker skips without DATABASE_URL, so CI must set it."""
    jobs = _postgres_jobs()

    assert any(
        "-m integration" in cmd for job in jobs.values() for cmd in _run_commands(job)
    )


def test_ci_validates_compose_and_builds_both_images() -> None:
    """A broken compose edit or Dockerfile must not ship green."""
    commands = [cmd for job in _jobs().values() for cmd in _run_commands(job)]

    def compose(verb: str) -> bool:
        return any(
            "docker compose" in cmd
            and "--profile ingestion" in cmd
            and "--profile fund" in cmd
            and f" {verb}" in cmd
            for cmd in commands
        )

    assert compose("config")
    assert compose("build")


def test_fund_pyproject_does_not_claim_a_removed_branch_coverage_gate() -> None:
    """CI gates fund on line coverage only; the config must not promise more."""
    text = (_REPO_ROOT / "fund" / "pyproject.toml").read_text(encoding="utf-8")

    assert "independent branch-rate" not in text


_FULL_SHA = re.compile(r"[0-9a-f]{40}")


@pytest.mark.parametrize("name", ["ci.yml", "release.yml"])
def test_every_action_is_pinned_to_a_full_commit_sha(name: str) -> None:
    """A moved tag must not be able to swap code into a job holding the token."""
    unpinned = [
        step["uses"]
        for job in _jobs(name).values()
        for step in job["steps"]
        if "uses" in step and not _FULL_SHA.fullmatch(step["uses"].partition("@")[2])
    ]

    assert unpinned == []


def test_release_build_tool_is_version_pinned() -> None:
    """The ``build`` backend produces the artifacts published to PyPI."""
    installs = [
        cmd
        for job in _jobs("release.yml").values()
        for cmd in _run_commands(job)
        if "pip install" in cmd
    ]

    assert installs
    assert all(re.search(r"\w==\d", cmd) for cmd in installs)


_SETUP_UV = "astral-sh/setup-uv@"
_DOCKER_UV_PIN = re.compile(r"astral-sh/uv:(\d+\.\d+\.\d+)")


def _setup_uv_steps() -> list[tuple[str, dict[str, Any]]]:
    return [
        (name, step)
        for name, job in _jobs().items()
        for step in job["steps"]
        if step.get("uses", "").startswith(_SETUP_UV)
    ]


def _docker_uv_pins() -> set[str]:
    return {
        pin
        for dockerfile in ("ingestion/Dockerfile", "fund/Dockerfile")
        for pin in _DOCKER_UV_PIN.findall(
            (_REPO_ROOT / dockerfile).read_text(encoding="utf-8")
        )
    }


def test_ci_uses_the_uv_version_the_docker_images_ship() -> None:
    """CI must resolve and lock with the uv that builds the images."""
    pins = _docker_uv_pins()
    steps = _setup_uv_steps()

    assert len(pins) == 1, f"the two Dockerfiles disagree on the uv pin: {pins}"
    assert steps
    assert {step.get("with", {}).get("version") for _, step in steps} == pins


def test_every_uv_job_names_its_python_version() -> None:
    """A runner-image Python bump must not silently change what CI tests."""
    missing = [
        name
        for name, step in _setup_uv_steps()
        if not step.get("with", {}).get("python-version")
    ]

    assert missing == []


@pytest.mark.parametrize("name", ["ci.yml", "release.yml"])
def test_every_job_has_a_timeout(name: str) -> None:
    """A hung job must not burn the six-hour runner default."""
    untimed = [
        job for job, body in _jobs(name).items() if "timeout-minutes" not in body
    ]

    assert untimed == []


def _ruff_configured_members() -> list[str]:
    """Workspace members that carry their own ruff.toml, as repo-relative paths."""
    configs = [
        *_REPO_ROOT.glob("*/ruff.toml"),
        *_REPO_ROOT.glob("packages/*/ruff.toml"),
    ]
    return sorted(
        config.parent.relative_to(_REPO_ROOT).as_posix() for config in configs
    )


def test_lint_job_covers_every_member_that_has_a_ruff_config() -> None:
    """A lint or format regression in any member must fail CI, not only in core."""
    members = _ruff_configured_members()
    commands = _run_commands(_jobs()["lint"])

    assert {"fund", "ingestion", "packages/portopt-db"} <= set(members)
    for member in members:
        for tool in ("ruff check", "ruff format --check"):
            assert any(
                tool in cmd
                and re.search(rf"(?<![\w/-]){re.escape(member)}/?(?!\w)", cmd)
                for cmd in commands
            ), f"{tool} never runs on {member}"


def test_lint_job_runs_shellcheck_on_every_tracked_shell_script() -> None:
    """The entrypoints and installers run on user machines and in containers."""
    commands = _run_commands(_jobs()["lint"])

    assert any("shellcheck" in cmd and "git ls-files" in cmd for cmd in commands)
