"""Guards on the GitHub Actions workflows.

The workflows are the only gate that runs the lint, type-check, coverage and
import-boundary suites, so a trigger that never fires or a pin that floats
silently removes that gate. These tests parse the YAML instead of grepping it so
a reformat cannot hide a regression.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import tomllib
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


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    # A job that calls a reusable workflow (``uses:``) has no steps of its own.
    return job.get("steps", [])


def _run_commands(job: dict[str, Any]) -> list[str]:
    return [step["run"] for step in _steps(job) if "run" in step]


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


def test_ci_points_the_integration_tests_at_a_separate_scratch_database() -> None:
    """The fund integration tests run only on ``FUND_TEST_DATABASE_URL``.

    They ignore ``DATABASE_URL`` (the live database on a developer machine) and
    refuse a target equal to it, so CI must create a second database and point the
    integration run at it, or every integration case silently skips.
    """
    for job in _postgres_jobs().values():
        steps = [
            step for step in job["steps"] if "-m integration" in step.get("run", "")
        ]
        if not steps:
            continue
        env = {**job.get("env", {}), **steps[0].get("env", {})}
        scratch = env.get("FUND_TEST_DATABASE_URL", "")
        assert scratch.startswith("postgresql://"), env
        assert scratch.rsplit("/", 1)[-1] != env["DATABASE_URL"].rsplit("/", 1)[-1]
        database = scratch.rsplit("/", 1)[-1]
        assert any(
            f"CREATE DATABASE {database}" in cmd for cmd in _run_commands(job)
        ), "the scratch database is never created"
        return
    pytest.fail("no postgres job runs the integration tests")


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
        for step in _steps(job)
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
    """A hung job must not burn the six-hour runner default.

    A job that calls a reusable workflow cannot set its own timeout; the called
    workflow's jobs carry theirs.
    """
    untimed = [
        job
        for job, body in _jobs(name).items()
        if "uses" not in body and "timeout-minutes" not in body
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


def _members_with_mypy_config() -> list[str]:
    """Workspace members whose own pyproject.toml declares a ``[tool.mypy]`` gate."""
    pyprojects = [
        *_REPO_ROOT.glob("*/pyproject.toml"),
        *_REPO_ROOT.glob("packages/*/pyproject.toml"),
    ]
    return sorted(
        pyproject.parent.relative_to(_REPO_ROOT).as_posix()
        for pyproject in pyprojects
        if "mypy"
        in tomllib.loads(pyproject.read_text(encoding="utf-8")).get("tool", {})
    )


def _mentions(command: str, member: str) -> bool:
    return re.search(rf"(?<![\w/-]){re.escape(member)}/?(?!\w)", command) is not None


def test_typecheck_job_covers_every_member_that_configures_mypy() -> None:
    """A member that ships a mypy config must be held to it in CI."""
    members = _members_with_mypy_config()
    commands = _run_commands(_jobs()["typecheck"])

    assert {"ingestion", "packages/portopt-db"} <= set(members)
    for member in members:
        assert any("mypy" in cmd and _mentions(cmd, member) for cmd in commands), (
            f"mypy never runs on {member}"
        )


def test_makefile_runs_the_same_lint_and_typecheck_gates_as_ci() -> None:
    """``make lint`` / ``make typecheck`` are what contributors run before a PR."""
    lines = (_REPO_ROOT / "Makefile").read_text(encoding="utf-8").splitlines()

    for member in _ruff_configured_members():
        for tool in ("ruff check", "ruff format --check"):
            assert any(tool in line and _mentions(line, member) for line in lines), (
                f"make lint never runs {tool} on {member}"
            )
    for member in _members_with_mypy_config():
        assert any("mypy" in line and _mentions(line, member) for line in lines), (
            f"make typecheck never runs mypy on {member}"
        )


def _codecov_steps() -> list[dict[str, Any]]:
    return [
        step
        for job in _jobs().values()
        for step in job["steps"]
        if step.get("uses", "").startswith("codecov/codecov-action@")
    ]


def test_codecov_upload_uses_inputs_the_action_declares() -> None:
    """codecov-action v5 reads ``files``; ``file`` is dropped with only a warning."""
    steps = _codecov_steps()

    assert steps
    for step in steps:
        inputs = step.get("with", {})
        assert "file" not in inputs
        assert inputs.get("files")


def test_codecov_upload_is_authenticated_and_cannot_fail_silently_on_main() -> None:
    """A rejected upload on the main branch must turn the job red.

    Without a token (or OIDC) Codecov rejects the upload of a protected branch,
    and ``fail_ci_if_error: false`` made that invisible, so coverage reporting
    could stay empty indefinitely.
    """
    for step in _codecov_steps():
        inputs = step["with"]
        assert "CODECOV_TOKEN" in str(inputs.get("token", ""))
        assert "refs/heads/main" in str(inputs.get("fail_ci_if_error", False))


_CI_WORKFLOW_REF = "./.github/workflows/ci.yml"


def _needs_closure(name: str, job: str) -> set[str]:
    """Every job that must finish before ``job`` starts, transitively."""
    jobs = _jobs(name)
    seen: set[str] = set()
    pending = [job]
    while pending:
        needs = jobs[pending.pop()].get("needs", [])
        for dependency in [needs] if isinstance(needs, str) else needs:
            if dependency not in seen:
                seen.add(dependency)
                pending.append(dependency)
    return seen


def _release_step(marker: str) -> tuple[str, dict[str, Any]]:
    """The one release-workflow step whose script contains ``marker``."""
    matches = [
        (job_name, step)
        for job_name, job in _jobs("release.yml").items()
        for step in _steps(job)
        if marker in step.get("run", "")
    ]
    assert len(matches) == 1, f"expected one release step running {marker!r}"
    return matches[0]


def _run_release_step(
    marker: str, cwd: Path, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Execute a release step's script the way the runner's default shell does."""
    _, step = _release_step(marker)
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not installed")
    path = os.pathsep.join([str(Path(sys.executable).parent), os.environ["PATH"]])
    return subprocess.run(  # noqa: S603
        [bash, "-e", "-c", step["run"]],
        cwd=cwd,
        env={**os.environ, "PATH": path, **env},
        capture_output=True,
        text=True,
        check=False,
    )


def test_ci_workflow_is_reusable_so_the_release_can_run_it() -> None:
    """``workflow_call`` lets the release workflow run the whole CI as a gate."""
    assert "workflow_call" in _triggers(_load("ci.yml"))


def test_release_publish_waits_for_the_full_ci_workflow() -> None:
    """A tag must not reach PyPI without the lint, type and test gates passing."""
    ci_jobs = {
        name
        for name, job in _jobs("release.yml").items()
        if job.get("uses") == _CI_WORKFLOW_REF
    }

    assert ci_jobs, "release.yml never calls the CI workflow"
    assert ci_jobs <= _needs_closure("release.yml", "publish")


def test_release_checks_run_before_the_build_and_publish() -> None:
    """The tag checks must gate publishing, not merely exist in the workflow."""
    for marker in ("merge-base", "pyproject.toml"):
        job_name, _ = _release_step(marker)
        assert job_name in _needs_closure("release.yml", "publish")


def test_release_tag_check_sees_the_full_history_of_the_tagged_commit() -> None:
    """A shallow checkout cannot tell whether the tag is an ancestor of main."""
    job_name, _ = _release_step("merge-base")
    checkouts = [
        step
        for step in _steps(_jobs("release.yml")[job_name])
        if step.get("uses", "").startswith("actions/checkout@")
    ]

    assert checkouts
    assert all(step.get("with", {}).get("fetch-depth") == 0 for step in checkouts)


def _git(cwd: Path, *args: str) -> str:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    identity = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
    }
    return subprocess.run(  # noqa: S603
        [git, *args],
        cwd=cwd,
        env={**os.environ, **identity},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def tagged_clone(tmp_path: Path) -> tuple[Path, str, str]:
    """A clone of a remote whose ``main`` and ``feature`` branches have diverged.

    Returns the clone, the SHA on ``main`` and the SHA only on ``feature``.
    """
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    (origin / "file.txt").write_text("base\n", encoding="utf-8")
    _git(origin, "add", ".")
    _git(origin, "commit", "-q", "-m", "base")
    on_main = _git(origin, "rev-parse", "HEAD")
    _git(origin, "checkout", "-q", "-b", "feature")
    (origin / "file.txt").write_text("feature\n", encoding="utf-8")
    _git(origin, "commit", "-q", "-am", "feature")
    off_main = _git(origin, "rev-parse", "HEAD")
    _git(origin, "checkout", "-q", "main")

    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    return clone, on_main, off_main


@pytest.mark.parametrize(("which", "ok"), [("on_main", True), ("off_main", False)])
def test_release_only_accepts_tags_reachable_from_main(
    tagged_clone: tuple[Path, str, str], which: str, ok: bool
) -> None:
    """A ``v*`` tag pushed on a feature branch must not be published."""
    clone, on_main, off_main = tagged_clone
    sha = on_main if which == "on_main" else off_main
    _git(clone, "fetch", "-q", "origin", "feature")
    _git(clone, "checkout", "-q", "--detach", sha)

    result = _run_release_step(
        "merge-base", clone, {"GITHUB_SHA": sha, "GITHUB_REF_NAME": "v0.1.0"}
    )

    assert (result.returncode == 0) is ok, result.stdout + result.stderr


def _pyproject_version() -> str:
    text = (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return tomllib.loads(text)["project"]["version"]


@pytest.mark.parametrize(
    ("tag_template", "ok"),
    [
        ("v{version}", True),
        ("v{version}1", False),
        ("v9.9.9", False),
        ("{version}", False),
    ],
)
def test_release_tag_must_equal_the_pyproject_version(
    tmp_path: Path, tag_template: str, ok: bool
) -> None:
    """A wheel whose version differs from its tag cannot be replaced on PyPI."""
    (tmp_path / "pyproject.toml").write_text(
        (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    tag = tag_template.format(version=_pyproject_version())

    result = _run_release_step("pyproject.toml", tmp_path, {"GITHUB_REF_NAME": tag})

    assert (result.returncode == 0) is ok, result.stdout + result.stderr


# portopt-core is not on PyPI until a tagged release has actually published it,
# so prose may say what ``release.yml`` does but must not state it as a done deal.
_PYPI_PRESENT_FACT = re.compile(
    r"\b(?:is|are|been)\s+published\s+to\s+pypi"
    r"|only\s+the\s+`?portopt-core`?\s+library\s+is\b",
    re.IGNORECASE,
)


def _prose(path: str) -> str:
    """Return a doc's text with comment markers dropped and line breaks joined."""
    text = (_REPO_ROOT / path).read_text(encoding="utf-8")
    return re.sub(r"\s+", " ", re.sub(r"^\s*#", "", text, flags=re.MULTILINE))


def test_docs_attribute_the_pypi_release_to_the_workflow() -> None:
    """``release.yml`` publishes ``portopt-core``; docs describe it, not claim it."""
    publishes = any(
        "pypa/gh-action-pypi-publish" in step.get("uses", "")
        for job in _jobs("release.yml").values()
        for step in _steps(job)
    )
    assert publishes
    assert "`release.yml` publishes `portopt-core` to PyPI" in _prose("README.md")
    assert "release workflow" in _prose("CONTRIBUTING.md")

    claims = [
        doc
        for doc in ("README.md", "CONTRIBUTING.md", "install.sh", "install.ps1")
        if _PYPI_PRESENT_FACT.search(_prose(doc))
    ]

    assert claims == []
