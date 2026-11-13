"""The changelog check: shipped code may not change without a CHANGELOG entry.

Standard 5.1.2 makes a CHANGELOG update mandatory for every code change. The check
runs on pull requests only and is advisory (ADR-0007): a push without a pull request
never carries it, so it cannot be a required status.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "ci" / "check_changelog.py"
_CI_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_changelog", _SCRIPT)
    assert spec and spec.loader, _SCRIPT
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "changed",
    [
        ["fund/src/fund/actions.py"],
        ["ingestion/app/services/jobs/scheduler.py"],
        ["packages/portopt-core/optimizer/factors/_validation.py"],
        ["packages/portopt-db/alembic/versions/abc123_x.py"],
        ["docker-compose.yml"],
        ["install.sh"],
        ["docs/adr/0001-x.md", "fund/src/fund/config.py"],
    ],
)
def test_shipped_changes_need_a_changelog_entry(changed: list[str]) -> None:
    """Source, migrations, images and installers are what a user receives."""
    assert _load().needs_changelog(changed) is True


@pytest.mark.parametrize(
    "changed",
    [
        ["docs/nfr.md"],
        ["README.md", "CONTRIBUTING.md"],
        ["fund/tests/unit/test_risk.py"],
        ["packages/portopt-db/tests/test_x.py"],
        ["tests/test_docs_match_repo.py"],
        ["packages/portopt-db/README.md"],
        [".github/PULL_REQUEST_TEMPLATE.md"],
        [],
    ],
)
def test_documentation_and_test_only_changes_do_not(changed: list[str]) -> None:
    """Tests and prose are not shipped, so they need no entry."""
    assert _load().needs_changelog(changed) is False


def test_exit_code_is_failure_only_for_shipped_code_without_an_entry() -> None:
    """The four outcomes of the check."""
    check = _load()
    code = ["fund/src/fund/actions.py"]

    assert check.evaluate(code, skip=False) == 1
    assert check.evaluate([*code, "CHANGELOG.md"], skip=False) == 0
    assert check.evaluate(code, skip=True) == 0
    assert check.evaluate(["docs/nfr.md"], skip=False) == 0


def test_the_skip_label_value_is_parsed() -> None:
    """The workflow hands the label test over as the string 'true' or 'false'."""
    check = _load()

    assert check.parse_skip("true") is True
    assert check.parse_skip("false") is False
    assert check.parse_skip("") is False


def _job() -> dict[str, object]:
    workflow = yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8"))
    return workflow["jobs"]["changelog"]


def test_ci_has_a_changelog_job_that_runs_only_on_pull_requests() -> None:
    """A push has no base to diff against, so the job is limited to pull requests."""
    assert _job()["if"] == "github.event_name == 'pull_request'"


def test_ci_changelog_job_fetches_history_and_passes_input_through_env() -> None:
    """The base branch needs full history, and no `${{ }}` may sit in a `run:` line."""
    job = _job()
    steps: list[dict[str, object]] = job["steps"]  # type: ignore[assignment]
    checkout = next(s for s in steps if "actions/checkout" in str(s.get("uses", "")))

    assert checkout["with"] == {"fetch-depth": 0}  # type: ignore[comparison-overlap]
    for step in steps:
        assert "${{" not in str(step.get("run", "")), step
