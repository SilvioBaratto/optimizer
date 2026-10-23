"""Guards on the GitHub Actions workflows.

The workflows are the only gate that runs the lint, type-check, coverage and
import-boundary suites, so a trigger that never fires or a pin that floats
silently removes that gate. These tests parse the YAML instead of grepping it so
a reformat cannot hide a regression.
"""

from __future__ import annotations

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
