"""Guard: CI runs on every branch the branch protection lets a change travel through.

``main`` and ``development`` require the CI job statuses on the exact commit being
pushed, and a pull request is not mandatory. A branch pushed without a pull request
only carries those statuses when ``ci.yml`` also triggers on ``push`` for it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_CI_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"

_PROTECTED = {"main", "development"}
_PREFIXED = {"feature/**", "bugfix/**", "hotfix/**", "release/**"}


def _push_branches() -> set[str]:
    workflow: dict[Any, Any] = yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8"))
    # PyYAML follows YAML 1.1, which reads the bare key ``on`` as the boolean True.
    triggers = workflow["on"] if "on" in workflow else workflow[True]
    return set(triggers["push"]["branches"])


def test_ci_runs_on_push_to_the_protected_branches() -> None:
    """Pushes to ``main`` and ``development`` keep running the full CI."""
    assert _push_branches() >= _PROTECTED


def test_ci_runs_on_push_to_every_prefixed_branch() -> None:
    """A feature, bugfix, hotfix or release branch gets statuses without a PR."""
    assert _push_branches() >= _PREFIXED
