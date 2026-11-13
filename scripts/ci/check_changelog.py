"""Fail a pull request that changes shipped code without touching CHANGELOG.md.

Standard 5.1.2 makes a changelog entry mandatory for every code change. "Shipped"
means what a user receives or runs: the package sources, the migrations, the image
definitions and entrypoints, the installers, the scheduler scripts, the dependency
pins, and the markdown the agents load as prompt content. Tests and prose are not
shipped. A pull request that really needs no entry carries the ``skip-changelog``
label.

Usage: ``check_changelog.py --base origin/development [--skip-label true|false]``.
Standard library only, so the CI job needs no environment.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Sequence

_SHIPPED_PREFIXES = (
    "packages/",
    "ingestion/app/",
    "fund/src/",
    "docker/",
    "scheduler/",
)
_SHIPPED_FILES = frozenset(
    {
        "docker-compose.yml",
        "install.sh",
        "install.ps1",
        "setup.sh",
        "setup.ps1",
        "setup.cmd",
        "pyproject.toml",
        "uv.lock",
        "ingestion/Dockerfile",
        "ingestion/entrypoint.sh",
        "ingestion/pyproject.toml",
        "fund/Dockerfile",
        "fund/entrypoint.sh",
        "fund/pyproject.toml",
    }
)
# The skills under fund/src/ are markdown, but the profiler agent loads them as prompt
# content, so editing one changes behaviour.
_PROMPT_MARKDOWN_PREFIX = "fund/src/"
_CHANGELOG = "CHANGELOG.md"


def _is_shipped(path: str) -> bool:
    if "/tests/" in f"/{path}":
        return False
    if path.endswith(".md") and not path.startswith(_PROMPT_MARKDOWN_PREFIX):
        return False
    return path in _SHIPPED_FILES or path.startswith(_SHIPPED_PREFIXES)


def needs_changelog(changed: Sequence[str]) -> bool:
    """Whether any changed path is shipped code that calls for a changelog entry.

    Args:
        changed: Repository-relative paths changed by the pull request.

    Returns:
        True when at least one path is shipped (see the module docstring). Prose
        markdown and anything under a ``tests`` directory never count; markdown under
        ``fund/src/`` does, because the agents load it.
    """
    return any(_is_shipped(path) for path in changed)


def parse_skip(value: str) -> bool:
    """Read the ``skip-changelog`` label flag the workflow passes as text."""
    return value.strip().lower() == "true"


def evaluate(changed: Sequence[str], *, skip: bool) -> int:
    """Return the process exit code for a set of changed paths.

    Args:
        changed: Repository-relative paths changed by the pull request.
        skip: Whether the pull request carries the ``skip-changelog`` label.

    Returns:
        0 when no entry is needed, one was added or the label is set; 1 otherwise.
    """
    if skip or not needs_changelog(changed) or _CHANGELOG in changed:
        return 0
    print(
        f"{_CHANGELOG} is not updated, but this pull request changes shipped code. "
        "Add an entry under [Unreleased], or label the pull request 'skip-changelog'.",
        file=sys.stderr,
    )
    return 1


def changed_files(base: str) -> list[str]:
    """List the paths changed on this branch since it left ``base``."""
    git = shutil.which("git")
    if git is None:
        raise SystemExit("git is required to list the changed files")
    listing = subprocess.run(  # noqa: S603 - fixed argv, base comes from the workflow
        [git, "diff", "--name-only", f"{base}...HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return listing.stdout.splitlines()


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point: diff against ``--base`` and apply :func:`evaluate`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="branch the pull request targets")
    parser.add_argument(
        "--skip-label",
        default="false",
        help="'true' when the pull request carries the skip-changelog label",
    )
    args = parser.parse_args(argv)
    return evaluate(changed_files(args.base), skip=parse_skip(args.skip_label))


if __name__ == "__main__":
    sys.exit(main())
