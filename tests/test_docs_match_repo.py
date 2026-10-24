"""Guard: the root developer docs describe the repository as it actually is.

``CLAUDE.md`` and ``README.md`` are the first thing a contributor (or an agent)
follows. Every fact checked here has already drifted once and sent someone down
a dead end, so each check derives the truth from the repo itself (pyproject
files, test configs, alembic revisions, the source tree) instead of
re-stating it.
"""

from __future__ import annotations

import configparser
import shlex
from collections.abc import Iterator
from pathlib import Path

import pytest
import tomllib

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CLAUDE_MD = _REPO_ROOT / "CLAUDE.md"
_README = _REPO_ROOT / "README.md"
_ROOT_DOCS = [_CLAUDE_MD, _README]

# ``uv run --package <dist>`` only selects the dependency set; it never changes
# the working directory. Maps each workspace dist to the suite it must run.
_SUITE_DIR_BY_PACKAGE = {
    "portopt-core": "tests",
    "portopt-db": "packages/portopt-db/tests",
    "portopt": "ingestion/tests",
    "portopt-fund": "fund/tests",
}

# pytest options that consume the following token as their value.
_PYTEST_VALUE_FLAGS = frozenset(
    {"-k", "-m", "-p", "-o", "-c", "--deselect", "--ignore", "--rootdir", "--junitxml"}
)


def _fenced_lines(text: str) -> Iterator[str]:
    """Yield the lines of ``text`` that sit inside a fenced code block."""
    inside = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            inside = not inside
        elif inside:
            yield line


def _pytest_config(start: Path) -> tuple[Path, list[str]] | None:
    """Return ``(rootdir, testpaths)`` of the pytest config governing ``start``.

    Mirrors pytest's upward search for ``pytest.ini`` or a ``pyproject.toml``
    carrying ``[tool.pytest.ini_options]``.
    """
    for directory in (start, *start.parents):
        ini = directory / "pytest.ini"
        if ini.is_file():
            parser = configparser.ConfigParser()
            parser.read(ini, encoding="utf-8")
            raw = parser.get("pytest", "testpaths", fallback="")
            return directory, raw.split()
        pyproject = directory / "pyproject.toml"
        if pyproject.is_file():
            options = (
                tomllib.loads(pyproject.read_text(encoding="utf-8"))
                .get("tool", {})
                .get("pytest", {})
                .get("ini_options")
            )
            if options is not None:
                return directory, list(options.get("testpaths", []))
    return None


def _collected_dirs(cwd: Path, pytest_args: list[str]) -> list[Path]:
    """Return the directories ``pytest <pytest_args>`` collects when run in ``cwd``."""
    positional: list[str] = []
    skip_next = False
    for arg in pytest_args:
        if skip_next:
            skip_next = False
        elif arg in _PYTEST_VALUE_FLAGS:
            skip_next = True
        elif not arg.startswith("-"):
            positional.append(arg.split("::")[0])
    if positional:
        return [(cwd / arg).resolve() for arg in positional]
    config = _pytest_config(cwd)
    # ``testpaths`` only applies when pytest is started from the rootdir itself.
    if config is not None and config[0] == cwd and config[1]:
        return [(cwd / path).resolve() for path in config[1]]
    return [cwd]


def _within(collected: list[Path], suite: Path) -> bool:
    """Return whether every collected directory is ``suite`` or inside it."""
    return all(path == suite or suite in path.parents for path in collected)


def _pytest_runs(text: str) -> Iterator[tuple[str, Path, list[Path]]]:
    """Yield ``(line, claimed suite, collected dirs)`` per documented pytest command.

    The claimed suite is the member named by ``--package``, else the directory
    the command is run from.
    """
    for raw in _fenced_lines(text):
        line = raw.split(" #")[0].strip().strip("()")
        if "pytest" not in line:
            continue
        cwd = _REPO_ROOT
        argv: list[str] = []
        for part in line.split("&&"):
            tokens = shlex.split(part)
            if tokens[:1] == ["cd"]:
                cwd = (cwd / tokens[1]).resolve()
            elif "pytest" in tokens:
                argv = tokens
        if not argv:
            continue
        package = argv[argv.index("--package") + 1] if "--package" in argv else None
        claimed = (
            (_REPO_ROOT / _SUITE_DIR_BY_PACKAGE[package]).resolve() if package else cwd
        )
        yield (
            raw.strip(),
            claimed,
            _collected_dirs(cwd, argv[argv.index("pytest") + 1 :]),
        )


def _wrong_suite_commands(text: str) -> list[str]:
    """Return documented pytest commands that would not run their member's suite."""
    return [
        raw
        for raw, claimed, collected in _pytest_runs(text)
        if not _within(collected, claimed)
    ]


def _covered_suites(text: str) -> set[str]:
    """Return the workspace dists whose suite a documented command really runs."""
    covered: set[str] = set()
    for _raw, _claimed, collected in _pytest_runs(text):
        for package, suite_dir in _SUITE_DIR_BY_PACKAGE.items():
            if _within(collected, (_REPO_ROOT / suite_dir).resolve()):
                covered.add(package)
    return covered


@pytest.mark.parametrize("doc", _ROOT_DOCS, ids=lambda doc: doc.name)
def test_documented_pytest_commands_run_the_members_own_suite(doc: Path) -> None:
    """Assert each documented ``pytest`` command collects the suite it names."""
    assert _wrong_suite_commands(doc.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("doc", _ROOT_DOCS, ids=lambda doc: doc.name)
def test_docs_document_a_pytest_command_for_every_package(doc: Path) -> None:
    """Assert each doc has a command that really runs each of the four suites.

    Without this the check above passes vacuously when a member's command is
    deleted, since it can only judge the commands that remain.
    """
    covered = _covered_suites(doc.read_text(encoding="utf-8"))
    assert covered == set(_SUITE_DIR_BY_PACKAGE)


def test_guard_flags_a_member_package_run_from_the_repo_root() -> None:
    """The matcher flags ``--package portopt pytest`` run from the repo root."""
    doc = "```bash\nuv run --package portopt pytest\n```\n"
    assert _wrong_suite_commands(doc) == ["uv run --package portopt pytest"]


def test_guard_accepts_a_member_suite_run_from_its_own_directory() -> None:
    """The matcher accepts the commands CI itself runs from each member dir."""
    doc = (
        "```bash\n"
        "cd ingestion && uv run pytest tests/\n"
        "cd packages/portopt-db && uv run pytest\n"
        "cd fund && uv run pytest tests/\n"
        "uv run --package portopt-core pytest tests/ -v\n"
        "```\n"
    )
    assert _wrong_suite_commands(doc) == []
    assert _covered_suites(doc) == set(_SUITE_DIR_BY_PACKAGE)


def test_guard_notices_a_member_suite_with_no_command() -> None:
    """Prose, comments and other members' commands never stand in for a suite."""
    doc = (
        "```bash\n"
        "uv run --package portopt-core pytest tests/ -v\n"
        "# the fund suite: cd fund && uv run pytest tests/\n"
        "cd ingestion && uv run pytest tests/\n"
        "cd packages/portopt-db && uv run pytest\n"
        "uv run --package portopt-fund pytest\n"
        "```\n"
        "Run `cd fund && uv run pytest tests/` for the fund.\n"
    )
    assert _covered_suites(doc) == {"portopt-core", "portopt", "portopt-db"}
