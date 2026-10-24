"""Guard: the root developer docs describe the repository as it actually is.

``CLAUDE.md`` and ``README.md`` are the first thing a contributor (or an agent)
follows. Every fact checked here has already drifted once and sent someone down
a dead end, so each check derives the truth from the repo itself (pyproject
files, test configs, alembic revisions, the source tree) instead of
re-stating it.
"""

from __future__ import annotations

import ast
import configparser
import re
import shlex
from collections.abc import Iterator
from pathlib import Path

import pytest
import tomllib

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CLAUDE_MD = _REPO_ROOT / "CLAUDE.md"
_README = _REPO_ROOT / "README.md"
_ROOT_DOCS = [_CLAUDE_MD, _README]
_INGESTION = _REPO_ROOT / "ingestion"
_ALEMBIC_VERSIONS = _REPO_ROOT / "packages" / "portopt-db" / "alembic" / "versions"
_PYPROJECTS = [
    _REPO_ROOT / "pyproject.toml",
    _REPO_ROOT / "packages" / "portopt-db" / "pyproject.toml",
    _INGESTION / "pyproject.toml",
    _REPO_ROOT / "fund" / "pyproject.toml",
]

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


def _doc_id(path: Path) -> str:
    """Return a repo-relative test id; bare names collide (two README.md)."""
    return path.relative_to(_REPO_ROOT).as_posix()


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


# --- Trading 212 is an optional annotation add-on, not the universe source ---

_T212_ABSENT_DOCS = [
    _CLAUDE_MD,
    _README,
    _INGESTION / "README.md",
    _INGESTION / ".env.example",
]
# Files whose prose or labels describe where the instrument universe comes from.
_T212_SOURCE_CLAIM_FILES = [
    _INGESTION / "README.md",
    _INGESTION / "app" / "cli.py",
    _INGESTION / "app" / "config.py",
    _INGESTION / "app" / "services" / "__init__.py",
    _INGESTION / "app" / "services" / "jobs" / "scheduler.py",
    _INGESTION / "app" / "services" / "universe" / "__init__.py",
    _INGESTION / "app" / "setup" / "compose_secrets.py",
]
_T212_ABSENT_WORDS = re.compile(r"\b(absent|missing|unset|not set)\b", re.IGNORECASE)
_UNIVERSE_BUILD_SKIPS = re.compile(
    r"universe[_ -]build`?\s+(?:job\s+|step\s+)?skips", re.IGNORECASE
)
# The sources spell the vendor both "Trading 212" and "Trading212".
_T212_SOURCES_UNIVERSE = re.compile(
    r"trading ?212 (?:instrument[- ])?universe", re.IGNORECASE
)


def _passages(text: str) -> Iterator[str]:
    """Yield whitespace-normalised paragraphs and list items of ``text``.

    A passage ends at a blank line or where the next ``-``/``*`` list item
    starts; ``#`` comment markers (``.env`` files) are dropped.
    """
    current: list[str] = []
    for line in [*text.splitlines(), ""]:
        if not line.strip() or re.match(r"\s*[-*] ", line):
            if current:
                yield " ".join(" ".join(current).replace("#", " ").split())
            current = []
        if line.strip():
            current.append(line)


def _absent_t212_passages(text: str) -> list[str]:
    """Return passages describing what happens when the Trading 212 key is unset."""
    return [
        passage
        for passage in _passages(text)
        if re.search(r"TRADING_212_API_KEY|Trading 212", passage)
        and _T212_ABSENT_WORDS.search(passage)
    ]


@pytest.mark.parametrize("doc", _T212_ABSENT_DOCS, ids=_doc_id)
def test_absent_trading_212_key_only_skips_the_annotation_step(doc: Path) -> None:
    """Assert docs say a missing T212 key skips annotation, never the universe build.

    ``run_universe_step`` runs unconditionally from the yfinance Screener; only
    the follow-on ``t212_annotate`` step skips when no key is configured.
    """
    passages = _absent_t212_passages(doc.read_text(encoding="utf-8"))
    assert passages, f"{doc.name} no longer documents the optional Trading 212 key"
    wrong = [
        passage
        for passage in passages
        if _UNIVERSE_BUILD_SKIPS.search(passage) or "annotat" not in passage.lower()
    ]
    assert wrong == []


@pytest.mark.parametrize("path", _T212_SOURCE_CLAIM_FILES, ids=_doc_id)
def test_universe_is_not_described_as_a_trading_212_build(path: Path) -> None:
    """Assert no doc or label names Trading 212 as the instrument-universe source."""
    text = " ".join(path.read_text(encoding="utf-8").split())
    assert _T212_SOURCES_UNIVERSE.findall(text) == []


def test_guard_flags_the_old_claim_that_a_missing_key_skips_the_universe_build() -> (
    None
):
    """The matchers flag the pre-fix wording in each of its four shapes."""
    old = [
        "absent ⇒ `universe_build` skips without claiming a slot",
        "Absent Trading 212 ⇒ the universe-build step skips cleanly.",
        "# Absent => the universe_build job skips without claiming a job slot",
    ]
    assert all(_UNIVERSE_BUILD_SKIPS.search(claim) for claim in old)
    assert _T212_SOURCES_UNIVERSE.search("Weekly Trading 212 universe rebuild")
    assert _T212_SOURCES_UNIVERSE.search("Trading 212 instrument universe (Saturday)")
    assert _T212_SOURCES_UNIVERSE.search("Trading 212 instrument-universe build.")
    assert _T212_SOURCES_UNIVERSE.search("secrets (Trading212 universe + FRED macro)")
    assert not _T212_SOURCES_UNIVERSE.search("Trading 212 ticker annotation")


# --- Facts that must track the repo: alembic head, pins, gates, module paths ---

_HEAD_MENTION = re.compile(r"\bhead\W{0,6}([0-9a-f]{12})\b")
# `name==1.2.3` or `name[extra]==1.2.3` in backticks, as the docs quote pins.
_PIN_TOKEN = re.compile(r"`([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?==([0-9][^`,\s)]*)`")
_PIN_REQUIREMENT = re.compile(r"^([A-Za-z0-9_.\-]+)(?:\[[^\]]*\])?==(.+)$")
_BARE_LINT_TARGET = re.compile(r"\b(?:ruff check|ruff format|mypy)\s+optimizer/")
_LINE_ONLY_GATE = re.compile(
    r"\bline\b[^.\n|]{0,40}?(?:--cov-fail-under|≥|>=)", re.IGNORECASE
)
_BRANCH_GATE = re.compile(
    r"\bbranch(?:-rate)?\b[^.\n|]{0,20}?(?:≥|>=)\s*0?\.?\d|independent branch",
    re.IGNORECASE,
)
_INGESTION_MODULE = re.compile(r"`(app[./]services[./][A-Za-z0-9_./]*)`")
_RETIRED_TERMS = re.compile(r"reference[- ]ind(?:ex|ices)", re.IGNORECASE)


def _alembic_heads() -> set[str]:
    """Return the revisions of the alembic tree that no other revision follows."""
    revisions: set[str] = set()
    parents: set[str] = set()
    for path in _ALEMBIC_VERSIONS.glob("*.py"):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            name = getattr(target, "id", None)
            if name not in ("revision", "down_revision") or node.value is None:
                continue
            value = ast.literal_eval(node.value)
            if name == "revision":
                revisions.add(value)
            elif isinstance(value, (tuple, list)):
                parents.update(value)
            elif value is not None:
                parents.add(value)
    return revisions - parents


def _declared_pins() -> dict[str, set[str]]:
    """Return every exact (``==``) dependency pin declared by a workspace member."""
    pins: dict[str, set[str]] = {}
    for pyproject in _PYPROJECTS:
        project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
        requirements = [
            *project.get("dependencies", []),
            *(
                r
                for group in project.get("optional-dependencies", {}).values()
                for r in group
            ),
        ]
        for requirement in requirements:
            match = _PIN_REQUIREMENT.match(requirement)
            if match:
                name = match.group(1).lower().replace("_", "-")
                pins.setdefault(name, set()).add(match.group(2))
    return pins


def _stale_pins(text: str) -> list[str]:
    """Return documented ``name==version`` pins that differ from the pyprojects."""
    declared = _declared_pins()
    stale = []
    for match in _PIN_TOKEN.finditer(text):
        name = match.group(1).lower().replace("_", "-")
        if declared.get(name) != {match.group(2)}:
            stale.append(match.group(0))
    return stale


def _missing_ingestion_modules(text: str) -> list[str]:
    """Return documented ``app.services...`` paths that do not exist in ingestion/."""
    missing = []
    for token in _INGESTION_MODULE.findall(text):
        relative = Path(*re.split(r"[./]", token.removesuffix(".py").strip("./")))
        if not (
            (_INGESTION / relative).is_dir()
            or (_INGESTION / relative).with_suffix(".py").is_file()
        ):
            missing.append(token)
    return missing


def test_alembic_tree_has_a_single_head() -> None:
    """Assert the revision graph the docs quote a head for is linear at the tip."""
    assert len(_alembic_heads()) == 1


@pytest.mark.parametrize("doc", _ROOT_DOCS, ids=_doc_id)
def test_documented_alembic_head_is_the_actual_head(doc: Path) -> None:
    """Assert every alembic head quoted in the docs is the tree's real head."""
    quoted = set(_HEAD_MENTION.findall(doc.read_text(encoding="utf-8")))
    assert quoted == _alembic_heads()


@pytest.mark.parametrize("doc", _ROOT_DOCS, ids=_doc_id)
def test_documented_dependency_pins_match_the_pyprojects(doc: Path) -> None:
    """Assert each ``name==version`` quoted in the docs is what a pyproject pins."""
    assert _stale_pins(doc.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("doc", _ROOT_DOCS, ids=_doc_id)
def test_docs_do_not_pass_a_bare_optimizer_path_to_lint_tools(doc: Path) -> None:
    """Assert docs never show ruff/mypy run on the removed root ``optimizer/`` dir."""
    assert _BARE_LINT_TARGET.findall(doc.read_text(encoding="utf-8")) == []


def test_readme_does_not_claim_the_makefile_and_ci_use_a_bare_path() -> None:
    """Assert the README no longer says the build config targets ``optimizer/``.

    ``tests/test_build_config_paths.py`` guarantees the Makefile and CI already
    use the src-layout path, so a caveat saying otherwise is simply wrong.
    """
    text = " ".join(_README.read_text(encoding="utf-8").split())
    assert "bare path" not in text


@pytest.mark.parametrize(
    "path",
    [_CLAUDE_MD, _INGESTION / "README.md", _REPO_ROOT / "fund" / "pyproject.toml"],
    ids=_doc_id,
)
def test_coverage_floor_is_documented_as_one_blended_gate(path: Path) -> None:
    """Assert docs describe the coverage floor as CI enforces it.

    Each CI job passes ``--cov-branch`` with a single ``--cov-fail-under``, which
    coverage.py applies to the blended statement+branch percentage; there is no
    separate line gate and no separate branch gate.
    """
    text = path.read_text(encoding="utf-8")
    assert _LINE_ONLY_GATE.findall(text) + _BRANCH_GATE.findall(text) == []


def test_ci_gates_coverage_with_one_floor_per_job() -> None:
    """Assert the premise of the check above: one ``--cov-fail-under`` per command."""
    workflow = (_REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(
        encoding="utf-8"
    )
    gated = [line for line in workflow.splitlines() if "--cov-fail-under" in line]
    assert gated
    assert all(line.count("--cov-fail-under") == 1 for line in gated)


@pytest.mark.parametrize("doc", _ROOT_DOCS, ids=_doc_id)
def test_documented_ingestion_module_paths_exist(doc: Path) -> None:
    """Assert every ``app.services...`` path quoted in the docs resolves to code."""
    assert _missing_ingestion_modules(doc.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    "path",
    [
        _README,
        _INGESTION / "README.md",
        _INGESTION / "app" / "services" / "__init__.py",
        _INGESTION / "app" / "services" / "jobs" / "scheduler.py",
        _INGESTION / "app" / "repositories" / "jobs" / "background_job_repository.py",
    ],
    ids=_doc_id,
)
def test_retired_reference_index_feature_is_not_documented(path: Path) -> None:
    """Assert docs no longer cite the reference-index seed removed from the daemon.

    CLAUDE.md is exempt: it deliberately says there is no ``reference-indices``
    command so nobody goes looking for one.
    """
    assert _RETIRED_TERMS.findall(path.read_text(encoding="utf-8")) == []


def test_guards_catch_the_old_wording() -> None:
    """The matchers flag each pre-fix phrasing (guards the guards)."""
    assert _LINE_ONLY_GATE.search("Coverage floors (line `--cov-fail-under`)")
    assert _LINE_ONLY_GATE.search("Coverage floor line ≥ 90%")
    assert _BRANCH_GATE.search("CI gate (line ≥80% and branch ≥0.80)")
    assert _BRANCH_GATE.search("plus the independent branch-rate 0.80 check")
    assert _HEAD_MENTION.findall("# head: c4d5e6f7a8b9") == ["c4d5e6f7a8b9"]
    assert _HEAD_MENTION.findall("(**head `d8f0b2c4e6a1`**,") == ["d8f0b2c4e6a1"]
    assert _stale_pins("`numpy==0.0.1`") == ["`numpy==0.0.1`"]
    assert _missing_ingestion_modules("`app.services._shared._gone`") == [
        "app.services._shared._gone"
    ]
    assert _missing_ingestion_modules("`app/services/_shared/__init__.py`") == []
