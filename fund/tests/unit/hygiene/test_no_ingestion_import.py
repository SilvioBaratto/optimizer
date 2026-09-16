"""Guard: the fund bridge never imports the ingestion daemon (``app``).

``fund`` is *allowed* to import ``optimizer`` and ``portopt_db`` — it is the
bridge that turns LLM decisions into optimizer inputs against DB data. What it
must NOT reach into is the ingestion daemon's ``app`` package: ingestion and
fund are sibling apps in one shared venv, and letting fund call into ``app``
would couple the fund's runtime to the daemon's scheduler/services and collapse
the boundary the workspace is built around. The reverse direction
(``ingestion``/``portopt-db`` importing ``fund``/``deepagents``) is held by the
existing optimizer-import guards on those packages.

Source-blind by construction: scans raw tracked-file text for the forbidden
import marker and a forbidden dependency declaration. No implementation module
is imported, so the guard cannot itself drag ``deepagents`` into a clean tree.

``Path(__file__).resolve().parents[3]`` resolves directly to the ``fund/``
directory from this file (hygiene → unit → tests → fund), never ``Path.cwd()``.
The scans take an injectable ``root``/text so the fail-injection tests never
mutate real source.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

_FUND_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _FUND_ROOT / "src" / "fund"
_PYPROJECT_FILE = _FUND_ROOT / "pyproject.toml"
_SELF = Path(__file__).resolve()

_EXCLUDE_DIR_PARTS = {"__pycache__"}

# The ingestion daemon's import package is `app` (dist `portopt`). Forbid both
# `import app` and `from app ...`; a bare `app = typer.Typer()` assignment or a
# `fund.tui.app` submodule reference does not match (anchored on the keyword).
_INGESTION_IMPORT_PATTERN = re.compile(
    r"^\s*(from app\b|import app\b)", re.MULTILINE
)

# The ingestion distribution is exactly `portopt` — NOT `portopt-core`,
# `portopt-db`, or `portopt-fund` (the lookahead stops at the `-` those carry).
_INGESTION_DIST_PATTERN = re.compile(r"""["']portopt(?=["'\s<>=!~])""")


def _iter_python_files(root: Path, *, exclude: Path | None = None) -> Iterator[Path]:
    """Yield every ``.py`` file under ``root``, skipping generated noise dirs.

    Args:
        root: Directory tree to walk (a ``src/fund``-shaped root).
        exclude: A resolved file path to skip — this guard passes its own
            ``__file__`` so a scan never self-matches the marker it forbids.

    Yields:
        Each ``.py`` file under ``root`` not inside an excluded directory and
        not equal to ``exclude``.
    """
    for path in root.rglob("*.py"):
        if _EXCLUDE_DIR_PARTS.intersection(path.parts):
            continue
        if exclude is not None and path.resolve() == exclude:
            continue
        yield path


def find_ingestion_import_violations(root: Path) -> list[str]:
    """Scan ``root`` for ``app`` (ingestion) import statements.

    Args:
        root: Directory tree to scan (a ``src/fund``-shaped root).

    Returns:
        One relative path per offending file, empty when clean.
    """
    offending: list[str] = []
    for path in _iter_python_files(root, exclude=_SELF):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if _INGESTION_IMPORT_PATTERN.search(text):
            offending.append(str(path.relative_to(root)))
    return offending


def find_ingestion_dependency(dependencies_text: str) -> bool:
    """Return whether a dependency manifest declares the ingestion dist.

    Args:
        dependencies_text: Raw contents of a dependency manifest (fund's
            ``pyproject.toml``).

    Returns:
        ``True`` if the ``portopt`` ingestion distribution is declared as a
        dependency, ``False`` otherwise. ``portopt-core`` / ``portopt-db`` /
        ``portopt-fund`` do not count.
    """
    return _INGESTION_DIST_PATTERN.search(dependencies_text) is not None


def test_when_fund_source_is_scanned_then_no_ingestion_import_is_found():
    assert find_ingestion_import_violations(_SRC_ROOT) == []


def test_when_fund_pyproject_is_read_then_no_ingestion_dependency_is_declared():
    content = _PYPROJECT_FILE.read_text(encoding="utf-8")
    assert find_ingestion_dependency(content) is False


def test_when_bridge_dependencies_are_declared_then_the_guard_allows_them():
    """fund IS allowed portopt-core / portopt-db — they must not trip the
    ingestion-dist guard, which targets the bare `portopt` dist only."""
    assert find_ingestion_dependency('"portopt-core",\n"portopt-db",\n') is False


def test_when_fund_root_is_resolved_then_it_points_at_the_fund_directory():
    assert _FUND_ROOT.name == "fund"


def test_when_an_ingestion_import_is_injected_then_the_guard_fails(tmp_path):
    seeded = tmp_path / "offender.py"
    seeded.write_text("from app.services import worker\n", encoding="utf-8")

    violations = find_ingestion_import_violations(tmp_path)
    seeded.unlink()  # revert — tmp_path is ephemeral, cleanup made explicit

    assert violations, "injecting `from app` must fail the guard"


def test_when_a_bare_import_app_is_injected_then_the_guard_fails(tmp_path):
    seeded = tmp_path / "offender.py"
    seeded.write_text("import app\n", encoding="utf-8")

    violations = find_ingestion_import_violations(tmp_path)
    seeded.unlink()

    assert violations, "injecting `import app` must fail the guard"


def test_when_a_non_utf8_file_is_scanned_then_it_is_skipped(tmp_path):
    (tmp_path / "binary.py").write_bytes(b"\xff\xfe import app\n")
    assert find_ingestion_import_violations(tmp_path) == []


def test_when_the_ingestion_dist_is_injected_then_the_guard_fails():
    assert find_ingestion_dependency('"portopt==0.1.0",\n') is True


def test_when_own_source_is_scanned_then_it_is_excluded_from_the_tree():
    assert _SELF not in set(_iter_python_files(_FUND_ROOT, exclude=_SELF))
