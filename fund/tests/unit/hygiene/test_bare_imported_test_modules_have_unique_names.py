"""Guard: no two bare-imported test-tree modules share a file name.

The fund and portopt-db test trees have no ``__init__.py``, so pytest puts each
tree's own directory on ``sys.path`` and imports every module there by its bare
file name (the guards' ``_fund_import_scan`` / ``_import_scan`` helpers and the
``test_*`` modules alike). Two such modules with the same name collide in
``sys.modules`` as soon as both trees are collected in one session: whichever
loads second is rejected as an "import file mismatch", or silently resolves to
the first one's API and dies with an unrelated-looking ``ImportError``. CI runs
each package separately, so only a combined local run
(``pytest fund packages/portopt-db``) exposes it.

Modules inside a package (the root ``tests/`` and ``ingestion/tests`` trees) are
imported by their dotted name and cannot collide this way, and ``conftest.py``
is special-cased by pytest itself, so neither is counted.

``Path(__file__).resolve().parents[4]`` is the repository root (hygiene -> unit
-> tests -> fund -> root). Only the four first-party trees are walked so a nested
virtualenv or ``node_modules`` never contributes a stray match.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FIRST_PARTY_TREES = ("tests", "fund", "packages", "ingestion")
_SKIPPED_DIR_PARTS = {"__pycache__", ".venv", "node_modules", "build"}


def _bare_imported_test_modules(
    repo_root: Path, trees: Iterable[str] = _FIRST_PARTY_TREES
) -> Iterator[Path]:
    """Yield each test-tree module that pytest imports by its bare file name.

    Args:
        repo_root: Repository root that the ``trees`` are resolved against.
        trees: Directory names under ``repo_root`` to walk.

    Yields:
        ``.py`` files under a ``tests`` directory whose own directory is not a
        package, excluding ``conftest.py``.
    """
    for tree in trees:
        for path in (repo_root / tree).rglob("*.py"):
            parts = path.relative_to(repo_root).parts
            if _SKIPPED_DIR_PARTS.intersection(parts) or "tests" not in parts:
                continue
            if path.name == "conftest.py" or (path.parent / "__init__.py").exists():
                continue
            yield path


def _duplicated_module_names(paths: Iterable[Path]) -> dict[str, list[Path]]:
    """Group module files by name and keep only the colliding groups.

    Args:
        paths: Module files to group.

    Returns:
        Module name to its files, for every name used by more than one file.
    """
    by_name: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        by_name[path.stem].append(path)
    return {name: files for name, files in by_name.items() if len(files) > 1}


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def test_when_repo_test_trees_are_scanned_then_no_bare_module_name_repeats():
    """The real repository has no two bare-imported test modules with one name."""
    duplicated = _duplicated_module_names(_bare_imported_test_modules(_REPO_ROOT))

    assert duplicated == {}, (
        "bare-imported test-tree modules share a name and collide in "
        f"sys.modules: {sorted(duplicated)}"
    )


def test_when_two_trees_ship_the_same_module_name_then_the_collision_is_reported(
    tmp_path,
):
    """Two non-package test trees with ``_import_scan.py`` are flagged."""
    for tree in ("alpha", "beta"):
        _touch(tmp_path / tree / "tests" / "_import_scan.py")

    found = _duplicated_module_names(
        _bare_imported_test_modules(tmp_path, ("alpha", "beta"))
    )

    assert list(found) == ["_import_scan"]
    assert len(found["_import_scan"]) == 2


def test_when_two_trees_ship_the_same_test_file_name_then_the_collision_is_reported(
    tmp_path,
):
    """A repeated ``test_*`` module is flagged just like a repeated helper."""
    for tree in ("alpha", "beta"):
        _touch(tmp_path / tree / "tests" / "test_no_ingestion_import.py")

    found = _duplicated_module_names(
        _bare_imported_test_modules(tmp_path, ("alpha", "beta"))
    )

    assert list(found) == ["test_no_ingestion_import"]


def test_when_one_module_lives_in_a_package_then_it_is_not_counted(tmp_path):
    """A module inside a package is imported by dotted name, so it cannot collide."""
    _touch(tmp_path / "bare" / "tests" / "_import_scan.py")
    _touch(tmp_path / "packaged" / "tests" / "_import_scan.py")
    _touch(tmp_path / "packaged" / "tests" / "__init__.py")

    found = _duplicated_module_names(
        _bare_imported_test_modules(tmp_path, ("bare", "packaged"))
    )

    assert found == {}


def test_when_conftest_files_repeat_then_they_are_not_counted(tmp_path):
    """Pytest special-cases ``conftest.py``, so repeating it is legitimate."""
    for tree in ("alpha", "beta"):
        _touch(tmp_path / tree / "tests" / "conftest.py")

    found = _duplicated_module_names(
        _bare_imported_test_modules(tmp_path, ("alpha", "beta"))
    )

    assert found == {}


def test_when_helpers_have_distinct_names_then_nothing_is_reported(tmp_path):
    """Differently named bare helpers coexist, which is the intended layout."""
    _touch(tmp_path / "alpha" / "tests" / "_fund_import_scan.py")
    _touch(tmp_path / "beta" / "tests" / "_import_scan.py")

    found = _duplicated_module_names(
        _bare_imported_test_modules(tmp_path, ("alpha", "beta"))
    )

    assert found == {}
