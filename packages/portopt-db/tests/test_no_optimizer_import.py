"""Guard: portopt-db must never import the ``optimizer`` library or declare the
optimization stack (``skfolio``, ``scikit-learn`` or ``portopt-core``, which
pulls both in).

portopt-db is the shared data layer (models / repositories / engine / Alembic).
It sits *below* ingestion and any future consumer; pulling in the optimizer /
skfolio stack would couple the database package to portfolio optimization.
Source-blind scan of the package source and its Alembic tree (migrations run
in the same environment) + its pyproject. Imports are read from the syntax
tree, so ``import os, optimizer`` and ``importlib.import_module("optimizer")``
are caught and a docstring quoting an import is not.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from _import_scan import declared_distribution_names, imported_module_names

_PKG_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PKG_ROOT / "src" / "portopt_db"
_ALEMBIC = _PKG_ROOT / "alembic"
_PYPROJECT = _PKG_ROOT / "pyproject.toml"
_SELF = Path(__file__).resolve()

# ``portopt-core`` (import name ``optimizer``) is listed because it is how the
# stack would arrive: its manifest names skfolio/scikit-learn, ours would not.
_FORBIDDEN_PACKAGES = ("skfolio", "scikit-learn", "portopt-core", "optimizer")


def _iter_python_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*.py"):
        # Exclude self so the guard pattern in this file never trips the scan.
        if "__pycache__" in path.parts or path.resolve() == _SELF:
            continue
        yield path


def find_optimizer_import_violations(root: Path) -> list[str]:
    """Return paths of files that import the optimizer library, relative to root.

    Args:
        root: Directory tree to scan recursively for ``.py`` files.

    Returns:
        Relative path strings for every file containing a bare
        ``import optimizer`` or ``from optimizer …`` statement.
    """
    offending: list[str] = []
    for path in _iter_python_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(
            module.split(".", 1)[0] == "optimizer"
            for module in imported_module_names(text, str(path))
        ):
            offending.append(str(path.relative_to(root)))
    return offending


def find_forbidden_stack_dependencies(text: str) -> list[str]:
    """Return forbidden distributions required by a pyproject.toml body.

    Args:
        text: Raw text content of a pyproject.toml file.

    Returns:
        Names from ``_FORBIDDEN_PACKAGES`` declared as a requirement (name
        compared PEP 503-normalized, so case and ``-``/``_`` spelling do not
        matter). Comments and lookalike names do not count.
    """
    declared = declared_distribution_names(text)
    return [pkg for pkg in _FORBIDDEN_PACKAGES if pkg in declared]


@pytest.mark.parametrize("tree", [_SRC, _ALEMBIC], ids=["src", "alembic"])
def test_when_a_package_tree_is_scanned_then_it_is_non_empty_and_has_no_optimizer_import(
    tree,
):
    assert any(_iter_python_files(tree)), f"{tree} yields no files to scan"
    assert find_optimizer_import_violations(tree) == []


def test_when_pyproject_is_read_then_no_optimization_stack_dependency_is_declared():
    assert (
        find_forbidden_stack_dependencies(_PYPROJECT.read_text(encoding="utf-8")) == []
    )


def test_when_an_optimizer_import_is_injected_then_the_guard_fails(tmp_path):
    (tmp_path / "offender.py").write_text("import optimizer\n", encoding="utf-8")
    assert find_optimizer_import_violations(tmp_path)


def _pyproject(
    *,
    dependencies: str = "[]",
    extras: str = "",
    groups: str = "",
    description: str = "db layer",
) -> str:
    """Build a minimal ``pyproject.toml`` body for the manifest fail-injection tests."""
    return (
        "[project]\n"
        'name = "x"\n'
        f'description = "{description}"\n'
        f"dependencies = {dependencies}\n"
        f"[project.optional-dependencies]\n{extras}\n"
        f"[dependency-groups]\n{groups}\n"
    )


def test_when_a_forbidden_stack_package_is_injected_then_the_guard_fails():
    manifest = _pyproject(dependencies='["skfolio==1.0.0"]')
    assert find_forbidden_stack_dependencies(manifest) == ["skfolio"]


@pytest.mark.parametrize(
    ("manifest", "expected"),
    [
        (_pyproject(dependencies='["portopt-core"]'), ["portopt-core"]),
        (_pyproject(dependencies='["portopt-core==0.1.0"]'), ["portopt-core"]),
        (_pyproject(dependencies='["optimizer>=1"]'), ["optimizer"]),
        (_pyproject(dependencies='["scikit-learn==1.9.1"]'), ["scikit-learn"]),
        (
            _pyproject(
                dependencies="[\"Portopt_Core[extra]>=0.1; python_version >= '3.12'\"]"
            ),
            ["portopt-core"],
        ),
        (
            _pyproject(dependencies='["portopt-core @ file:///tmp/portopt-core"]'),
            ["portopt-core"],
        ),
        (_pyproject(extras='stack = ["portopt-core"]'), ["portopt-core"]),
        (
            _pyproject(groups='dev = ["pytest", {include-group = "x"}, "skfolio"]'),
            ["skfolio"],
        ),
        (
            _pyproject(dependencies='["skfolio", "portopt-core"]'),
            ["skfolio", "portopt-core"],
        ),
    ],
    ids=[
        "portopt-core",
        "pinned",
        "optimizer",
        "scikit-learn",
        "extras-marker-and-case",
        "direct-reference",
        "optional-dependencies",
        "dependency-groups",
        "several",
    ],
)
def test_when_a_forbidden_distribution_is_declared_in_any_manifest_table_then_the_guard_fails(
    manifest, expected
):
    assert find_forbidden_stack_dependencies(manifest) == expected


def test_when_a_manifest_only_mentions_the_stack_in_prose_then_the_guard_allows_it():
    manifest = "# never add skfolio or portopt-core here\n" + _pyproject(
        dependencies='["pandas==3.0.6"]',
        description="runs without skfolio, portopt-core or optimizer",
    )
    assert find_forbidden_stack_dependencies(manifest) == []


def test_when_a_distribution_merely_resembles_a_forbidden_name_then_the_guard_allows_it():
    manifest = _pyproject(dependencies='["skfolio-extras==1", "portopt-core-docs"]')
    assert find_forbidden_stack_dependencies(manifest) == []


@pytest.mark.parametrize(
    "source",
    [
        "import numpy, optimizer\n",
        "from  optimizer import x\n",
        "from\toptimizer import x\n",
        "x = 1; import optimizer\n",
        "if True: import optimizer\n",
        "import numpy, \\\n    optimizer\n",
        "import importlib\nimportlib.import_module('optimizer.moments')\n",
        "__import__('optimizer')\n",
    ],
    ids=[
        "comma-list",
        "double-space",
        "tab",
        "semicolon",
        "inline-if",
        "backslash-continuation",
        "import_module",
        "dunder-import",
    ],
)
def test_when_a_non_canonical_optimizer_import_is_injected_then_the_guard_fails(
    tmp_path, source
):
    (tmp_path / "offender.py").write_text(source, encoding="utf-8")
    assert find_optimizer_import_violations(tmp_path) == ["offender.py"]


@pytest.mark.parametrize(
    "source",
    [
        '"""Usage:\n\nfrom optimizer import build_mean_risk\n"""\n',
        "from optimizer_extras import x\n",
        "import importlib\nimportlib.import_module('sqlalchemy')\n",
        "import importlib\nimportlib.import_module(name)\n",
    ],
    ids=["docstring", "lookalike-name", "other-dynamic-import", "non-literal-name"],
)
def test_when_source_only_mentions_or_resembles_an_optimizer_import_then_the_guard_allows_it(
    tmp_path, source
):
    (tmp_path / "clean.py").write_text(source, encoding="utf-8")
    assert find_optimizer_import_violations(tmp_path) == []
