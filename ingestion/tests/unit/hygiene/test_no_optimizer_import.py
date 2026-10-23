"""Guard: the ingestion daemon never depends on ``optimizer`` or the portfolio-
optimization stack (``skfolio``).

``scikit-learn`` is intentionally NOT forbidden: yfinance's price-repair path
(``repair=True``) imports ``sklearn.cluster.DBSCAN`` to reconstruct bad prints,
so it is a required data-layer dependency (see
``app/services/market_data/yfinance/_facade.py`` and
``tests/unit/test_price_history_repair_dependency.py``). ``scipy`` arrives
transitively through it. The boundary this guard protects is *optimization*, not
the presence of any ML library — the daemon ingests, it does not optimize.

Source-blind by construction: parses each file's syntax tree for import
statements (and string-literal ``import_module`` / ``__import__`` calls) and
scans the dependency manifest text. No implementation module is imported.

Anchoring reuses the shape established by ``test_no_http_surface.py`` (the
checkpoint guard): ``Path(__file__).resolve().parents[3]`` resolves directly
to the ``ingestion/`` directory from this file's location, never
``Path.cwd()``. The scans take an injectable ``root``/text so the
fail-injection tests below never mutate real source.

``_iter_python_files`` / ``_EXCLUDE_DIR_PARTS`` live in ``_shared_scan.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.hygiene._shared_scan import (
    _imported_module_names,
    _iter_python_files,
)

_INGESTION_ROOT = Path(__file__).resolve().parents[3]
_APP_ROOT = _INGESTION_ROOT / "app"
_PYPROJECT_FILE = _INGESTION_ROOT / "pyproject.toml"
_SELF = Path(__file__).resolve()

# Every tree whose code ships with or is driven by the daemon: the package, its
# operator scripts and its own tests. The boundary is about what the daemon's
# environment may import, not only what its entrypoint does.
_SCANNED_TREES = ("app", "scripts", "tests")
# Only the optimization stack is forbidden. scikit-learn is a required data-layer
# dep (yfinance price-repair) and scipy is its transitive dep, so neither is here.
_FORBIDDEN_STACK_PACKAGES = ("skfolio",)


def find_optimizer_import_violations(root: Path) -> list[str]:
    """Scan ``root`` for ``optimizer`` import statements.

    Injectable-root sibling of the checkpoint guard's ``find_http_violations``.

    Args:
        root: Directory tree to scan (an ``app/``-shaped root).

    Returns:
        One relative path per offending file, empty when clean.
    """
    offending: list[str] = []
    for path in _iter_python_files(root, exclude=_SELF):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(
            module.split(".", 1)[0] == "optimizer"
            for module in _imported_module_names(text, str(path))
        ):
            offending.append(str(path.relative_to(root)))
    return offending


def find_forbidden_stack_dependencies(dependencies_text: str) -> list[str]:
    """Return which sklearn-stack packages a dependency manifest declares.

    Args:
        dependencies_text: Raw contents of a dependency manifest (the daemon's
            ``pyproject.toml``).

    Returns:
        The subset of the forbidden optimization stack (``skfolio``) present,
        empty when the manifest declares none of them.
    """
    lowered = dependencies_text.lower()
    return [pkg for pkg in _FORBIDDEN_STACK_PACKAGES if pkg in lowered]


@pytest.mark.criterion("global-6")
def test_when_ingestion_app_is_scanned_then_no_optimizer_import_is_found():
    assert find_optimizer_import_violations(_APP_ROOT) == []


@pytest.mark.parametrize("tree", _SCANNED_TREES)
def test_when_an_ingestion_tree_is_scanned_then_it_is_non_empty_and_clean(tree):
    root = _INGESTION_ROOT / tree
    assert any(_iter_python_files(root)), f"{root} yields no files to scan"
    assert find_optimizer_import_violations(root) == []


@pytest.mark.criterion("global-6")
def test_when_ingestion_pyproject_is_read_then_no_optimization_stack_dependency_is_declared():
    content = _PYPROJECT_FILE.read_text(encoding="utf-8")
    assert find_forbidden_stack_dependencies(content) == []


def test_when_scikit_learn_is_declared_then_the_guard_allows_it():
    """scikit-learn is required by yfinance price-repair — it must NOT trip the
    optimization-stack guard."""
    assert find_forbidden_stack_dependencies("scikit-learn==1.9.0\n") == []


def test_when_ingestion_root_is_resolved_then_it_points_at_the_ingestion_directory():
    assert _INGESTION_ROOT.name == "ingestion"


def test_when_an_optimizer_import_is_injected_then_the_guard_fails(tmp_path):
    seeded = tmp_path / "offender.py"
    seeded.write_text("import optimizer\n", encoding="utf-8")

    violations = find_optimizer_import_violations(tmp_path)
    seeded.unlink()  # revert — tmp_path is ephemeral, cleanup made explicit

    assert violations, "injecting `import optimizer` must fail the guard"


def test_when_a_forbidden_stack_package_is_injected_then_the_guard_fails():
    violations = find_forbidden_stack_dependencies("skfolio==0.20.1\n")
    assert violations == ["skfolio"]


def test_when_own_source_is_scanned_then_it_is_excluded_from_the_tree():
    assert _SELF not in set(_iter_python_files(_INGESTION_ROOT, exclude=_SELF))


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
        "import importlib\nimportlib.import_module('sklearn.cluster')\n",
        "import importlib\nimportlib.import_module(name)\n",
    ],
    ids=["docstring", "lookalike-name", "other-dynamic-import", "non-literal-name"],
)
def test_when_source_only_mentions_or_resembles_an_optimizer_import_then_the_guard_allows_it(
    tmp_path, source
):
    (tmp_path / "clean.py").write_text(source, encoding="utf-8")
    assert find_optimizer_import_violations(tmp_path) == []
