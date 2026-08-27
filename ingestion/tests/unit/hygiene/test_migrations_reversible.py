"""Guard: every Alembic revision under ``ingestion/alembic/versions/``
defines both ``upgrade()`` and ``downgrade()``, and no ``downgrade()`` is
a bare no-op — unless ``upgrade()`` is itself bare, in which case there is
nothing to reverse and a bare ``downgrade()`` is correct.

Parses each revision with ``ast`` rather than regex — an AST walk
distinguishes a bare ``pass`` (or docstring-only) body from a
docstring-plus-``raise`` body cleanly, where a regex would not. A
``downgrade()`` that explicitly raises (declaring itself irreversible, e.g.
``d1e2f3a4b5c6_drop_non_ingestion_tables.py``) is a defined, non-bare
downgrade and satisfies the invariant; only a missing function, or one
whose body reduces to nothing but ``pass``/a docstring, is a violation —
and only when ``upgrade()`` performed real work (e.g.
``ecf0a9a2bfdc_add_macro_regime_tables.py``, whose ``upgrade()`` is a bare
auto-generated stub that creates nothing, so its bare ``downgrade()`` is
exempt rather than flagged).

The scan enumerates whatever revisions are present — it does not require a
migration to exist, so it passes vacuously in a cycle that adds none.

Anchored on ``Path(__file__).resolve().parents[3]`` to reach ``ingestion/``
(never ``Path.cwd()``, per the issue's explicit instruction), then
``alembic/versions``. ``__pycache__`` is skipped by construction: the
directory glob only matches ``*.py`` file siblings, never descends into it.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

# Alembic moved to packages/portopt-db (single migration owner). parents[4] is
# the repo root from tests/unit/hygiene/.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_VERSIONS_DIR = _REPO_ROOT / "packages" / "portopt-db" / "alembic" / "versions"


def _iter_revision_files() -> Iterator[Path]:
    if not _VERSIONS_DIR.exists():
        return
    for path in sorted(_VERSIONS_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        yield path


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _is_docstring_expr(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _is_bare_body(body: list[ast.stmt]) -> bool:
    """A body is bare when, after dropping any leading docstring, nothing
    remains but ``pass`` statements (or nothing at all).
    """
    meaningful = [stmt for stmt in body if not _is_docstring_expr(stmt)]
    if not meaningful:
        return True
    return all(isinstance(stmt, ast.Pass) for stmt in meaningful)


def _find_downgrade_violation(text: str) -> str | None:
    """Return a short reason string when ``text`` violates the reversible-
    migration invariant, else ``None``.

    A revision whose ``upgrade()`` body is itself bare is exempt outright:
    it changes no schema, so there is nothing a ``downgrade()`` could
    legitimately reverse, and a bare (or even missing) ``downgrade()`` is
    correct rather than a violation. The "every downgrade must be non-bare"
    rule below only applies once ``upgrade()`` has done real work.
    """
    tree = ast.parse(text)
    upgrade = _find_function(tree, "upgrade")
    if upgrade is not None and _is_bare_body(upgrade.body):
        return None
    downgrade = _find_function(tree, "downgrade")
    if downgrade is None:
        return "missing downgrade()"
    if _is_bare_body(downgrade.body):
        return "downgrade() body is bare"
    return None


def _scan_texts(texts: dict[str, str]) -> list[str]:
    return [name for name, text in texts.items() if _find_downgrade_violation(text)]


def _well_formed_revision_text(index: int) -> str:
    return (
        "def upgrade() -> None:\n    pass\n\n"
        f"def downgrade() -> None:\n    raise RuntimeError('irreversible-{index}')\n"
    )


@pytest.mark.criterion("global-8")
def test_when_the_real_versions_directory_is_scanned_then_all_current_revisions_pass():
    real_files = list(_iter_revision_files())

    assert len(real_files) > 0

    offending = [
        str(path.relative_to(_REPO_ROOT))
        for path in real_files
        if _find_downgrade_violation(path.read_text(encoding="utf-8"))
    ]
    assert offending == []


@pytest.mark.criterion("scope-1")
def test_when_the_real_versions_directory_is_scanned_then_each_revision_defines_upgrade_and_downgrade():
    offending: list[str] = []
    for path in _iter_revision_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        has_upgrade = _find_function(tree, "upgrade") is not None
        has_downgrade = _find_function(tree, "downgrade") is not None
        if not (has_upgrade and has_downgrade):
            offending.append(str(path.relative_to(_REPO_ROOT)))

    assert offending == []


_NON_BARE_UPGRADE = "def upgrade() -> None:\n    op.create_table('macro_regime')\n"


@pytest.mark.criterion("scope-2")
def test_when_a_revision_with_non_bare_upgrade_is_missing_downgrade_then_it_is_reported_as_a_violation():
    texts = {"missing_downgrade.py": _NON_BARE_UPGRADE}

    assert _scan_texts(texts) == ["missing_downgrade.py"]


@pytest.mark.criterion("scope-2")
def test_when_a_revision_with_non_bare_upgrade_has_a_bare_pass_downgrade_then_it_is_reported_as_a_violation():
    texts = {
        "bare_pass_downgrade.py": (
            f"{_NON_BARE_UPGRADE}\ndef downgrade() -> None:\n    pass\n"
        )
    }

    assert _scan_texts(texts) == ["bare_pass_downgrade.py"]


@pytest.mark.criterion("scope-2")
def test_when_a_revision_with_non_bare_upgrade_has_a_docstring_only_downgrade_then_it_is_reported_as_a_violation():
    texts = {
        "docstring_only_downgrade.py": (
            f'{_NON_BARE_UPGRADE}\ndef downgrade() -> None:\n    """Not implemented."""\n'
        )
    }

    assert _scan_texts(texts) == ["docstring_only_downgrade.py"]


@pytest.mark.criterion("scope-5")
def test_when_a_revision_has_a_bare_upgrade_then_its_bare_downgrade_is_exempt_not_a_violation():
    texts = {
        "bare_upgrade_bare_downgrade.py": (
            "def upgrade() -> None:\n    pass\n\ndef downgrade() -> None:\n    pass\n"
        )
    }

    assert _scan_texts(texts) == []


@pytest.mark.criterion("scope-5")
def test_when_a_revision_has_a_bare_upgrade_then_a_missing_downgrade_is_also_exempt():
    texts = {"bare_upgrade_missing_downgrade.py": "def upgrade() -> None:\n    pass\n"}

    assert _scan_texts(texts) == []


@pytest.mark.criterion("scope-3")
def test_when_the_drop_non_ingestion_tables_revision_is_scanned_then_its_raising_downgrade_is_not_reported():
    matches = list(_VERSIONS_DIR.glob("d1e2f3a4b5c6*.py"))
    assert len(matches) == 1, "expected the drop_non_ingestion_tables revision to exist"

    text = matches[0].read_text(encoding="utf-8")

    assert _find_downgrade_violation(text) is None


@pytest.mark.criterion("scope-4")
@pytest.mark.parametrize("revision_count", range(6))
def test_when_any_number_of_well_formed_revisions_are_scanned_then_none_are_reported(
    revision_count,
):
    """Cover 0..5 well-formed revisions, including the empty-directory case.

    Replaces a prior ``hypothesis`` ``@given`` property: the CI job installs
    only ``ingestion/pyproject.toml`` (deps + the ``[test]`` extra), which
    does not declare ``hypothesis``, so collecting a module-level
    ``from hypothesis import given`` raised
    ``ModuleNotFoundError`` and failed the whole ``ingestion-test`` job before
    a single test ran. ``_well_formed_revision_text`` is a hardcoded template
    whose ``downgrade()`` is a ``raise`` by construction, so the property added
    no coverage a fixed 0..5 sweep does not already give — including
    ``revision_count=0``, which exercises the same empty-directory path a
    dedicated vacuous test previously asserted in isolation
    (``assert _scan_texts({}) == []``, unconditionally true and unable to
    fail regardless of implementation).
    """
    texts = {
        f"revision_{i}.py": _well_formed_revision_text(i) for i in range(revision_count)
    }

    assert _scan_texts(texts) == []
