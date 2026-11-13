"""Guard: a migration that cannot be downgraded says why, in a machine-readable place.

Standard 3.7 asks destructive migrations to be handled safely and documented. A
``downgrade`` that raises ``NotImplementedError`` is a deliberate one-way door; the
revision must name the reason in a module-level ``IRREVERSIBLE_REASON`` so a reviewer
and the migration process doc can see at a glance what was destroyed and where the
restore source is.
"""

from __future__ import annotations

import ast
from pathlib import Path

_VERSIONS = Path(__file__).resolve().parents[1] / "alembic" / "versions"

_EXPECTED_IRREVERSIBLE = {
    "a1b2c3d4e5f6",
    "a5ac0fa9b0ec",
    "a6f1c8d3e2b7",
    "b2c3d4e5f6a7",
    "b9c8d7e6f5a4",
    "d1e2f3a4b5c6",
}


def raises_in_downgrade(tree: ast.Module) -> bool:
    """Whether the module's ``downgrade`` raises ``NotImplementedError``."""
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "downgrade":
            for child in ast.walk(node):
                if not isinstance(child, ast.Raise) or child.exc is None:
                    continue
                raised = (
                    child.exc.func if isinstance(child.exc, ast.Call) else child.exc
                )
                if isinstance(raised, ast.Name) and raised.id == "NotImplementedError":
                    return True
    return False


def declared_reason(tree: ast.Module) -> str | None:
    """The module-level ``IRREVERSIBLE_REASON`` string, or None when absent."""
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(t, ast.Name) and t.id == "IRREVERSIBLE_REASON"
            for t in node.targets
        ):
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                return None
            return value if isinstance(value, str) else None
    return None


def _irreversible_revisions() -> dict[str, ast.Module]:
    found: dict[str, ast.Module] = {}
    for path in sorted(_VERSIONS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if raises_in_downgrade(tree):
            found[path.name.split("_", 1)[0]] = tree
    return found


def test_the_scan_finds_exactly_the_known_irreversible_revisions() -> None:
    """A new one-way revision must be added here on purpose, with its reason."""
    assert set(_irreversible_revisions()) == _EXPECTED_IRREVERSIBLE


def test_every_irreversible_revision_declares_a_reason() -> None:
    """Each one-way revision carries a non-empty ``IRREVERSIBLE_REASON``."""
    missing = [
        revision
        for revision, tree in _irreversible_revisions().items()
        if not (declared_reason(tree) or "").strip()
    ]
    assert missing == []


def test_a_revision_without_the_constant_is_reported() -> None:
    """The check can fail: a synthetic one-way revision has no reason."""
    tree = ast.parse("def downgrade():\n    raise NotImplementedError('x')\n")

    assert raises_in_downgrade(tree)
    assert declared_reason(tree) is None


def test_a_reversible_revision_is_not_flagged() -> None:
    """A downgrade that does work needs no reason."""
    tree = ast.parse("def downgrade():\n    op.drop_table('t')\n")

    assert not raises_in_downgrade(tree)
