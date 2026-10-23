"""[scope-7]: every invariant guarded before the helper consolidation is still
guarded after it, demonstrated by the fail-injection tests in
``test_guards_detect_violations.py``.

Source-blind by construction: parses ``test_guards_detect_violations.py``
and checks its import statements for each hygiene guard module it must
exercise. No implementation module is imported.

Ambiguity resolution: "demonstrated by the fail-injection tests" is read as
the checkpoint module importing a symbol from every guard module (the same
pattern it already uses for the HTTP-surface and legacy-API-reference
guards), rather than requiring a specific function-naming scheme this test
cannot see without reading the checkpoint's implementation. Imports are read
from the syntax tree, never from raw text, so a guard that is merely named in
a comment does not count as exercised.

Guards that read one tracked file's text instead of scanning a tree
(``test_fastapi_dependency_removed``, ``test_ruff_toml_fastapi_reference_removed``,
``test_pytest_ini_header_describes_daemon``) are not listed here. They have no
walk, exclusion list or pattern that could silently match nothing: a missing
file raises instead of passing, and a reintroduced token fails the assertion
directly. They are deliberately not fault-injected; the only parsing helper
among them, ``declared_runtime_dependencies``, has injection tests in its own
module.
The migration-reversibility guard moved to ``packages/portopt-db`` together
with the Alembic tree and is fault-injected there.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_HYGIENE_DIR = Path(__file__).resolve().parent
_CHECKPOINT_MODULE = _HYGIENE_DIR / "test_guards_detect_violations.py"

_GUARD_MODULES = (
    "test_no_http_surface",
    "test_no_legacy_api_references",
    "test_synchronous_only",
    "test_no_optimizer_import",
    "test_shared_init_exports",
    "test_no_agent_stack_import",
    "test_no_llm_reference",
    "test_hypothesis_dependency_is_pinned_when_used",
)


@pytest.mark.criterion("scope-7")
@pytest.mark.parametrize("guard_module", _GUARD_MODULES)
def test_when_checkpoint_module_is_read_then_it_references_every_guard_module(
    guard_module,
):
    tree = ast.parse(_CHECKPOINT_MODULE.read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert f"tests.unit.hygiene.{guard_module}" in imported
