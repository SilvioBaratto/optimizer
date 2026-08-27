"""Guard: the migrations-reversible hygiene guard contains no tautological
"vacuous" test — a test whose entire body is a single assertion that holds
for every possible implementation, so it can never turn red (issue #11,
scope-3).

Source-blind by construction: parses the guard file's raw text with ``ast``
and inspects test-function bodies structurally. No implementation module is
imported.

Ambiguity resolution: the criterion names one concrete example
(``assert _scan_texts({}) == []``, previously at
``test_migrations_reversible.py:162-163``) rather than a line range, which
would drift the moment the fix lands. This guard does not anchor on that
line number — it walks every ``test_``-prefixed function in the file and
flags any whose body reduces to exactly one ``assert`` comparing a call to
``_scan_texts`` against a *literal empty container* (``{}``, ``[]``, or
``()``) passed as the sole argument. That is the general shape of "scanning
nothing always finds nothing" the criterion objects to, not the specific
line — so the guard keeps working however the fix is written.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# The reversibility guard is now a sibling in packages/portopt-db/tests/.
_GUARD_FILE = Path(__file__).resolve().parent / "test_migrations_reversible.py"


def _is_empty_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Dict):
        return not node.keys
    if isinstance(node, (ast.List, ast.Tuple)):
        return not node.elts
    return False


def _is_vacuous_scan_assert(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.Assert):
        return False
    test = stmt.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return False
    left = test.left
    if not (isinstance(left, ast.Call) and isinstance(left.func, ast.Name)):
        return False
    if left.func.id != "_scan_texts":
        return False
    if len(left.args) != 1 or not _is_empty_literal(left.args[0]):
        return False
    return _is_empty_literal(test.comparators[0])


def _is_docstring_expr(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def find_vacuous_test_functions(source_text: str) -> list[str]:
    """Return names of test functions that assert only "empty scan is empty".

    Args:
        source_text: Raw Python source of the guard module.

    Returns:
        One function name per vacuous test found, empty when none remain.
    """
    tree = ast.parse(source_text)
    offending: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        meaningful = [stmt for stmt in node.body if not _is_docstring_expr(stmt)]
        if len(meaningful) == 1 and _is_vacuous_scan_assert(meaningful[0]):
            offending.append(node.name)
    return offending


@pytest.mark.criterion("scope-3")
def test_when_migrations_guard_source_is_scanned_then_no_vacuous_scan_assertion_remains():
    content = _GUARD_FILE.read_text(encoding="utf-8")
    assert find_vacuous_test_functions(content) == []


@pytest.mark.criterion("scope-3")
def test_when_a_function_body_is_a_single_assert_on_empty_scan_then_it_is_detected():
    source = (
        "def test_when_versions_dir_is_empty_then_scan_is_vacuous():\n"
        "    assert _scan_texts({}) == []\n"
    )
    assert find_vacuous_test_functions(source) == [
        "test_when_versions_dir_is_empty_then_scan_is_vacuous"
    ]


@pytest.mark.criterion("scope-3")
def test_when_a_function_body_asserts_on_a_non_empty_scan_then_it_is_not_detected():
    source = (
        "def test_when_versions_dir_has_a_revision_then_scan_reports_it():\n"
        "    assert _scan_texts({'a.py': 'bad'}) == ['a.py']\n"
    )
    assert find_vacuous_test_functions(source) == []


@pytest.mark.criterion("scope-3")
def test_when_a_function_has_additional_assertions_beyond_the_empty_scan_then_it_is_not_detected():
    source = (
        "def test_when_versions_dir_is_empty_then_scan_is_still_checked():\n"
        "    assert _scan_texts({}) == []\n"
        "    assert _scan_texts({'a.py': 'bad'}) == ['a.py']\n"
    )
    assert find_vacuous_test_functions(source) == []
