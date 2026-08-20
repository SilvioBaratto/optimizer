"""Example test for issue #15, [scope-7]: every invariant guarded before the
change is still guarded after it, demonstrated by the fail-injection tests in
``test_guards_detect_violations.py``.

Source-blind by construction: scans the raw text of
``test_guards_detect_violations.py`` for references to each of the six
hygiene guard modules it must exercise. No implementation module is
imported.

Ambiguity resolution: "demonstrated by the fail-injection tests" is read as
the checkpoint module importing a symbol from every guard module (the same
pattern it already uses for the HTTP-surface and legacy-API-reference
guards), rather than requiring a specific function-naming scheme this test
cannot see without reading the checkpoint's implementation.
"""

from __future__ import annotations

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
    "test_migrations_reversible",
)


@pytest.mark.criterion("scope-7")
@pytest.mark.parametrize("guard_module", _GUARD_MODULES)
def test_when_checkpoint_module_is_read_then_it_references_every_guard_module(
    guard_module,
):
    text = _CHECKPOINT_MODULE.read_text(encoding="utf-8")
    assert guard_module in text
