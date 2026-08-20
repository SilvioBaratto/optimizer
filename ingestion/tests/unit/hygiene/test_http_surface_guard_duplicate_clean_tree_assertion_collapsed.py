"""Example test for issue #15, [scope-5]: the duplicate assertion pair at
``test_no_http_surface.py:160-161`` and ``:184-186`` (both asserting
``find_http_violations(_INGESTION_ROOT) == []``) is collapsed to one.

Source-blind by construction: counts literal occurrences of the duplicated
assertion expression in the tracked source text of the named module. No
implementation module is imported.

Ambiguity resolution: "collapsed to one" is read as: the exact expression
``find_http_violations(_INGESTION_ROOT) == []`` must appear exactly once in
the file's source text, regardless of which of the two original test
functions is kept or how it is renamed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_HYGIENE_DIR = Path(__file__).resolve().parent
_TARGET_MODULE = _HYGIENE_DIR / "test_no_http_surface.py"
_DUPLICATED_ASSERTION = "find_http_violations(_INGESTION_ROOT) == []"


@pytest.mark.criterion("scope-5")
def test_when_no_http_surface_module_is_read_then_the_clean_tree_assertion_appears_once():
    text = _TARGET_MODULE.read_text(encoding="utf-8")
    assert text.count(_DUPLICATED_ASSERTION) == 1
