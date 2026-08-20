"""Example test for issue #15, [scope-1]: the five
``test_legacy_api_reference_*.py`` satellite modules and
``test_legacy_api_comment_purge.py`` are removed from
``tests/unit/hygiene/``, with any genuinely unique assertion folded into
``test_no_legacy_api_references.py`` or ``test_guards_detect_violations.py``.

Source-blind by construction: this test only checks for the *absence* of
named files on disk. It does not import or execute any of them.

Ambiguity resolution: the criterion names the satellite files by role
("the five ... modules") without spelling out each filename. The five taken
here are every file matching ``test_legacy_api_reference_*.py`` present in
the hygiene directory at authoring time, which is the literal reading of
"satellite modules" for the ``test_no_legacy_api_references.py`` guard.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_HYGIENE_DIR = Path(__file__).resolve().parent

_REMOVED_SATELLITE_MODULES = (
    "test_legacy_api_reference_guard_is_collected.py",
    "test_legacy_api_reference_allowlist_covers_known_files.py",
    "test_legacy_api_reference_guard_path_resolution_is_cwd_independent.py",
    "test_legacy_api_reference_guard_enforces_no_stale_paths.py",
    "test_legacy_api_reference_pattern_excludes_safe_urls.py",
    "test_legacy_api_comment_purge.py",
)


@pytest.mark.criterion("scope-1")
@pytest.mark.parametrize("filename", _REMOVED_SATELLITE_MODULES)
def test_when_hygiene_directory_is_listed_then_satellite_module_is_absent(filename):
    assert not (_HYGIENE_DIR / filename).exists()


@pytest.mark.criterion("scope-1")
def test_when_hygiene_directory_is_listed_then_exactly_six_satellites_are_gone():
    surviving = [
        name for name in _REMOVED_SATELLITE_MODULES if (_HYGIENE_DIR / name).exists()
    ]
    assert surviving == []
