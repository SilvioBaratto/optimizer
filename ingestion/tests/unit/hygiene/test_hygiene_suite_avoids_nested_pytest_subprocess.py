"""Example test for issue #15, [scope-2]: no test under ``tests/unit/hygiene/``
spawns a nested ``pytest`` subprocess, except the cwd-independence check in
``test_guards_detect_violations.py``.

Source-blind by construction: scans the raw text of tracked hygiene modules
for the ``"-m", "pytest"`` subprocess-argument shape. No implementation
module is imported or executed.

Ambiguity resolution: "spawns a nested pytest subprocess" is read as a file
whose text contains the literal ``[..., "-m", "pytest", ...]`` invocation
shape every real spawn in this suite uses — not a bare co-occurrence of the
words ``subprocess`` and ``pytest``, which would false-positive on every
hygiene module (each one does ``import pytest`` for its own test functions)
and on legitimate non-pytest subprocess use (``_shared_scan.py`` shells out
to ``git``; ``test_ruff_clean_after_fastapi_purge.py`` shells out to
``ruff``).

Scope resolution: issue #15's Context section enumerates the exact ten
nested-pytest launches it exists to remove — all ten sit in the six files
named there (the five deleted ``test_legacy_api_reference_*.py`` satellites
plus ``test_guards_detect_violations.py``). The deployment-suite guards
(``test_deployment_*.py``) and ``test_unit_guards_survive_not_integration_
filter.py`` predate this issue, pin unrelated acceptance criteria from
issues #7/#12 (verifying real subprocess pytest collection/exec behaviour
is their whole point), and are not named anywhere in issue #15 — they are
out of scope and allowlisted rather than modified.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_NESTED_PYTEST_INVOCATION = re.compile(r'"-m",\s*"pytest"')

_HYGIENE_DIR = Path(__file__).resolve().parent
_SELF = Path(__file__).resolve()
_ALLOWED_EXCEPTIONS = frozenset(
    {
        "test_guards_detect_violations.py",
        "test_deployment_suite_default_run_never_fails.py",
        "test_deployment_suite_default_run_skips_docker_teardown.py",
        "test_deployment_tests_excluded_by_default.py",
        "test_unit_guards_survive_not_integration_filter.py",
    }
)


def find_nested_pytest_subprocess_spawners(root: Path) -> list[str]:
    """Return hygiene-directory filenames that spawn a nested pytest process.

    Args:
        root: The hygiene test directory to scan.

    Returns:
        Relative filenames (other than the allowed exceptions) whose text
        contains the ``"-m", "pytest"`` subprocess-argument shape, empty
        when clean.
    """
    offending: list[str] = []
    for path in root.glob("*.py"):
        if path.resolve() == _SELF or path.name in _ALLOWED_EXCEPTIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if _NESTED_PYTEST_INVOCATION.search(text):
            offending.append(path.name)
    return offending


@pytest.mark.criterion("scope-2")
def test_when_hygiene_directory_is_scanned_then_no_disallowed_file_spawns_nested_pytest():
    assert find_nested_pytest_subprocess_spawners(_HYGIENE_DIR) == []


@pytest.mark.criterion("scope-2")
def test_when_a_synthetic_file_spawns_nested_pytest_then_the_scan_detects_it(tmp_path):
    seeded = tmp_path / "offender.py"
    seeded.write_text(
        'import subprocess\nsubprocess.run(["pytest", "-m", "pytest"])\n',
        encoding="utf-8",
    )

    offending = find_nested_pytest_subprocess_spawners(tmp_path)
    seeded.unlink()

    assert offending == ["offender.py"]
