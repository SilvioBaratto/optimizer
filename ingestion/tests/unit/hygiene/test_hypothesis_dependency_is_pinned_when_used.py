"""Guard: ``hypothesis`` is either unused anywhere under ``ingestion/tests``,
or declared with an exact-pinned version in ``ingestion/requirements.txt``
(issue #11, scope-2).

Source-blind by construction: scans raw tracked-file text for the
``hypothesis`` import marker and for a pinned requirements-file entry. No
implementation module is imported.

Ambiguity resolution: "pinned version" is read as an exact pin (``==``)
rather than a floor (``>=``) or a bare unconstrained name — an unpinned
``hypothesis`` line would let a new Hypothesis release silently change
shrinking/example-generation behaviour between CI runs, which is exactly
what "pinned" is written to prevent.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest

_INGESTION_ROOT = Path(__file__).resolve().parents[3]
_TESTS_ROOT = _INGESTION_ROOT / "tests"
_REQUIREMENTS_FILE = _INGESTION_ROOT / "requirements.txt"
_SELF = Path(__file__).resolve()

_HYPOTHESIS_IMPORT_PATTERN = re.compile(
    r"^\s*(from hypothesis\b|import hypothesis\b)", re.MULTILINE
)
_HYPOTHESIS_PIN_PATTERN = re.compile(r"^hypothesis==\S+", re.MULTILINE)
_EXCLUDE_DIR_PARTS = {"__pycache__"}


def _iter_python_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*.py"):
        if _EXCLUDE_DIR_PARTS.intersection(path.parts):
            continue
        if path.resolve() == _SELF:
            continue
        yield path


def find_hypothesis_usages(root: Path) -> list[str]:
    """Return the relative paths of files under ``root`` importing hypothesis.

    Args:
        root: Directory tree to scan (a ``tests/``-shaped root).

    Returns:
        One relative path per file importing ``hypothesis``, empty when none.
    """
    usages: list[str] = []
    for path in _iter_python_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if _HYPOTHESIS_IMPORT_PATTERN.search(text):
            usages.append(str(path.relative_to(root)))
    return usages


def is_hypothesis_pinned(requirements_text: str) -> bool:
    """Return whether ``requirements_text`` declares an exact-pinned hypothesis.

    Args:
        requirements_text: Raw contents of a ``requirements.txt``.

    Returns:
        True when a ``hypothesis==<version>`` line is present.
    """
    return bool(_HYPOTHESIS_PIN_PATTERN.search(requirements_text))


@pytest.mark.criterion("scope-2")
def test_when_hypothesis_is_used_under_tests_then_it_is_pinned_in_requirements():
    usages = find_hypothesis_usages(_TESTS_ROOT)
    if not usages:
        pytest.skip("hypothesis is not used anywhere under ingestion/tests")

    content = _REQUIREMENTS_FILE.read_text(encoding="utf-8")
    assert is_hypothesis_pinned(content), (
        f"hypothesis used in {usages} but not pinned (==) in requirements.txt"
    )


@pytest.mark.criterion("scope-2")
def test_when_requirements_pins_hypothesis_then_the_pin_is_detected():
    assert is_hypothesis_pinned("hypothesis==6.100.0\n") is True


@pytest.mark.criterion("scope-2")
def test_when_requirements_declares_hypothesis_without_exact_pin_then_it_is_not_detected():
    assert is_hypothesis_pinned("hypothesis>=6.0\n") is False
    assert is_hypothesis_pinned("hypothesis\n") is False


@pytest.mark.criterion("scope-2")
def test_when_own_source_is_scanned_then_it_is_excluded_from_the_tree():
    assert _SELF not in set(_iter_python_files(_INGESTION_ROOT))
