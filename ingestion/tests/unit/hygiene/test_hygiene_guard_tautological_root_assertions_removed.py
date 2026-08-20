"""Example test for issue #15, [scope-4]: all six tautological
``parents[3] == _INGESTION_ROOT`` assertions are removed from the hygiene
guard suite; the meaningful ``.name == "ingestion"`` companion assertions are
retained.

Source-blind by construction: scans the raw text of tracked hygiene modules
for the two assertion shapes. No implementation module is imported.

Ambiguity resolution: "tautological" is read as literally comparing a value
derived from ``Path(__file__).resolve().parents[3]`` back to the
module-level constant that was itself defined as
``Path(__file__).resolve().parents[3]`` — an assertion that can never fail
given the module imported successfully. The companion ``.name == "ingestion"``
assertion is meaningful (it fails if the anchor ever points at the wrong
directory) and must survive.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_HYGIENE_DIR = Path(__file__).resolve().parent
_SELF = Path(__file__).resolve()

_TAUTOLOGICAL_PATTERN = re.compile(r"parents\[3\]\s*==\s*_INGESTION_ROOT")
_MEANINGFUL_NAME_PATTERN = re.compile(r'\.name\s*==\s*"ingestion"')


def _hygiene_python_texts() -> dict[str, str]:
    texts = {}
    for path in _HYGIENE_DIR.glob("*.py"):
        if path.resolve() == _SELF:
            continue
        try:
            texts[path.name] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    return texts


@pytest.mark.criterion("scope-4")
def test_when_hygiene_directory_is_scanned_then_no_tautological_parents_assertion_survives():
    texts = _hygiene_python_texts()
    offending = [
        name for name, text in texts.items() if _TAUTOLOGICAL_PATTERN.search(text)
    ]
    assert offending == []


@pytest.mark.criterion("scope-4")
def test_when_hygiene_directory_is_scanned_then_meaningful_name_assertion_is_retained():
    texts = _hygiene_python_texts()
    surviving = [
        name for name, text in texts.items() if _MEANINGFUL_NAME_PATTERN.search(text)
    ]
    assert surviving != []
