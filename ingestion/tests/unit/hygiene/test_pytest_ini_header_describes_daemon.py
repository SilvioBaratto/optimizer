"""Example test for issue #3, criterion scope-1.

``ingestion/pytest.ini`` carries a header comment. The criterion requires the
header to no longer describe the tree as "FastAPI application" and to instead
describe the headless ingestion daemon it actually is.

Ambiguity resolved in this test: the criterion names the phrase "FastAPI
application" verbatim as the one to remove, and "headless ingestion daemon" as
the replacement description in spirit. ``pytest.ini`` is INI syntax, so a
``[section]`` marker starts on line 1 before any header comment — scoping the
check to "before the first bracket" would inspect an empty string. This test
instead scans the whole file for the absence of the removed phrase and the
presence of both "headless" and "ingestion daemon", rather than requiring one
exact sentence — the criterion does not mandate specific wording beyond that
meaning.

Source-blind by construction: reads the raw text of a tracked config file, no
implementation module is imported or exercised.

Anchored on ``Path(__file__).resolve().parents[4]`` for the repository root:
from ``ingestion/tests/unit/hygiene/test_pytest_ini_header_describes_daemon.py``,
parents[0]=hygiene, [1]=unit, [2]=tests, [3]=ingestion, [4]=repository root.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PYTEST_INI = _REPO_ROOT / "ingestion" / "pytest.ini"


@pytest.mark.criterion("scope-1")
def test_when_pytest_ini_header_is_read_then_it_describes_the_headless_daemon():
    text = _PYTEST_INI.read_text(encoding="utf-8").lower()

    assert "fastapi application" not in text
    assert "headless" in text
    assert "ingestion daemon" in text
