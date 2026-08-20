"""Example test for issue #3, criterion scope-3.

``ingestion/tests/fixtures/smoke_prices.sql`` is deleted, and the now-empty
``ingestion/tests/fixtures/`` directory is removed along with it.

Source-blind by construction: asserts on filesystem paths under the tracked
tree, no implementation module is imported or exercised.

Anchored on ``Path(__file__).resolve().parents[4]`` for the repository root:
from ``ingestion/tests/unit/hygiene/test_smoke_fixture_deleted.py``,
parents[0]=hygiene, [1]=unit, [2]=tests, [3]=ingestion, [4]=repository root.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FIXTURES_DIR = _REPO_ROOT / "ingestion" / "tests" / "fixtures"


@pytest.mark.criterion("scope-3")
def test_when_fixtures_directory_is_checked_then_smoke_prices_sql_is_absent():
    assert not (_FIXTURES_DIR / "smoke_prices.sql").exists()


@pytest.mark.criterion("scope-3")
def test_when_fixtures_directory_is_checked_then_it_no_longer_exists():
    assert not _FIXTURES_DIR.exists()
