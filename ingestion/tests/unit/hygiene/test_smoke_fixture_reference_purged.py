"""Example test for issue #3, criterion scope-5.

``cd ingestion && pytest tests/`` remains green after the smoke fixture is
deleted — meaning nothing in the tracked tree still imports or references
``smoke_prices.sql`` by name. A dangling reference to a deleted fixture is
what would make the suite fail, so this test asserts the absence of the
reference directly rather than re-running the whole suite from within itself.

Source-blind by construction: scans the raw text of tracked files for the
absence of a dead filename. No implementation module is imported or
exercised.

Anchored on ``Path(__file__).resolve().parents[4]`` for the repository root:
from ``ingestion/tests/unit/hygiene/test_smoke_fixture_reference_purged.py``,
parents[0]=hygiene, [1]=unit, [2]=tests, [3]=ingestion, [4]=repository root.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.unit.hygiene._shared_scan import _iter_git_tracked_relpaths

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SELF_RELATIVE = Path(__file__).resolve().relative_to(_REPO_ROOT).as_posix()


def _iter_repo_files() -> Iterator[str]:
    tracked = _iter_git_tracked_relpaths(_REPO_ROOT)
    assert tracked is not None, "git executable not found on PATH"
    for rel in tracked:
        if rel == _SELF_RELATIVE:
            continue
        yield rel


@pytest.mark.criterion("scope-5")
def test_when_repository_is_scanned_then_no_file_references_the_deleted_fixture():
    offending: list[str] = []
    for rel in _iter_repo_files():
        path = _REPO_ROOT / rel
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "smoke_prices" in text:
            offending.append(rel)

    assert offending == []
