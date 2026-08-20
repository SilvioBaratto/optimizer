"""Guard: the ingestion daemon stays synchronous-only — no ``AsyncSession``,
``asyncpg``, or asyncio worker primitive anywhere under ``ingestion/app``
(issue #8).

Source-blind by construction: scans the raw text of tracked ``app/`` files
for dead async markers. No implementation module is imported — this is a
pure text-content guard.

Anchoring reuses the shape established by ``test_no_http_surface.py`` (issue
#5, the checkpoint guard): ``Path(__file__).resolve().parents[3]`` resolves
directly to the ``ingestion/`` directory from this file's location
(``ingestion/tests/unit/hygiene/``) — no separate climb to the repository
root, and never ``Path.cwd()``. The scan takes an injectable ``root`` so the
fail-injection test below can point it at a synthetic tree instead of
mutating real source.

``_iter_python_files`` / ``_EXCLUDE_DIR_PARTS`` live in ``_shared_scan.py``
(issue #15) — this module used to carry a byte-identical copy of both,
shared verbatim with ``test_no_optimizer_import.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.unit.hygiene._shared_scan import _iter_python_files

_INGESTION_ROOT = Path(__file__).resolve().parents[3]
_APP_ROOT = _INGESTION_ROOT / "app"
_SELF = Path(__file__).resolve()

_FORBIDDEN_MARKERS = ("AsyncSession", "asyncpg", "asyncio.", "async def ")


def find_async_violations(root: Path) -> list[str]:
    """Scan ``root`` for dead async markers.

    Injectable-root sibling of the checkpoint guard's
    ``find_http_violations`` (issue #5): a caller can point the scan at a
    synthetic tree instead of the real ``ingestion/app`` root.

    Args:
        root: Directory tree to scan (an ``app/``-shaped root).

    Returns:
        One string per offending marker occurrence, empty when clean.
    """
    offending: list[str] = []
    for path in _iter_python_files(root, exclude=_SELF):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative = path.relative_to(root)
        for marker in _FORBIDDEN_MARKERS:
            if marker in text:
                offending.append(f"{relative}: {marker!r}")
    return offending


@pytest.mark.criterion("global-5")
def test_when_ingestion_app_is_scanned_then_no_async_marker_is_found():
    assert find_async_violations(_APP_ROOT) == []


def test_when_ingestion_root_is_resolved_then_it_points_at_the_ingestion_directory():
    assert _INGESTION_ROOT.name == "ingestion"


def test_when_an_async_session_marker_is_injected_then_the_guard_fails(tmp_path):
    seeded = tmp_path / "offender.py"
    seeded.write_text("session: AsyncSession = ...\n", encoding="utf-8")

    violations = find_async_violations(tmp_path)
    seeded.unlink()  # revert — tmp_path is ephemeral, cleanup made explicit

    assert violations, "injecting AsyncSession must fail the guard"


def test_when_an_asyncpg_marker_is_injected_then_the_guard_fails(tmp_path):
    seeded = tmp_path / "offender.py"
    seeded.write_text("import asyncpg\n", encoding="utf-8")

    violations = find_async_violations(tmp_path)
    seeded.unlink()

    assert violations, "injecting asyncpg must fail the guard"


def test_when_own_source_is_scanned_then_it_is_excluded_from_the_tree():
    # This file's own text names AsyncSession/asyncpg to forbid them, which
    # would self-match if a scan root ever widened to include it.
    assert _SELF not in set(_iter_python_files(_INGESTION_ROOT, exclude=_SELF))
