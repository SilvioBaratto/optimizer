"""Guard: ``uv.lock`` must not pin third-party packages below their advisory fixes.

Both Docker images install with ``uv sync --frozen``, so whatever the lockfile pins is
exactly what ships; CI's ``pip-audit`` only runs on pull requests to ``main`` and so
never sees a stale lock on ``development``. Each floor below is the first release
that fixes the published advisories for that package (``pip-audit`` fix versions).
Raise a floor when a new advisory lands; never lower one.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import tomllib
from packaging.version import Version

_UV_LOCK = Path(__file__).resolve().parents[1] / "uv.lock"

_ADVISORY_FLOORS = {
    # PYSEC-2026-4320: bundle trusts emSign roots whose intermediates were revoked.
    "certifi": "2026.10.19",
    # PYSEC-2026-4175/4176/4177: proxy TLS policy override, stream loop, chunk DoS.
    "urllib3": "2.8.0",
    "aiohttp": "3.14.4",
    # GHSA-4f8h-vr72-p32c: 21.14.9 is still flagged; 21.14.10 audits clean.
    "virtualenv": "21.14.10",
}


def _locked_versions(lock_text: str) -> dict[str, list[Version]]:
    lock = tomllib.loads(lock_text)
    versions: dict[str, list[Version]] = {}
    for package in lock["package"]:
        if "version" in package:
            versions.setdefault(package["name"], []).append(Version(package["version"]))
    return versions


def _stale_copies(lock_text: str, name: str) -> list[str] | None:
    """Return the locked versions of ``name`` below its floor, or None if unlocked."""
    locked = _locked_versions(lock_text).get(name)
    if locked is None:
        return None
    floor = Version(_ADVISORY_FLOORS[name])
    return sorted(str(v) for v in locked if v < floor)


@pytest.mark.parametrize(("name", "floor"), sorted(_ADVISORY_FLOORS.items()))
def test_locked_package_is_at_or_above_its_advisory_fix(name: str, floor: str) -> None:
    """Every locked copy of the package is at or above the advisory-fix release."""
    stale = _stale_copies(_UV_LOCK.read_text(encoding="utf-8"), name)
    if stale is None:
        pytest.skip(f"{name} is no longer in uv.lock")
    assert not stale, f"uv.lock pins {name} {stale}; advisory fix is {floor}"


def test_virtualenv_floor_rejects_the_last_vulnerable_release() -> None:
    """virtualenv 21.14.9 still carries GHSA-4f8h-vr72-p32c; only 21.14.10 clears it."""
    lock_text = '[[package]]\nname = "virtualenv"\nversion = "21.14.9"\n'
    assert _stale_copies(lock_text, "virtualenv") == ["21.14.9"]
