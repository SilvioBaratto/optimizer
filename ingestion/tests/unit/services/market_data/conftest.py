"""Shared fixtures for the market-data service unit tests."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator

import pytest


@pytest.fixture
def host_timezone(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[str], None]]:
    """Let a test switch the process-wide local timezone via ``TZ``.

    Yields:
        A callable taking an IANA zone name. The original zone is restored
        after the test, including the C library's cached copy.
    """
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is unavailable on this platform")

    def _set(name: str) -> None:
        monkeypatch.setenv("TZ", name)
        time.tzset()

    yield _set
    monkeypatch.undo()
    time.tzset()
