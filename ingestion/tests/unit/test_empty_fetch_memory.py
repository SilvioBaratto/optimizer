"""EmptyFetchMemory: remembers fetches Yahoo answered with nothing.

Targets: app/services/market_data/empty_fetch_memory.py
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from app.services.market_data.empty_fetch_memory import EmptyFetchMemory
from app.services.market_data.yfinance_data_service import StalenessThresholds

NOW = datetime(2026, 3, 1, 7, 0, tzinfo=timezone.utc)


class TestIsRecent:
    """A remembered fetch counts as recent for exactly the requested window."""

    def test_unknown_fetch_is_not_recent(self) -> None:
        """Return False for a fetch that was never remembered."""
        memory = EmptyFetchMemory()

        assert not memory.is_recent(uuid4(), "eps_trend_updated_at", 72, NOW)

    def test_fetch_inside_the_window_is_recent(self) -> None:
        """Return True while the remembered stamp is younger than the window."""
        memory = EmptyFetchMemory()
        instrument = uuid4()
        memory.remember(instrument, "eps_trend_updated_at", NOW - timedelta(hours=71))

        assert memory.is_recent(instrument, "eps_trend_updated_at", 72, NOW)

    def test_fetch_outside_the_window_is_not_recent(self) -> None:
        """Return False once the stamp is as old as the window."""
        memory = EmptyFetchMemory()
        instrument = uuid4()
        memory.remember(instrument, "eps_trend_updated_at", NOW - timedelta(hours=72))

        assert not memory.is_recent(instrument, "eps_trend_updated_at", 72, NOW)

    def test_memory_is_keyed_per_instrument_and_per_key(self) -> None:
        """Keep one entry per instrument and marker key."""
        memory = EmptyFetchMemory()
        instrument = uuid4()
        memory.remember(instrument, "eps_trend_updated_at", NOW)

        assert not memory.is_recent(uuid4(), "eps_trend_updated_at", 72, NOW)
        assert not memory.is_recent(instrument, "eps_revisions_updated_at", 72, NOW)

    def test_remembering_again_restarts_the_window(self) -> None:
        """Replace an older stamp with the newer one."""
        memory = EmptyFetchMemory()
        instrument = uuid4()
        memory.remember(instrument, "eps_trend_updated_at", NOW - timedelta(hours=100))
        memory.remember(instrument, "eps_trend_updated_at", NOW)

        assert memory.is_recent(instrument, "eps_trend_updated_at", 72, NOW)

    def test_concurrent_writers_do_not_lose_entries(self) -> None:
        """Survive the parallel bulk fetch, where every worker thread writes."""
        memory = EmptyFetchMemory()
        instruments = [uuid4() for _ in range(64)]
        keys = ("a", "b", "c", "d")

        def _remember(instrument: UUID) -> None:
            for key in keys:
                memory.remember(instrument, key, NOW)

        threads = [threading.Thread(target=_remember, args=(i,)) for i in instruments]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert all(
            memory.is_recent(instrument, key, 72, NOW)
            for instrument in instruments
            for key in keys
        )


class TestNoDataWindow:
    """The retry interval for empty answers is a configurable threshold."""

    def test_default_is_shorter_than_the_fundamentals_window(self) -> None:
        """Retry a fetch that merely failed well before a stored one expires."""
        thresholds = StalenessThresholds()

        assert 24 < thresholds.no_data_hours < thresholds.financials_hours
