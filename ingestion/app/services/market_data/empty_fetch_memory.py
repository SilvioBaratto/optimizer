"""In-process memory of yfinance fetches that Yahoo answered with nothing."""

import threading
from datetime import datetime, timedelta
from uuid import UUID


class EmptyFetchMemory:
    """Remember which ``(instrument, marker key)`` fetches last came back empty.

    Freshness markers in the database advance only when rows are written, so a
    category Yahoo holds nothing for (a ticker without analyst coverage) looks
    stale on every run and would be requested every time. The staleness gate
    consults this memory so a recent empty answer counts as settled.

    The memory lives in the process on purpose: it needs no schema, and a restart
    costs one extra request per empty category. The yfinance client reports
    "no data" and "retries exhausted" the same way, so an empty answer may hide a
    failed fetch; the gate's window bounds how long.

    Safe to share between threads: the parallel bulk fetch writes from every
    worker.
    """

    def __init__(self) -> None:
        self._stamps: dict[tuple[UUID, str], datetime] = {}
        self._lock = threading.Lock()

    def remember(self, instrument_id: UUID, key: str, at: datetime) -> None:
        """Record that the fetch behind *key* came back empty at *at*.

        Args:
            instrument_id: Instrument the fetch was for.
            key: Staleness-marker key identifying the category or statement
                group, e.g. ``"eps_trend_updated_at"``.
            at: Timezone-aware time of the empty answer; replaces any earlier one.
        """
        with self._lock:
            self._stamps[(instrument_id, key)] = at

    def is_recent(
        self, instrument_id: UUID, key: str, within_hours: int, now: datetime
    ) -> bool:
        """Report whether the fetch behind *key* came back empty recently.

        Args:
            instrument_id: Instrument the fetch was for.
            key: Staleness-marker key identifying the category or statement group.
            within_hours: Length of the window, measured back from *now*.
            now: Timezone-aware reference time.

        Returns:
            True when an empty answer was remembered less than *within_hours*
            before *now*; False when none was, or it has aged out.
        """
        with self._lock:
            stamp = self._stamps.get((instrument_id, key))
        return stamp is not None and (now - stamp) < timedelta(hours=within_hours)
