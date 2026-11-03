"""Shared memory of whether the ticker news feed still answers."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

DEFAULT_FAILURE_THRESHOLD = 5
DEFAULT_COOLDOWN_SECONDS = 30 * 60.0
DEFAULT_PROBE_WINDOW_SECONDS = 60.0


class TickerFeedHealth:
    """Stop calling a ticker news feed that keeps failing, and test it now and then.

    Yahoo retired the route behind ``Ticker.news``. Its error page makes yfinance
    raise a generic "Yahoo Finance is currently down" error for every symbol, so a
    bulk sweep would spend several requests per ticker (the request, a cookie
    strategy flip and a crumb fetch, then a retry) only to fall back to search each
    time. After ``failure_threshold`` failures in a row the gate opens and callers
    go straight to the fallback. Once ``cooldown_seconds`` have passed one caller is
    let through as a probe while the others keep skipping the feed, so a feed that
    is fixed upstream is picked up again without a restart.

    The state lives in the process on purpose, like
    :class:`~app.services.market_data.empty_fetch_memory.EmptyFetchMemory`: it needs
    no schema and a restart costs at most ``failure_threshold`` wasted requests.

    Safe to share between threads: the parallel bulk fetch calls it from every
    worker.

    Args:
        failure_threshold: Consecutive failures that open the gate.
        cooldown_seconds: How long an open gate turns callers away before probing.
        probe_window_seconds: How long a probe has to report before another caller
            may take its place; it covers a probe that died on an unrelated error
            and never reported.
        clock: Monotonic time source in seconds; replaced in tests.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        probe_window_seconds: float = DEFAULT_PROBE_WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._probe_window_seconds = probe_window_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._failures = 0
        self._open = False
        self._retry_at = 0.0

    def allows_request(self) -> bool:
        """Report whether the caller should try the feed.

        Returns:
            True while the gate is closed. When it is open, False until the
            cooldown has passed; then True for exactly one caller, the probe, and
            False for the rest until the probe reports or its window runs out.
        """
        with self._lock:
            if not self._open:
                return True
            now = self._clock()
            if now < self._retry_at:
                return False
            self._retry_at = now + self._probe_window_seconds
            return True

    def record_success(self) -> None:
        """Note that the feed answered, however few articles it returned."""
        with self._lock:
            recovered = self._open
            self._failures = 0
            self._open = False
        if recovered:
            logger.info("ticker news feed recovered; using it again")

    def record_failure(self) -> None:
        """Note that the feed failed; the failure that reaches the threshold opens it."""
        with self._lock:
            self._failures += 1
            opened = False
            if self._failures >= self._failure_threshold:
                opened = not self._open
                self._open = True
                self._retry_at = self._clock() + self._cooldown_seconds
        if opened:
            logger.warning(
                "ticker news feed failed %d times in a row; falling back to Yahoo "
                "search and retrying the feed in %.0f min",
                self._failure_threshold,
                self._cooldown_seconds / 60,
            )
