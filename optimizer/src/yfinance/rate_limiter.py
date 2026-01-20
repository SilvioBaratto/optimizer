"""
Rate limiter for yfinance API calls.

Thread-safe rate limiter with per-key tracking to avoid
API throttling.
"""

import threading
import time
from dataclasses import dataclass, field


@dataclass
class RateLimiter:
    """
    Thread-safe rate limiter with per-key tracking.

    Ensures minimum delay between requests for the same key
    (e.g., ticker symbol) to avoid API throttling.

    Attributes:
        delay: Minimum delay between requests in seconds

    Example:
        limiter = RateLimiter(delay=0.1)
        limiter.acquire("AAPL")  # First call - no delay
        limiter.acquire("AAPL")  # Waits if < 0.1s since last call
    """

    delay: float = 0.1
    _last_request: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def acquire(self, key: str) -> None:
        """
        Acquire rate limit slot for a key.

        Blocks if necessary to enforce minimum delay between
        requests for the same key.

        Args:
            key: Rate limit key (e.g., ticker symbol)
        """
        with self._lock:
            now = time.time()
            last = self._last_request.get(key, 0)
            elapsed = now - last

            if elapsed < self.delay:
                sleep_time = self.delay - elapsed
                time.sleep(sleep_time)

            self._last_request[key] = time.time()

    def get_last_request_time(self, key: str) -> float | None:
        """
        Get last request timestamp for a key.

        Args:
            key: Rate limit key

        Returns:
            Unix timestamp of last request or None if never requested
        """
        with self._lock:
            return self._last_request.get(key)

    def clear(self) -> None:
        """Clear all tracked request times."""
        with self._lock:
            self._last_request.clear()
