"""Per-key minimum-interval rate limiter."""

import threading
import time
from dataclasses import dataclass, field


@dataclass
class RateLimiter:
    """Minimum-interval rate limiter keyed by arbitrary string tokens.

    Args:
        delay: Minimum seconds between successive calls for the same key.
    """

    delay: float = 0.1
    _last_request: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def acquire(self, key: str) -> None:
        """Block until the minimum interval has elapsed for *key*."""
        with self._lock:
            now = time.time()
            last = self._last_request.get(key, 0)
            elapsed = now - last

            if elapsed < self.delay:
                sleep_time = self.delay - elapsed
                time.sleep(sleep_time)

            self._last_request[key] = time.time()

    def get_last_request_time(self, key: str) -> float | None:
        """Return the monotonic time of the last ``acquire`` for *key*, or ``None``."""
        with self._lock:
            return self._last_request.get(key)

    def clear(self) -> None:
        """Reset all per-key timestamps."""
        with self._lock:
            self._last_request.clear()
