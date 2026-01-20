"""
Circuit breaker for yfinance API calls.

Implements circuit breaker pattern with exponential backoff
to handle rate limiting from Yahoo Finance API.
"""

import threading
import time
from dataclasses import dataclass, field


@dataclass
class CircuitBreaker:
    """
    Global circuit breaker with exponential backoff.

    When rate limiting is detected, blocks ALL API calls with
    exponentially increasing wait times:
    - Attempt 1: 2 minutes
    - Attempt 2: 4 minutes
    - Attempt 3: 8 minutes
    - And so on...

    Attributes:
        max_attempts: Maximum number of retry attempts before failing
        base_wait_minutes: Base wait time in minutes (default: 2)

    Example:
        breaker = CircuitBreaker()
        breaker.check()      # Blocks if active
        try:
            # API call
        except RateLimitError:
            breaker.trigger()  # Activates backoff
    """

    max_attempts: int = 10
    base_wait_minutes: float = 2.0
    _active: bool = field(default=False, repr=False)
    _until: float = field(default=0.0, repr=False)
    _attempt: int = field(default=0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def trigger(self) -> None:
        """
        Trigger global circuit breaker when rate limit is detected.

        Blocks ALL API calls with exponential backoff.
        Thread-safe - multiple threads detecting rate limit simultaneously
        will not increment the counter multiple times.
        """
        with self._lock:
            # If circuit breaker is already active, don't increment counter again
            if self._active:
                now = time.time()
                if now < self._until:
                    return

            self._attempt += 1
            wait_seconds = (2 ** self._attempt) * 60 * self.base_wait_minutes / 2
            resume_time = time.time() + wait_seconds

            self._until = resume_time
            self._active = True

    def check(self) -> None:
        """
        Check if circuit breaker is active and wait if necessary.

        Blocks the calling thread if circuit breaker is active,
        waiting until the cooldown period expires.

        Raises:
            RuntimeError: If circuit breaker has exceeded max attempts
        """
        should_wait = False
        wait_time = 0.0

        with self._lock:
            if self._active:
                now = time.time()
                if now < self._until:
                    should_wait = True
                    wait_time = self._until - now
                else:
                    self._active = False

            if self._attempt >= self.max_attempts:
                raise RuntimeError(
                    f"Yahoo Finance rate limit persists after {self._attempt} attempts. "
                    "Total wait time exceeded safety limit. Aborting to prevent infinite loop."
                )

        # Sleep outside the lock so other threads can check status
        if should_wait:
            time.sleep(wait_time)

    def reset(self) -> None:
        """
        Gradually reset circuit breaker after successful API calls.

        Should be called after successful requests to gradually
        reduce the backoff counter.
        """
        with self._lock:
            if self._attempt > 0:
                self._attempt = max(0, self._attempt - 1)

    @property
    def is_active(self) -> bool:
        """Check if circuit breaker is currently active."""
        with self._lock:
            if not self._active:
                return False
            now = time.time()
            if now >= self._until:
                self._active = False
                return False
            return True

    @property
    def attempt_count(self) -> int:
        """Get current attempt count."""
        with self._lock:
            return self._attempt

    def force_reset(self) -> None:
        """
        Force complete reset of circuit breaker.

        Use with caution - mainly for testing.
        """
        with self._lock:
            self._active = False
            self._until = 0.0
            self._attempt = 0
