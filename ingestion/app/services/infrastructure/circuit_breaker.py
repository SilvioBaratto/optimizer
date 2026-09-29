"""Exponential-backoff circuit breaker for rate-limited external services."""

import threading
import time
from dataclasses import dataclass, field


@dataclass
class CircuitBreaker:
    """Exponential-backoff circuit breaker for rate-limited services.

    Tracks consecutive failures and blocks callers via ``check()`` for a
    back-off window proportional to ``base_wait_minutes * 2**attempt``.
    Decays the attempt counter on each successful recovery so the breaker
    self-heals over time.

    Args:
        service_name: Human-readable name used in error messages.
        max_attempts: Failure count at which ``check()`` raises
            ``RuntimeError`` instead of sleeping.
        base_wait_minutes: Scaling factor for the back-off formula.
    """

    service_name: str = "external service"
    max_attempts: int = 10
    base_wait_minutes: float = 2.0
    _active: bool = field(default=False, repr=False)
    _until: float = field(default=0.0, repr=False)
    _attempt: int = field(default=0, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def trigger(self) -> None:
        """Record a failure and extend the back-off window.

        Idempotent while the current window is still open — a second trigger
        during an active back-off does not reset the window.
        """
        with self._lock:
            if self._active:
                now = time.time()
                if now < self._until:
                    return

            self._attempt += 1
            wait_seconds = (2**self._attempt) * 60 * self.base_wait_minutes / 2
            resume_time = time.time() + wait_seconds

            self._until = resume_time
            self._active = True

    def check(self) -> None:
        """Block until the back-off window elapses, then return.

        Raises:
            RuntimeError: If *max_attempts* has been reached.
        """
        should_wait = False
        wait_time = 0.0

        with self._lock:
            if self._attempt >= self.max_attempts:
                raise RuntimeError(
                    f"{self.service_name} rate limit persists after "
                    f"{self._attempt} attempts. Backing off until cooldown "
                    "elapses, then retrying."
                )

            if self._active:
                now = time.time()
                if now < self._until:
                    should_wait = True
                    wait_time = self._until - now
                else:
                    # Cooldown elapsed: half-open. Clear the breaker AND decay
                    # the attempt counter so a stuck breaker self-heals even
                    # when no success has been recorded. ``reset`` is the only
                    # other decrementer and it is unreachable while ``check``
                    # raises before the fetch runs — without this decay the
                    # breaker latches open permanently for the life of the
                    # process.
                    self._active = False
                    self._attempt = max(0, self._attempt - 1)

        # Sleep outside the lock so other threads can check status
        if should_wait:
            time.sleep(wait_time)

    def reset(self) -> None:
        """Decay the attempt counter by one on a successful outcome."""
        with self._lock:
            if self._attempt > 0:
                self._attempt = max(0, self._attempt - 1)

    @property
    def is_active(self) -> bool:
        """True while the back-off window has not yet elapsed."""
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
        """Current consecutive-failure count."""
        with self._lock:
            return self._attempt

    def force_reset(self) -> None:
        """Clear all state unconditionally (for testing or administrative recovery)."""
        with self._lock:
            self._active = False
            self._until = 0.0
            self._attempt = 0
