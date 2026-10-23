"""Exponential-backoff circuit breaker for rate-limited external services."""

import threading
import time
from dataclasses import dataclass, field


class CircuitOpenError(RuntimeError):
    """The breaker refused a call because the service keeps rate-limiting us.

    A ``RuntimeError`` subclass so callers that already catch the generic error
    keep working, while resilience wrappers can tell a refused call from any
    other failure.
    """


@dataclass
class CircuitBreaker:
    """Exponential-backoff circuit breaker for rate-limited services.

    Tracks consecutive failures and blocks callers via ``check()`` for a
    back-off window proportional to ``base_wait_minutes * 2**attempt``, capped
    at ``max_wait_minutes``.
    Decays the attempt counter on each successful recovery so the breaker
    self-heals over time.

    Args:
        service_name: Human-readable name used in error messages.
        max_attempts: Failure count at which ``check()`` raises
            ``RuntimeError`` instead of sleeping, for as long as the current
            back-off window is open.
        base_wait_minutes: Scaling factor for the back-off formula.
        max_wait_minutes: Upper bound on a single back-off window. Uncapped,
            the window reaches hours, and every worker sleeping through it
            keeps its job's heartbeat alive and outlasts the upstream ban.
    """

    service_name: str = "external service"
    max_attempts: int = 10
    base_wait_minutes: float = 2.0
    max_wait_minutes: float = 30.0
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
            wait_seconds = min(
                (2**self._attempt) * 60 * self.base_wait_minutes / 2,
                self.max_wait_minutes * 60,
            )
            resume_time = time.time() + wait_seconds

            self._until = resume_time
            self._active = True

    def check(self) -> None:
        """Block until the back-off window elapses, then return.

        Once the window has elapsed the breaker is half-open: the attempt
        counter decays by one and the caller proceeds as a probe, even when
        *max_attempts* had been reached.

        Raises:
            CircuitOpenError: If *max_attempts* has been reached and the back-off
                window is still open.
        """
        should_wait = False
        wait_time = 0.0

        with self._lock:
            now = time.time()
            if self._active and now < self._until:
                if self._attempt >= self.max_attempts:
                    raise CircuitOpenError(
                        f"{self.service_name} rate limit persists after "
                        f"{self._attempt} attempts. Backing off until cooldown "
                        "elapses, then retrying."
                    )
                should_wait = True
                wait_time = self._until - now
            elif self._active or self._attempt >= self.max_attempts:
                # Half-open: ``reset`` is the only other decrementer and it
                # needs a fetch to succeed, which a raising ``check`` never
                # lets run, so without this decay the breaker would stay
                # latched for the life of the process. ``is_active`` may have
                # cleared ``_active`` already, hence the attempt-count test.
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
