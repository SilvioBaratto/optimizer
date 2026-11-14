"""Circuit breaker that stops a dead database from being hammered.

``DatabaseManager`` holds one ``DbCircuitBreaker`` and consults it on every
``get_session``. After ``failure_threshold`` consecutive connectivity failures the
breaker opens: sessions are refused at once with ``DatabaseUnavailableError`` and
the engine is never touched, so a daemon whose database is gone fails its work in
milliseconds instead of waiting out a connect timeout per call. Once
``reset_timeout_s`` has passed, exactly one probe session is admitted; its outcome
closes the breaker or opens it for another full timeout.

Only connectivity failures count. SQLAlchemy wraps driver errors, and psycopg2's
statement-timeout (``QueryCanceled``, 57014) and lock-timeout (``LockNotAvailable``,
55P03) errors subclass the driver's ``OperationalError``, so the class alone would
let one slow query open the breaker on a healthy database. The SQLSTATE decides
instead: an error the server answered with proves it is reachable.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.exc import DBAPIError, DisconnectionError, OperationalError

logger = logging.getLogger(__name__)

# SQLSTATE class 08 is "connection exception". The 57P0x codes are the server
# shutting down, crashing or still starting up: the session is gone or never came.
_CONNECTION_EXCEPTION_CLASS = "08"
_SERVER_GONE_SQLSTATES = frozenset({"57P01", "57P02", "57P03"})

__all__ = [
    "BreakerConfig",
    "BreakerState",
    "DatabaseUnavailableError",
    "DbCircuitBreaker",
    "is_connectivity_failure",
]


class BreakerState(str, Enum):
    """Where the breaker stands.

    Attributes:
        CLOSED: Sessions pass; connectivity failures are counted.
        OPEN: Sessions are refused until the reset timeout has passed.
        HALF_OPEN: One probe session is admitted; every other caller is refused.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class BreakerConfig:
    """When the breaker opens and how long it stays open.

    Attributes:
        failure_threshold: Consecutive connectivity failures that open the breaker.
        reset_timeout_s: Seconds an open breaker refuses sessions before it admits
            one probe; a failed probe restarts the full timeout.

    Raises:
        ValueError: Either value is not a finite number above zero.
    """

    failure_threshold: int = 5
    reset_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        for name in ("failure_threshold", "reset_timeout_s"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a finite number > 0, got {value!r}")


class DatabaseUnavailableError(RuntimeError):
    """A session was refused because the database circuit breaker is open.

    Raised by ``DatabaseManager.get_session`` before any engine or session is
    created, so nothing was read or written. Its message is one line without
    host, credentials or SQL, safe to show an operator.
    """


def _sqlstate(orig: object) -> str | None:
    # psycopg2 names it ``pgcode``, psycopg 3 ``sqlstate``.
    code = getattr(orig, "pgcode", None) or getattr(orig, "sqlstate", None)
    return code if isinstance(code, str) else None


def is_connectivity_failure(exc: BaseException) -> bool:
    """Report whether ``exc`` means the database could not be reached.

    Counted: ``DisconnectionError`` (raised by the pool's pre-ping), any
    ``DBAPIError`` SQLAlchemy marked ``connection_invalidated``, a driver error
    whose SQLSTATE is in class 08 or is 57P01/57P02/57P03, and an
    ``OperationalError`` without a SQLSTATE raised outside a statement, which is
    how a refused or timed-out connect at pool checkout surfaces.

    Not counted: any other SQLSTATE (statement and lock timeouts, integrity,
    programming and data errors: the server answered), an ``OperationalError``
    raised by a statement without a SQLSTATE, a pool-exhaustion timeout, and
    every non-driver exception.

    Args:
        exc: The exception that ended a database call.

    Returns:
        True when the failure should count toward opening the breaker.
    """
    if isinstance(exc, DisconnectionError):
        return True
    if not isinstance(exc, DBAPIError):
        return False
    if exc.connection_invalidated:
        return True
    sqlstate = _sqlstate(exc.orig)
    if sqlstate is not None:
        return (
            sqlstate.startswith(_CONNECTION_EXCEPTION_CLASS)
            or sqlstate in _SERVER_GONE_SQLSTATES
        )
    return isinstance(exc, OperationalError) and exc.statement is None


class DbCircuitBreaker:
    """Thread-safe circuit breaker around one database's sessions.

    Every state change happens under one lock: the state, the consecutive failure
    count, the time the breaker opened and the probe slot. Outcomes are not tied
    to the call that produced them. While OPEN they are ignored, so a session
    admitted before the breaker opened can neither close it nor extend it; only
    the timed probe leaves OPEN. While HALF_OPEN the first outcome to land settles
    the probe, which may be a session admitted before the breaker opened: its
    result is equally fresh evidence about the same database.

    Args:
        config: Threshold and reset timeout.
        clock: Monotonic seconds; injectable so tests never sleep.
    """

    def __init__(
        self, config: BreakerConfig, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._config = config
        self._clock = clock
        self._lock = threading.Lock()
        self._state = BreakerState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._probe_in_flight = False

    @property
    def config(self) -> BreakerConfig:
        """The threshold and reset timeout this breaker was built with."""
        return self._config

    @property
    def state(self) -> BreakerState:
        """The current state.

        An OPEN breaker whose timeout has passed still reads OPEN until a call
        arrives and is admitted as the probe.
        """
        with self._lock:
            return self._state

    def refusal(self) -> DatabaseUnavailableError | None:
        """Return the error a call made now would be refused with, or None.

        Read-only: unlike ``before_call`` it never admits the probe, so a caller
        can grey out work while the database is known to be unreachable.
        """
        with self._lock:
            return self._refusal_locked()

    def before_call(self) -> None:
        """Admit a call or refuse it.

        CLOSED admits. OPEN refuses until the reset timeout has passed, then moves
        to HALF_OPEN and admits this call as the single probe. HALF_OPEN refuses
        while the probe is in flight.

        Raises:
            DatabaseUnavailableError: The call is refused; nothing was attempted.
        """
        with self._lock:
            refusal = self._refusal_locked()
            probing = refusal is None and self._state is not BreakerState.CLOSED
            if probing:
                self._state = BreakerState.HALF_OPEN
                self._probe_in_flight = True
        if refusal is not None:
            raise refusal
        if probing:
            logger.info("database circuit breaker half-open: admitting one probe")

    def record_success(self) -> None:
        """Record a call that reached the database.

        Resets the consecutive failure count; a HALF_OPEN breaker closes. Ignored
        while OPEN.
        """
        with self._lock:
            if self._state is BreakerState.OPEN:
                return
            recovered = self._state is BreakerState.HALF_OPEN
            self._state = BreakerState.CLOSED
            self._failures = 0
            self._probe_in_flight = False
        if recovered:
            logger.info("database circuit breaker closed: the database answered")

    def record_failure(self, exc: BaseException) -> None:
        """Record a call that ended by raising ``exc``.

        A connectivity failure (see ``is_connectivity_failure``) counts toward
        opening, and a failed probe reopens the breaker for a full timeout. Any
        other driver error proves the database answered and is recorded as a
        success. A non-driver exception, such as the caller's own bug or
        ``GeneratorExit``, says nothing about the database: the count is left
        alone and a probe slot is released so the next caller probes instead.

        Args:
            exc: The exception that ended the call.
        """
        if is_connectivity_failure(exc):
            self._count_failure()
        elif isinstance(exc, DBAPIError):
            self.record_success()
        else:
            with self._lock:
                self._probe_in_flight = False

    def _count_failure(self) -> None:
        with self._lock:
            if self._state is BreakerState.OPEN:
                return
            self._failures += 1
            reopened = self._state is BreakerState.HALF_OPEN
            if not reopened and self._failures < self._config.failure_threshold:
                return
            self._state = BreakerState.OPEN
            self._opened_at = self._clock()
            self._probe_in_flight = False
            failures = self._failures
        logger.warning(
            "database circuit breaker opened after %d consecutive connection "
            "failure(s)%s; refusing sessions for %.0f s",
            failures,
            " (probe failed)" if reopened else "",
            self._config.reset_timeout_s,
        )

    def _refusal_locked(self) -> DatabaseUnavailableError | None:
        if self._state is BreakerState.CLOSED:
            return None
        if self._state is BreakerState.HALF_OPEN:
            if not self._probe_in_flight:
                return None
            return DatabaseUnavailableError(
                "database unavailable: a probe is testing the connection; retry shortly"
            )
        remaining = self._opened_at + self._config.reset_timeout_s - self._clock()
        if remaining <= 0:
            return None
        return DatabaseUnavailableError(
            "database unavailable: repeated connection failures opened the "
            f"circuit breaker; next attempt in {math.ceil(remaining)} s"
        )
