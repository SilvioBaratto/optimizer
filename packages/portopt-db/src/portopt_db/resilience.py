"""Circuit breaker that stops a dead database from being hammered.

``DatabaseManager`` holds one ``CircuitBreaker`` (``DbCircuitBreaker`` by default)
and consults it on every ``get_session``. After ``failure_threshold`` consecutive
connectivity failures the breaker opens: sessions are refused at once with
``DatabaseUnavailableError`` and the engine is never touched, so a daemon whose
database is gone fails its work in milliseconds instead of waiting out a connect
timeout per call. Once ``reset_timeout_s`` has passed, exactly one probe is
admitted. ``before_call`` hands that caller a ``ProbeTicket``, and only an outcome
recorded with that ticket decides the probe: success closes the breaker, a
connectivity failure opens it for another full timeout.

Only connectivity failures count. SQLAlchemy wraps driver errors, and psycopg2's
statement-timeout (``QueryCanceled``, 57014) and lock-timeout (``LockNotAvailable``,
55P03) errors subclass the driver's ``OperationalError``, so the class alone would
let one slow query open the breaker on a healthy database. The SQLSTATE decides
instead: an error the server answered with a SQLSTATE proves it is reachable.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from sqlalchemy.exc import DBAPIError, DisconnectionError, OperationalError

logger = logging.getLogger(__name__)

DEFAULT_FAILURE_THRESHOLD = 5
DEFAULT_RESET_TIMEOUT_S = 30.0

# SQLSTATE class 08 is "connection exception". The 57P0x codes are the server
# shutting down, crashing or still starting up: the session is gone or never came.
_CONNECTION_EXCEPTION_CLASS = "08"
_SERVER_GONE_SQLSTATES = frozenset({"57P01", "57P02", "57P03"})

__all__ = [
    "DEFAULT_FAILURE_THRESHOLD",
    "DEFAULT_RESET_TIMEOUT_S",
    "BreakerConfig",
    "BreakerState",
    "CircuitBreaker",
    "DatabaseUnavailableError",
    "DbCircuitBreaker",
    "ProbeTicket",
    "is_connectivity_failure",
    "parse_failure_threshold",
    "parse_reset_timeout",
    "validate_breaker_settings",
]


class BreakerState(str, Enum):
    """Where the breaker stands.

    Attributes:
        CLOSED: Sessions pass; connectivity failures are counted.
        OPEN: Sessions are refused until the reset timeout has passed.
        HALF_OPEN: One probe is admitted; every other caller is refused until the
            probe is decided.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_finite_positive(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value > 0


def validate_breaker_settings(
    failure_threshold: object, reset_timeout_s: object, *, prefix: str = ""
) -> None:
    """Refuse breaker settings that cannot drive a breaker.

    The one place the rule lives: ``BreakerConfig`` and ``DbConfig`` both call it.

    Args:
        failure_threshold: Consecutive connectivity failures that open the breaker;
            must be an ``int`` above zero (``bool`` is refused).
        reset_timeout_s: Seconds an open breaker waits before its probe; must be a
            finite ``int`` or ``float`` above zero (``bool`` is refused).
        prefix: Put before the setting names in the message, so a caller whose
            fields carry a prefix reports its own names.

    Raises:
        ValueError: A setting has the wrong type or is out of range.
    """
    if not _is_positive_int(failure_threshold):
        raise ValueError(
            f"{prefix}failure_threshold must be an integer > 0, "
            f"got {failure_threshold!r}"
        )
    if not _is_finite_positive(reset_timeout_s):
        raise ValueError(
            f"{prefix}reset_timeout_s must be a finite number > 0, "
            f"got {reset_timeout_s!r}"
        )


# Python's int() and float() also take signs, digit underscores, exponents and
# "inf", and pydantic's lax int takes "5.0", so each daemon would accept a different
# set of spellings for the same variable. Both parse with these instead.
_WHOLE_NUMBER_TEXT = re.compile(r"[0-9]+")
_DECIMAL_TEXT = re.compile(r"[0-9]+(?:\.[0-9]+)?")


def parse_failure_threshold(text: str, *, name: str) -> int:
    """Read a failure threshold from environment text, strictly.

    Args:
        text: The variable's value; blanks around the number are ignored.
        name: The variable's name, put at the start of the error message.

    Returns:
        The threshold, an integer above zero.

    Raises:
        ValueError: ``text`` is not a whole number above zero written in plain
            digits (``5.0``, ``+3``, ``1_0`` and an empty value are refused).
    """
    stripped = text.strip()
    if _WHOLE_NUMBER_TEXT.fullmatch(stripped) and _is_positive_int(int(stripped)):
        return int(stripped)
    raise ValueError(f"{name} must be an integer > 0, got {text!r}")


def parse_reset_timeout(text: str, *, name: str) -> float:
    """Read a reset timeout in seconds from environment text, strictly.

    Args:
        text: The variable's value; blanks around the number are ignored.
        name: The variable's name, put at the start of the error message.

    Returns:
        The timeout, a finite number above zero.

    Raises:
        ValueError: ``text`` is not plain digits with an optional fraction above
            zero (a sign, an exponent, ``inf``, ``nan`` and an empty value are
            refused).
    """
    stripped = text.strip()
    if _DECIMAL_TEXT.fullmatch(stripped) and _is_finite_positive(float(stripped)):
        return float(stripped)
    raise ValueError(f"{name} must be a finite number > 0, got {text!r}")


@dataclass(frozen=True)
class BreakerConfig:
    """When the breaker opens and how long it stays open.

    Attributes:
        failure_threshold: Consecutive connectivity failures that open the breaker.
        reset_timeout_s: Seconds an open breaker refuses sessions before it admits
            one probe; a failed probe restarts the full timeout.

    Raises:
        ValueError: See ``validate_breaker_settings``.
    """

    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD
    reset_timeout_s: float = DEFAULT_RESET_TIMEOUT_S

    def __post_init__(self) -> None:
        validate_breaker_settings(self.failure_threshold, self.reset_timeout_s)


class DatabaseUnavailableError(RuntimeError):
    """A session was refused because the database circuit breaker is open.

    Raised by ``DatabaseManager.get_session`` before any engine or session is
    created, so nothing was read or written. Its message is one line without
    host, credentials or SQL, safe to show an operator.
    """


class ProbeTicket:
    """Marks the one call a half-open breaker admitted as its probe.

    Opaque and compared by identity: only an outcome recorded with the ticket the
    breaker issued can decide or release the probe.
    """

    __slots__ = ()


@runtime_checkable
class CircuitBreaker(Protocol):
    """What ``DatabaseManager`` and the breaker's monitors need from a breaker."""

    @property
    def state(self) -> BreakerState:
        """The current state."""
        ...

    def refusal(self) -> DatabaseUnavailableError | None:
        """Return the error a call made now would be refused with, or None."""
        ...

    def before_call(self) -> ProbeTicket | None:
        """Admit a call, returning a ticket when it is the probe, or refuse it.

        Raises:
            DatabaseUnavailableError: The call is refused.
        """
        ...

    def record_success(self, ticket: ProbeTicket | None = None) -> None:
        """Record a call that reached the database."""
        ...

    def record_failure(
        self, exc: BaseException, ticket: ProbeTicket | None = None
    ) -> None:
        """Record a call that ended by raising ``exc``."""
        ...


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


def _proves_the_server_answered(exc: BaseException) -> bool:
    return isinstance(exc, DBAPIError) and _sqlstate(exc.orig) is not None


class DbCircuitBreaker:
    """Thread-safe circuit breaker around one database's sessions.

    Every state change happens under one lock: the state, the consecutive failure
    count, the time the breaker opened and the probe ticket in flight. While
    CLOSED, outcomes recorded without a ticket count or reset the failures. While
    OPEN every outcome is ignored, so a session admitted before the breaker opened
    can neither close it nor extend it. While HALF_OPEN only the outcome carrying
    the probe's ticket decides it; a late outcome from any other session changes
    nothing, and in particular cannot free the probe's slot.

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
        self._probe: ProbeTicket | None = None

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

    def before_call(self) -> ProbeTicket | None:
        """Admit a call or refuse it.

        CLOSED admits. OPEN refuses until the reset timeout has passed, then moves
        to HALF_OPEN and admits this call as the single probe. HALF_OPEN refuses
        while a probe is in flight.

        Returns:
            The probe's ticket when this call is the probe, else ``None``. The
            caller passes the ticket back with the probe's outcome.

        Raises:
            DatabaseUnavailableError: The call is refused; nothing was attempted.
        """
        with self._lock:
            refusal = self._refusal_locked()
            ticket = None
            if refusal is None and self._state is not BreakerState.CLOSED:
                ticket = ProbeTicket()
                self._state = BreakerState.HALF_OPEN
                self._probe = ticket
        if refusal is not None:
            raise refusal
        if ticket is not None:
            logger.info("database circuit breaker half-open: admitting one probe")
        return ticket

    def record_success(self, ticket: ProbeTicket | None = None) -> None:
        """Record a call that reached the database.

        While CLOSED a call without a ticket resets the consecutive failure count;
        the probe's ticket closes a HALF_OPEN breaker. Anything else is ignored.

        Args:
            ticket: The ticket ``before_call`` returned for this call, if any.
        """
        with self._lock:
            if not self._decides_locked(ticket):
                return
            recovered = self._state is BreakerState.HALF_OPEN
            self._state = BreakerState.CLOSED
            self._failures = 0
            self._probe = None
        if recovered:
            logger.info("database circuit breaker closed: the database answered")

    def record_failure(
        self, exc: BaseException, ticket: ProbeTicket | None = None
    ) -> None:
        """Record a call that ended by raising ``exc``.

        A connectivity failure (see ``is_connectivity_failure``) counts toward
        opening, and a failed probe reopens the breaker for a full timeout. A
        driver error that carries a SQLSTATE proves the server answered and is
        recorded as a success. Anything else (a driver error without a SQLSTATE,
        the caller's own bug, ``GeneratorExit``) says nothing about the database:
        the count is left alone, and the probe named by ``ticket`` releases its
        slot so the next caller probes instead.

        Args:
            exc: The exception that ended the call.
            ticket: The ticket ``before_call`` returned for this call, if any.
        """
        if is_connectivity_failure(exc):
            self._count_failure(ticket)
        elif _proves_the_server_answered(exc):
            self.record_success(ticket)
        else:
            self._release_probe(ticket)

    def _decides_locked(self, ticket: ProbeTicket | None) -> bool:
        if self._state is BreakerState.CLOSED:
            return ticket is None
        return ticket is not None and ticket is self._probe

    def _release_probe(self, ticket: ProbeTicket | None) -> None:
        with self._lock:
            if ticket is not None and ticket is self._probe:
                self._probe = None

    def _count_failure(self, ticket: ProbeTicket | None) -> None:
        with self._lock:
            if not self._decides_locked(ticket):
                return
            self._failures += 1
            reopened = self._state is BreakerState.HALF_OPEN
            if not reopened and self._failures < self._config.failure_threshold:
                return
            self._state = BreakerState.OPEN
            self._opened_at = self._clock()
            self._probe = None
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
            if self._probe is None:
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
