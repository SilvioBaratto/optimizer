"""Tests for the database circuit breaker and its wiring into ``get_session``.

Validates ``portopt_db.resilience`` (``DbCircuitBreaker``, ``BreakerConfig``,
``validate_breaker_settings``, the ``CircuitBreaker`` protocol) and
``DatabaseManager.get_session``. Time comes from an injected fake clock, so no test
sleeps; threads meet on events and barriers, each wait bounded so a regression fails
instead of hanging. Driver errors are SQLAlchemy wrappers around a fake ``orig``
that carries a PostgreSQL ``pgcode``, so the SQLite-only CI exercises the
classification a psycopg2 error would get.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from contextlib import suppress
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import text
from sqlalchemy.exc import (
    DataError,
    DBAPIError,
    DisconnectionError,
    IntegrityError,
    InterfaceError,
    OperationalError,
    ProgrammingError,
)
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from portopt_db.config import DbConfig
from portopt_db.engine import DatabaseManager
from portopt_db.resilience import (
    BreakerConfig,
    BreakerState,
    CircuitBreaker,
    DatabaseUnavailableError,
    DbCircuitBreaker,
    ProbeTicket,
)

_URL = "postgresql://u:p@h:5432/db"
_TIMEOUT_S = 30.0
_THRESHOLD = 5
_WAIT_S = 5.0


class _Clock:
    """Monotonic fake clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the clock forward by ``seconds``."""
        self.now += seconds


class _FakeOrig(Exception):
    """Stands in for a psycopg2 error: a message and an optional SQLSTATE."""

    def __init__(self, message: str, pgcode: str | None = None) -> None:
        super().__init__(message)
        self.pgcode = pgcode


def _driver_error(
    cls: type[DBAPIError],
    pgcode: str | None,
    *,
    statement: str | None = "SELECT 1",
    invalidated: bool = False,
) -> DBAPIError:
    return cls(
        statement,
        {},
        _FakeOrig(f"driver error {pgcode}", pgcode),
        connection_invalidated=invalidated,
    )


def _connect_refused() -> OperationalError:
    """What SQLAlchemy raises when the pool cannot open a connection at all."""
    return OperationalError(None, None, _FakeOrig("Connection refused"))


def _breaker(
    clock: _Clock, threshold: int = _THRESHOLD, timeout: float = _TIMEOUT_S
) -> DbCircuitBreaker:
    return DbCircuitBreaker(
        BreakerConfig(failure_threshold=threshold, reset_timeout_s=timeout),
        clock=clock,
    )


def _trip(breaker: DbCircuitBreaker, times: int = _THRESHOLD) -> None:
    for _ in range(times):
        breaker.before_call()
        breaker.record_failure(_connect_refused())


def _half_open(breaker: DbCircuitBreaker, clock: _Clock) -> ProbeTicket:
    """Open the breaker, wait out the timeout and admit the probe."""
    _trip(breaker)
    clock.advance(_TIMEOUT_S)
    ticket = breaker.before_call()
    assert ticket is not None
    return ticket


# --- classification -----------------------------------------------------------

_COUNTED = "counted"
_NOT_COUNTED = "not counted"

_CLASSIFICATION_CASES: list[tuple[str, Callable[[], BaseException], str]] = [
    ("pool pre-ping disconnect", lambda: DisconnectionError("ping failed"), _COUNTED),
    ("connect refused (no statement)", _connect_refused, _COUNTED),
    (
        "connect timeout (no statement)",
        lambda: OperationalError(None, None, _FakeOrig("timeout expired")),
        _COUNTED,
    ),
    (
        "08006 connection_failure",
        lambda: _driver_error(OperationalError, "08006"),
        _COUNTED,
    ),
    (
        "08001 unable to establish",
        lambda: _driver_error(OperationalError, "08001"),
        _COUNTED,
    ),
    (
        "08003 connection_does_not_exist",
        lambda: _driver_error(DBAPIError, "08003"),
        _COUNTED,
    ),
    (
        "57P01 admin_shutdown",
        lambda: _driver_error(OperationalError, "57P01"),
        _COUNTED,
    ),
    (
        "57P02 crash_shutdown",
        lambda: _driver_error(OperationalError, "57P02"),
        _COUNTED,
    ),
    (
        "57P03 cannot_connect_now",
        lambda: _driver_error(OperationalError, "57P03"),
        _COUNTED,
    ),
    (
        "connection_invalidated",
        lambda: _driver_error(InterfaceError, None, invalidated=True),
        _COUNTED,
    ),
    (
        "57014 QueryCanceled (statement timeout)",
        lambda: _driver_error(OperationalError, "57014"),
        _NOT_COUNTED,
    ),
    (
        "57014 QueryCanceled without a statement",
        lambda: _driver_error(OperationalError, "57014", statement=None),
        _NOT_COUNTED,
    ),
    (
        "55P03 LockNotAvailable (lock timeout)",
        lambda: _driver_error(OperationalError, "55P03"),
        _NOT_COUNTED,
    ),
    (
        "23505 IntegrityError",
        lambda: _driver_error(IntegrityError, "23505"),
        _NOT_COUNTED,
    ),
    (
        "42P01 ProgrammingError",
        lambda: _driver_error(ProgrammingError, "42P01"),
        _NOT_COUNTED,
    ),
    ("22003 DataError", lambda: _driver_error(DataError, "22003"), _NOT_COUNTED),
    (
        "statement error without SQLSTATE (SQLite)",
        lambda: OperationalError("SELECT 1", {}, _FakeOrig("database is locked")),
        _NOT_COUNTED,
    ),
    ("pool exhausted", lambda: PoolTimeoutError("QueuePool limit"), _NOT_COUNTED),
    ("caller bug", lambda: ValueError("bad input"), _NOT_COUNTED),
    (
        "refused by an open breaker",
        lambda: DatabaseUnavailableError("database unavailable"),
        _NOT_COUNTED,
    ),
]


@pytest.mark.parametrize(
    ("make_error", "expected"),
    [(make, expected) for _, make, expected in _CLASSIFICATION_CASES],
    ids=[case_id for case_id, _, _ in _CLASSIFICATION_CASES],
)
def test_only_connectivity_failures_count_toward_opening(
    make_error: Callable[[], BaseException], expected: str
) -> None:
    """A slow query, a lock wait or a constraint must never open the breaker."""
    breaker = _breaker(_Clock(), threshold=1)

    breaker.before_call()
    breaker.record_failure(make_error())

    counted = breaker.state is BreakerState.OPEN
    assert (_COUNTED if counted else _NOT_COUNTED) == expected


_RESETS = "resets"
_KEEPS = "keeps"

_ANSWER_CASES: list[tuple[str, Callable[[], BaseException], str]] = [
    ("23505 constraint", lambda: _driver_error(IntegrityError, "23505"), _RESETS),
    (
        "57014 statement timeout",
        lambda: _driver_error(OperationalError, "57014"),
        _RESETS,
    ),
    ("55P03 lock timeout", lambda: _driver_error(OperationalError, "55P03"), _RESETS),
    ("42P01 missing table", lambda: _driver_error(ProgrammingError, "42P01"), _RESETS),
    (
        "statement error without SQLSTATE",
        lambda: OperationalError("SELECT 1", {}, _FakeOrig("database is locked")),
        _KEEPS,
    ),
    (
        "driver error without SQLSTATE",
        lambda: _driver_error(ProgrammingError, None),
        _KEEPS,
    ),
    ("pool exhausted", lambda: PoolTimeoutError("QueuePool limit"), _KEEPS),
    ("caller bug", lambda: ValueError("bad input"), _KEEPS),
]


@pytest.mark.parametrize(
    ("make_error", "expected"),
    [(make, expected) for _, make, expected in _ANSWER_CASES],
    ids=[case_id for case_id, _, _ in _ANSWER_CASES],
)
def test_only_an_error_with_a_sqlstate_proves_the_server_answered(
    make_error: Callable[[], BaseException], expected: str
) -> None:
    """A SQLSTATE resets the count; an error without one proves nothing."""
    breaker = _breaker(_Clock())
    _trip(breaker, _THRESHOLD - 1)

    breaker.record_failure(make_error())
    _trip(breaker, 1)

    reset = breaker.state is BreakerState.CLOSED
    assert (_RESETS if reset else _KEEPS) == expected


# --- state machine -------------------------------------------------------------


def test_starts_closed_and_admits_calls_without_a_ticket() -> None:
    """A fresh breaker is closed, refuses nothing and issues no probe ticket."""
    breaker = _breaker(_Clock())

    ticket = breaker.before_call()

    assert ticket is None
    assert breaker.state is BreakerState.CLOSED
    assert breaker.refusal() is None


def test_opens_after_threshold_consecutive_counted_failures() -> None:
    """The threshold-th consecutive connectivity failure opens it, not one earlier."""
    breaker = _breaker(_Clock())

    _trip(breaker, _THRESHOLD - 1)
    assert breaker.state is BreakerState.CLOSED

    _trip(breaker, 1)
    assert breaker.state is BreakerState.OPEN


def test_a_success_resets_the_failure_count() -> None:
    """Failures separated by a success are not consecutive."""
    breaker = _breaker(_Clock())

    _trip(breaker, _THRESHOLD - 1)
    breaker.before_call()
    breaker.record_success()
    _trip(breaker, _THRESHOLD - 1)

    assert breaker.state is BreakerState.CLOSED


def test_open_refuses_until_the_reset_timeout_elapses() -> None:
    """Open refuses every call until the reset timeout has passed."""
    clock = _Clock()
    breaker = _breaker(clock)
    _trip(breaker)

    clock.advance(_TIMEOUT_S - 0.1)

    with pytest.raises(DatabaseUnavailableError, match="database unavailable"):
        breaker.before_call()
    assert breaker.state is BreakerState.OPEN


def test_refusal_names_when_the_next_attempt_is_due() -> None:
    """The refusal tells the reader how long until the next attempt."""
    clock = _Clock()
    breaker = _breaker(clock)
    _trip(breaker)
    clock.advance(10.0)

    refusal = breaker.refusal()

    assert refusal is not None
    assert "20 s" in str(refusal)


def test_after_the_timeout_exactly_one_probe_is_admitted_with_a_ticket() -> None:
    """Once the timeout passes one probe goes through and the next caller waits."""
    clock = _Clock()
    breaker = _breaker(clock)
    _trip(breaker)
    clock.advance(_TIMEOUT_S)

    ticket = breaker.before_call()

    assert isinstance(ticket, ProbeTicket)
    assert breaker.state is BreakerState.HALF_OPEN
    with pytest.raises(DatabaseUnavailableError):
        breaker.before_call()


def test_refusal_peeks_without_admitting_the_probe() -> None:
    """``refusal`` reports without taking the probe slot or changing the state."""
    clock = _Clock()
    breaker = _breaker(clock)
    _trip(breaker)
    clock.advance(_TIMEOUT_S)

    assert breaker.refusal() is None
    assert breaker.state is BreakerState.OPEN
    breaker.before_call()
    assert breaker.refusal() is not None


def test_probe_success_closes_the_breaker_and_resets_the_count() -> None:
    """A successful probe closes the breaker with a clean failure count."""
    clock = _Clock()
    breaker = _breaker(clock)
    ticket = _half_open(breaker, clock)

    breaker.record_success(ticket)

    assert breaker.state is BreakerState.CLOSED
    _trip(breaker, _THRESHOLD - 1)
    assert breaker.state is BreakerState.CLOSED


def test_probe_failure_reopens_for_a_full_cooldown_from_the_failure() -> None:
    """A failed probe reopens, and the cooldown restarts at the failure."""
    clock = _Clock()
    breaker = _breaker(clock)
    ticket = _half_open(breaker, clock)
    clock.advance(7.0)

    breaker.record_failure(_connect_refused(), ticket)

    assert breaker.state is BreakerState.OPEN
    clock.advance(_TIMEOUT_S - 0.1)
    with pytest.raises(DatabaseUnavailableError):
        breaker.before_call()
    clock.advance(0.1)
    breaker.before_call()
    assert breaker.state is BreakerState.HALF_OPEN


@pytest.mark.parametrize(
    "answer",
    [
        _driver_error(IntegrityError, "23505"),
        _driver_error(OperationalError, "57014"),
        _driver_error(OperationalError, "55P03"),
    ],
    ids=["integrity", "statement-timeout", "lock-timeout"],
)
def test_probe_answered_by_the_database_closes_the_breaker(
    answer: DBAPIError,
) -> None:
    """The server answered the probe, so connectivity is proven."""
    clock = _Clock()
    breaker = _breaker(clock)
    ticket = _half_open(breaker, clock)

    breaker.record_failure(answer, ticket)

    assert breaker.state is BreakerState.CLOSED


@pytest.mark.parametrize(
    "inconclusive",
    [
        ValueError("caller bug"),
        GeneratorExit(),
        KeyboardInterrupt(),
        _driver_error(ProgrammingError, None),
    ],
    ids=["caller-error", "generator-exit", "interrupt", "no-sqlstate"],
)
def test_inconclusive_probe_releases_the_slot_for_the_next_caller(
    inconclusive: BaseException,
) -> None:
    """A probe that ended without a database verdict frees the slot, once."""
    clock = _Clock()
    breaker = _breaker(clock)
    ticket = _half_open(breaker, clock)

    breaker.record_failure(inconclusive, ticket)

    assert breaker.state is BreakerState.HALF_OPEN
    assert breaker.before_call() is not None
    with pytest.raises(DatabaseUnavailableError):
        breaker.before_call()


def test_outcomes_without_the_probe_ticket_never_settle_the_probe() -> None:
    """A session admitted before the breaker opened cannot free or decide the probe."""
    clock = _Clock()
    breaker = _breaker(clock)
    ticket = _half_open(breaker, clock)

    breaker.record_failure(ValueError("late caller bug"))
    breaker.record_success()
    breaker.record_failure(_connect_refused())

    assert breaker.state is BreakerState.HALF_OPEN
    with pytest.raises(DatabaseUnavailableError):
        breaker.before_call()
    breaker.record_success(ticket)
    assert breaker.state is BreakerState.CLOSED


def test_a_stale_ticket_from_an_earlier_probe_settles_nothing() -> None:
    """Only the ticket of the probe in flight counts; an older one is ignored."""
    clock = _Clock()
    breaker = _breaker(clock)
    first = _half_open(breaker, clock)
    breaker.record_failure(ValueError("first probe proved nothing"), first)
    second = breaker.before_call()

    breaker.record_success(first)

    assert second is not None
    assert breaker.state is BreakerState.HALF_OPEN
    breaker.record_success(second)
    assert breaker.state is BreakerState.CLOSED


def test_outcomes_landing_while_open_change_nothing() -> None:
    """A session admitted before the breaker opened neither closes it nor extends it."""
    clock = _Clock()
    breaker = _breaker(clock)
    _trip(breaker)

    breaker.record_success()
    clock.advance(_TIMEOUT_S - 1.0)
    breaker.record_failure(_connect_refused())
    clock.advance(1.0)

    assert breaker.state is BreakerState.OPEN
    breaker.before_call()
    assert breaker.state is BreakerState.HALF_OPEN


def test_db_circuit_breaker_satisfies_the_protocol() -> None:
    """The default implementation is a ``CircuitBreaker``."""
    assert isinstance(_breaker(_Clock()), CircuitBreaker)


# --- thread safety ---------------------------------------------------------------


def test_two_threads_racing_for_the_probe_admit_exactly_one() -> None:
    """Two callers that race for the probe slot: one wins, one is refused."""
    clock = _Clock()
    breaker = _breaker(clock)
    _trip(breaker)
    clock.advance(_TIMEOUT_S)
    start = threading.Barrier(2, timeout=_WAIT_S)
    outcomes: list[str] = []
    lock = threading.Lock()

    def contend() -> None:
        """Ask for admission and record whether it was granted."""
        start.wait()
        try:
            breaker.before_call()
            outcome = "admitted"
        except DatabaseUnavailableError:
            outcome = "refused"
        with lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=contend) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=_WAIT_S)

    assert sorted(outcomes) == ["admitted", "refused"]


def test_a_second_caller_is_refused_while_the_probe_runs() -> None:
    """While the probe is still running a second caller is turned away."""
    clock = _Clock()
    breaker = _breaker(clock)
    _trip(breaker)
    clock.advance(_TIMEOUT_S)
    admitted, finish = threading.Event(), threading.Event()

    def probe() -> None:
        """Hold the probe slot until the test lets it finish."""
        ticket = breaker.before_call()
        admitted.set()
        finish.wait(timeout=_WAIT_S)
        breaker.record_success(ticket)

    thread = threading.Thread(target=probe)
    thread.start()
    try:
        assert admitted.wait(timeout=_WAIT_S)
        with pytest.raises(DatabaseUnavailableError):
            breaker.before_call()
    finally:
        finish.set()
        thread.join(timeout=_WAIT_S)

    assert breaker.state is BreakerState.CLOSED
    assert breaker.before_call() is None


# --- configuration ---------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"failure_threshold": 0},
        {"failure_threshold": -1},
        {"failure_threshold": True},
        {"failure_threshold": 2.5},
        {"failure_threshold": "5"},
        {"reset_timeout_s": 0.0},
        {"reset_timeout_s": -5.0},
        {"reset_timeout_s": math.nan},
        {"reset_timeout_s": math.inf},
        {"reset_timeout_s": True},
        {"reset_timeout_s": "30"},
    ],
)
def test_breaker_config_rejects_settings_that_cannot_drive_a_breaker(
    kwargs: dict,
) -> None:
    """A threshold must be a positive int and a timeout a finite positive number."""
    with pytest.raises(ValueError, match="must be"):
        BreakerConfig(**kwargs)


def test_breaker_config_defaults_are_five_failures_and_thirty_seconds() -> None:
    """The defaults are 5 failures and 30 seconds."""
    config = BreakerConfig()

    assert config.failure_threshold == 5
    assert config.reset_timeout_s == 30.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"breaker_failure_threshold": 0},
        {"breaker_failure_threshold": -3},
        {"breaker_failure_threshold": True},
        {"breaker_failure_threshold": 2.5},
        {"breaker_reset_timeout_s": 0.0},
        {"breaker_reset_timeout_s": -1.0},
        {"breaker_reset_timeout_s": math.nan},
        {"breaker_reset_timeout_s": math.inf},
        {"breaker_reset_timeout_s": "30"},
    ],
)
def test_db_config_rejects_breaker_settings_that_cannot_drive_a_breaker(
    kwargs: dict,
) -> None:
    """``DbConfig`` names its own field when it refuses a breaker setting."""
    with pytest.raises(ValueError, match="breaker_"):
        DbConfig(url=_URL, **kwargs)


def test_db_config_breaker_defaults_match_the_breaker_defaults() -> None:
    """``DbConfig`` turns the breaker on by default with 5 failures and 30 s."""
    config = DbConfig(url=_URL)
    defaults = BreakerConfig()

    assert config.breaker_failure_threshold == defaults.failure_threshold == 5
    assert config.breaker_reset_timeout_s == defaults.reset_timeout_s == 30.0


def test_manager_builds_its_breaker_from_the_config() -> None:
    """The manager's breaker takes its threshold and timeout from ``DbConfig``."""
    mgr = DatabaseManager(
        DbConfig(url=_URL, breaker_failure_threshold=2, breaker_reset_timeout_s=9.0)
    )

    assert isinstance(mgr.breaker, DbCircuitBreaker)
    assert mgr.breaker.config.failure_threshold == 2
    assert mgr.breaker.config.reset_timeout_s == 9.0


def test_each_manager_has_its_own_breaker() -> None:
    """Tripping one manager's breaker leaves another manager's closed."""
    first = DatabaseManager(DbConfig(url=_URL))
    second = DatabaseManager(DbConfig(url=_URL))

    _trip(first.breaker)  # type: ignore[arg-type]

    assert first.breaker.state is BreakerState.OPEN
    assert second.breaker.state is BreakerState.CLOSED


# --- get_session wiring ------------------------------------------------------------


class _SqliteConfig(DbConfig):
    """SQLite config: the psycopg2 connect args would be rejected."""

    def connect_args(self) -> dict:
        """Return no driver arguments, since sqlite3 accepts none of psycopg2's."""
        return {}


def _ready_manager(clock: _Clock) -> tuple[DatabaseManager, MagicMock]:
    """A manager that looks initialized and hands out one mock session."""
    mgr = DatabaseManager(DbConfig(url=_URL), breaker=_breaker(clock))
    session = MagicMock()
    mgr._is_initialized = True
    mgr._session_factory = MagicMock(return_value=session)
    return mgr, session


def _probing_manager(clock: _Clock) -> tuple[DatabaseManager, MagicMock]:
    """A ready manager whose breaker has opened and waited out its timeout."""
    mgr, session = _ready_manager(clock)
    _trip(mgr.breaker)  # type: ignore[arg-type]
    clock.advance(_TIMEOUT_S)
    return mgr, session


def _unreachable_manager(tmp_path, clock: _Clock) -> DatabaseManager:
    """A real SQLite engine whose every connect fails: the file's folder is missing."""
    url = f"sqlite:///{tmp_path / 'missing' / 'db.sqlite'}"
    return DatabaseManager(_SqliteConfig(url=url), breaker=_breaker(clock))


def test_open_breaker_refuses_a_session_without_touching_the_engine() -> None:
    """An open breaker refuses before any session is built."""
    clock = _Clock()
    mgr, _ = _ready_manager(clock)
    _trip(mgr.breaker)  # type: ignore[arg-type]
    factory = mgr._session_factory

    with pytest.raises(DatabaseUnavailableError), mgr.get_session():
        pass

    assert isinstance(factory, MagicMock)
    factory.assert_not_called()


def test_unreachable_database_opens_the_breaker_then_stops_connecting(
    tmp_path,
) -> None:
    """Failed connects open the breaker; after that no engine is built."""
    clock = _Clock()
    mgr = _unreachable_manager(tmp_path, clock)
    for _ in range(_THRESHOLD):
        with pytest.raises(OperationalError), mgr.get_session() as session:
            session.execute(text("SELECT 1"))
    assert mgr.breaker.state is BreakerState.OPEN

    with (
        patch.object(mgr, "initialize", wraps=mgr.initialize) as initialize,
        pytest.raises(DatabaseUnavailableError),
        mgr.get_session(),
    ):
        pass

    initialize.assert_not_called()


def test_a_closed_breaker_leaves_the_connection_lazy() -> None:
    """Only the probe connects early; a normal session opens on first use."""
    mgr, session = _ready_manager(_Clock())

    with mgr.get_session():
        pass

    session.connection.assert_not_called()


def test_a_body_that_completes_is_recorded_as_one_success() -> None:
    """A session whose body completes records exactly one success."""
    mgr, _ = _ready_manager(_Clock())

    with (
        patch.object(mgr.breaker, "record_success") as success,
        patch.object(mgr.breaker, "record_failure") as failure,
        mgr.get_session(),
    ):
        pass

    success.assert_called_once_with()
    failure.assert_not_called()


def test_a_connectivity_failure_in_the_body_is_recorded_once() -> None:
    """The connection opens lazily inside the body, so that is where it fails."""
    mgr, session = _ready_manager(_Clock())
    error = _connect_refused()

    with (
        patch.object(mgr.breaker, "record_success") as success,
        patch.object(mgr.breaker, "record_failure") as failure,
        pytest.raises(OperationalError),
        mgr.get_session(),
    ):
        raise error

    failure.assert_called_once_with(error)
    success.assert_not_called()
    session.rollback.assert_called_once()
    session.close.assert_called_once()


def test_a_failing_rollback_still_leaves_the_outcome_recorded() -> None:
    """The outcome is recorded before a rollback that may itself fail."""
    mgr, session = _ready_manager(_Clock())
    _trip(mgr.breaker, _THRESHOLD - 1)  # type: ignore[arg-type]
    session.rollback.side_effect = _connect_refused()

    with pytest.raises(OperationalError), mgr.get_session():
        raise _connect_refused()

    assert mgr.breaker.state is BreakerState.OPEN


def test_threshold_failures_through_get_session_open_the_breaker() -> None:
    """Connectivity failures raised in session bodies open the breaker."""
    mgr, _ = _ready_manager(_Clock())

    for _ in range(_THRESHOLD):
        with pytest.raises(OperationalError), mgr.get_session():
            raise _connect_refused()

    assert mgr.breaker.state is BreakerState.OPEN


def test_statement_timeouts_through_get_session_never_open_the_breaker() -> None:
    """Statement timeouts on a healthy database never open the breaker."""
    mgr, _ = _ready_manager(_Clock())

    for _ in range(_THRESHOLD * 2):
        with pytest.raises(OperationalError), mgr.get_session():
            raise _driver_error(OperationalError, "57014")

    assert mgr.breaker.state is BreakerState.CLOSED


def test_probe_is_settled_by_its_connect_before_the_body_runs() -> None:
    """The probe connects before the body and is decided there, not at exit."""
    clock = _Clock()
    mgr, session = _probing_manager(clock)
    seen_at_connect: list[BreakerState] = []
    session.connection.side_effect = lambda: seen_at_connect.append(mgr.breaker.state)

    with mgr.get_session():
        assert mgr.breaker.state is BreakerState.CLOSED

    assert seen_at_connect == [BreakerState.HALF_OPEN]


def test_probe_whose_connect_proves_nothing_does_not_close() -> None:
    """No evidence, no recovery: the slot is freed and the breaker stays half-open."""
    clock = _Clock()
    mgr, session = _probing_manager(clock)
    session.connection.side_effect = ValueError("not a database answer")
    ran: list[bool] = []

    with pytest.raises(ValueError), mgr.get_session():
        ran.append(True)

    assert ran == []
    assert mgr.breaker.state is BreakerState.HALF_OPEN
    assert mgr.breaker.refusal() is None
    session.close.assert_called_once()


def test_probe_whose_connect_fails_reopens_for_a_full_cooldown() -> None:
    """A probe that cannot connect reopens the breaker; its body never runs."""
    clock = _Clock()
    mgr, session = _probing_manager(clock)
    session.connection.side_effect = _connect_refused()

    with pytest.raises(OperationalError), mgr.get_session():
        pass

    assert mgr.breaker.state is BreakerState.OPEN
    session.close.assert_called_once()
    clock.advance(_TIMEOUT_S - 0.1)
    with pytest.raises(DatabaseUnavailableError), mgr.get_session():
        pass


def test_a_probe_body_cannot_turn_a_failed_connect_into_a_success() -> None:
    """A body that would swallow the connection error never gets the chance."""
    clock = _Clock()
    mgr, session = _probing_manager(clock)
    session.connection.side_effect = _connect_refused()
    swallowed: list[bool] = []

    with pytest.raises(OperationalError), mgr.get_session():
        with suppress(OperationalError):
            swallowed.append(True)

    assert swallowed == []
    assert mgr.breaker.state is BreakerState.OPEN


def test_a_long_probe_session_does_not_hold_other_sessions_back() -> None:
    """Once the probe has connected, other sessions run while it is still open."""
    clock = _Clock()
    mgr, _ = _probing_manager(clock)
    inside, finish = threading.Event(), threading.Event()

    def probe() -> None:
        """Connect as the probe, then keep the session open until released."""
        with mgr.get_session():
            inside.set()
            finish.wait(timeout=_WAIT_S)

    thread = threading.Thread(target=probe)
    thread.start()
    try:
        assert inside.wait(timeout=_WAIT_S)
        with mgr.get_session():
            pass
    finally:
        finish.set()
        thread.join(timeout=_WAIT_S)

    assert mgr.breaker.state is BreakerState.CLOSED


def test_a_second_session_is_refused_while_the_probe_connects() -> None:
    """Until the probe's connect returns, every other session is turned away."""
    clock = _Clock()
    mgr, session = _probing_manager(clock)
    connecting, finish = threading.Event(), threading.Event()

    def slow_connect() -> None:
        """Signal that the probe is connecting, then wait to be released."""
        connecting.set()
        finish.wait(timeout=_WAIT_S)

    session.connection.side_effect = slow_connect

    def probe() -> None:
        """Open the probe session; its connect blocks until released."""
        with mgr.get_session():
            pass

    thread = threading.Thread(target=probe)
    thread.start()
    try:
        assert connecting.wait(timeout=_WAIT_S)
        with pytest.raises(DatabaseUnavailableError), mgr.get_session():
            pass
    finally:
        finish.set()
        thread.join(timeout=_WAIT_S)

    assert mgr.breaker.state is BreakerState.CLOSED


@pytest.mark.parametrize(
    "raised",
    [ValueError("caller bug"), _driver_error(IntegrityError, "23505")],
    ids=["caller-error", "integrity"],
)
def test_a_probe_body_error_after_a_good_connect_leaves_it_closed(
    raised: Exception,
) -> None:
    """After the connect has decided the probe, the body is an ordinary session."""
    clock = _Clock()
    mgr, _ = _probing_manager(clock)

    with pytest.raises(type(raised)), mgr.get_session():
        raise raised

    assert mgr.breaker.state is BreakerState.CLOSED


def test_an_abandoned_probe_session_leaves_nothing_stuck() -> None:
    """A probe context closed without being exited still closes its session."""
    clock = _Clock()
    mgr, session = _probing_manager(clock)
    context = mgr.get_session()
    context.__enter__()

    context.gen.close()

    session.close.assert_called_once()
    assert mgr.breaker.state is BreakerState.CLOSED


def test_probe_refused_by_a_closed_manager_frees_the_slot() -> None:
    """A probe that a closed manager refuses frees the slot."""
    clock = _Clock()
    mgr, _ = _probing_manager(clock)
    mgr.close()

    with pytest.raises(RuntimeError, match="closed"), mgr.get_session():
        pass

    assert mgr.breaker.state is BreakerState.HALF_OPEN
    assert mgr.breaker.refusal() is None


class _RecordingBreaker:
    """A ``CircuitBreaker`` that admits everything and records the calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def state(self) -> BreakerState:
        """Always closed."""
        return BreakerState.CLOSED

    def refusal(self) -> DatabaseUnavailableError | None:
        """Never refuses."""
        return None

    def before_call(self) -> ProbeTicket | None:
        """Admit without a probe."""
        self.calls.append("before_call")
        return None

    def record_success(self, ticket: ProbeTicket | None = None) -> None:
        """Record a success."""
        self.calls.append("success")

    def record_failure(
        self, exc: BaseException, ticket: ProbeTicket | None = None
    ) -> None:
        """Record a failure."""
        self.calls.append("failure")


def test_manager_accepts_any_breaker_that_satisfies_the_protocol() -> None:
    """``DatabaseManager`` depends on the protocol, not on the default class."""
    breaker = _RecordingBreaker()
    mgr = DatabaseManager(DbConfig(url=_URL), breaker=breaker)
    mgr._is_initialized = True
    mgr._session_factory = MagicMock(return_value=MagicMock())

    with mgr.get_session():
        pass

    assert isinstance(breaker, CircuitBreaker)
    assert mgr.breaker is breaker
    assert breaker.calls == ["before_call", "success"]


def test_health_check_reports_false_while_the_breaker_is_open() -> None:
    """The health check reads an open breaker as unhealthy."""
    mgr, _ = _ready_manager(_Clock())
    _trip(mgr.breaker)  # type: ignore[arg-type]

    assert mgr.health_check() is False
