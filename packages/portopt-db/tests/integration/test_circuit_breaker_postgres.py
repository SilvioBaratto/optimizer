"""The database circuit breaker against a real PostgreSQL server and psycopg2.

Validates ``portopt_db.engine.DatabaseManager.get_session`` with
``portopt_db.resilience.DbCircuitBreaker``: the classification the unit suite pins
with fake ``pgcode`` objects holds for the errors psycopg2 really raises. A
statement timeout leaves the breaker closed, while a refused connect and a killed
backend open it at the threshold. The breaker runs on a fake clock, so the
recovery probe needs no wait.
"""

from __future__ import annotations

import socket
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import patch

import pytest
from _fake_clock import FakeClock
from sqlalchemy import make_url, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

from portopt_db.config import DbConfig
from portopt_db.engine import DatabaseManager
from portopt_db.resilience import (
    BreakerConfig,
    BreakerState,
    DatabaseUnavailableError,
    DbCircuitBreaker,
    is_connectivity_failure,
)

pytestmark = pytest.mark.integration

_THRESHOLD = 2
_TIMEOUT_S = 30.0
_STATEMENT_TIMEOUT_MS = 50
# pg_terminate_backend waits up to this long for the backend to exit.
_TERMINATE_WAIT_MS = 5_000

_MakeManager = Callable[..., DatabaseManager]


@pytest.fixture
def clock() -> FakeClock:
    """A fresh fake clock per test."""
    return FakeClock()


@pytest.fixture
def make_manager(clock: FakeClock) -> Iterator[_MakeManager]:
    """Build managers whose breakers open after two failures; close them after."""
    managers: list[DatabaseManager] = []

    def make(url: str, **config: Any) -> DatabaseManager:
        """Return a manager on ``url`` with a fake-clock breaker."""
        breaker = DbCircuitBreaker(
            BreakerConfig(failure_threshold=_THRESHOLD, reset_timeout_s=_TIMEOUT_S),
            clock=clock,
        )
        manager = DatabaseManager(
            DbConfig(url=url, connect_timeout=2, **config), breaker=breaker
        )
        managers.append(manager)
        return manager

    yield make
    for manager in managers:
        manager.close()


def _closed_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _terminate(admin: Engine, pid: int) -> None:
    with admin.connect() as connection:
        ended = connection.execute(
            text("SELECT pg_terminate_backend(:pid, :wait_ms)"),
            {"pid": pid, "wait_ms": _TERMINATE_WAIT_MS},
        ).scalar_one()
    assert ended, f"backend {pid} did not exit"


def test_statement_timeouts_never_open_the_breaker(
    pg_url: str, make_manager: _MakeManager
) -> None:
    """A real QueryCanceled (57014) proves the server answered, so it never counts."""
    manager = make_manager(pg_url, statement_timeout_ms=_STATEMENT_TIMEOUT_MS)

    for _ in range(_THRESHOLD + 1):
        with (
            pytest.raises(OperationalError) as raised,
            manager.get_session() as session,
        ):
            session.execute(text("SELECT pg_sleep(1)"))
        assert getattr(raised.value.orig, "pgcode", None) == "57014"

    assert manager.breaker.state is BreakerState.CLOSED
    with manager.get_session() as session:
        assert session.execute(text("SELECT 1")).scalar_one() == 1


def test_a_refused_connect_counts_and_opens_the_breaker(
    pg_url: str, make_manager: _MakeManager
) -> None:
    """Nothing listens on the port: each connect fails, and the threshold opens it."""
    url = make_url(pg_url).set(host="127.0.0.1", port=_closed_port())
    manager = make_manager(url.render_as_string(hide_password=False))

    for _ in range(_THRESHOLD):
        with (
            pytest.raises(OperationalError) as raised,
            manager.get_session() as session,
        ):
            session.execute(text("SELECT 1"))
        assert is_connectivity_failure(raised.value)

    assert manager.breaker.state is BreakerState.OPEN
    with (
        patch.object(manager, "initialize") as initialize,
        pytest.raises(DatabaseUnavailableError),
        manager.get_session(),
    ):
        pass
    initialize.assert_not_called()


def test_a_killed_backend_counts_opens_the_breaker_and_a_probe_recovers(
    pg_url: str, pg_engine: Engine, make_manager: _MakeManager, clock: FakeClock
) -> None:
    """A backend terminated mid-session fails its next statement as a lost connection."""
    manager = make_manager(pg_url)

    for _ in range(_THRESHOLD):
        with (
            pytest.raises(OperationalError) as raised,
            manager.get_session() as session,
        ):
            pid = session.execute(text("SELECT pg_backend_pid()")).scalar_one()
            _terminate(pg_engine, pid)
            session.execute(text("SELECT 1"))
        assert is_connectivity_failure(raised.value)

    assert manager.breaker.state is BreakerState.OPEN
    with pytest.raises(DatabaseUnavailableError), manager.get_session():
        pass

    clock.advance(_TIMEOUT_S)
    with manager.get_session() as session:
        assert manager.breaker.state is BreakerState.CLOSED
        assert session.execute(text("SELECT 1")).scalar_one() == 1
