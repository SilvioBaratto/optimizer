"""Pytest configuration for the fund test suite.

``fund`` is installed editable into the shared workspace venv (src-layout), so
``import fund`` resolves with no ``sys.path`` surgery. The Phase 1 hygiene guard
is source-blind (it scans tracked file text, imports no implementation module),
so it needs no fixtures.

Fase 2+ persistence/tool tests need a real (in-memory) database. The ``db_session``
fixture mirrors the portopt-db harness: StaticPool SQLite with a per-test
SAVEPOINT rollback, built off ``portopt_db.models`` so every table — including
the ``agent_runs`` / ``agent_decisions`` audit tables — is on ``Base.metadata``
before ``create_all``.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

# Importing the models package registers every model on Base.metadata.
from portopt_db.models import Base
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(scope="session")
def test_engine() -> Generator[Engine, None, None]:
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def db_session(test_engine: Engine) -> Generator[Session, None, None]:
    """New session per test; app-code ``commit()`` only releases the SAVEPOINT,
    and the outer transaction is rolled back on teardown."""
    connection = test_engine.connect()
    transaction = connection.begin()
    testing_session_local = sessionmaker(
        autocommit=False, autoflush=False, bind=connection
    )
    session = testing_session_local()
    session.begin_nested()

    @event.listens_for(session, "after_transaction_end")
    def restart_savepoint(sess: Session, trans: object) -> None:
        if trans.nested and not trans._parent.nested:  # type: ignore[attr-defined]
            sess.begin_nested()

    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
