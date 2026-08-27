"""Pytest fixtures for portopt-db: in-memory SQLite with the SAVEPOINT pattern.

Mirrors the ingestion daemon's DB test harness (StaticPool + per-test SAVEPOINT
rollback), but built off ``portopt_db.models`` — the models package is imported
so every table is registered on ``Base.metadata`` before ``create_all``.
"""

from collections.abc import Generator

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

# Importing the models package registers every model on Base.metadata.
from portopt_db.models import Base

TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(scope="session")
def test_engine():
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def db_session(test_engine) -> Generator[Session, None, None]:
    """New session per test; ``commit()`` in app code only releases the
    savepoint, and the outer transaction is rolled back on teardown."""
    connection = test_engine.connect()
    transaction = connection.begin()
    testing_session_local = sessionmaker(
        autocommit=False, autoflush=False, bind=connection
    )
    session = testing_session_local()
    session.begin_nested()

    @event.listens_for(session, "after_transaction_end")
    def restart_savepoint(sess, trans):
        if trans.nested and not trans._parent.nested:
            sess.begin_nested()

    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
