"""Pytest configuration and shared fixtures for the ingestion daemon.

There is no HTTP layer, so there is no ``client`` fixture: tests drive the
service, repository, and scheduler functions directly against an in-memory
SQLite database.
"""

from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.models._shared import Base
from tests._fixtures import seed_macro, seed_market_data, seed_universe

# Test database URL - use SQLite for fast tests
TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(scope="session")
def test_engine():
    """Create a test database engine (session-scoped for performance)."""
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
    """
    Create a new database session for each test function.

    Uses SQLAlchemy's SAVEPOINT pattern so that ``session.commit()``
    inside app code only releases the savepoint — the outer transaction
    is rolled back on teardown, keeping the shared in-memory DB clean.
    """
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


@pytest.fixture(scope="function")
def patched_session_factory(db_session: Session, monkeypatch):
    """Point ``database_manager.get_session`` at the test session.

    Service functions and ``BackgroundJobService`` open their own sessions via
    ``database_manager.get_session`` rather than receiving one — they bypass any
    session a test holds. Tests that exercise those paths need this fixture, or
    they silently hit the real PostgreSQL URL.
    """
    from contextlib import contextmanager

    from app import database as db_module

    @contextmanager
    def _test_session_cm():
        yield db_session

    monkeypatch.setattr(db_module.database_manager, "get_session", _test_session_cm)
    return db_session


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock()
    settings.debug = True
    settings.environment = "test"
    settings.project_name = "Test App"
    return settings


# ---------------------------------------------------------------------------
# Per-domain seed fixtures
# ---------------------------------------------------------------------------
# Thin wiring over the ``tests._fixtures`` seed builders. All are
# function-scoped so each runs inside the per-test SAVEPOINT and is rolled back
# on teardown.


@pytest.fixture
def seeded_universe(db_session: Session):
    """Seed an Exchange + Instrument."""
    return seed_universe(db_session)


@pytest.fixture
def seeded_market_data(db_session: Session):
    """Seed an Exchange + Instrument + TickerProfile."""
    return seed_market_data(db_session)


@pytest.fixture
def seeded_macro(db_session: Session):
    """Seed an EconomicIndicator + MacroCalibration + FredObservation."""
    return seed_macro(db_session)


@pytest.fixture
def job_service_mock():
    """Expose the BackgroundJobService mock context manager.

    Lazy import keeps test collection decoupled from the helper module.
    """
    from tests._fixtures.job_service_mock import mock_job_service

    return mock_job_service
