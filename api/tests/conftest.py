"""
Pytest Configuration and Shared Fixtures
=========================================

This module provides shared fixtures for testing the FastAPI application.
"""

from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import get_db
from app.main import app
from app.models._shared import Base
from tests._fixtures import (
    seed_factors,
    seed_macro,
    seed_market_data,
    seed_portfolio,
    seed_rebalancing,
    seed_risk,
    seed_universe,
)

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
    # Create all tables
    Base.metadata.create_all(bind=engine)
    yield engine
    # Drop all tables after tests
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def db_session(test_engine) -> Generator[Session, None, None]:
    """
    Create a new database session for each test function.

    Uses SQLAlchemy's SAVEPOINT pattern so that ``session.commit()``
    inside tests only releases the savepoint — the outer transaction
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
def client(db_session: Session) -> Generator[TestClient, None, None]:
    """
    Create a test client with overridden database dependency.
    """

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    # Patch database_manager so lifespan doesn't try to connect to PostgreSQL
    from contextlib import contextmanager

    from app import database as db_module

    @contextmanager
    def _test_session_cm():
        yield db_session

    # Create a mock initialize that sets the flags but doesn't try to connect
    def _mock_initialize():
        db_module.database_manager._is_initialized = True
        db_module.database_manager._engine = MagicMock()
        # Ensure session factory is set
        db_module.database_manager._session_factory = lambda: db_session

    with patch.object(db_module.database_manager, "get_session", _test_session_cm):
        with patch.object(
            db_module.database_manager, "initialize", side_effect=_mock_initialize
        ):
            with patch.object(db_module.database_manager, "create_all_tables"):
                with patch.object(
                    db_module.database_manager, "health_check", return_value=True
                ):
                    with patch.object(db_module.database_manager, "close"):
                        app.dependency_overrides[get_db] = override_get_db
                        test_client = TestClient(app)
                        try:
                            yield test_client
                        finally:
                            app.dependency_overrides.clear()


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    settings = MagicMock()
    settings.debug = True
    settings.environment = "test"
    settings.project_name = "Test App"
    settings.api_v1_str = "/api/v1"
    return settings


# ---------------------------------------------------------------------------
# Per-domain seed fixtures (issue #805)
# ---------------------------------------------------------------------------
# Thin wiring over the ``tests._fixtures`` seed builders (issue #804).  All are
# function-scoped so each runs inside the per-test SAVEPOINT and is rolled back
# on teardown.  FK-dependent fixtures depend on ``seeded_portfolio`` so the
# parent row is flushed before the child builder runs.


@pytest.fixture
def seeded_portfolio(db_session: Session):
    """Seed a Portfolio + snapshot + position + account."""
    return seed_portfolio(db_session)


@pytest.fixture
def seeded_universe(db_session: Session):
    """Seed an Exchange + Instrument."""
    return seed_universe(db_session)


@pytest.fixture
def seeded_market_data(db_session: Session):
    """Seed an Exchange + Instrument + TickerProfile."""
    return seed_market_data(db_session)


@pytest.fixture
def seeded_factors(db_session: Session):
    """Seed a FactorScore + FactorValidationReport."""
    return seed_factors(db_session)


@pytest.fixture
def seeded_macro(db_session: Session):
    """Seed an EconomicIndicator + MacroCalibration + FredObservation."""
    return seed_macro(db_session)


@pytest.fixture
def seeded_risk(db_session: Session, seeded_portfolio):
    """Seed a RiskLimit for the seeded portfolio."""
    return seed_risk(db_session, seeded_portfolio.portfolio)


@pytest.fixture
def seeded_rebalancing(db_session: Session, seeded_portfolio):
    """Seed a RebalancingPolicy for the seeded portfolio."""
    return seed_rebalancing(db_session, seeded_portfolio.portfolio)


@pytest.fixture
def job_service_mock():
    """Expose the BackgroundJobService mock context manager (issue #806).

    Lazy import keeps test collection decoupled from the helper module.
    """
    from tests._fixtures.job_service_mock import mock_job_service

    return mock_job_service
