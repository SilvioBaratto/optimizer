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
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=connection
    )
    session = TestingSessionLocal()
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


# Add more fixtures as needed:
# - authenticated_client (with JWT token)
# - sample_user (creates a test user)
# - sample_data (populates test data)
