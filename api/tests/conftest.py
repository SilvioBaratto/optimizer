"""
Pytest Configuration and Shared Fixtures
=========================================

This module provides shared fixtures for testing the FastAPI application.
"""

from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from app.database import get_db
from app.main import app
from app.models.base import Base
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

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

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

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
