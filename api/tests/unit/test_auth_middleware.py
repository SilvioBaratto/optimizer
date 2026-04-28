"""Tests for ApiKeyAuthMiddleware and ApiKey model.

Covers:
- ApiKey model: table creation, UUID PK, timestamps, key_hash, name, is_active
- ApiKeyAuthMiddleware: public path passthrough, OPTIONS passthrough,
  missing header → 401, invalid key → 401, valid key → 200
- Revoked key (is_active=False) → 401
"""

from __future__ import annotations

import hashlib
import secrets
from contextlib import contextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.models.api_key import ApiKey
from app.models.base import Base

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=eng)
    yield eng
    Base.metadata.drop_all(bind=eng)


@pytest.fixture(scope="module")
def session_factory(engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture
def db(session_factory):
    session = session_factory()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


def _make_key() -> tuple[str, str]:
    """Return (plaintext, sha256_hex) pair."""
    plaintext = secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(plaintext.encode()).hexdigest()
    return plaintext, key_hash


def _pinned_factory(session: Session):
    """Return a session_factory that always yields the given session."""
    @contextmanager
    def factory():
        yield session
    return factory


# ---------------------------------------------------------------------------
# ApiKey model tests
# ---------------------------------------------------------------------------


class TestApiKeyModel:
    def test_table_created(self, engine):
        inspector = inspect(engine)
        assert "api_keys" in inspector.get_table_names()

    def test_columns_exist(self, engine):
        inspector = inspect(engine)
        cols = {c["name"] for c in inspector.get_columns("api_keys")}
        assert {"id", "key_hash", "name", "is_active", "created_at", "updated_at"} <= cols

    def test_insert_and_retrieve(self, db: Session):
        _, key_hash = _make_key()
        api_key = ApiKey(key_hash=key_hash, name="test-key")
        db.add(api_key)
        db.flush()

        found = db.query(ApiKey).filter_by(id=api_key.id).one()
        assert found.key_hash == key_hash
        assert found.name == "test-key"
        assert found.is_active is True

    def test_default_is_active_true(self, db: Session):
        _, key_hash = _make_key()
        api_key = ApiKey(key_hash=key_hash, name="active-by-default")
        db.add(api_key)
        db.flush()
        assert api_key.is_active is True

    def test_soft_delete_sets_is_active_false(self, db: Session):
        _, key_hash = _make_key()
        api_key = ApiKey(key_hash=key_hash, name="to-revoke")
        db.add(api_key)
        db.flush()

        api_key.is_active = False
        db.flush()

        found = db.query(ApiKey).filter_by(id=api_key.id).one()
        assert found.is_active is False


# ---------------------------------------------------------------------------
# Middleware integration tests
# ---------------------------------------------------------------------------


def _make_test_app(session: Session) -> tuple[FastAPI, str]:
    """Build a minimal FastAPI app with ApiKeyAuthMiddleware wired up."""
    from app.middleware.auth import ApiKeyAuthMiddleware

    plaintext, key_hash = _make_key()
    api_key = ApiKey(key_hash=key_hash, name="integration-key")
    session.add(api_key)
    session.flush()

    test_app = FastAPI()

    @test_app.get("/api/v1/protected")
    def protected():
        return {"ok": True}

    for path in ("/health", "/", "/docs", "/redoc", "/openapi.json", "/metrics"):
        # Register each public path as an endpoint
        test_app.add_api_route(path, lambda: {"ok": True})

    test_app.add_middleware(ApiKeyAuthMiddleware, session_factory=_pinned_factory(session))

    return test_app, plaintext


class TestApiKeyAuthMiddleware:
    @pytest.fixture
    def setup(self, session_factory):
        session = session_factory()
        app, plaintext = _make_test_app(session)
        client = TestClient(app, raise_server_exceptions=False)
        yield client, plaintext
        session.rollback()
        session.close()

    def test_missing_api_key_returns_401(self, setup):
        client, _ = setup
        resp = client.get("/api/v1/protected")
        assert resp.status_code == 401
        assert "detail" in resp.json()

    def test_401_response_is_json_with_detail(self, setup):
        client, _ = setup
        resp = client.get("/api/v1/protected")
        assert resp.headers["content-type"].startswith("application/json")
        assert "detail" in resp.json()

    def test_invalid_api_key_returns_401(self, setup):
        client, _ = setup
        resp = client.get("/api/v1/protected", headers={"X-API-Key": "bad-key"})
        assert resp.status_code == 401

    def test_valid_api_key_returns_200(self, setup):
        client, plaintext = setup
        resp = client.get("/api/v1/protected", headers={"X-API-Key": plaintext})
        assert resp.status_code == 200

    def test_health_path_no_auth_required(self, setup):
        client, _ = setup
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_root_path_no_auth_required(self, setup):
        client, _ = setup
        resp = client.get("/")
        assert resp.status_code == 200

    def test_docs_path_no_auth_required(self, setup):
        client, _ = setup
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_redoc_path_no_auth_required(self, setup):
        client, _ = setup
        resp = client.get("/redoc")
        assert resp.status_code == 200

    def test_openapi_json_no_auth_required(self, setup):
        client, _ = setup
        resp = client.get("/openapi.json")
        assert resp.status_code == 200

    def test_metrics_path_no_auth_required(self, setup):
        client, _ = setup
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_options_request_passes_through(self, setup):
        client, _ = setup
        resp = client.options("/api/v1/protected")
        assert resp.status_code != 401

    def test_revoked_key_returns_401(self, session_factory):
        session = session_factory()
        plaintext, key_hash = _make_key()
        api_key = ApiKey(key_hash=key_hash, name="revoked-key", is_active=False)
        session.add(api_key)
        session.flush()

        from app.middleware.auth import ApiKeyAuthMiddleware

        app = FastAPI()
        app.add_api_route("/api/v1/protected", lambda: {"ok": True})
        app.add_middleware(ApiKeyAuthMiddleware, session_factory=_pinned_factory(session))

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/v1/protected", headers={"X-API-Key": plaintext})
        assert resp.status_code == 401
        session.rollback()
        session.close()

    def test_empty_string_api_key_returns_401(self, setup):
        client, _ = setup
        resp = client.get("/api/v1/protected", headers={"X-API-Key": ""})
        assert resp.status_code == 401

    def test_whitespace_only_api_key_returns_401(self, setup):
        client, _ = setup
        resp = client.get("/api/v1/protected", headers={"X-API-Key": "   "})
        assert resp.status_code == 401

    def test_valid_format_key_not_in_db_returns_401(self, setup):
        """A properly-formatted token that was never stored must return 401."""
        client, _ = setup
        unknown_key = secrets.token_urlsafe(32)  # same format, not in DB
        resp = client.get("/api/v1/protected", headers={"X-API-Key": unknown_key})
        assert resp.status_code == 401

    def test_middleware_runs_before_route_handler(self, session_factory):
        """Route handler must NOT be called when auth fails."""
        from app.middleware.auth import ApiKeyAuthMiddleware

        handler_called = []

        session = session_factory()
        app = FastAPI()

        @app.get("/api/v1/guarded")
        def guarded():
            handler_called.append(True)
            return {"ok": True}

        app.add_middleware(ApiKeyAuthMiddleware, session_factory=_pinned_factory(session))

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/api/v1/guarded")

        assert resp.status_code == 401
        assert handler_called == [], "route handler must not execute when auth fails"
        session.close()
