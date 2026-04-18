"""Integration tests for ApiKeyAuthMiddleware activation in production (issue #429).

Activation policy (verified against ``api/app/main.py:169``):

    if settings.is_production_like:
        app.add_middleware(ApiKeyAuthMiddleware, ...)

``settings.is_production_like`` is ``True`` when ``ENVIRONMENT`` is
``"production"`` or ``"staging"`` (``app/config.py:131``). These tests patch
``settings.environment = "production"`` for the fixture duration, build a
fresh FastAPI app via ``create_application()``, and exercise the real app
under the production middleware stack.

Coverage:

* unauthed ``GET /api/v1/portfolio/`` → 401
* exempt paths (``/``, ``/health``, ``/metrics``, ``/docs``, ``/redoc``,
  ``/openapi.json``) → 2xx regardless of header
* valid seeded API key → 200
* missing / malformed / not-in-DB / revoked (``is_active=False``) key → 401

The ``ApiKey`` model only carries a boolean ``is_active`` flag — there is
no dedicated ``expires_at`` column — so the 'revoked' and 'expired' cases
collapse into a single data shape. The test covers revoked-via-``is_active``
and not-in-DB as the two distinct 401 code paths actually supported by the
schema.
"""

from __future__ import annotations

import hashlib
import secrets
from collections.abc import Generator
from contextlib import contextmanager

import pytest
from app.config import settings
from app.database import get_db
from app.main import create_application
from app.models.api_key import ApiKey
from app.models.base import Base
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


# ---------------------------------------------------------------------------
# Fixtures — isolated SQLite engine + production app factory
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def auth_engine():
    """Module-scoped SQLite engine for the production-mode app under test."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def auth_session(auth_engine) -> Generator[Session, None, None]:
    """Fresh session per test; rolled back on teardown so state never leaks."""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=auth_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        # Delete seeded rows explicitly so they don't leak across tests
        # (the shared engine persists rows across function-scoped sessions).
        session.rollback()
        session.query(ApiKey).delete()
        session.commit()
        session.close()


@pytest.fixture
def production_app(
    auth_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[FastAPI, None, None]:
    """Build a fresh FastAPI app under ``ENVIRONMENT=production``.

    The middleware reads ``database_manager.get_session`` at construction
    time, so the patch MUST precede ``create_application()``.
    """
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "rate_limit_requests", 10_000)  # never trip in tests

    @contextmanager
    def _session_cm() -> Generator[Session, None, None]:
        yield auth_session

    from app import database as db_module

    monkeypatch.setattr(db_module.database_manager, "get_session", _session_cm)

    app = create_application()
    app.dependency_overrides[get_db] = lambda: auth_session

    yield app

    app.dependency_overrides.clear()


@pytest.fixture
def client(production_app: FastAPI) -> Generator[TestClient, None, None]:
    with TestClient(production_app, raise_server_exceptions=False) as c:
        yield c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_key(session: Session, *, is_active: bool = True, name: str = "int-key") -> str:
    """Seed an ``ApiKey`` row and return the plaintext token."""
    plaintext = secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(plaintext.encode()).hexdigest()
    row = ApiKey(key_hash=key_hash, name=name, is_active=is_active)
    session.add(row)
    session.commit()
    return plaintext


_EXEMPT_PATHS = ("/", "/health", "/metrics", "/docs", "/redoc", "/openapi.json")


# ---------------------------------------------------------------------------
# Activation — unauthed protected call is rejected
# ---------------------------------------------------------------------------


class TestProductionRejectsUnauthed:
    """Middleware is live: protected routes require a valid X-API-Key."""

    def test_unauthed_get_portfolio_returns_401(self, client: TestClient) -> None:
        resp = client.get("/api/v1/portfolio/")
        assert resp.status_code == 401

    def test_unauthed_401_body_has_detail(self, client: TestClient) -> None:
        resp = client.get("/api/v1/portfolio/")
        assert "detail" in resp.json()


# ---------------------------------------------------------------------------
# Exempt list — public paths stay reachable
# ---------------------------------------------------------------------------


class TestExemptPathsStayReachable:
    """Public paths declared in ``ApiKeyAuthMiddleware.PUBLIC_PATHS`` must not 401."""

    @pytest.mark.parametrize("path", _EXEMPT_PATHS)
    def test_path_returns_2xx_without_api_key(
        self, client: TestClient, path: str
    ) -> None:
        resp = client.get(path)
        assert resp.status_code < 400, (
            f"exempt path {path!r} must not 401 under production middleware "
            f"(got {resp.status_code})"
        )


# ---------------------------------------------------------------------------
# Valid key — authorised access
# ---------------------------------------------------------------------------


class TestValidApiKeyAuthorises:
    """A seeded, active key grants access to protected routes."""

    def test_valid_key_returns_200_on_list_portfolios(
        self, client: TestClient, auth_session: Session
    ) -> None:
        plaintext = _make_key(auth_session)
        resp = client.get("/api/v1/portfolio/", headers={"X-API-Key": plaintext})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Invalid keys — every rejection path the schema supports
# ---------------------------------------------------------------------------


class TestInvalidApiKeyIsRejected:
    """Missing / malformed / revoked API keys must all return 401."""

    def test_missing_header_returns_401(self, client: TestClient) -> None:
        resp = client.get("/api/v1/portfolio/")
        assert resp.status_code == 401

    def test_empty_header_returns_401(self, client: TestClient) -> None:
        resp = client.get("/api/v1/portfolio/", headers={"X-API-Key": ""})
        assert resp.status_code == 401

    def test_malformed_header_returns_401(self, client: TestClient) -> None:
        resp = client.get("/api/v1/portfolio/", headers={"X-API-Key": "not-a-real-key"})
        assert resp.status_code == 401

    def test_properly_shaped_but_not_in_db_returns_401(
        self, client: TestClient
    ) -> None:
        """A token with the right shape that was never stored is still 401."""
        unknown = secrets.token_urlsafe(32)
        resp = client.get("/api/v1/portfolio/", headers={"X-API-Key": unknown})
        assert resp.status_code == 401

    def test_revoked_key_returns_401(
        self, client: TestClient, auth_session: Session
    ) -> None:
        """``is_active=False`` simulates revocation and (by schema) expiration."""
        plaintext = _make_key(auth_session, is_active=False, name="revoked-key")
        resp = client.get("/api/v1/portfolio/", headers={"X-API-Key": plaintext})
        assert resp.status_code == 401
