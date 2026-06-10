"""Branch coverage for app/exceptions.py.

Covers every exception class constructor (optional-argument present/absent),
every branch in create_error_response, and all async handler dispatch paths.
Pure-unit: no DB, no real HTTP transport. Handlers are called directly with a
MagicMock Request.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.exceptions import (
    AuthenticationError,
    AuthorizationError,
    AuthServiceUnavailableError,
    BaseAPIException,
    CacheError,
    ConflictError,
    DatabaseConnectionError,
    DatabaseError,
    DatabaseTimeoutError,
    ExternalServiceError,
    InsufficientPermissionsError,
    InvalidTokenFormatError,
    NotFoundError,
    RateLimitError,
    SecurityViolationError,
    SuspiciousActivityError,
    TokenExpiredError,
    TokenValidationError,
    UserInactiveError,
    ValidationError,
    create_error_response,
    setup_exception_handlers,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_request(
    path: str = "/test",
    method: str = "GET",
    client_host: str | None = "127.0.0.1",
    user_agent: str = "pytest",
    request_id: str | None = None,
) -> MagicMock:
    """Return a MagicMock that mimics a FastAPI ``Request``."""
    req = MagicMock()
    req.url.path = path
    req.method = method
    req.headers.get = lambda key, default="": {
        "user-agent": user_agent,
    }.get(key, default)

    if client_host is None:
        req.client = None
    else:
        req.client = MagicMock()
        req.client.host = client_host

    # state.request_id attribute
    state = MagicMock()
    state.request_id = request_id
    req.state = state

    return req


def _run(coro: Any) -> Any:
    """Run a coroutine synchronously in the test process."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Exception class construction — optional-argument branches
# ---------------------------------------------------------------------------


class TestBaseAPIException:
    def test_when_error_code_omitted_then_uses_class_name(self) -> None:
        exc = BaseAPIException("boom")
        assert exc.error_code == "BaseAPIException"
        assert exc.details == {}

    def test_when_error_code_provided_then_stored(self) -> None:
        exc = BaseAPIException("boom", error_code="CUSTOM", details={"k": "v"})
        assert exc.error_code == "CUSTOM"
        assert exc.details == {"k": "v"}


class TestValidationError:
    def test_when_errors_omitted_then_empty_dict(self) -> None:
        exc = ValidationError("bad input")
        assert exc.details == {"errors": {}}
        assert exc.status_code == 422

    def test_when_errors_provided_then_stored(self) -> None:
        exc = ValidationError("bad input", errors={"field": ["required"]})
        assert exc.details["errors"] == {"field": ["required"]}


class TestAuthenticationError:
    def test_default_message(self) -> None:
        exc = AuthenticationError()
        assert exc.message == "Authentication failed"
        assert exc.status_code == 401

    def test_custom_message(self) -> None:
        exc = AuthenticationError("token missing")
        assert exc.message == "token missing"


class TestAuthorizationError:
    def test_default_message(self) -> None:
        exc = AuthorizationError()
        assert exc.message == "Insufficient permissions"
        assert exc.status_code == 403

    def test_custom_message(self) -> None:
        exc = AuthorizationError("denied")
        assert exc.message == "denied"


class TestNotFoundError:
    def test_when_identifier_absent_then_simple_message(self) -> None:
        exc = NotFoundError("Portfolio")
        assert exc.message == "Portfolio not found"
        assert exc.status_code == 404

    def test_when_identifier_present_then_includes_id(self) -> None:
        exc = NotFoundError("Portfolio", identifier=42)
        assert "42" in exc.message


class TestConflictError:
    def test_construction(self) -> None:
        exc = ConflictError("already exists")
        assert exc.status_code == 409
        assert exc.error_code == "CONFLICT_ERROR"


class TestRateLimitError:
    def test_when_retry_after_absent_then_empty_details(self) -> None:
        exc = RateLimitError()
        assert exc.details == {}
        assert exc.status_code == 429

    def test_when_retry_after_present_then_stored_in_details(self) -> None:
        exc = RateLimitError(retry_after=30)
        assert exc.details == {"retry_after": 30}


class TestExternalServiceError:
    def test_construction(self) -> None:
        exc = ExternalServiceError("Stripe", "timeout")
        assert "Stripe" in exc.message
        assert exc.details == {"service": "Stripe"}
        assert exc.status_code == 503


class TestDatabaseError:
    def test_default_message(self) -> None:
        exc = DatabaseError()
        assert exc.message == "Database operation failed"
        assert exc.status_code == 500


class TestDatabaseTimeoutError:
    def test_when_timeout_duration_absent_then_no_details(self) -> None:
        exc = DatabaseTimeoutError()
        assert exc.error_code == "DATABASE_TIMEOUT_ERROR"
        assert exc.details == {}

    def test_when_timeout_duration_present_then_stored(self) -> None:
        exc = DatabaseTimeoutError(timeout_duration=5.0)
        assert exc.details == {"timeout_duration": 5.0}


class TestDatabaseConnectionError:
    def test_when_connection_info_absent_then_no_extra_details(self) -> None:
        exc = DatabaseConnectionError()
        assert exc.error_code == "DATABASE_CONNECTION_ERROR"
        assert exc.details == {}

    def test_when_connection_info_present_then_stored(self) -> None:
        info = {"host": "localhost", "port": 5432}
        exc = DatabaseConnectionError(connection_info=info)
        assert exc.details == {"connection_info": info}


class TestTokenValidationError:
    def test_when_token_info_absent_then_empty_dict(self) -> None:
        exc = TokenValidationError()
        assert exc.details == {"token_info": {}}

    def test_when_token_info_present_then_stored(self) -> None:
        exc = TokenValidationError(token_info={"sub": "user1"})
        assert exc.details["token_info"] == {"sub": "user1"}


class TestTokenExpiredError:
    def test_when_expired_at_absent_then_no_detail(self) -> None:
        exc = TokenExpiredError()
        # details may be {} or whatever AuthenticationError leaves; what matters
        # is that no "expired_at" key was injected
        assert "expired_at" not in exc.details

    def test_when_expired_at_present_then_stored(self) -> None:
        exc = TokenExpiredError(expired_at="2024-01-01T00:00:00Z")
        assert exc.details.get("expired_at") == "2024-01-01T00:00:00Z"


class TestInvalidTokenFormatError:
    def test_has_expected_format_detail(self) -> None:
        exc = InvalidTokenFormatError()
        assert exc.details["expected_format"] == "Bearer <JWT_TOKEN>"


class TestUserInactiveError:
    def test_when_user_id_absent_then_no_user_id_detail(self) -> None:
        exc = UserInactiveError()
        assert "user_id" not in exc.details

    def test_when_user_id_present_then_stored(self) -> None:
        exc = UserInactiveError(user_id="abc-123")
        assert exc.details.get("user_id") == "abc-123"


class TestInsufficientPermissionsError:
    def test_when_user_permissions_absent_then_empty_list(self) -> None:
        exc = InsufficientPermissionsError(required_permissions=["admin"])
        assert exc.details["user_permissions"] == []
        assert "admin" in exc.details["required_permissions"]

    def test_when_user_permissions_provided_then_stored(self) -> None:
        exc = InsufficientPermissionsError(
            required_permissions=["admin"],
            user_permissions=["read"],
        )
        assert exc.details["user_permissions"] == ["read"]


class TestSecurityViolationError:
    def test_when_details_absent_then_empty(self) -> None:
        exc = SecurityViolationError("sql_injection")
        assert exc.details == {}
        assert exc.status_code == 403

    def test_when_details_present_then_stored(self) -> None:
        exc = SecurityViolationError("xss", details={"payload": "<script>"})
        assert exc.details == {"payload": "<script>"}


class TestSuspiciousActivityError:
    def test_when_client_ip_absent_then_no_ip_detail(self) -> None:
        exc = SuspiciousActivityError("brute_force")
        assert "client_ip" not in exc.details

    def test_when_client_ip_present_then_stored(self) -> None:
        exc = SuspiciousActivityError("brute_force", client_ip="1.2.3.4")
        assert exc.details.get("client_ip") == "1.2.3.4"


class TestAuthServiceUnavailableError:
    def test_default_service_name(self) -> None:
        exc = AuthServiceUnavailableError()
        assert "Authentication Service" in exc.message


class TestCacheError:
    def test_when_message_absent_then_no_suffix(self) -> None:
        exc = CacheError("get")
        assert exc.message == "Cache get failed"
        assert exc.details == {"operation": "get"}

    def test_when_message_present_then_appended(self) -> None:
        exc = CacheError("set", message="redis down")
        assert exc.message == "Cache set failed: redis down"


# ---------------------------------------------------------------------------
# create_error_response — details / request_id branches
# ---------------------------------------------------------------------------


class TestCreateErrorResponse:
    def test_when_details_and_request_id_omitted_then_minimal_body(self) -> None:
        resp = create_error_response(400, "bad", "BAD_REQUEST")
        body = resp.body  # bytes
        import json

        data = json.loads(body)
        assert "details" not in data["error"]
        assert "request_id" not in data["error"]
        assert data["error"]["code"] == "BAD_REQUEST"

    def test_when_details_provided_then_included(self) -> None:
        import json

        resp = create_error_response(400, "bad", "BAD_REQUEST", details={"k": "v"})
        data = json.loads(resp.body)
        assert data["error"]["details"] == {"k": "v"}

    def test_when_request_id_provided_then_included(self) -> None:
        import json

        resp = create_error_response(
            400, "bad", "BAD_REQUEST", request_id="req-abc-123"
        )
        data = json.loads(resp.body)
        assert data["error"]["request_id"] == "req-abc-123"

    def test_status_code_forwarded(self) -> None:
        resp = create_error_response(503, "down", "SVC_DOWN")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Async exception handlers — called directly (not via HTTP transport)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _handlers() -> dict:
    """
    Build a minimal FastAPI app, wire handlers, then extract handler callables
    from the app's exception_handlers dict so we can call them directly.
    """
    from fastapi import FastAPI

    app = FastAPI()
    setup_exception_handlers(app)
    return app.exception_handlers


class TestValidationErrorHandler:
    def test_when_validation_error_then_returns_422(self, _handlers: dict) -> None:
        handler = _handlers[ValidationError]
        req = _fake_request(request_id="rid-1")
        exc = ValidationError("bad input", errors={"field": ["required"]})
        resp = _run(handler(req, exc))
        assert resp.status_code == 422

    def test_when_no_request_id_then_no_request_id_in_body(
        self, _handlers: dict
    ) -> None:
        import json

        handler = _handlers[ValidationError]
        req = _fake_request(request_id=None)
        req.state.request_id = None
        resp = _run(handler(req, ValidationError("bad")))
        data = json.loads(resp.body)
        assert "request_id" not in data["error"]


class TestSecurityViolationHandler:
    def test_when_security_violation_then_403_with_headers(
        self, _handlers: dict
    ) -> None:
        handler = _handlers[SecurityViolationError]
        req = _fake_request()
        exc = SecurityViolationError("xss", details={"payload": "<s>"})
        resp = _run(handler(req, exc))
        assert resp.status_code == 403
        assert resp.headers.get("X-Security-Event") == "violation_detected"
        assert resp.headers.get("X-Request-Blocked") == "true"

    def test_when_no_client_then_ip_is_unknown(self, _handlers: dict) -> None:
        handler = _handlers[SecurityViolationError]
        req = _fake_request(client_host=None)
        exc = SecurityViolationError("csrf")
        # should not raise even without client
        resp = _run(handler(req, exc))
        assert resp.status_code == 403


class TestAuthenticationErrorHandler:
    def test_when_auth_error_then_401_with_www_authenticate(
        self, _handlers: dict
    ) -> None:
        handler = _handlers[AuthenticationError]
        req = _fake_request()
        exc = AuthenticationError("token missing")
        resp = _run(handler(req, exc))
        assert resp.status_code == 401
        assert resp.headers.get("WWW-Authenticate") == "Bearer"
        assert resp.headers.get("X-Auth-Error") == "true"

    def test_when_no_client_then_no_error(self, _handlers: dict) -> None:
        handler = _handlers[AuthenticationError]
        req = _fake_request(client_host=None)
        resp = _run(handler(req, AuthenticationError()))
        assert resp.status_code == 401


class TestAuthorizationErrorHandler:
    def test_when_authz_error_then_403_with_permission_header(
        self, _handlers: dict
    ) -> None:
        handler = _handlers[AuthorizationError]
        req = _fake_request()
        exc = InsufficientPermissionsError(required_permissions=["admin"])
        resp = _run(handler(req, exc))
        assert resp.status_code == 403
        assert resp.headers.get("X-Permission-Required") == "true"

    def test_when_no_client_then_fallback(self, _handlers: dict) -> None:
        handler = _handlers[AuthorizationError]
        req = _fake_request(client_host=None)
        resp = _run(handler(req, AuthorizationError()))
        assert resp.status_code == 403


class TestRateLimitHandler:
    def test_when_rate_limit_error_then_429_with_retry_header(
        self, _handlers: dict
    ) -> None:
        handler = _handlers[RateLimitError]
        req = _fake_request()
        exc = RateLimitError(retry_after=60)
        resp = _run(handler(req, exc))
        assert resp.status_code == 429
        assert resp.headers.get("Retry-After") == "60"
        assert resp.headers.get("X-RateLimit-Exceeded") == "true"

    def test_when_retry_after_absent_then_defaults_to_60(
        self, _handlers: dict
    ) -> None:
        handler = _handlers[RateLimitError]
        req = _fake_request()
        exc = RateLimitError()  # no retry_after → details == {}
        resp = _run(handler(req, exc))
        assert resp.headers.get("Retry-After") == "60"


class TestDatabaseTimeoutHandler:
    def test_when_db_timeout_then_500_with_timeout_header(
        self, _handlers: dict
    ) -> None:
        handler = _handlers[DatabaseTimeoutError]
        req = _fake_request()
        exc = DatabaseTimeoutError(timeout_duration=30.0)
        resp = _run(handler(req, exc))
        assert resp.status_code == 500
        assert resp.headers.get("X-Database-Error") == "timeout"
        assert resp.headers.get("Retry-After") == "5"


class TestDatabaseConnectionHandler:
    def test_when_db_connection_error_then_500_with_connection_header(
        self, _handlers: dict
    ) -> None:
        handler = _handlers[DatabaseConnectionError]
        req = _fake_request()
        exc = DatabaseConnectionError(connection_info={"host": "localhost"})
        resp = _run(handler(req, exc))
        assert resp.status_code == 500
        assert resp.headers.get("X-Database-Error") == "connection_failed"
        assert resp.headers.get("Retry-After") == "10"


class TestExternalServiceHandler:
    def test_when_external_service_error_then_503_with_service_header(
        self, _handlers: dict
    ) -> None:
        handler = _handlers[ExternalServiceError]
        req = _fake_request()
        exc = ExternalServiceError("Redis", "connection refused")
        resp = _run(handler(req, exc))
        assert resp.status_code == 503
        assert resp.headers.get("X-Service-Error") == "Redis"


class TestDatabaseErrorHandler:
    def test_when_database_error_then_500(self, _handlers: dict) -> None:
        handler = _handlers[DatabaseError]
        req = _fake_request()
        exc = DatabaseError("constraint violation")
        resp = _run(handler(req, exc))
        assert resp.status_code == 500


class TestBaseAPIExceptionHandler:
    def test_when_base_api_exception_then_correct_status(
        self, _handlers: dict
    ) -> None:
        handler = _handlers[BaseAPIException]
        req = _fake_request()
        exc = BaseAPIException("something broke", status_code=418)
        resp = _run(handler(req, exc))
        assert resp.status_code == 418


class TestGeneralExceptionHandler:
    def test_when_debug_false_then_generic_message(self, _handlers: dict) -> None:
        import json

        from fastapi import FastAPI

        app = FastAPI(debug=False)
        setup_exception_handlers(app)
        # call the handler that was registered on this non-debug app
        non_debug_handler = app.exception_handlers[Exception]
        req = _fake_request()
        resp = _run(non_debug_handler(req, RuntimeError("internal detail")))
        data = json.loads(resp.body)
        assert data["error"]["message"] == "An unexpected error occurred"

    def test_when_debug_true_then_actual_message(self) -> None:
        import json

        from fastapi import FastAPI

        app = FastAPI(debug=True)
        setup_exception_handlers(app)
        handler = app.exception_handlers[Exception]
        req = _fake_request()
        resp = _run(handler(req, RuntimeError("real detail")))
        data = json.loads(resp.body)
        assert data["error"]["message"] == "real detail"


class TestRequestValidationHandler:
    def test_when_pydantic_validation_error_then_422_with_structured_errors(
        self, _handlers: dict
    ) -> None:
        import json

        from fastapi.exceptions import RequestValidationError

        handler = _handlers[RequestValidationError]
        req = _fake_request()

        # Build a minimal pydantic validation error list
        pydantic_errors = [
            {"loc": ("body", "field_a"), "msg": "field required", "type": "missing"}
        ]
        exc = RequestValidationError(errors=pydantic_errors)
        resp = _run(handler(req, exc))
        assert resp.status_code == 422
        data = json.loads(resp.body)
        assert "validation_errors" in data["error"]["details"]

    def test_when_error_loc_is_empty_then_empty_field_key(
        self, _handlers: dict
    ) -> None:
        """loc with only one element (e.g. ``('body',)``) produces empty field string."""
        import json

        from fastapi.exceptions import RequestValidationError

        handler = _handlers[RequestValidationError]
        req = _fake_request()
        pydantic_errors = [
            {"loc": ("body",), "msg": "required", "type": "missing"}
        ]
        exc = RequestValidationError(errors=pydantic_errors)
        resp = _run(handler(req, exc))
        assert resp.status_code == 422
        data = json.loads(resp.body)
        # field is "" (loc[1:] is empty)
        assert "" in data["error"]["details"]["validation_errors"]


class TestStarletteHTTPExceptionHandler:
    def test_when_starlette_http_exception_then_proxies_status_code(
        self, _handlers: dict
    ) -> None:
        from starlette.exceptions import HTTPException as StarletteHTTPException

        handler = _handlers[StarletteHTTPException]
        req = _fake_request()
        exc = StarletteHTTPException(status_code=404, detail="not here")
        resp = _run(handler(req, exc))
        assert resp.status_code == 404


class TestRequestValidationHandlerDuplicateField:
    def test_when_same_field_has_multiple_errors_then_all_messages_accumulated(
        self, _handlers: dict
    ) -> None:
        """Covers the ``if field not in errors`` false-branch (field already seen)."""
        import json

        from fastapi.exceptions import RequestValidationError

        handler = _handlers[RequestValidationError]
        req = _fake_request()
        # Two errors on the same field — second hits the ``else`` branch
        pydantic_errors = [
            {"loc": ("body", "email"), "msg": "field required", "type": "missing"},
            {"loc": ("body", "email"), "msg": "invalid format", "type": "value_error"},
        ]
        exc = RequestValidationError(errors=pydantic_errors)
        resp = _run(handler(req, exc))
        data = json.loads(resp.body)
        email_msgs = data["error"]["details"]["validation_errors"]["email"]
        assert len(email_msgs) == 2
