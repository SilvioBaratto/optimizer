"""Pure-unit branch coverage for RateLimitingMiddleware and LoggingMiddleware."""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

from starlette.responses import Response

from app.middleware.logging import LoggingMiddleware
from app.middleware.rate_limiting import RateLimitingMiddleware

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MW_MOD = "app.middleware.rate_limiting"


def _make_request(path: str = "/api/v1/x", client_host: str = "1.2.3.4") -> MagicMock:
    req = MagicMock()
    req.client.host = client_host
    req.url.path = path
    return req


def _call_next_200() -> AsyncMock:
    return AsyncMock(return_value=Response(status_code=200))


def _mock_settings(is_development: bool) -> MagicMock:
    s = MagicMock()
    s.is_development = is_development
    return s


def _asgi_app(status: int, headers: list, body: bytes):
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})

    return app


_HTTP_SCOPE = {
    "type": "http",
    "method": "GET",
    "path": "/x",
    "query_string": b"",
    "headers": [],
    "client": ("1.2.3.4", 123),
}


# ---------------------------------------------------------------------------
# RateLimitingMiddleware
# ---------------------------------------------------------------------------


class TestRateLimitingMiddleware:
    async def test_is_development_bypasses_rate_limit(self):
        mw = RateLimitingMiddleware(app=MagicMock(), requests=5, window=60)
        call_next = _call_next_200()
        req = _make_request()
        with patch(f"{_MW_MOD}.settings", _mock_settings(is_development=True)):
            resp = await mw.dispatch(req, call_next)
        call_next.assert_awaited_once()
        assert resp is call_next.return_value

    async def test_health_path_skips_rate_limit(self):
        mw = RateLimitingMiddleware(app=MagicMock(), requests=5, window=60)
        call_next = _call_next_200()
        req = _make_request(path="/health")
        with patch(f"{_MW_MOD}.settings", _mock_settings(is_development=False)):
            resp = await mw.dispatch(req, call_next)
        call_next.assert_awaited_once()
        assert resp.status_code == 200

    async def test_under_limit_records_request_and_adds_headers(self):
        mw = RateLimitingMiddleware(app=MagicMock(), requests=5, window=60)
        call_next = _call_next_200()
        req = _make_request()
        with patch(f"{_MW_MOD}.settings", _mock_settings(is_development=False)):
            resp = await mw.dispatch(req, call_next)
        assert resp.headers["X-RateLimit-Limit"] == "5"
        assert len(mw.client_requests["1.2.3.4"]) == 1

    async def test_stale_timestamps_pruned_before_check(self):
        mw = RateLimitingMiddleware(app=MagicMock(), requests=5, window=60)
        now = time.time()
        mw.client_requests["1.2.3.4"] = [now - 10_000]
        call_next = _call_next_200()
        req = _make_request()
        with patch(f"{_MW_MOD}.settings", _mock_settings(is_development=False)):
            await mw.dispatch(req, call_next)
        # stale entry removed, only the newly-added one remains
        assert len(mw.client_requests["1.2.3.4"]) == 1
        assert mw.client_requests["1.2.3.4"][0] > now - 1

    async def test_over_limit_returns_429_with_json_body(self):
        mw = RateLimitingMiddleware(app=MagicMock(), requests=1, window=60)
        mw.client_requests["1.2.3.4"] = [time.time()]  # already at limit
        call_next = _call_next_200()
        req = _make_request()
        with patch(f"{_MW_MOD}.settings", _mock_settings(is_development=False)):
            resp = await mw.dispatch(req, call_next)
        assert resp.status_code == 429
        body = json.loads(bytes(resp.body))
        assert body["error"]["code"] == "RATE_LIMIT_EXCEEDED"
        assert "Retry-After" in resp.headers

    async def test_none_client_uses_unknown_identifier(self):
        mw = RateLimitingMiddleware(app=MagicMock(), requests=5, window=60)
        call_next = _call_next_200()
        req = _make_request()
        req.client = None
        with patch(f"{_MW_MOD}.settings", _mock_settings(is_development=False)):
            await mw.dispatch(req, call_next)
        call_next.assert_awaited_once()
        assert "unknown" in mw.client_requests

    def test_cleanup_removes_stale_and_keeps_fresh(self):
        mw = RateLimitingMiddleware(app=MagicMock(), requests=5, window=60)
        now = time.time()
        mw.client_requests = {
            "old": [now - 10_000],
            "fresh": [now - 10_000, now],
        }
        mw.cleanup_old_entries()
        assert "old" not in mw.client_requests
        assert "fresh" in mw.client_requests
        assert all(ts > now - 1 for ts in mw.client_requests["fresh"])


# ---------------------------------------------------------------------------
# LoggingMiddleware
# ---------------------------------------------------------------------------


class TestLoggingMiddleware:
    async def test_non_http_scope_passes_through_unchanged(self):
        inner_app = AsyncMock()
        mw = LoggingMiddleware(app=inner_app)
        scope = {"type": "lifespan"}
        receive = AsyncMock()
        send = AsyncMock()
        await mw(scope, receive, send)
        inner_app.assert_awaited_once_with(scope, receive, send)

    async def test_valid_json_response_body_logged_without_error(self):
        inner = _asgi_app(200, [(b"content-type", b"application/json")], b'{"ok": true}')
        mw = LoggingMiddleware(app=inner)
        send = AsyncMock()
        await mw(_HTTP_SCOPE, AsyncMock(), send)
        assert send.await_count >= 2  # start + body messages forwarded

    async def test_invalid_json_body_caught_without_raise(self):
        inner = _asgi_app(200, [(b"content-type", b"application/json")], b"{bad json")
        mw = LoggingMiddleware(app=inner)
        await mw(_HTTP_SCOPE, AsyncMock(), AsyncMock())  # must not raise

    async def test_json_dumps_failure_caught_without_raise(self):
        inner = _asgi_app(200, [(b"content-type", b"application/json")], b'{"ok": true}')
        mw = LoggingMiddleware(app=inner)
        with patch("app.middleware.logging.json.dumps", side_effect=TypeError("boom")):
            await mw(_HTTP_SCOPE, AsyncMock(), AsyncMock())  # must not raise

    async def test_scope_without_client_uses_unknown(self):
        scope_no_client = {k: v for k, v in _HTTP_SCOPE.items() if k != "client"}
        inner = _asgi_app(200, [(b"content-type", b"application/json")], b'{"ok": true}')
        mw = LoggingMiddleware(app=inner)
        await mw(scope_no_client, AsyncMock(), AsyncMock())  # must not raise
