"""Branch coverage for app/dependencies.py.

Covers every conditional branch:
- get_current_user: header present vs absent
- get_optional_user: header present vs absent
- RateLimiter._check_memory_rate_limit: per_user/IP key selection, cache
  miss/hit, rate-exceeded raise
- RateLimiter._get_client_ip: X-Forwarded-For, X-Real-IP, direct client IP,
  no-client fallback
- PaginationParams.__init__: skip clamping, limit clamping (low/high), large-
  skip warning, cursor auto-activates use_cursor
- PaginationParams.decode_cursor: no cursor, valid cursor, corrupt cursor

Pure-unit: no DB session, no real HTTP transport.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from app.dependencies import (
    FilterParams,
    PaginationParams,
    RateLimiter,
    get_current_user,
    get_optional_user,
    get_user_id,
    require_user,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_request(
    forwarded_for: str | None = None,
    real_ip: str | None = None,
    client_host: str | None = "127.0.0.1",
) -> MagicMock:
    """Build a minimal MagicMock that mimics a FastAPI Request."""
    req = MagicMock()
    headers: dict[str, str] = {}
    if forwarded_for is not None:
        headers["X-Forwarded-For"] = forwarded_for
    if real_ip is not None:
        headers["X-Real-IP"] = real_ip

    req.headers.get = lambda key, default=None: headers.get(key, default)

    if client_host is None:
        req.client = None
    else:
        req.client = MagicMock()
        req.client.host = client_host
    return req


# ---------------------------------------------------------------------------
# get_current_user
# ---------------------------------------------------------------------------


class TestGetCurrentUser:
    def test_when_header_absent_then_default_stub_returned(self) -> None:
        result = get_current_user(x_user_id=None)
        assert result["id"] == "default"
        assert result["is_active"] is True

    def test_when_header_present_then_user_id_used(self) -> None:
        result = get_current_user(x_user_id="user-42")
        assert result["id"] == "user-42"
        assert "user-42" in result["email"]
        assert "user-42" in result["username"]


# ---------------------------------------------------------------------------
# get_optional_user
# ---------------------------------------------------------------------------


class TestGetOptionalUser:
    def test_when_header_absent_then_returns_none(self) -> None:
        db = MagicMock()
        result = get_optional_user(x_user_id=None, db=db)
        assert result is None

    def test_when_header_present_then_returns_user_dict(self) -> None:
        db = MagicMock()
        result = get_optional_user(x_user_id="alice", db=db)
        assert result is not None
        assert result["id"] == "alice"


# ---------------------------------------------------------------------------
# require_user / get_user_id (simple pass-through delegates)
# ---------------------------------------------------------------------------


class TestRequireUser:
    def test_returns_whatever_get_current_user_returns(self) -> None:
        user = {"id": "bob", "is_active": True}
        assert require_user(current_user=user) == user


class TestGetUserId:
    def test_returns_string_id(self) -> None:
        user = {"id": 99}
        assert get_user_id(current_user=user) == "99"


# ---------------------------------------------------------------------------
# RateLimiter._get_client_ip — three header branches + no-client fallback
# ---------------------------------------------------------------------------


class TestRateLimiterGetClientIp:
    def setup_method(self) -> None:
        self.limiter = RateLimiter(requests=100, window=60)

    def test_when_x_forwarded_for_present_then_first_ip_returned(self) -> None:
        req = _fake_request(forwarded_for="10.0.0.1, 10.0.0.2")
        assert self.limiter._get_client_ip(req) == "10.0.0.1"

    def test_when_x_real_ip_present_and_no_forwarded_for_then_real_ip_returned(
        self,
    ) -> None:
        req = _fake_request(real_ip="192.168.1.5")
        assert self.limiter._get_client_ip(req) == "192.168.1.5"

    def test_when_neither_header_present_then_client_host_returned(self) -> None:
        req = _fake_request(client_host="172.16.0.3")
        assert self.limiter._get_client_ip(req) == "172.16.0.3"

    def test_when_no_client_and_no_headers_then_unknown_returned(self) -> None:
        req = _fake_request(client_host=None)
        assert self.limiter._get_client_ip(req) == "unknown"


# ---------------------------------------------------------------------------
# RateLimiter._check_memory_rate_limit — key selection + exceed branch
# ---------------------------------------------------------------------------


class TestRateLimiterCheckMemoryRateLimit:
    def test_when_per_user_and_user_present_then_user_key_used(self) -> None:
        limiter = RateLimiter(requests=100, window=60, per_user=True)
        req = _fake_request()
        user = {"id": "carol"}
        limiter._check_memory_rate_limit(req, current_user=user)
        assert any("user:carol" in k for k in limiter._in_memory_cache)

    def test_when_per_user_but_no_user_then_ip_key_used(self) -> None:
        limiter = RateLimiter(requests=100, window=60, per_user=True)
        req = _fake_request(client_host="1.2.3.4")
        limiter._check_memory_rate_limit(req, current_user=None)
        assert any("ip:" in k for k in limiter._in_memory_cache)

    def test_when_not_per_user_then_ip_key_used(self) -> None:
        limiter = RateLimiter(requests=100, window=60, per_user=False)
        req = _fake_request(client_host="5.6.7.8")
        limiter._check_memory_rate_limit(req, current_user={"id": "dave"})
        assert any("ip:5.6.7.8" in k for k in limiter._in_memory_cache)

    def test_when_below_limit_then_returns_true(self) -> None:
        limiter = RateLimiter(requests=5, window=60)
        req = _fake_request(client_host="10.0.0.1")
        result = limiter._check_memory_rate_limit(req)
        assert result is True

    def test_when_limit_exceeded_then_raises_rate_limit_error(self) -> None:
        from app.exceptions import RateLimitError

        limiter = RateLimiter(requests=2, window=60)
        req = _fake_request(client_host="10.0.0.2")
        limiter._check_memory_rate_limit(req)
        limiter._check_memory_rate_limit(req)
        with pytest.raises(RateLimitError):
            limiter._check_memory_rate_limit(req)

    def test_old_timestamps_are_evicted_before_check(self) -> None:
        """Entries older than the window are removed, freeing capacity."""
        limiter = RateLimiter(requests=2, window=1)
        req = _fake_request(client_host="10.0.0.3")
        # Prime two stale entries directly
        stale_time = time.time() - 10  # well outside 1-second window
        limiter._in_memory_cache["ip:10.0.0.3"] = [stale_time, stale_time]
        # Should not raise — stale entries were removed
        result = limiter._check_memory_rate_limit(req)
        assert result is True


# ---------------------------------------------------------------------------
# RateLimiter.__call__
# ---------------------------------------------------------------------------


class TestRateLimiterCall:
    def test_call_delegates_to_check_memory_rate_limit(self) -> None:
        limiter = RateLimiter(requests=10, window=60)
        req = _fake_request(client_host="9.9.9.9")
        assert limiter(req) is True


# ---------------------------------------------------------------------------
# PaginationParams — constructor branches
# ---------------------------------------------------------------------------


class TestPaginationParams:
    def test_when_skip_negative_then_clamped_to_zero(self) -> None:
        p = PaginationParams(skip=-10)
        assert p.skip == 0

    def test_when_limit_zero_then_clamped_to_one(self) -> None:
        p = PaginationParams(limit=0)
        assert p.limit == 1

    def test_when_limit_exceeds_max_then_clamped_to_1000(self) -> None:
        p = PaginationParams(limit=9999)
        assert p.limit == 1000

    def test_when_cursor_provided_then_use_cursor_is_true(self) -> None:
        p = PaginationParams(cursor="abc123")
        assert p.use_cursor is True

    def test_when_use_cursor_explicit_true_and_no_cursor_then_true(self) -> None:
        p = PaginationParams(use_cursor=True)
        assert p.use_cursor is True

    def test_when_skip_large_and_no_cursor_then_warning_logged(self) -> None:
        with patch("app.dependencies.logger") as mock_log:
            PaginationParams(skip=10001, use_cursor=False)
            mock_log.warning.assert_called_once()

    def test_when_skip_large_but_use_cursor_then_no_warning(self) -> None:
        with patch("app.dependencies.logger") as mock_log:
            PaginationParams(skip=10001, use_cursor=True)
            mock_log.warning.assert_not_called()

    def test_normal_params_stored_as_is(self) -> None:
        p = PaginationParams(
            skip=5, limit=20, order_by="created_at", order_desc=True
        )
        assert p.skip == 5
        assert p.limit == 20
        assert p.order_by == "created_at"
        assert p.order_desc is True


# ---------------------------------------------------------------------------
# PaginationParams.encode_cursor / decode_cursor
# ---------------------------------------------------------------------------


class TestPaginationParamsCursor:
    def test_encode_then_decode_round_trip(self) -> None:
        p = PaginationParams(order_desc=False)
        token = p.encode_cursor("row-id-999")
        decoded_value, decoded_desc = p.decode_cursor.__func__(
            PaginationParams(cursor=token, order_desc=False)
        )
        assert decoded_value == "row-id-999"
        assert decoded_desc is False

    def test_when_no_cursor_then_decode_returns_none(self) -> None:
        p = PaginationParams()
        value, desc = p.decode_cursor()
        assert value is None
        assert desc is p.order_desc

    def test_when_corrupt_cursor_then_decode_returns_none_with_warning(self) -> None:
        with patch("app.dependencies.logger") as mock_log:
            p = PaginationParams(cursor="!!!not-valid-base64!!!")
            value, desc = p.decode_cursor()
        assert value is None
        mock_log.warning.assert_called_once()


# ---------------------------------------------------------------------------
# FilterParams — simple value storage
# ---------------------------------------------------------------------------


class TestGetRateLimiter:
    def test_returns_rate_limiter_with_configured_values(self) -> None:
        from app.dependencies import get_rate_limiter

        limiter = get_rate_limiter(requests=50, window=30)
        assert limiter.requests == 50
        assert limiter.window == 30


class TestFilterParams:
    def test_when_all_none_then_defaults(self) -> None:
        f = FilterParams()
        assert f.search is None
        assert f.is_active is None
        assert f.created_after is None
        assert f.created_before is None

    def test_when_values_provided_then_stored(self) -> None:
        f = FilterParams(
            search="AAPL",
            is_active=True,
            created_after="2024-01-01",
            created_before="2024-12-31",
        )
        assert f.search == "AAPL"
        assert f.is_active is True
        assert f.created_after == "2024-01-01"
        assert f.created_before == "2024-12-31"
