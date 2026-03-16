"""Tests for shared resilience infrastructure (circuit breaker, retry, rate limiter)."""

import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from app.services.infrastructure import (
    CircuitBreaker,
    RateLimiter,
    retry_with_backoff,
)
from app.services.infrastructure.retry import is_transient_network_error


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_initial_state(self):
        cb = CircuitBreaker(service_name="test")
        assert cb.attempt_count == 0
        assert not cb.is_active

    def test_trigger_increments_attempt(self):
        cb = CircuitBreaker(service_name="test")
        cb.trigger()
        assert cb.attempt_count == 1
        assert cb.is_active

    def test_check_raises_after_max_attempts(self):
        cb = CircuitBreaker(service_name="TestService", max_attempts=1, base_wait_minutes=0.0001)
        cb.trigger()
        # Wait for the short backoff to expire so check() evaluates attempt count
        time.sleep(0.05)
        with pytest.raises(RuntimeError, match="TestService rate limit persists"):
            cb.check()

    def test_reset_decrements(self):
        cb = CircuitBreaker(service_name="test", base_wait_minutes=0.0001)
        cb.trigger()
        # Wait for backoff to expire so second trigger re-arms
        time.sleep(0.05)
        cb.trigger()
        assert cb.attempt_count == 2
        cb.reset()
        assert cb.attempt_count == 1

    def test_reset_does_not_go_below_zero(self):
        cb = CircuitBreaker(service_name="test")
        cb.reset()
        assert cb.attempt_count == 0

    def test_force_reset(self):
        cb = CircuitBreaker(service_name="test")
        cb.trigger()
        cb.trigger()
        cb.force_reset()
        assert cb.attempt_count == 0
        assert not cb.is_active

    def test_service_name_in_error(self):
        cb = CircuitBreaker(service_name="FRED API", max_attempts=1)
        cb.trigger()
        with pytest.raises(RuntimeError, match="FRED API"):
            cb.check()

    def test_default_service_name(self):
        cb = CircuitBreaker(max_attempts=1)
        cb.trigger()
        with pytest.raises(RuntimeError, match="external service"):
            cb.check()

    def test_check_waits_when_active(self):
        cb = CircuitBreaker(service_name="test", base_wait_minutes=0.001)
        cb.trigger()
        # Should not raise — just wait the short backoff period
        cb.check()


# ---------------------------------------------------------------------------
# is_transient_network_error
# ---------------------------------------------------------------------------


class TestTransientErrorDetection:
    @pytest.mark.parametrize(
        "error",
        [
            requests.exceptions.HTTPError("429 Too Many Requests"),
            ConnectionResetError("Connection reset by peer"),
            requests.exceptions.ReadTimeout("ReadTimeout"),
            requests.exceptions.ConnectTimeout("ConnectTimeout"),
            Exception("ChunkedEncodingError: incomplete"),
            Exception("RemoteDisconnected: peer closed"),
            Exception("IncompleteRead(0 bytes read)"),
            Exception("Rate limited by server"),
        ],
    )
    def test_detects_transient_errors(self, error):
        assert is_transient_network_error(error) is True

    @pytest.mark.parametrize(
        "error",
        [
            ValueError("invalid literal"),
            KeyError("missing_key"),
            Exception("404 Not Found"),
            TypeError("unexpected type"),
        ],
    )
    def test_rejects_non_transient_errors(self, error):
        assert is_transient_network_error(error) is False


# ---------------------------------------------------------------------------
# retry_with_backoff
# ---------------------------------------------------------------------------


class TestRetryWithBackoff:
    @patch("app.services.infrastructure.retry._full_jitter", return_value=0.0)
    def test_returns_on_first_success(self, _mock_jitter):
        action = MagicMock(return_value="ok")
        result = retry_with_backoff(action, max_retries=3)
        assert result == "ok"
        assert action.call_count == 1

    @patch("app.services.infrastructure.retry._full_jitter", return_value=0.0)
    def test_retries_on_exception(self, _mock_jitter):
        action = MagicMock(side_effect=[Exception("fail"), "ok"])
        result = retry_with_backoff(action, max_retries=3)
        assert result == "ok"
        assert action.call_count == 2

    @patch("app.services.infrastructure.retry._full_jitter", return_value=0.0)
    def test_returns_none_after_exhaustion(self, _mock_jitter):
        action = MagicMock(side_effect=Exception("always fails"))
        result = retry_with_backoff(action, max_retries=3)
        assert result is None
        assert action.call_count == 3

    @patch("app.services.infrastructure.retry._full_jitter", return_value=0.0)
    def test_calls_on_rate_limit(self, _mock_jitter):
        on_rate_limit = MagicMock()
        action = MagicMock(
            side_effect=[
                requests.exceptions.HTTPError("429 Too Many Requests"),
                "ok",
            ]
        )
        result = retry_with_backoff(
            action,
            max_retries=3,
            is_rate_limit_error=is_transient_network_error,
            on_rate_limit=on_rate_limit,
        )
        assert result == "ok"
        on_rate_limit.assert_called_once()

    @patch("app.services.infrastructure.retry._full_jitter", return_value=0.0)
    def test_calls_on_success(self, _mock_jitter):
        on_success = MagicMock()
        result = retry_with_backoff(
            lambda: "data",
            max_retries=1,
            on_success=on_success,
        )
        assert result == "data"
        on_success.assert_called_once_with("data")

    @patch("app.services.infrastructure.retry._full_jitter", return_value=0.0)
    def test_is_valid_rejects_bad_results(self, _mock_jitter):
        action = MagicMock(side_effect=[None, "good"])
        result = retry_with_backoff(
            action,
            max_retries=3,
            is_valid=lambda r: r is not None,
        )
        assert result == "good"
        assert action.call_count == 2


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_first_acquire_does_not_block(self):
        rl = RateLimiter(delay=1.0)
        start = time.monotonic()
        rl.acquire("key1")
        elapsed = time.monotonic() - start
        assert elapsed < 0.1

    def test_second_acquire_blocks(self):
        rl = RateLimiter(delay=0.2)
        rl.acquire("key1")
        start = time.monotonic()
        rl.acquire("key1")
        elapsed = time.monotonic() - start
        assert elapsed >= 0.15  # Should have waited ~0.2s

    def test_different_keys_independent(self):
        rl = RateLimiter(delay=0.5)
        rl.acquire("key1")
        start = time.monotonic()
        rl.acquire("key2")  # Different key — should not wait
        elapsed = time.monotonic() - start
        assert elapsed < 0.1

    def test_clear(self):
        rl = RateLimiter(delay=0.5)
        rl.acquire("key1")
        rl.clear()
        assert rl.get_last_request_time("key1") is None
