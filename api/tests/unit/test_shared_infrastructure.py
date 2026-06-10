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
from app.services.infrastructure.retry import _full_jitter, is_transient_network_error

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
        cb = CircuitBreaker(
            service_name="TestService", max_attempts=1, base_wait_minutes=0.0001
        )
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

    def test_backoff_doubles_per_attempt(self):
        # wait_seconds = (2**attempt) * 60 * base_wait_minutes / 2
        # attempt 1 → 120s, attempt 2 → 240s (geometric 2^attempt growth).
        with patch("app.services.infrastructure.circuit_breaker.time.time") as mt:
            mt.return_value = 0.0
            cb = CircuitBreaker(service_name="t", base_wait_minutes=2.0)
            cb.trigger()  # attempt 1
            first_wait = cb._until - 0.0
            mt.return_value = 1000.0  # past _until so the next trigger re-arms
            cb.trigger()  # attempt 2
            second_wait = cb._until - 1000.0

        assert first_wait == 120.0
        assert second_wait == 240.0
        assert second_wait == 2 * first_wait

    # ------------------------------------------------------------------
    # Uncovered branch: lines 57-58 — cooldown-elapsed half-open decay
    # ------------------------------------------------------------------

    def test_check_when_cooldown_elapsed_clears_active_and_decrements_attempt(self):
        mock_time = MagicMock()
        with patch("app.services.infrastructure.circuit_breaker.time", mock_time):
            # Phase 1: trigger() — time=0.0 so _until = 0.0 + wait_seconds
            mock_time.time.return_value = 0.0
            cb = CircuitBreaker(service_name="test", base_wait_minutes=1.0)
            cb.trigger()
            assert cb.attempt_count == 1
            assert cb._active is True

            # Phase 2: advance past _until so cooldown has elapsed
            mock_time.time.return_value = cb._until + 1.0

            # check() must NOT raise, must NOT sleep, and must enter the
            # half-open decay branch (lines 57-58).
            cb.check()

        mock_time.sleep.assert_not_called()
        assert cb.attempt_count == 0
        assert cb.is_active is False

    # ------------------------------------------------------------------
    # Uncovered branch: lines 76-77 — is_active expiry-clearing
    # ------------------------------------------------------------------

    def test_is_active_when_time_past_until_clears_and_returns_false(self):
        mock_time = MagicMock()
        with patch("app.services.infrastructure.circuit_breaker.time", mock_time):
            mock_time.time.return_value = 0.0
            cb = CircuitBreaker(service_name="test", base_wait_minutes=1.0)
            cb.trigger()
            assert cb._active is True

            # Advance past _until
            mock_time.time.return_value = cb._until + 1.0
            result = cb.is_active

        assert result is False
        assert cb._active is False

    # ------------------------------------------------------------------
    # AC: concurrent trigger() calls must not cause runaway increments
    # ------------------------------------------------------------------

    def test_concurrent_triggers_do_not_exceed_small_attempt_count(self):
        import threading

        # Large base_wait_minutes keeps the window open for the whole test
        cb = CircuitBreaker(service_name="test", base_wait_minutes=60.0)
        barrier = threading.Barrier(2)
        call_count = 10

        def hammer() -> None:
            barrier.wait()
            for _ in range(call_count):
                cb.trigger()

        t1 = threading.Thread(target=hammer)
        t2 = threading.Thread(target=hammer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # First trigger arms the breaker; all subsequent calls inside the same
        # active window are no-ops (guard at line 19-22).  With two threads
        # firing simultaneously exactly ONE trigger may land before the guard
        # is set, meaning attempt_count must be <= 2, not 20.
        assert cb.attempt_count <= 2


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

    # ------------------------------------------------------------------
    # Uncovered branch: line 106 False path — on_rate_limit is None
    # ------------------------------------------------------------------

    @patch("app.services.infrastructure.retry._full_jitter", return_value=0.0)
    def test_rate_limit_error_with_no_on_rate_limit_callback_returns_none(
        self, _mock_jitter
    ):
        # is_rate_limit_error fires, but on_rate_limit=None — line 106 False
        # branch must be taken without AttributeError or crash.
        action = MagicMock(
            side_effect=requests.exceptions.HTTPError("429 Too Many Requests")
        )
        result = retry_with_backoff(
            action,
            max_retries=3,
            is_rate_limit_error=is_transient_network_error,
            on_rate_limit=None,
        )
        assert result is None
        assert action.call_count == 3

    # ------------------------------------------------------------------
    # AC: _full_jitter bounds — cap and exponential ceiling respected
    # ------------------------------------------------------------------

    def test_full_jitter_bounded_by_base_when_cap_not_reached(self):
        # attempt=0 → uniform(0, min(120, 2.0*1)) = uniform(0, 2.0)
        for _ in range(20):
            val = _full_jitter(0, base=2.0, cap=120.0)
            assert 0.0 <= val <= 2.0

    def test_full_jitter_bounded_by_cap_when_exponential_exceeds_cap(self):
        # attempt=10 → base*2**10 = 2048 >> cap=5.0 → uniform(0, 5.0)
        for _ in range(20):
            val = _full_jitter(10, base=2.0, cap=5.0)
            assert 0.0 <= val <= 5.0


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
