"""Unit tests for Trading212Client — all HTTP calls mocked via requests.

The client's surface is intentionally tiny: the two metadata endpoints the
ingestion daemon actually uses, plus the shared ``_get`` retry helper and the
``from_settings`` factory. The account / portfolio / order / dividend methods
were removed (daemon ingests, does not trade), so their tests are gone too.

Waits are measured on a fake clock: ``time.sleep`` advances ``time.time``, and
the retry helper's small jitter is pinned to zero, so the gap between two
requests is the wait the client chose.
"""

from __future__ import annotations

import base64
import itertools
import logging
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests
from prometheus_client import REGISTRY

from app import metrics
from app.config import Settings, settings
from app.services.infrastructure import CircuitOpenError
from app.services.infrastructure.external_calls import (
    ExternalService,
    build_breaker,
    process_breaker,
)
from app.services.universe.trading212.client import (
    _MAX_RATE_LIMIT_WAIT_SECONDS,
    Trading212Client,
)

_NOW = 1_700_000_000.0
_SERVICE = ExternalService.TRADING_212.value

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(mode: str = "live", max_retries: int = 3) -> Trading212Client:
    return Trading212Client(
        api_key="test-key-abc",
        api_secret="test-secret-xyz",
        mode=mode,
        max_retries=max_retries,
    )


def _mock_response(
    json_data, status_code: int = 200, headers: dict[str, str] | None = None
) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    resp.headers = headers if headers is not None else {}
    return resp


def _mock_429_error(retry_after: str | None = None) -> requests.HTTPError:
    response = MagicMock()
    response.status_code = 429
    response.headers = {"Retry-After": retry_after} if retry_after else {}
    err = requests.HTTPError("429 Too Many Requests")
    err.response = response
    return err


def _mock_429_with_reset(reset_epoch: float) -> requests.HTTPError:
    response = MagicMock()
    response.status_code = 429
    response.headers = {"x-ratelimit-reset": str(int(reset_epoch))}
    err = requests.HTTPError("429 Too Many Requests")
    err.response = response
    return err


def _mock_http_error(status_code: int) -> requests.HTTPError:
    response = MagicMock()
    response.status_code = status_code
    response.headers = {}
    err = requests.HTTPError(f"{status_code} Error")
    err.response = response
    return err


def _raise(err):
    """Helper: make side_effect raise an exception from a mock_get call."""
    resp = MagicMock()
    resp.raise_for_status.side_effect = err
    return resp


class _Clock:
    """Stand-in for ``time.time`` / ``time.sleep``; sleeping advances the time."""

    def __init__(self) -> None:
        self.now = _NOW

    def time(self) -> float:
        """Return the fake current time."""
        return self.now

    def sleep(self, seconds: float) -> None:
        """Advance the fake time instead of blocking."""
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    """Run the client, its breaker and the retry helper on one fake clock."""
    fake = _Clock()
    monkeypatch.setattr(time, "time", fake.time)
    monkeypatch.setattr(time, "sleep", fake.sleep)
    monkeypatch.setattr(
        "app.services.infrastructure.retry._full_jitter", lambda *_: 0.0
    )
    return fake


@pytest.fixture
def mock_get(clock: _Clock):
    """Patch ``requests.get``; ``mock_get.sent_at`` lists the time of each request."""
    with patch("app.services.universe.trading212.client.requests.get") as get:
        get.sent_at = []
        answers: list[Any] = []

        def answer(*_args: Any, **_kwargs: Any) -> Any:
            """Note the send time, then give the next queued answer or raise it."""
            get.sent_at.append(clock.now)
            item = answers.pop(0) if answers else get.return_value
            if isinstance(item, Exception):
                raise item
            return item

        def queue(items: list[Any]) -> None:
            """Answer the next requests with ``items``, in order."""
            answers.extend(items)

        get.side_effect = answer
        get.queue = queue
        yield get


def _gaps(sent_at: list[float]) -> list[float]:
    return [later - earlier for earlier, later in itertools.pairwise(sent_at)]


# ---------------------------------------------------------------------------
# Credentials never reach logs
# ---------------------------------------------------------------------------


class TestCredentialsAreNotRendered:
    """The API key and secret stay out of every rendering of the client."""

    def test_repr_hides_api_key_and_secret(self):
        """``repr`` omits both credentials."""
        text = repr(_make_client())

        assert "test-key-abc" not in text
        assert "test-secret-xyz" not in text

    def test_repr_still_identifies_the_client(self):
        """``repr`` keeps the mode."""
        assert "mode='live'" in repr(_make_client())

    def test_enclosing_dataclass_repr_hides_credentials(self):
        """A dataclass holding the client does not leak them either."""
        from app.services.universe.trading212.builder import UniverseBuilder
        from app.services.universe.trading212.config import UniverseBuilderConfig

        builder = UniverseBuilder(
            config=UniverseBuilderConfig(),
            api_client=_make_client(),
            ticker_mapper=MagicMock(),
            repository=MagicMock(),
        )

        assert "test-secret-xyz" not in repr(builder)

    def test_a_failed_call_logs_no_credential(self, mock_get, caplog):
        """The exhaustion log line names the error, never the key or the header."""
        mock_get.queue([requests.exceptions.ConnectionError("refused")] * 2)
        encoded = base64.b64encode(b"test-key-abc:test-secret-xyz").decode()

        with (
            caplog.at_level(logging.DEBUG),
            pytest.raises(requests.exceptions.ConnectionError),
        ):
            _make_client(max_retries=2)._get("/p")

        assert "refused" in caplog.text
        for secret in ("test-key-abc", "test-secret-xyz", encoded):
            assert secret not in caplog.text


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestTrading212ClientInit:
    """Construction picks the host and reads its knobs from settings."""

    def test_live_mode_sets_live_base_url(self):
        """``live`` points at the live host."""
        c = _make_client(mode="live")
        assert c.base_url == "https://live.trading212.com"

    def test_demo_mode_sets_demo_base_url(self):
        """``demo`` points at the demo host."""
        c = _make_client(mode="demo")
        assert c.base_url == "https://demo.trading212.com"

    def test_headers_contain_basic_auth(self):
        """Basic auth is Base64 of key and secret."""
        c = _make_client()
        expected = base64.b64encode(b"test-key-abc:test-secret-xyz").decode()
        assert c.headers["Authorization"] == f"Basic {expected}"

    def test_default_max_retries(self):
        """Five attempts unless the setting says otherwise."""
        c = Trading212Client(api_key="k", api_secret="s")
        assert c.max_retries == 5

    def test_attempts_and_timeout_come_from_settings(self, monkeypatch, mock_get):
        """``TRADING_212_REQUEST_ATTEMPTS`` and its timeout reach the request."""
        monkeypatch.setattr(settings, "trading_212_request_attempts", 7)
        monkeypatch.setattr(settings, "trading_212_request_timeout_seconds", 12.5)
        mock_get.return_value = _mock_response({"ok": True})
        c = Trading212Client(api_key="k", api_secret="s")

        c._get("/p")

        assert c.max_retries == 7
        assert mock_get.call_args.kwargs["timeout"] == 12.5


# ---------------------------------------------------------------------------
# Browser headers — Cloudflare 403 dodge on the metadata endpoints
# ---------------------------------------------------------------------------


class TestBrowserHeaders:
    """Every request carries the headers that clear Cloudflare."""

    def test_headers_include_browser_user_agent(self):
        """A browser User-Agent."""
        c = _make_client()
        assert "Mozilla/5.0" in c.headers["User-Agent"]

    def test_headers_include_json_accept(self):
        """A JSON Accept header."""
        c = _make_client()
        assert c.headers["Accept"] == "application/json"


# ---------------------------------------------------------------------------
# Metadata endpoints — the live surface
# ---------------------------------------------------------------------------


class TestMetadataEndpoints:
    """The two metadata endpoints map to their paths."""

    @patch.object(Trading212Client, "_get")
    def test_get_exchanges_calls_correct_endpoint(self, mock_get):
        """Exchanges come from the exchanges path."""
        mock_get.return_value = [{"id": 1, "name": "NASDAQ"}]
        c = _make_client()
        assert c.get_exchanges() == [{"id": 1, "name": "NASDAQ"}]
        mock_get.assert_called_once_with("/api/v0/equity/metadata/exchanges")

    @patch.object(Trading212Client, "_get")
    def test_get_instruments_calls_correct_endpoint(self, mock_get):
        """Instruments come from the instruments path."""
        mock_get.return_value = [{"ticker": "AAPL_US_EQ", "type": "STOCK"}]
        c = _make_client()
        assert c.get_instruments() == [{"ticker": "AAPL_US_EQ", "type": "STOCK"}]
        mock_get.assert_called_once_with("/api/v0/equity/metadata/instruments")


# ---------------------------------------------------------------------------
# _get — retry / error behaviour
# ---------------------------------------------------------------------------


class TestGetInternal:
    """``_get`` returns the body, retries what can recover and raises the rest."""

    def test_success_on_first_attempt(self, mock_get):
        """One answer, one request."""
        mock_get.return_value = _mock_response({"ok": True})
        c = _make_client()
        assert c._get("/path") == {"ok": True}
        assert mock_get.call_count == 1

    def test_passes_params_to_requests(self, mock_get):
        """Query parameters reach ``requests.get``."""
        mock_get.return_value = _mock_response([])
        c = _make_client()
        c._get("/path", params={"limit": 50})
        assert mock_get.call_args.kwargs["params"] == {"limit": 50}

    def test_a_null_body_is_returned_as_none(self, mock_get):
        """A JSON ``null`` is an answer, not a failed call."""
        mock_get.return_value = _mock_response(None)

        assert _make_client()._get("/p") is None
        assert mock_get.call_count == 1

    def test_retries_on_network_error(self, mock_get):
        """A transport error is retried."""
        mock_get.queue(
            [
                requests.exceptions.ConnectionError("fail"),
                _mock_response({"retry": True}),
            ]
        )
        c = _make_client()
        assert c._get("/p") == {"retry": True}
        assert mock_get.call_count == 2

    def test_raises_after_max_retries_network_error(self, mock_get):
        """The last transport error is raised once every attempt failed."""
        mock_get.queue([requests.exceptions.ConnectionError("fail")] * 2)
        c = _make_client(max_retries=2)
        with pytest.raises(requests.exceptions.ConnectionError):
            c._get("/p")
        assert mock_get.call_count == 2

    def test_handles_429_with_retry_after_header(self, mock_get):
        """A 429's ``Retry-After`` is the wait before the next request."""
        mock_get.queue(
            [_raise(_mock_429_error(retry_after="3")), _mock_response({"ok": True})]
        )
        c = _make_client()
        assert c._get("/p") == {"ok": True}
        assert _gaps(mock_get.sent_at) == [pytest.approx(3)]

    def test_429_without_a_usable_header_waits_the_first_breaker_window(self, mock_get):
        """No usable header: the breaker's first window, 2 s."""
        mock_get.queue([_raise(_mock_429_error()), _mock_response({"ok": True})])
        c = _make_client()
        assert c._get("/p") == {"ok": True}
        assert _gaps(mock_get.sent_at) == [pytest.approx(2)]

    def test_raises_non_429_http_error_immediately(self, mock_get):
        """A 404 is not retried."""
        mock_get.queue([_raise(_mock_http_error(404))])
        c = _make_client()
        with pytest.raises(requests.HTTPError):
            c._get("/p")
        assert mock_get.call_count == 1

    def test_raises_on_last_429_attempt(self, mock_get):
        """The last 429 is raised once the attempts run out."""
        mock_get.queue([_raise(_mock_429_error()), _raise(_mock_429_error())])
        c = _make_client(max_retries=2)
        with pytest.raises(requests.HTTPError):
            c._get("/p")

    def test_fewer_than_one_attempt_is_refused_without_a_request(self, mock_get):
        """``max_retries`` below 1 is a configuration error, not a silent no-op."""
        with pytest.raises(ValueError, match="max_retries"):
            _make_client(max_retries=0)._get("/p")
        assert mock_get.call_count == 0


class TestRateLimitWindowBackoff:
    """T212 allows about one /instruments call per 50 s and reports the window
    end in ``x-ratelimit-reset`` (epoch seconds), not ``Retry-After``."""

    def test_429_waits_until_the_reset_header_instead_of_a_short_backoff(
        self, mock_get
    ):
        """The wait is the time to the reset plus a one-second margin."""
        mock_get.queue(
            [_raise(_mock_429_with_reset(_NOW + 45)), _mock_response({"ok": True})]
        )
        c = _make_client()

        assert c._get("/p") == {"ok": True}

        assert _gaps(mock_get.sent_at) == [pytest.approx(46)]

    def test_wait_for_the_reset_is_capped(self, mock_get):
        """A reset far in the future is held to the 120 s ceiling."""
        mock_get.queue(
            [
                _raise(_mock_429_with_reset(_NOW + 100_000)),
                _mock_response({"ok": True}),
            ]
        )
        c = _make_client()

        c._get("/p")

        assert _gaps(mock_get.sent_at) == [pytest.approx(_MAX_RATE_LIMIT_WAIT_SECONDS)]

    def test_reset_already_in_the_past_falls_back_to_the_breaker_window(self, mock_get):
        """A stale reset says nothing; the breaker's first window applies."""
        mock_get.queue(
            [_raise(_mock_429_with_reset(_NOW - 30)), _mock_response({"ok": True})]
        )
        c = _make_client()

        c._get("/p")

        assert _gaps(mock_get.sent_at) == [pytest.approx(2)]

    def test_retry_after_still_wins_over_the_reset_header(self, mock_get):
        """``Retry-After`` takes precedence over ``x-ratelimit-reset``."""
        err = _mock_429_with_reset(_NOW + 45)
        err.response.headers["Retry-After"] = "3"
        mock_get.queue([_raise(err), _mock_response({"ok": True})])
        c = _make_client()

        c._get("/p")

        assert _gaps(mock_get.sent_at) == [pytest.approx(3)]


class TestTransientServerErrorsAreRetried:
    """408 and 5xx are retried after the breaker's window; other 4xx are not."""

    @pytest.mark.parametrize("status", [408, 500, 502, 503, 504])
    def test_transient_status_is_retried_with_backoff(self, mock_get, status):
        """One transient answer, then success, two seconds apart."""
        mock_get.queue([_raise(_mock_http_error(status)), _mock_response({"ok": True})])
        c = _make_client()

        assert c._get("/p") == {"ok": True}
        assert mock_get.call_count == 2
        assert _gaps(mock_get.sent_at) == [pytest.approx(2)]

    def test_persistent_server_error_is_raised_after_max_retries(self, mock_get):
        """The last 5xx is raised once every attempt failed."""
        mock_get.queue([_raise(_mock_http_error(503)) for _ in range(2)])
        c = _make_client(max_retries=2)

        with pytest.raises(requests.HTTPError):
            c._get("/p")
        assert mock_get.call_count == 2

    def test_an_http_error_without_a_response_is_raised_at_once(self, mock_get):
        """Without a status there is nothing to judge, so nothing is retried."""
        mock_get.queue([_raise(requests.HTTPError("no response attached"))])
        c = _make_client()

        with pytest.raises(requests.HTTPError):
            c._get("/p")
        assert mock_get.call_count == 1
        assert c.breaker.attempt_count == 0

    @pytest.mark.parametrize("status", [400, 401, 403, 404])
    def test_client_errors_are_not_retried(self, mock_get, status):
        """An answer about the request itself is final and trips nothing."""
        mock_get.queue([_raise(_mock_http_error(status))])
        c = _make_client()

        with pytest.raises(requests.HTTPError):
            c._get("/p")
        assert mock_get.call_count == 1
        assert c.breaker.attempt_count == 0


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


def _client_with_breaker(max_attempts: int, max_retries: int = 5) -> Trading212Client:
    breaker = build_breaker(
        ExternalService.TRADING_212,
        Settings(_env_file=None, BREAKER_TRADING_212_MAX_ATTEMPTS=max_attempts),
    )
    return Trading212Client(
        api_key="k", api_secret="s", max_retries=max_retries, breaker=breaker
    )


def _gauge() -> float | None:
    return REGISTRY.get_sample_value("external_breaker_open", {"service": _SERVICE})


@pytest.fixture
def process_breaker_on_the_gauge():
    """Track the process-wide T212 breaker on the gauge, reset before and after."""
    breaker = process_breaker(ExternalService.TRADING_212)
    breaker.force_reset()
    metrics.track_external_breaker(_SERVICE, breaker)
    yield breaker
    breaker.force_reset()


class TestCircuitBreaker:
    """Every request passes the Trading 212 breaker, which refuses once open."""

    def test_consecutive_failures_wait_two_four_eight_sixteen_seconds(self, mock_get):
        """The breaker's windows space the attempts of one call."""
        mock_get.queue([requests.exceptions.ConnectionError("down")] * 5)
        c = _client_with_breaker(max_attempts=5)

        with pytest.raises(requests.exceptions.ConnectionError):
            c._get("/p")

        assert _gaps(mock_get.sent_at) == pytest.approx([2, 4, 8, 16])
        assert c.breaker.attempt_count == 5

    def test_after_n_failures_the_client_refuses_without_a_request(self, mock_get):
        """With ``max_attempts`` N, attempt N+1 raises ``CircuitOpenError`` unsent."""
        mock_get.queue([requests.exceptions.ConnectionError("down")] * 5)
        c = _client_with_breaker(max_attempts=2)

        with pytest.raises(CircuitOpenError):
            c._get("/p")
        assert mock_get.call_count == 2

        with pytest.raises(CircuitOpenError):
            c.get_instruments()
        assert mock_get.call_count == 2

    def test_an_open_process_breaker_reads_one_on_the_gauge(
        self, mock_get, monkeypatch, process_breaker_on_the_gauge
    ):
        """A client from settings trips the breaker the daemon's gauge reads."""
        monkeypatch.setattr(settings, "trading_212_api_key", "k")
        monkeypatch.setattr(settings, "trading_212_secret_key", "s")
        mock_get.queue([requests.exceptions.ConnectionError("down")] * 5)
        client = Trading212Client.from_settings()
        assert client is not None
        assert _gauge() == 0

        with pytest.raises(requests.exceptions.ConnectionError):
            client.get_instruments()
        with pytest.raises(CircuitOpenError):
            Trading212Client.from_settings().get_instruments()

        assert _gauge() == 1
        assert mock_get.call_count == process_breaker_on_the_gauge.max_attempts

    def test_a_success_decays_the_failure_count(self, mock_get):
        """A call that ends in an answer gives back one failure."""
        mock_get.queue(
            [requests.exceptions.ConnectionError("down"), _mock_response({"ok": 1})]
        )
        c = _client_with_breaker(max_attempts=5)

        c._get("/p")

        assert c.breaker.attempt_count == 0

    def test_a_call_inside_an_earlier_calls_window_waits_out_the_rest(self, mock_get):
        """Below ``max_attempts`` a new call first sleeps through the open window."""
        mock_get.queue(
            [_raise(_mock_429_error(retry_after="30")), _mock_response({"ok": True})]
        )
        c = _client_with_breaker(max_attempts=5, max_retries=1)

        with pytest.raises(requests.HTTPError):
            c._get("/p")
        assert c._get("/p") == {"ok": True}

        assert _gaps(mock_get.sent_at) == [pytest.approx(30)]
        assert c.breaker.attempt_count == 0

    def test_from_settings_shares_the_process_breaker(self, monkeypatch):
        """Every client the daemon builds trips the same breaker."""
        monkeypatch.setattr(settings, "trading_212_api_key", "k")
        monkeypatch.setattr(settings, "trading_212_secret_key", "s")

        first = Trading212Client.from_settings()
        second = Trading212Client.from_settings()

        assert first is not None
        assert second is not None
        assert first.breaker is second.breaker
        assert first.breaker is process_breaker(ExternalService.TRADING_212)

    def test_a_directly_built_client_gets_its_own_breaker(self, monkeypatch):
        """Outside ``from_settings`` the breaker is fresh but sized from settings."""
        monkeypatch.setattr(settings, "breaker_trading_212_max_attempts", 3)

        first = Trading212Client(api_key="k", api_secret="s")
        second = Trading212Client(api_key="k", api_secret="s")

        assert first.breaker is not second.breaker
        assert first.breaker.max_attempts == 3
        assert first.breaker.service_name == _SERVICE

    def test_each_attempt_is_timed_on_the_request_histogram(self, mock_get):
        """A failed attempt counts as ``error``, the answered one as ``ok``."""
        errors, oks = _request_count("error"), _request_count("ok")
        mock_get.queue([_raise(_mock_http_error(503)), _mock_response({"ok": True})])

        _make_client()._get("/p")

        assert _request_count("error") == errors + 1
        assert _request_count("ok") == oks + 1


def _request_count(outcome: str) -> float:
    labels = {"service": _SERVICE, "outcome": outcome}
    value = REGISTRY.get_sample_value("external_request_seconds_count", labels)
    return value or 0.0


# ---------------------------------------------------------------------------
# _fetch_json
# ---------------------------------------------------------------------------


class TestFetchJson:
    """``_fetch_json`` is a thin alias of ``_get``."""

    @patch.object(Trading212Client, "_get", return_value=[{"id": 1}])
    def test_delegates_to_get(self, mock_get):
        """It forwards the path and returns the body."""
        c = _make_client()
        result = c._fetch_json("/path")
        mock_get.assert_called_once_with("/path")
        assert result == [{"id": 1}]


# ---------------------------------------------------------------------------
# get_instruments — caching + rate-limit capture
# ---------------------------------------------------------------------------


class TestInstrumentCachingAndRateLimit:
    """The catalogue is fetched once per client; rate-limit headers are kept."""

    def test_second_call_serves_from_cache(self, mock_get):
        """The second read costs no request."""
        mock_get.return_value = _mock_response([{"ticker": "AAPL_US_EQ"}])
        c = _make_client()
        first = c.get_instruments()
        second = c.get_instruments()
        assert first == second == [{"ticker": "AAPL_US_EQ"}]
        assert mock_get.call_count == 1

    def test_force_refresh_bypasses_cache(self, mock_get):
        """``force_refresh`` fetches again."""
        mock_get.queue(
            [
                _mock_response([{"ticker": "AAPL_US_EQ"}]),
                _mock_response([{"ticker": "MSFT_US_EQ"}]),
            ]
        )
        c = _make_client()
        assert c.get_instruments() == [{"ticker": "AAPL_US_EQ"}]
        assert c.get_instruments(force_refresh=True) == [{"ticker": "MSFT_US_EQ"}]
        assert mock_get.call_count == 2

    def test_exchanges_are_not_cached(self, mock_get):
        """Exchanges are fetched on every call."""
        mock_get.return_value = _mock_response([{"id": 1, "name": "NASDAQ"}])
        c = _make_client()
        c.get_exchanges()
        c.get_exchanges()
        assert mock_get.call_count == 2

    def test_get_sends_browser_user_agent(self, mock_get):
        """The request carries the browser headers."""
        mock_get.return_value = _mock_response([])
        c = _make_client()
        c.get_instruments()
        headers = mock_get.call_args.kwargs["headers"]
        assert "Mozilla/5.0" in headers["User-Agent"]
        assert headers["Accept"] == "application/json"

    def test_captures_rate_limit_headers(self, mock_get):
        """Only the ``x-ratelimit-*`` headers are kept."""
        mock_get.return_value = _mock_response(
            [{"ticker": "AAPL_US_EQ"}],
            headers={
                "x-ratelimit-remaining": "0",
                "x-ratelimit-reset": "1700000000",
                "content-type": "application/json",
            },
        )
        c = _make_client()
        c.get_instruments()
        assert c.rate_limit == {
            "x-ratelimit-remaining": "0",
            "x-ratelimit-reset": "1700000000",
        }

    def test_rate_limit_defaults_empty_without_headers(self, mock_get):
        """No headers, nothing captured."""
        mock_get.return_value = _mock_response([{"ticker": "AAPL_US_EQ"}])
        c = _make_client()
        c.get_instruments()
        assert c.rate_limit == {}


# ---------------------------------------------------------------------------
# from_settings factory
# ---------------------------------------------------------------------------


class TestFromSettings:
    """``from_settings`` builds a client only when both credentials are set."""

    @patch("app.services.universe.trading212.client.settings")
    def test_returns_none_when_no_api_key(self, mock_settings):
        """No key, no client."""
        mock_settings.trading_212_api_key = ""
        assert Trading212Client.from_settings() is None

    @patch("app.services.universe.trading212.client.settings")
    def test_returns_none_when_no_secret_key(self, mock_settings):
        """No secret, no client."""
        mock_settings.trading_212_api_key = "key123"
        mock_settings.trading_212_secret_key = ""
        assert Trading212Client.from_settings() is None

    @patch("app.services.universe.trading212.client.settings")
    def test_returns_client_when_key_set(self, mock_settings):
        """Both set, a client with the key."""
        mock_settings.trading_212_api_key = "key123"
        mock_settings.trading_212_mode = "live"
        client = Trading212Client.from_settings()
        assert client is not None
        assert client.api_key == "key123"

    @patch("app.services.universe.trading212.client.settings")
    def test_mode_from_settings_when_no_override(self, mock_settings):
        """The mode comes from settings by default."""
        mock_settings.trading_212_api_key = "key123"
        mock_settings.trading_212_mode = "demo"
        client = Trading212Client.from_settings()
        assert client is not None
        assert client.base_url == "https://demo.trading212.com"

    @patch("app.services.universe.trading212.client.settings")
    def test_mode_override_takes_precedence(self, mock_settings):
        """An explicit mode wins."""
        mock_settings.trading_212_api_key = "key123"
        mock_settings.trading_212_mode = "live"
        client = Trading212Client.from_settings(mode="demo")
        assert client is not None
        assert client.base_url == "https://demo.trading212.com"


class TestMissingCredentials:
    """``missing_credentials`` names the unset variables."""

    @patch("app.services.universe.trading212.client.settings")
    def test_empty_when_both_set(self, mock_settings):
        """Nothing missing."""
        mock_settings.trading_212_api_key = "key123"
        mock_settings.trading_212_secret_key = "secret123"  # noqa: S105 - test value, not a secret
        assert Trading212Client.missing_credentials() == []

    @patch("app.services.universe.trading212.client.settings")
    def test_names_the_unset_variables(self, mock_settings):
        """Each unset variable is named, in order."""
        mock_settings.trading_212_api_key = ""
        mock_settings.trading_212_secret_key = "secret123"  # noqa: S105 - test value, not a secret
        assert Trading212Client.missing_credentials() == ["TRADING_212_API_KEY"]

        mock_settings.trading_212_api_key = "key123"
        mock_settings.trading_212_secret_key = ""
        assert Trading212Client.missing_credentials() == ["TRADING_212_SECRET_KEY"]

        mock_settings.trading_212_api_key = ""
        assert Trading212Client.missing_credentials() == [
            "TRADING_212_API_KEY",
            "TRADING_212_SECRET_KEY",
        ]


class TestRetryAfterWait:
    """A server-sent ``Retry-After`` must never stall the build or crash the sleep."""

    @pytest.mark.parametrize(
        ("header", "expected"),
        [
            ("86400", _MAX_RATE_LIMIT_WAIT_SECONDS),
            ("-5", 0),
            ("0", 0),
            ("abc", 0),
            ("Wed, 21 Oct 2015 07:28:00 GMT", 0),
            ("7", 7),
        ],
    )
    def test_retry_after_is_clamped_or_ignored(
        self, header: str, expected: float
    ) -> None:
        """Huge values are capped; unusable ones report no wait (0)."""
        waited = Trading212Client._rate_limit_wait({"Retry-After": header})

        assert waited == expected

    def test_headers_that_are_not_a_mapping_report_no_wait(self) -> None:
        """A response without readable headers sets no floor."""
        assert Trading212Client._rate_limit_wait(None) == 0.0
