"""Yahoo's "no data" answers end a request at once instead of being retried or logged.

Yahoo answers 404 per requested data module, not per symbol: ``0NMK.IL`` serves
its profile and institutional holders but 404s its EPS trend, and even a fully
covered European listing 404s SEC filings. yfinance logs such a 404 at ``ERROR``
and returns empty data, and the empty result used to cost two more attempts
with backoff sleeps. These tests pin the replacement contract: one request, a
quiet ``None``, and transient failures still retried.

Tickers are mocks; ``yf.Ticker`` is never constructed against the network.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import pandas as pd
import pytest
import requests
import yfinance as yf
from yfinance.data import YfData
from yfinance.exceptions import (
    YFDataException,
    YFEarningsDateMissing,
    YFPricesMissingError,
    YFRateLimitError,
)

from app.services.infrastructure.cache import LRUCache
from app.services.infrastructure.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
)
from app.services.market_data.yfinance import configure_yfinance
from app.services.market_data.yfinance._facade import YFinanceClient
from app.services.market_data.yfinance.infrastructure import (
    FaultWatch,
    YahooTransientError,
    install_fault_witness,
    is_no_data_error,
    is_symbol_not_found,
    is_transport_error,
    track_request_outcomes,
    watch_for_faults,
)
from app.services.market_data.yfinance.ticker import (
    AnalysisClient,
    FinancialsClient,
    HoldersClient,
    MetadataClient,
)


@pytest.fixture(autouse=True)
def _no_backoff_sleep() -> Iterator[None]:
    with patch("app.services.infrastructure.retry._full_jitter", return_value=0.0):
        yield


def _http_error(status: int) -> requests.exceptions.HTTPError:
    response = requests.Response()
    response.status_code = status
    return requests.exceptions.HTTPError(f"HTTP Error {status}", response=response)


def _ticker_with(**props: PropertyMock) -> MagicMock:
    ticker = MagicMock()
    for attr, prop in props.items():
        setattr(type(ticker), attr, prop)
    return ticker


def _sub_client(cls: type, ticker: MagicMock):
    cache = MagicMock()
    cache.get.return_value = ticker
    return cls(
        cache=cache,
        rate_limiter=MagicMock(),
        circuit_breaker=MagicMock(),
        default_max_retries=3,
    )


class TestClassification:
    """Which exceptions are Yahoo's final answer and which are worth a retry."""

    def test_http_404_means_not_found(self) -> None:
        """quoteSummary answers 404 when the requested module has nothing."""
        assert is_symbol_not_found(_http_error(404))

    @pytest.mark.parametrize("status", [429, 500, 503])
    def test_other_http_statuses_are_not_not_found(self, status: int) -> None:
        """Throttling and server faults say nothing about the data itself."""
        assert not is_symbol_not_found(_http_error(status))

    def test_errors_without_a_response_are_not_not_found(self) -> None:
        """A bare exception carries no HTTP status to judge by."""
        assert not is_symbol_not_found(RuntimeError("boom"))

    @pytest.mark.parametrize(
        "error",
        [
            _http_error(404),
            YFPricesMissingError("ZZZZ", ""),
            YFEarningsDateMissing("ZZZZ"),
            YFDataException("No upgrade/downgrade history found for ZZZZ"),
        ],
    )
    def test_definitive_answers_are_no_data(self, error: Exception) -> None:
        """Repeating these requests returns the same empty answer."""
        assert is_no_data_error(error)

    @pytest.mark.parametrize(
        "error",
        [
            _http_error(429),
            _http_error(503),
            YFRateLimitError(),
            ConnectionError("reset"),
            TimeoutError(),
        ],
    )
    def test_transient_failures_are_not_no_data(self, error: Exception) -> None:
        """These may succeed on a later attempt, so they stay retryable."""
        assert not is_no_data_error(error)


class TestTransportClassification:
    """Failures to reach Yahoo, as opposed to what Yahoo says once reached."""

    @pytest.mark.parametrize(
        "error",
        [
            ConnectionError("reset"),
            TimeoutError(),
            YFRateLimitError(),
            requests.exceptions.ConnectionError("network down"),
            requests.exceptions.ReadTimeout("slow"),
            _http_error(429),
            _http_error(503),
            CircuitOpenError("Yahoo rate limit persists after 10 attempts"),
        ],
    )
    def test_unreachable_or_refusing_yahoo_is_a_transport_error(
        self, error: Exception
    ) -> None:
        assert is_transport_error(error)

    @pytest.mark.parametrize(
        "error",
        [
            _http_error(404),
            YFPricesMissingError("ZZZZ", ""),
            YFDataException("No upgrade/downgrade history found for ZZZZ"),
            KeyError("quoteSummary"),
            ValueError("unexpected payload"),
            requests.exceptions.JSONDecodeError("bad json", "<html>", 0),
        ],
    )
    def test_what_yahoo_answers_is_not_a_transport_error(
        self, error: Exception
    ) -> None:
        assert not is_transport_error(error)

    def test_a_curl_cffi_failure_is_a_transport_error(self) -> None:
        """yfinance 1.x talks through curl_cffi, whose errors share no base with
        ``requests``; it is recognised by origin since the package may not load."""
        error_type = type(
            "ConnectionError", (Exception,), {"__module__": "curl_cffi.requests"}
        )

        assert is_transport_error(error_type("Failed to perform"))


class TestRequestOutcomes:
    """Every yfinance request leaves a trace in the active tracking scope, so a
    caller can tell an unreachable Yahoo from one that has nothing to say."""

    def test_exhausted_network_retries_are_recorded_as_a_failure(
        self, client: YFinanceClient
    ) -> None:
        info = PropertyMock(side_effect=ConnectionError("network down"))
        with patch.object(client, "get_ticker", return_value=_ticker_with(info=info)):
            with track_request_outcomes() as outcomes:
                assert client.fetch_info("AAPL") is None

        assert info.call_count == 3
        assert outcomes.answered == 0
        assert outcomes.failures == ["ConnectionError: network down"]

    def test_a_retry_that_succeeds_is_an_answer_not_a_failure(
        self, client: YFinanceClient
    ) -> None:
        prop = PropertyMock(
            side_effect=[ConnectionError("reset"), pd.DataFrame({"holder": ["x"]})]
        )
        holders = _sub_client(HoldersClient, _ticker_with(major_holders=prop))

        with track_request_outcomes() as outcomes:
            assert holders.fetch_major_holders("AAPL") is not None

        assert outcomes.answered == 1
        assert outcomes.failures == []

    @pytest.mark.parametrize(
        "side_effect",
        [
            _http_error(404),
            YFDataException("No upgrade/downgrade history found for AAPL"),
            [pd.DataFrame()],
        ],
    )
    def test_yahoos_no_data_answers_count_as_answered(self, side_effect: Any) -> None:
        prop = PropertyMock(side_effect=side_effect)
        holders = _sub_client(HoldersClient, _ticker_with(major_holders=prop))

        with track_request_outcomes() as outcomes:
            assert holders.fetch_major_holders("AAPL") is None

        assert outcomes.answered == 1
        assert outcomes.failures == []

    def test_an_unparseable_payload_is_an_answer_not_a_transport_failure(
        self,
    ) -> None:
        prop = PropertyMock(side_effect=KeyError("quoteSummary"))
        holders = _sub_client(HoldersClient, _ticker_with(major_holders=prop))

        with track_request_outcomes() as outcomes:
            assert holders.fetch_major_holders("AAPL") is None

        assert outcomes.answered == 1
        assert outcomes.failures == []

    def test_an_open_circuit_breaker_is_recorded_as_a_failure(self) -> None:
        """Once rate limits latch the breaker no request reaches Yahoo at all."""
        breaker = CircuitBreaker(service_name="Yahoo", max_attempts=0)
        cache = MagicMock()
        cache.get.return_value = None
        yf_client = YFinanceClient(
            cache=cache,
            rate_limiter=MagicMock(),
            circuit_breaker=breaker,
            default_max_retries=2,
        )

        with track_request_outcomes() as outcomes:
            assert yf_client.fetch_info("AAPL") is None

        assert outcomes.answered == 0
        assert len(outcomes.failures) == 1
        assert "rate limit persists" in outcomes.failures[0]

    def test_requests_outside_a_tracking_scope_are_not_recorded(
        self, client: YFinanceClient
    ) -> None:
        info = PropertyMock(side_effect=ConnectionError("network down"))
        with patch.object(client, "get_ticker", return_value=_ticker_with(info=info)):
            assert client.fetch_info("AAPL") is None

        with track_request_outcomes() as outcomes:
            pass
        assert outcomes.answered == 0
        assert outcomes.failures == []


@pytest.fixture
def _restore_yfinance_config() -> Iterator[None]:
    hide = yf.config.debug.hide_exceptions
    level = logging.getLogger("yfinance").level
    yield
    yf.config.debug.hide_exceptions = hide
    logging.getLogger("yfinance").setLevel(level)


@pytest.mark.usefixtures("_restore_yfinance_config")
class TestConfigureYfinance:
    """The process-wide yfinance switches the daemon and CLI rely on."""

    def test_yfinance_error_handling_mode_is_left_alone(self) -> None:
        """Only yfinance's default mode salvages partial data, e.g. growth estimates.

        With exceptions switched on, a 404 on a symbol's own EPS trend aborts
        ``growth_estimates`` instead of returning the industry/sector rows.
        """
        yf.config.debug.hide_exceptions = True
        configure_yfinance(log_level="CRITICAL")
        assert yf.config.debug.hide_exceptions is True

    def test_library_errors_are_hidden_at_the_default_level(self) -> None:
        """yfinance's own ERROR lines never reach an operator by default."""
        configure_yfinance(log_level="CRITICAL")
        assert not logging.getLogger("yfinance").isEnabledFor(logging.ERROR)

    def test_debug_level_restores_library_diagnostics(self) -> None:
        """Troubleshooting can opt back in, and the level name is case-insensitive."""
        configure_yfinance(log_level="debug")
        assert logging.getLogger("yfinance").isEnabledFor(logging.DEBUG)

    def test_unknown_level_is_rejected(self) -> None:
        """A typo in YFINANCE_LOG_LEVEL fails at startup, not silently."""
        with pytest.raises(ValueError, match="Unknown log level"):
            configure_yfinance(log_level="LOUD")


class TestSubClientCalls:
    """Sub-client calls retry transient failures only."""

    def test_empty_answer_is_returned_without_retrying(self) -> None:
        """yfinance turns a 404 into an empty frame; asking again returns the same."""
        prop = PropertyMock(return_value=pd.DataFrame())
        client = _sub_client(HoldersClient, _ticker_with(major_holders=prop))

        assert client.fetch_major_holders("AAPL") is None
        assert prop.call_count == 1

    def test_404_returns_none_after_one_request(self) -> None:
        """The module has nothing for the symbol; asking again cannot change that."""
        prop = PropertyMock(side_effect=_http_error(404))
        client = _sub_client(HoldersClient, _ticker_with(insider_transactions=prop))

        assert client.fetch_insider_transactions("0NMK.IL") is None
        assert prop.call_count == 1

    def test_a_404_on_one_category_leaves_the_others_untouched(self) -> None:
        """0NMK.IL 404s its EPS trend yet still serves institutional holders."""
        eps = PropertyMock(side_effect=_http_error(404))
        holders = PropertyMock(return_value=pd.DataFrame({"Holder": ["x"]}))
        ticker = _ticker_with(eps_trend=eps, institutional_holders=holders)
        analysis = _sub_client(AnalysisClient, ticker)
        holders_client = _sub_client(HoldersClient, ticker)

        assert analysis.fetch_eps_trend("0NMK.IL") is None
        assert holders_client.fetch_institutional_holders("0NMK.IL") is not None

    def test_an_empty_category_error_is_not_retried(self) -> None:
        """Yahoo raises this for a covered symbol that simply has no rating history."""
        prop = PropertyMock(
            side_effect=YFDataException("No upgrade/downgrade history found for AAPL")
        )
        client = _sub_client(AnalysisClient, _ticker_with(upgrades_downgrades=prop))

        assert client.fetch_upgrades_downgrades("AAPL") is None
        assert prop.call_count == 1

    def test_transient_errors_are_still_retried(self) -> None:
        """A dropped connection gets the normal retry budget."""
        prop = PropertyMock(
            side_effect=[ConnectionError("reset"), pd.DataFrame({"holder": ["x"]})]
        )
        client = _sub_client(HoldersClient, _ticker_with(major_holders=prop))

        result = client.fetch_major_holders("AAPL")

        assert result is not None
        assert prop.call_count == 2


@pytest.fixture
def client() -> YFinanceClient:
    """Return a facade with real retry logic and mocked cache, limiter and breaker."""
    cache = MagicMock()
    cache.get.return_value = None
    return YFinanceClient(
        cache=cache,
        rate_limiter=MagicMock(),
        circuit_breaker=MagicMock(),
        default_max_retries=3,
    )


class TestFacadeInfo:
    """``fetch_info`` stops on Yahoo's final answers and retries the rest."""

    def test_info_404_returns_none_after_one_request(
        self, client: YFinanceClient
    ) -> None:
        """A missing profile is final for this run."""
        info = PropertyMock(side_effect=_http_error(404))
        with patch.object(client, "get_ticker", return_value=_ticker_with(info=info)):
            assert client.fetch_info("ZZZZ") is None

        assert info.call_count == 1

    def test_a_stub_info_dict_is_dropped_without_retrying(
        self, client: YFinanceClient
    ) -> None:
        """A thin dict from a successful request is not stored and not re-asked."""
        info = PropertyMock(return_value={"symbol": "ZZZZ"})
        with patch.object(client, "get_ticker", return_value=_ticker_with(info=info)):
            assert client.fetch_info("ZZZZ", min_fields=10) is None

        assert info.call_count == 1


class TestFacadeHistory:
    """Price history keeps transient retries but drops retries on final answers."""

    def test_missing_prices_end_the_fetch_without_retrying(
        self, client: YFinanceClient
    ) -> None:
        """No chart data for the symbol is final for this run."""
        ticker = MagicMock()
        ticker.history.side_effect = YFPricesMissingError("ZZZZ", "")
        with patch.object(client, "get_ticker", return_value=ticker):
            assert client.fetch_history("ZZZZ") is None

        assert ticker.history.call_count == 1

    def test_empty_history_is_not_retried(self, client: YFinanceClient) -> None:
        """An empty frame from a successful request is Yahoo's answer."""
        ticker = MagicMock()
        ticker.history.return_value = pd.DataFrame()
        with patch.object(client, "get_ticker", return_value=ticker):
            assert client.fetch_history("AAPL") is None

        assert ticker.history.call_count == 1

    def test_transient_errors_are_still_retried(self, client: YFinanceClient) -> None:
        """A dropped connection gets the normal retry budget."""
        ticker = MagicMock()
        rows = pd.DataFrame(
            {"Close": range(1, 21)}, index=pd.date_range("2024-01-01", periods=20)
        )
        ticker.history.side_effect = [ConnectionError("reset"), rows]
        with patch.object(client, "get_ticker", return_value=ticker):
            out = client.fetch_history("AAPL")

        assert out is not None and len(out) == 20
        assert ticker.history.call_count == 2


class TestSettingsAndBoot:
    """Every entry point applies the yfinance switches before fetching."""

    def test_yfinance_library_logs_are_hidden_by_default(self) -> None:
        """Operators see our classified messages, not the library's raw output."""
        from app.config import Settings

        assert Settings.model_fields["yfinance_log_level"].default == "CRITICAL"

    @pytest.mark.usefixtures("_restore_yfinance_config")
    def test_cli_boot_configures_yfinance(self) -> None:
        """Manual CLI runs such as ``refetch-all`` get the same behaviour as the daemon."""
        from app import cli

        with (
            patch.object(cli, "init_db"),
            patch("app.services.market_data.yfinance.configure_yfinance") as cfg,
        ):
            cli._boot()

        cfg.assert_called_once_with(log_level=cli.settings.yfinance_log_level)

    @pytest.mark.usefixtures("_restore_yfinance_config")
    def test_worker_logging_setup_configures_yfinance(self) -> None:
        """The scheduler daemon applies the switches once at startup."""
        from app import worker

        with patch("app.services.market_data.yfinance.configure_yfinance") as cfg:
            worker._configure_logging()

        cfg.assert_called_once_with(log_level=worker.settings.yfinance_log_level)


_REVENUE_PAYLOAD = {
    "timeseries": {
        "result": [
            {
                "meta": {"symbol": ["AAPL"], "type": ["annualTotalRevenue"]},
                "timestamp": [1735603200],
                "annualTotalRevenue": [
                    {"asOfDate": "2024-12-31", "reportedValue": {"raw": 100.0}}
                ],
            }
        ]
    }
}


class _Response:
    def __init__(self, status_code: int, body: dict | None = None) -> None:
        self.status_code = status_code
        self.text = json.dumps(body if body is not None else {})
        self.url = "https://query2.finance.yahoo.com/ws/fundamentals-timeseries"


class _FakeYahoo:
    """Scriptable stand-in for yfinance's HTTP session.

    ``handler`` maps each request to a response or raises, so a test can make
    Yahoo throttle, fail or answer empty without any network access.
    """

    proxies = None

    def __init__(self) -> None:
        self.requests = 0
        self.handler: Callable[[], _Response] = lambda: _Response(200, {})

    def get(self, **_kwargs: object) -> _Response:
        self.requests += 1
        return self.handler()


@pytest.fixture
def yahoo(monkeypatch: pytest.MonkeyPatch) -> Iterator[_FakeYahoo]:
    """Run the real yfinance request path against a scripted session."""
    fake = _FakeYahoo()
    data = YfData()
    monkeypatch.setattr(data, "_session", fake)
    monkeypatch.setattr(data, "_cookie_strategy", data._cookie_strategy)
    monkeypatch.setattr(data, "fundamentals_use_chunked", False)
    monkeypatch.setattr(
        YfData, "_get_cookie_and_crumb", lambda _self, _timeout=30: (None, "basic")
    )
    YfData.cache_get.cache_clear()
    yield fake
    YfData.cache_get.cache_clear()


def _financials(breaker: MagicMock | None = None, retries: int = 3) -> FinancialsClient:
    return FinancialsClient(
        cache=LRUCache(),
        rate_limiter=MagicMock(),
        circuit_breaker=breaker or MagicMock(),
        default_max_retries=retries,
    )


def _metadata(breaker: MagicMock | None = None, retries: int = 3) -> MetadataClient:
    return MetadataClient(
        cache=LRUCache(),
        rate_limiter=MagicMock(),
        circuit_breaker=breaker or MagicMock(),
        default_max_retries=retries,
    )


class TestSwallowedTransientFaults:
    """yfinance folds throttling and server faults into empty frames.

    In its default mode the scrapers catch the failure, return an empty result
    and cache it on the ``yf.Ticker``, so an empty answer alone cannot be told
    from Yahoo's "no data". The sub-client must still see the fault.
    """

    def test_a_swallowed_rate_limit_trips_the_breaker_and_is_retried(
        self, yahoo: _FakeYahoo
    ) -> None:
        """Throttling must back the sweep off instead of reading as no data."""
        yahoo.handler = lambda: _Response(429)
        breaker = MagicMock()

        assert _financials(breaker).fetch_income_stmt("AAPL") is None

        assert breaker.trigger.call_count == 3

    def test_an_empty_result_after_a_server_error_recovers_on_retry(
        self, yahoo: _FakeYahoo
    ) -> None:
        """The poisoned empty frame and cached 503 must not outlive the retry."""
        yahoo.handler = lambda: _Response(503)

        def recover(*_args: float) -> float:
            yahoo.handler = lambda: _Response(200, _REVENUE_PAYLOAD)
            return 0.0

        with patch("app.services.infrastructure.retry._full_jitter", recover):
            frame = _financials().fetch_income_stmt("AAPL")

        assert frame is not None
        assert frame.loc["Total Revenue"].iloc[0] == 100.0

    def test_a_genuine_no_data_answer_is_still_final(self, yahoo: _FakeYahoo) -> None:
        """A 404 is Yahoo's answer; asking again must not cost another request."""
        yahoo.handler = lambda: _Response(404)
        single_attempt = _financials(retries=1)
        assert single_attempt.fetch_income_stmt("AAPL") is None
        requests_for_one_attempt = yahoo.requests

        YfData.cache_get.cache_clear()
        yahoo.requests = 0
        breaker = MagicMock()

        assert _financials(breaker, retries=3).fetch_income_stmt("AAPL") is None

        assert yahoo.requests == requests_for_one_attempt
        breaker.trigger.assert_not_called()

    def test_usable_data_is_kept_when_a_side_request_failed(
        self, yahoo: _FakeYahoo
    ) -> None:
        """yfinance falls back to chunked requests; a recovered fault is not an error."""
        pending_failures = [TimeoutError("timed out")]

        def handler() -> _Response:
            if pending_failures:
                raise pending_failures.pop()
            return _Response(200, _REVENUE_PAYLOAD)

        yahoo.handler = handler
        breaker = MagicMock()

        frame = _financials(breaker).fetch_income_stmt("AAPL")

        assert frame is not None
        assert frame.loc["Total Revenue"].iloc[0] == 100.0
        breaker.trigger.assert_not_called()

    def test_a_rate_limit_raised_out_of_the_scraper_does_not_poison_the_ticker(
        self, yahoo: _FakeYahoo
    ) -> None:
        """``Quote.info`` marks itself fetched before it asks Yahoo, so a cached
        ticker whose first fetch raised answers every retry from empty state."""
        yahoo.handler = lambda: _Response(429)
        breaker = MagicMock()

        assert _metadata(breaker).fetch_isin("AAPL") is None

        assert breaker.trigger.call_count == 3

    def test_a_rate_limit_outranks_an_earlier_server_error(self) -> None:
        """The breaker only trips on a rate limit, so that fault must be kept."""
        watch = FaultWatch()

        watch.record(YahooTransientError(503))
        watch.record(YFRateLimitError())
        watch.record(YahooTransientError(500))

        assert isinstance(watch.fault, YFRateLimitError)

    def test_installing_the_witness_twice_wraps_requests_once(self) -> None:
        """Every sub-client installs it, so repeats must not stack wrappers."""
        install_fault_witness()
        first = YfData._make_request

        install_fault_witness()

        assert YfData._make_request is first

    def test_faults_are_recorded_per_thread(self, yahoo: _FakeYahoo) -> None:
        """A concurrent worker's failed request is not this call's fault."""
        probe = "https://query2.finance.yahoo.com/probe"
        yahoo.handler = lambda: _Response(503)
        # Installed here, not left to an earlier test: without the wrapper no
        # watch could ever record a fault and the assertions below would hold
        # for any storage, thread-local or not.
        install_fault_witness()

        with watch_for_faults() as watch:
            worker = threading.Thread(target=lambda: YfData().get(probe))
            worker.start()
            worker.join()

        assert watch.fault is None
        with watch_for_faults() as own:
            YfData().get(probe)
        assert isinstance(own.fault, YahooTransientError)
