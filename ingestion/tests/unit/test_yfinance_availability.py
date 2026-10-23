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

import logging
from collections.abc import Iterator
from unittest.mock import MagicMock, PropertyMock, patch

import pandas as pd
import pytest
import requests
import yfinance as yf
from yfinance.exceptions import (
    YFDataException,
    YFEarningsDateMissing,
    YFPricesMissingError,
    YFRateLimitError,
)

from app.services.market_data.yfinance import configure_yfinance
from app.services.market_data.yfinance._facade import YFinanceClient
from app.services.market_data.yfinance.infrastructure import (
    is_no_data_error,
    is_symbol_not_found,
)
from app.services.market_data.yfinance.ticker import AnalysisClient, HoldersClient


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
