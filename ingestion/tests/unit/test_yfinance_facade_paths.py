"""Branch coverage for ``YFinanceClient`` — the fetch path under every ingestion step.

The daily pipeline funnels thousands of tickers through this class, so the
behaviours that matter are the degradations, not the happy path: a rate-limited
call must trip the circuit breaker rather than hammer Yahoo, a ticker with too
few fields must come back ``None`` rather than a half-populated dict, and a
failed bulk download must fall back to per-ticker fetches rather than losing the
whole batch.

``yf.Ticker`` and ``yf.download`` are patched — no network.
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yfinance as yf
from yfinance.base import TickerBase
from yfinance.scrapers.history import PriceHistory

from app.services.infrastructure.cache import LRUCache
from app.services.market_data.yfinance._base import TICKER_REQUEST_KEY
from app.services.market_data.yfinance._facade import YFinanceClient
from app.services.market_data.yfinance.infrastructure import RateLimiter
from app.services.market_data.yfinance.ticker import HoldersClient

M = "app.services.market_data.yfinance._facade"


@pytest.fixture
def client() -> YFinanceClient:
    """A client with real retry logic but mocked cache / limiter / breaker."""
    cache = MagicMock()
    cache.get.return_value = None
    return YFinanceClient(
        cache=cache,
        rate_limiter=MagicMock(),
        circuit_breaker=MagicMock(),
        default_max_retries=1,
    )


def _hist(rows: int) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="D")
    return pd.DataFrame({"Close": range(1, rows + 1)}, index=idx)


class TestGetTicker:
    def test_cache_hit_still_acquires_the_limiter(self, client: YFinanceClient) -> None:
        """Every request starts here, and a cached Ticker still sends requests."""
        cached = MagicMock()
        client.cache.get.return_value = cached

        assert client.get_ticker("AAPL") is cached
        client.rate_limiter.acquire.assert_called_once_with(TICKER_REQUEST_KEY)

    def test_cache_miss_acquires_the_limiter_and_caches(
        self, client: YFinanceClient
    ) -> None:
        with patch(f"{M}.yf.Ticker") as tk:
            result = client.get_ticker("AAPL")

        client.rate_limiter.acquire.assert_called_once_with(TICKER_REQUEST_KEY)
        client.cache.put.assert_called_once_with("AAPL", tk.return_value)
        assert result is tk.return_value


class TestRequestPacing:
    """The per-symbol delay applies to every request, not once per cached Ticker."""

    @staticmethod
    def _real_limiter_client() -> YFinanceClient:
        return YFinanceClient(
            cache=LRUCache(capacity=10, default_ttl=3600.0),
            rate_limiter=RateLimiter(delay=0.1),
            circuit_breaker=MagicMock(),
            default_max_retries=1,
        )

    def test_back_to_back_history_requests_for_one_symbol_are_spaced(self) -> None:
        client = self._real_limiter_client()
        ticker = MagicMock()
        ticker.history.return_value = _hist(20)

        with (
            patch(f"{M}.yf.Ticker", return_value=ticker),
            patch("app.services.infrastructure.rate_limiter.time.sleep") as nap,
        ):
            client.fetch_history("AAPL")
            client.fetch_history("AAPL")

        nap.assert_called_once()

    def test_back_to_back_sub_client_requests_for_one_symbol_are_spaced(self) -> None:
        client = HoldersClient(
            cache=LRUCache(capacity=10, default_ttl=3600.0),
            rate_limiter=RateLimiter(delay=0.1),
            circuit_breaker=MagicMock(),
            default_max_retries=1,
        )
        ticker = MagicMock()
        ticker.major_holders = pd.DataFrame({"holder": ["x"]})
        ticker.institutional_holders = pd.DataFrame({"holder": ["y"]})

        with (
            patch(
                "app.services.market_data.yfinance._base.yf.Ticker", lambda s: ticker
            ),
            patch("app.services.infrastructure.rate_limiter.time.sleep") as nap,
        ):
            client.fetch_major_holders("AAPL")
            client.fetch_institutional_holders("AAPL")

        nap.assert_called_once()

    def test_back_to_back_requests_for_different_symbols_are_spaced(self) -> None:
        """Parallel workers fetch distinct symbols, so the delay must be shared."""
        client = self._real_limiter_client()
        ticker = MagicMock()
        ticker.history.return_value = _hist(20)

        with (
            patch(f"{M}.yf.Ticker", return_value=ticker),
            patch("app.services.infrastructure.rate_limiter.time.sleep") as nap,
        ):
            client.fetch_history("AAPL")
            client.fetch_history("MSFT")

        nap.assert_called_once()

    def test_back_to_back_sub_client_requests_for_different_symbols_are_spaced(
        self,
    ) -> None:
        client = HoldersClient(
            cache=LRUCache(capacity=10, default_ttl=3600.0),
            rate_limiter=RateLimiter(delay=0.1),
            circuit_breaker=MagicMock(),
            default_max_retries=1,
        )
        ticker = MagicMock()
        ticker.major_holders = pd.DataFrame({"holder": ["x"]})

        with (
            patch(
                "app.services.market_data.yfinance._base.yf.Ticker", lambda s: ticker
            ),
            patch("app.services.infrastructure.rate_limiter.time.sleep") as nap,
        ):
            client.fetch_major_holders("AAPL")
            client.fetch_major_holders("MSFT")

        nap.assert_called_once()


class TestFetchInfo:
    def test_returns_info_when_field_count_is_sufficient(
        self, client: YFinanceClient
    ) -> None:
        info = {f"k{i}": i for i in range(12)}
        ticker = MagicMock()
        ticker.info = info

        with patch.object(client, "get_ticker", return_value=ticker):
            assert client.fetch_info("AAPL", min_fields=10) == info

        client.circuit_breaker.reset.assert_called()

    def test_too_few_fields_is_treated_as_a_failed_fetch(
        self, client: YFinanceClient
    ) -> None:
        """Yahoo returns a stub dict for delisted tickers; it must not be stored."""
        ticker = MagicMock()
        ticker.info = {"symbol": "ZZZZ"}

        with patch.object(client, "get_ticker", return_value=ticker):
            assert client.fetch_info("ZZZZ", min_fields=10) is None

    def test_rate_limit_error_trips_the_circuit_breaker(
        self, client: YFinanceClient
    ) -> None:
        with patch.object(
            client, "get_ticker", side_effect=RuntimeError("429 Too Many Requests")
        ):
            assert client.fetch_info("AAPL") is None

        client.circuit_breaker.trigger.assert_called()


class _MemoizingTicker:
    """Mimics yfinance's ``Ticker.info``: the fetched flag is set before the request.

    A request that dies leaves the object answering ``None`` (or ``{}`` when
    the error was swallowed) without another request until it is discarded.
    """

    requests = 0
    built = 0
    outcomes: list[Any] = []

    def __init__(self, symbol: str) -> None:
        type(self).built += 1
        self._fetched = False
        self._info: Any = None

    @property
    def info(self) -> Any:
        if not self._fetched:
            self._fetched = True
            outcome = type(self).outcomes[type(self).requests]
            type(self).requests += 1
            if isinstance(outcome, Exception):
                raise outcome
            self._info = outcome
        return self._info


@pytest.fixture
def memoizing_yahoo():
    """Patch ``yf.Ticker`` with a ticker that memoizes its first answer."""

    class _Ticker(_MemoizingTicker):
        requests = 0
        built = 0
        outcomes: list[Any] = []

    with patch(f"{M}.yf.Ticker", _Ticker):
        yield _Ticker


_FULL_INFO = {f"k{i}": i for i in range(12)}


class TestCachedTickerMemoization:
    """yfinance memoizes a failed ``.info``; the cache must not keep that Ticker."""

    @staticmethod
    def _real_cache_client(retries: int) -> YFinanceClient:
        return YFinanceClient(
            cache=LRUCache(capacity=10, default_ttl=3600.0),
            rate_limiter=MagicMock(),
            circuit_breaker=MagicMock(),
            default_max_retries=retries,
        )

    def test_a_retry_after_a_transient_error_makes_a_real_request(
        self, memoizing_yahoo: type[_MemoizingTicker]
    ) -> None:
        memoizing_yahoo.outcomes = [ConnectionError("reset"), _FULL_INFO]
        client = self._real_cache_client(retries=3)

        assert client.fetch_info("TEST", min_fields=10) == _FULL_INFO
        assert memoizing_yahoo.requests == 2

    def test_a_symbol_is_fetchable_again_after_its_retries_were_exhausted(
        self, memoizing_yahoo: type[_MemoizingTicker]
    ) -> None:
        """One transient fault must not hide the profile for the cache TTL."""
        memoizing_yahoo.outcomes = [ConnectionError("reset"), _FULL_INFO]
        client = self._real_cache_client(retries=1)

        assert client.fetch_info("TEST", min_fields=10) is None

        assert client.fetch_info("TEST", min_fields=10) == _FULL_INFO

    def test_an_empty_info_from_a_swallowed_error_is_not_kept(
        self, memoizing_yahoo: type[_MemoizingTicker]
    ) -> None:
        """yfinance leaves ``{}`` behind when it swallows an HTTP error."""
        memoizing_yahoo.outcomes = [{}, _FULL_INFO]
        client = self._real_cache_client(retries=1)

        assert client.fetch_info("TEST", min_fields=10) is None
        assert client.fetch_info("TEST", min_fields=10) == _FULL_INFO

    def test_a_good_profile_stays_cached_between_calls(
        self, memoizing_yahoo: type[_MemoizingTicker]
    ) -> None:
        memoizing_yahoo.outcomes = [_FULL_INFO]
        client = self._real_cache_client(retries=3)

        client.fetch_info("TEST", min_fields=10)
        client.fetch_info("TEST", min_fields=10)

        assert memoizing_yahoo.built == 1


class TestFetchHistoryTimeout:
    """The request timeout yfinance applies to a history call that sets none."""

    @staticmethod
    def _timeout_reaching_yfinance(client: YFinanceClient, **kwargs: Any) -> Any:
        """Return the ``timeout`` that yfinance's own history signature resolves to."""
        history_signature = inspect.signature(PriceHistory.history)
        seen: dict[str, Any] = {}

        def _spy(self: PriceHistory, *args: Any, **call_kwargs: Any) -> pd.DataFrame:
            bound = history_signature.bind(self, *args, **call_kwargs)
            bound.apply_defaults()
            seen["timeout"] = bound.arguments["timeout"]
            return _hist(20)

        with (
            patch.object(PriceHistory, "history", _spy),
            patch.object(TickerBase, "_get_ticker_tz", return_value="UTC"),
            patch.object(client, "get_ticker", return_value=yf.Ticker("TEST")),
        ):
            client.fetch_history("TEST", **kwargs)
        return seen["timeout"]

    def test_default_call_keeps_yfinances_finite_timeout(
        self, client: YFinanceClient
    ) -> None:
        """An explicit None would disable it, and the call could then hang forever."""
        timeout = self._timeout_reaching_yfinance(client)

        assert timeout is not None and timeout > 0

    def test_an_explicit_timeout_is_forwarded(self, client: YFinanceClient) -> None:
        assert self._timeout_reaching_yfinance(client, timeout=12.5) == 12.5


class TestFetchHistory:
    def test_returns_history_when_row_count_is_sufficient(
        self, client: YFinanceClient
    ) -> None:
        ticker = MagicMock()
        ticker.history.return_value = _hist(20)

        with patch.object(client, "get_ticker", return_value=ticker):
            out = client.fetch_history("AAPL", min_rows=10)

        assert out is not None and len(out) == 20

    def test_too_few_rows_is_treated_as_a_failed_fetch(
        self, client: YFinanceClient
    ) -> None:
        ticker = MagicMock()
        ticker.history.return_value = _hist(2)

        with patch.object(client, "get_ticker", return_value=ticker):
            assert client.fetch_history("AAPL", min_rows=10) is None

    def test_empty_history_is_a_failed_fetch(self, client: YFinanceClient) -> None:
        ticker = MagicMock()
        ticker.history.return_value = pd.DataFrame()

        with patch.object(client, "get_ticker", return_value=ticker):
            assert client.fetch_history("AAPL") is None

    def test_start_takes_precedence_over_period(self, client: YFinanceClient) -> None:
        """Incremental fetches pass start=; passing period= too would refetch history."""
        ticker = MagicMock()
        ticker.history.return_value = _hist(20)

        with patch.object(client, "get_ticker", return_value=ticker):
            client.fetch_history("AAPL", start="2024-01-01", end="2024-02-01")

        kwargs = ticker.history.call_args.kwargs
        assert kwargs["start"] == "2024-01-01"
        assert "period" not in kwargs

    def test_period_is_used_when_start_is_absent(self, client: YFinanceClient) -> None:
        ticker = MagicMock()
        ticker.history.return_value = _hist(20)

        with patch.object(client, "get_ticker", return_value=ticker):
            client.fetch_history("AAPL", period="1y")

        assert ticker.history.call_args.kwargs["period"] == "1y"

    def test_end_alone_bounds_the_period_window(self, client: YFinanceClient) -> None:
        """A backfill bounded by end= must not silently return the latest window."""
        ticker = MagicMock()
        ticker.history.return_value = _hist(20)

        with patch.object(client, "get_ticker", return_value=ticker):
            client.fetch_history("AAPL", period="1y", end="2020-01-01")

        kwargs = ticker.history.call_args.kwargs
        assert kwargs["end"] == "2020-01-01"
        assert kwargs["period"] == "1y"
        assert "start" not in kwargs

    def test_repair_is_enabled_by_default(self, client: YFinanceClient) -> None:
        """Price repair fixes bad Yahoo prints; it must stay on by default.
        It requires scikit-learn (yfinance uses DBSCAN) — with repair on and
        sklearn missing, any ticker yfinance tries to reconstruct raises
        ModuleNotFoundError -> empty result -> the universe filter silently drops
        a live stock (~22% of names). The dependency itself is guarded by
        test_price_history_repair_dependency.py."""
        ticker = MagicMock()
        ticker.history.return_value = _hist(20)

        with patch.object(client, "get_ticker", return_value=ticker):
            client.fetch_history("SU.PA", min_rows=1)

        assert ticker.history.call_args.kwargs["repair"] is True


class TestFetchPriceAndBenchmark:
    def test_when_stock_history_missing_then_all_none(
        self, client: YFinanceClient
    ) -> None:
        with patch.object(client, "fetch_history", return_value=None):
            with patch.object(client, "fetch_info", return_value={}):
                assert client.fetch_price_and_benchmark("AAPL") == (None, None, None)

    def test_when_benchmark_missing_then_stock_is_still_returned(
        self, client: YFinanceClient
    ) -> None:
        """Losing SPY must not cost us the stock's history."""
        stock = _hist(20)

        with patch.object(client, "fetch_history", side_effect=[stock, None]):
            with patch.object(client, "fetch_info", return_value={"a": 1}):
                hist, bench, info = client.fetch_price_and_benchmark("AAPL")

        assert hist is not None and bench is None and info == {"a": 1}

    def test_overlapping_dates_are_aligned_on_both_series(
        self, client: YFinanceClient
    ) -> None:
        stock = _hist(20)
        bench = _hist(10)

        with patch.object(client, "fetch_history", side_effect=[stock, bench]):
            with patch.object(client, "fetch_info", return_value={}):
                hist, benchmark, _ = client.fetch_price_and_benchmark("AAPL")

        assert hist is not None and benchmark is not None
        assert len(hist) == len(benchmark) == 10

    def test_when_no_dates_overlap_then_benchmark_is_dropped(
        self, client: YFinanceClient
    ) -> None:
        stock = _hist(5)
        bench = pd.DataFrame(
            {"Close": [1, 2]},
            index=pd.date_range("2030-01-01", periods=2, freq="D"),
        )

        with patch.object(client, "fetch_history", side_effect=[stock, bench]):
            with patch.object(client, "fetch_info", return_value={}):
                hist, benchmark, _ = client.fetch_price_and_benchmark("AAPL")

        assert hist is not None and benchmark is None


class TestBulkDownload:
    def test_returns_the_frame_on_success(self, client: YFinanceClient) -> None:
        frame = _hist(5)

        with patch(f"{M}.yf.download", return_value=frame) as dl:
            out = client.bulk_download(["AAPL", "MSFT"], period="1y")

        assert out is not None
        assert dl.call_args.kwargs["period"] == "1y"
        client.rate_limiter.acquire.assert_called_with("bulk_download")

    def test_start_replaces_period(self, client: YFinanceClient) -> None:
        with patch(f"{M}.yf.download", return_value=_hist(5)) as dl:
            client.bulk_download(["AAPL"], start="2024-01-01", end="2024-02-01")

        kwargs = dl.call_args.kwargs
        assert kwargs["start"] == "2024-01-01"
        assert "period" not in kwargs

    def test_end_alone_bounds_the_period_window(self, client: YFinanceClient) -> None:
        with patch(f"{M}.yf.download", return_value=_hist(5)) as dl:
            client.bulk_download(["AAPL"], period="1y", end="2020-01-01")

        kwargs = dl.call_args.kwargs
        assert kwargs["end"] == "2020-01-01"
        assert kwargs["period"] == "1y"
        assert "start" not in kwargs

    def test_empty_frame_is_a_failed_download(self, client: YFinanceClient) -> None:
        with patch(f"{M}.yf.download", return_value=pd.DataFrame()):
            assert client.bulk_download(["AAPL"]) is None

    def test_none_is_a_failed_download(self, client: YFinanceClient) -> None:
        with patch(f"{M}.yf.download", return_value=None):
            assert client.bulk_download(["AAPL"]) is None


class TestFetchPricesDataframe:
    def test_falls_back_to_individual_fetches_when_bulk_fails(
        self, client: YFinanceClient
    ) -> None:
        """One bad ticker must not cost the whole batch."""
        with patch.object(client, "bulk_download", return_value=None):
            with patch.object(client, "fetch_history", return_value=_hist(5)):
                out = client.fetch_prices_dataframe(["AAPL", "MSFT"])

        assert out is not None
        assert list(out.columns) == ["AAPL", "MSFT"]

    def test_individual_fallback_skips_tickers_with_no_history(
        self, client: YFinanceClient
    ) -> None:
        with patch.object(client, "bulk_download", return_value=None):
            with patch.object(client, "fetch_history", side_effect=[_hist(5), None]):
                out = client.fetch_prices_dataframe(["AAPL", "ZZZZ"])

        assert out is not None
        assert list(out.columns) == ["AAPL"]

    def test_when_nothing_can_be_fetched_then_none(
        self, client: YFinanceClient
    ) -> None:
        with patch.object(client, "bulk_download", return_value=None):
            with patch.object(client, "fetch_history", return_value=None):
                assert client.fetch_prices_dataframe(["ZZZZ"]) is None

    def test_multiindex_bulk_frame_is_flattened_to_close_columns(
        self, client: YFinanceClient
    ) -> None:
        idx = pd.date_range("2024-01-01", periods=4, freq="D")
        cols = pd.MultiIndex.from_product([["AAPL", "MSFT"], ["Close", "Open"]])
        frame = pd.DataFrame(1.0, index=idx, columns=cols)

        with patch.object(client, "bulk_download", return_value=frame):
            out = client.fetch_prices_dataframe(["AAPL", "MSFT"])

        assert out is not None
        assert sorted(out.columns) == ["AAPL", "MSFT"]


class TestCacheHelpers:
    def test_stats_are_read_from_the_cache(self, client: YFinanceClient) -> None:
        client.cache.size.return_value = 7
        assert client.get_cache_stats()["size"] == 7

    def test_clear_delegates_to_the_cache(self, client: YFinanceClient) -> None:
        client.clear_cache()
        client.cache.clear.assert_called_once()


class TestRateLimitDetection:
    @pytest.mark.parametrize(
        "message",
        [
            "429 Too Many Requests",
            "Rate limited",
            "Connection reset by peer",
            "RemoteDisconnected",
        ],
    )
    def test_transient_messages_are_recognised(
        self, client: YFinanceClient, message: str
    ) -> None:
        assert client._is_rate_limit_error(RuntimeError(message)) is True

    def test_detection_is_case_sensitive(self, client: YFinanceClient) -> None:
        """Documents current behaviour: TRANSIENT_NETWORK_INDICATORS is a
        case-sensitive substring match, so a lowercase variant is NOT detected
        and the circuit breaker will not trip on it."""
        assert client._is_rate_limit_error(RuntimeError("too many requests")) is False

    def test_unrelated_errors_are_not_transient(self, client: YFinanceClient) -> None:
        assert client._is_rate_limit_error(ValueError("bad ticker")) is False
