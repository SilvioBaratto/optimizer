"""Branch coverage for YFinanceTickerMapper and UniverseBuilder residual paths.

Targets:
- app/services/universe/trading212/ticker_mapper.py  (~19 branches)
- app/services/universe/trading212/builder.py        (~13 branches)

All network I/O is mocked. No time.sleep is exercised (patched throughout).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from app.services.universe.trading212.builder import (
    UniverseBuilder,
)
from app.services.universe.trading212.cache.ticker_cache import TickerMappingCache
from app.services.universe.trading212.config import UniverseBuilderConfig
from app.services.universe.trading212.ticker_mapper import YFinanceTickerMapper

CFG = UniverseBuilderConfig()

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_NYSE_EXCHANGE = {"name": "NYSE", "workingSchedules": [{"id": 1}]}
_AAPL_INSTRUMENT = {
    "ticker": "AAPL_US",
    "type": "STOCK",
    "shortName": "AAPL",
    "isin": "US0378331005",
    "currencyCode": "USD",
    "name": "Apple Inc.",
    "workingScheduleId": 1,
}


class _FakeRepo:
    """Minimal UniverseRepository stub."""

    def __init__(self) -> None:
        self._exchange_id = "ex-1"
        self.save_exchange_calls: list[dict[str, Any]] = []
        self.save_batch_calls: list[tuple[list[dict[str, Any]], Any]] = []
        self.mark_delisted_calls: list[dict[str, Any]] = []

    def save_exchange(self, exchange_data: dict[str, Any]) -> Any:
        self.save_exchange_calls.append(exchange_data)
        result = MagicMock()
        result.id = self._exchange_id
        return result

    def save_instruments_batch(
        self, instruments_data: list[dict[str, Any]], exchange_id: Any
    ) -> int:
        self.save_batch_calls.append((instruments_data, exchange_id))
        return len(instruments_data)

    def get_active_tickers(
        self, exchange_id: Any, instrument_type: str | None = None
    ) -> set[str]:
        return set()

    def mark_delisted(self, **kwargs: Any) -> bool:
        self.mark_delisted_calls.append(kwargs)
        return True

    def clear_all(self) -> tuple[int, int]:
        return 0, 0

    def get_instrument_count(self) -> int:
        return 0

    def get_exchange_count(self) -> int:
        return 0


def _make_api_client(
    exchanges: list[dict[str, Any]] | None = None,
    instruments: list[dict[str, Any]] | None = None,
) -> MagicMock:
    client = MagicMock()
    client.get_exchanges.return_value = exchanges or [_NYSE_EXCHANGE]
    client.get_instruments.return_value = instruments or [_AAPL_INSTRUMENT]
    return client


def _make_filter_pipeline(apply_return: tuple[bool, str] = (True, "ok")) -> MagicMock:
    pipeline = MagicMock()
    pipeline.apply.return_value = apply_return
    pipeline.get_summary.return_value = {}
    return pipeline


def _mapper_with_info(info: Any) -> YFinanceTickerMapper:
    mock_yf = MagicMock()
    mock_yf.fetch_info.return_value = info
    return YFinanceTickerMapper(
        config=CFG, cache=TickerMappingCache(), _yf_client=mock_yf
    )


# ===========================================================================
# YFinanceTickerMapper — uncovered branches
# ===========================================================================


class TestTickerMapperNameProperty:
    """Covers: ticker_mapper.py line 18 — __post_init__ cache init when None."""

    def test_when_cache_none_post_init_creates_default_cache(self) -> None:
        mock_yf = MagicMock()
        # Pass cache=None explicitly to trigger __post_init__ branch
        mapper = YFinanceTickerMapper(config=CFG, cache=None, _yf_client=mock_yf)
        assert mapper.cache is not None
        assert isinstance(mapper.cache, TickerMappingCache)


class TestTickerMapperLazyYfClient:
    """Covers: ticker_mapper.py line 23 — lazy yf_client singleton fallback."""

    def test_when_yf_client_none_property_calls_singleton(self) -> None:
        mapper = YFinanceTickerMapper(config=CFG, cache=TickerMappingCache())
        # _yf_client defaults to None via field(default=None)
        mock_instance = MagicMock()
        with patch(
            "app.services.universe.trading212.ticker_mapper.YFinanceClient"
        ) as mock_cls:
            mock_cls.get_instance.return_value = mock_instance
            client = mapper.yf_client
        mock_cls.get_instance.assert_called_once()
        assert client is mock_instance


class TestTickerMapperCacheHitButVerifyFails:
    """Covers: ticker_mapper.py line 29->35 — cache returns a ticker but verify fails."""

    def test_when_cache_hit_but_verify_fails_then_falls_through_to_discovery(
        self,
    ) -> None:
        cache = TickerMappingCache()
        cache.save_mapping("AAPL", "NYSE", "AAPL")

        mock_yf = MagicMock()
        mapper = YFinanceTickerMapper(config=CFG, cache=cache, _yf_client=mock_yf)

        # Patch _verify_ticker: first call (cache hit) fails, second succeeds
        verify_results = [False, True]
        mapper._verify_ticker = MagicMock(  # type: ignore[method-assign]
            side_effect=verify_results
        )

        result = mapper.discover("AAPL", "NYSE")

        # Falls through to ticker_attempts loop; second attempt succeeds
        assert result == "AAPL"
        assert mapper._verify_ticker.call_count == 2


class TestTickerMapperDiscoverNoExchangeNoCacheSave:
    """Covers: ticker_mapper.py line 43->45 — no exchange_name means cache.save skipped."""

    def test_when_exchange_name_is_none_then_cache_not_saved(self) -> None:
        cache = TickerMappingCache()
        mock_yf = MagicMock()
        mapper = YFinanceTickerMapper(config=CFG, cache=cache, _yf_client=mock_yf)
        mapper._verify_ticker = MagicMock(return_value=True)  # type: ignore[method-assign]

        result = mapper.discover("AAPL", None)

        assert result == "AAPL"
        # No exchange_name → no cache.save_mapping call
        assert cache.get_mapping("AAPL", "") is None


class TestVerifyTickerRateLimitRetry:
    """Covers: ticker_mapper.py lines 83-112 — retry paths in _verify_ticker."""

    def test_rate_limit_on_all_retries_returns_false(self) -> None:
        mock_yf = MagicMock()
        mock_yf.fetch_info.side_effect = RuntimeError("rate limit exceeded")
        mapper = YFinanceTickerMapper(
            config=CFG,
            cache=TickerMappingCache(),
            _yf_client=mock_yf,
            max_retries=3,
        )
        with patch("app.services.universe.trading212.ticker_mapper.time.sleep"):
            result = mapper._verify_ticker("AAPL")
        assert result is False
        assert mock_yf.fetch_info.call_count == 3

    def test_not_found_error_immediately_returns_false(self) -> None:
        mock_yf = MagicMock()
        mock_yf.fetch_info.side_effect = RuntimeError("not found")
        mapper = YFinanceTickerMapper(
            config=CFG,
            cache=TickerMappingCache(),
            _yf_client=mock_yf,
            max_retries=3,
        )
        result = mapper._verify_ticker("BADTICKER")
        assert result is False
        # not-found → returns on first attempt (no retry)
        assert mock_yf.fetch_info.call_count == 1

    def test_404_error_immediately_returns_false(self) -> None:
        mock_yf = MagicMock()
        mock_yf.fetch_info.side_effect = RuntimeError("404")
        mapper = YFinanceTickerMapper(
            config=CFG, cache=TickerMappingCache(), _yf_client=mock_yf, max_retries=3
        )
        result = mapper._verify_ticker("AAPL")
        assert result is False

    def test_generic_error_returns_false_immediately(self) -> None:
        mock_yf = MagicMock()
        mock_yf.fetch_info.side_effect = RuntimeError("unexpected json error")
        mapper = YFinanceTickerMapper(
            config=CFG, cache=TickerMappingCache(), _yf_client=mock_yf, max_retries=3
        )
        result = mapper._verify_ticker("AAPL")
        assert result is False

    def test_timeout_error_on_last_retry_returns_false(self) -> None:
        mock_yf = MagicMock()
        mock_yf.fetch_info.side_effect = RuntimeError("timed out")
        mapper = YFinanceTickerMapper(
            config=CFG,
            cache=TickerMappingCache(),
            _yf_client=mock_yf,
            max_retries=2,
        )
        with patch("app.services.universe.trading212.ticker_mapper.time.sleep"):
            result = mapper._verify_ticker("AAPL")
        assert result is False
        assert mock_yf.fetch_info.call_count == 2


class TestFetchBasicDataBranches:
    """Covers: ticker_mapper.py lines 123-124, 163-184."""

    def test_when_info_short_retries_then_returns_none_after_exhaustion(self) -> None:
        # info has < 10 fields every time → retries, then None
        mapper = _mapper_with_info({"a": 1})
        with patch("app.services.universe.trading212.ticker_mapper.time.sleep"):
            result = mapper.fetch_basic_data("AAPL", max_retries=3)
        assert result is None
        assert mapper.yf_client.fetch_info.call_count == 3

    def test_when_rate_limit_on_all_fetch_basic_data_retries_returns_none(self) -> None:
        mock_yf = MagicMock()
        mock_yf.fetch_info.side_effect = RuntimeError("rate limit exceeded")
        mapper = YFinanceTickerMapper(
            config=CFG, cache=TickerMappingCache(), _yf_client=mock_yf
        )
        with patch("app.services.universe.trading212.ticker_mapper.time.sleep"):
            result = mapper.fetch_basic_data("AAPL", max_retries=3)
        assert result is None
        assert mock_yf.fetch_info.call_count == 3

    def test_when_generic_error_on_first_attempt_retries_then_none(self) -> None:
        mock_yf = MagicMock()
        mock_yf.fetch_info.side_effect = RuntimeError("connection refused")
        mapper = YFinanceTickerMapper(
            config=CFG, cache=TickerMappingCache(), _yf_client=mock_yf
        )
        with patch("app.services.universe.trading212.ticker_mapper.time.sleep"):
            result = mapper.fetch_basic_data("AAPL", max_retries=2)
        assert result is None
        assert mock_yf.fetch_info.call_count == 2

    def test_when_info_short_on_last_retry_returns_none(self) -> None:
        # First attempt returns short dict; second also short → returns None
        mock_yf = MagicMock()
        mock_yf.fetch_info.return_value = {"x": 1}
        mapper = YFinanceTickerMapper(
            config=CFG, cache=TickerMappingCache(), _yf_client=mock_yf
        )
        with patch("app.services.universe.trading212.ticker_mapper.time.sleep"):
            result = mapper.fetch_basic_data("AAPL", max_retries=1)
        assert result is None


# ===========================================================================
# UniverseBuilder — residual uncovered branches
# ===========================================================================


class TestBuilderFetchMetadata:
    """Covers: builder.py lines 91-93 — fetch_metadata() public method."""

    def test_fetch_metadata_returns_exchanges_and_instruments(self) -> None:
        api_client = _make_api_client()
        builder = UniverseBuilder(
            config=CFG,
            api_client=api_client,
            ticker_mapper=MagicMock(),
            filter_pipeline=_make_filter_pipeline(),
            repository=_FakeRepo(),
        )
        exchanges, instruments = builder.fetch_metadata()
        assert exchanges == [_NYSE_EXCHANGE]
        assert instruments == [_AAPL_INSTRUMENT]


class TestBuilderGetExchangeStocks:
    """Covers: builder.py lines 98-99 — get_exchange_stocks() public method."""

    def test_get_exchange_stocks_returns_correct_structure(self) -> None:
        api_client = _make_api_client()
        builder = UniverseBuilder(
            config=CFG,
            api_client=api_client,
            ticker_mapper=MagicMock(),
            filter_pipeline=_make_filter_pipeline(),
            repository=_FakeRepo(),
        )
        exchange_stocks = builder.get_exchange_stocks(
            [_NYSE_EXCHANGE], [_AAPL_INSTRUMENT]
        )
        # NYSE is allowed; AAPL_INSTRUMENT is STOCK type
        assert len(exchange_stocks) == 1
        ex_data, stocks = exchange_stocks[0]
        assert ex_data["name"] == "NYSE"
        assert len(stocks) == 1


class TestBuilderExchangeNoName:
    """Covers: builder.py line 124 — exchange dict with no 'name' key is skipped."""

    def test_exchange_without_name_key_is_skipped(self) -> None:
        nameless_exchange = {"workingSchedules": [{"id": 99}]}
        instrument = {
            "ticker": "X_US",
            "type": "STOCK",
            "shortName": "X",
            "workingScheduleId": 99,
        }
        api_client = _make_api_client(
            exchanges=[nameless_exchange], instruments=[instrument]
        )
        repo = _FakeRepo()
        builder = UniverseBuilder(
            config=CFG,
            api_client=api_client,
            ticker_mapper=MagicMock(),
            filter_pipeline=_make_filter_pipeline(),
            repository=repo,
            skip_filters=True,
            max_workers=1,
        )
        result = builder.build()
        assert result.exchanges_saved == 0


class TestBuilderOnlyExchangesSkipPath:
    """Covers: builder.py line 129 — only_exchanges set but exchange not in list."""

    def test_exchange_not_in_only_exchanges_is_skipped(self) -> None:
        repo = _FakeRepo()
        mapper = MagicMock()
        mapper.discover.return_value = "AAPL"
        builder = UniverseBuilder(
            config=CFG,
            api_client=_make_api_client(),
            ticker_mapper=mapper,
            filter_pipeline=_make_filter_pipeline(),
            repository=repo,
            skip_filters=True,
            only_exchanges=["NASDAQ"],  # NYSE is not included
            max_workers=1,
        )
        result = builder.build()
        assert result.exchanges_saved == 0


class TestBuilderGetActiveTickersPath:
    """Covers: builder.py line 167 — repo.get_active_tickers is called when present."""

    def test_when_repo_has_get_active_tickers_then_it_is_called(self) -> None:
        repo = _FakeRepo()
        # _FakeRepo has get_active_tickers; track calls
        original = repo.get_active_tickers
        call_count = {"n": 0}

        def _tracking(exchange_id: Any, instrument_type: str | None = None) -> set[str]:
            call_count["n"] += 1
            return original(exchange_id, instrument_type)

        repo.get_active_tickers = _tracking  # type: ignore[assignment]

        mapper = MagicMock()
        mapper.discover.return_value = "AAPL"
        builder = UniverseBuilder(
            config=CFG,
            api_client=_make_api_client(),
            ticker_mapper=mapper,
            filter_pipeline=_make_filter_pipeline(),
            repository=repo,
            skip_filters=True,
            max_workers=1,
        )
        builder.build()
        assert call_count["n"] == 1


class TestBuilderMarkDelistedSkipWhenMethodAbsent:
    """Covers: builder.py line 200 — mark_delisted skipped when repo lacks the method."""

    def test_when_repo_lacks_mark_delisted_no_error(self) -> None:
        class _MinimalRepo:
            def save_exchange(self, data: dict[str, Any]) -> Any:
                r = MagicMock()
                r.id = "ex-1"
                return r

            def save_instruments_batch(
                self, data: list[dict[str, Any]], exchange_id: Any
            ) -> int:
                return len(data)

        repo = _MinimalRepo()
        mapper = MagicMock()
        mapper.discover.return_value = "AAPL"
        builder = UniverseBuilder(
            config=CFG,
            api_client=_make_api_client(),
            ticker_mapper=mapper,
            filter_pipeline=_make_filter_pipeline(),
            repository=repo,  # type: ignore[arg-type]
            skip_filters=True,
            max_workers=1,
        )
        # Must not raise even though repo has no mark_delisted
        result = builder.build()
        assert result.instruments_saved == 1


class TestBuilderDelistingDetection:
    """Covers: builder.py lines 206-216 — dropped tickers are marked as delisted."""

    def test_ticker_dropped_from_t212_response_is_marked_delisted(self) -> None:
        repo = _FakeRepo()

        # get_active_tickers returns an old ticker that won't be in the new response
        def get_active_tickers(
            exchange_id: Any, instrument_type: str | None = None
        ) -> set[str]:
            return {"OLD_TICKER"}

        repo.get_active_tickers = get_active_tickers  # type: ignore[assignment]

        mapper = MagicMock()
        mapper.discover.return_value = "AAPL"  # new response has only AAPL

        builder = UniverseBuilder(
            config=CFG,
            api_client=_make_api_client(),  # only _AAPL_INSTRUMENT
            ticker_mapper=mapper,
            filter_pipeline=_make_filter_pipeline(),
            repository=repo,
            skip_filters=True,
            max_workers=1,
        )
        builder.build()

        # OLD_TICKER was not in the new response → mark_delisted should be called
        assert len(repo.mark_delisted_calls) == 1
        assert repo.mark_delisted_calls[0]["ticker"] == "OLD_TICKER"


class TestBuilderProcessInstrumentsException:
    """Covers: builder.py lines 259-275 — exception from future.result() appended to errors.

    The exception-in-future path (lines 259-275) is reached when
    future.result() raises — i.e. _process_single_instrument itself raises
    rather than returning the (None, "error", str(e)) tuple.  We patch
    _process_single_instrument on the builder instance to raise directly.
    """

    def test_exception_in_future_is_appended_to_errors_and_progress_fired(
        self,
    ) -> None:
        repo = _FakeRepo()
        mapper = MagicMock()
        mapper.discover.return_value = "AAPL"
        progress_calls: list[Any] = []
        builder = UniverseBuilder(
            config=CFG,
            api_client=_make_api_client(),
            ticker_mapper=mapper,
            filter_pipeline=_make_filter_pipeline(),
            repository=repo,
            skip_filters=True,
            max_workers=1,
            progress_callback=lambda p: progress_calls.append(p),
        )

        # Patch _process_single_instrument to raise so future.result() re-raises
        def _raiser(instrument: Any, exchange_name: str) -> Any:
            raise RuntimeError("network failure")

        builder._process_single_instrument = _raiser  # type: ignore[method-assign]

        result = builder.build()

        # Exception path: error captured in BuildResult.errors
        assert len(result.errors) >= 1
        assert "network failure" in result.errors[0]
        # Progress callback fired with status="error"
        error_progress = [p for p in progress_calls if p.status == "error"]
        assert len(error_progress) >= 1


class TestBuilderProcessSingleInstrumentExceptionPath:
    """Covers: builder.py lines 323-326 — inner exception returns (None, error, str(e))."""

    def test_exception_inside_process_single_instrument_returns_none_error(
        self,
    ) -> None:
        repo = _FakeRepo()
        mapper = MagicMock()
        # discover itself raises an exception
        mapper.discover.side_effect = ValueError("bad symbol")

        builder = UniverseBuilder(
            config=CFG,
            api_client=_make_api_client(),
            ticker_mapper=mapper,
            filter_pipeline=_make_filter_pipeline(),
            repository=repo,
            skip_filters=True,
            max_workers=1,
        )
        # Call directly to inspect return value
        result_data, status, reason = builder._process_single_instrument(
            _AAPL_INSTRUMENT, "NYSE"
        )
        assert result_data is None
        assert status == "error"
        assert "bad symbol" in reason


class TestBuilderFetchFilterDataNonMapperPath:
    """Covers: builder.py lines 332-335 — _fetch_filter_data uses YFinanceClient.get_instance()
    when ticker_mapper is NOT a YFinanceTickerMapper instance.

    YFinanceClient is imported *locally* inside _fetch_filter_data (not at module
    top-level), so we patch it at its source module path.
    """

    def test_when_mapper_is_plain_mock_fetch_filter_data_uses_yfinance_singleton(
        self,
    ) -> None:
        repo = _FakeRepo()
        # Use a plain MagicMock (not YFinanceTickerMapper) so isinstance check fails
        mapper = MagicMock(
            spec=[]
        )  # spec=[] ensures it's not an instance of YFinanceTickerMapper

        mapper.discover = MagicMock(return_value="AAPL")
        mock_client = MagicMock()
        mock_client.fetch_info.return_value = {"marketCap": 5e9}

        builder = UniverseBuilder(
            config=CFG,
            api_client=_make_api_client(),
            ticker_mapper=mapper,
            filter_pipeline=_make_filter_pipeline(),
            repository=repo,
            skip_filters=False,
            max_workers=1,
        )
        # YFinanceClient is a local import inside _fetch_filter_data — patch at source
        with patch("app.services.market_data.yfinance.YFinanceClient") as mock_yf_cls:
            mock_yf_cls.get_instance.return_value = mock_client
            builder.build()

        mock_yf_cls.get_instance.assert_called()


class TestBuilderGetFilterStats:
    """Covers: builder.py line 338 — get_filter_stats() delegates to pipeline."""

    def test_get_filter_stats_delegates_to_pipeline(self) -> None:
        pipeline = _make_filter_pipeline()
        pipeline.get_summary.return_value = {"SomeFilter": {"passed": 5, "failed": 2}}
        builder = UniverseBuilder(
            config=CFG,
            api_client=_make_api_client(),
            ticker_mapper=MagicMock(),
            filter_pipeline=pipeline,
            repository=_FakeRepo(),
        )
        stats = builder.get_filter_stats()
        assert stats == {"SomeFilter": {"passed": 5, "failed": 2}}
        pipeline.get_summary.assert_called_once()
