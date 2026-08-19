"""Unit tests for UniverseBuilder.build().

All external I/O is mocked:
- Trading212ApiClient  → MagicMock
- UniverseRepository   → plain fake class (avoids optional-method edge-cases)
- YFinanceClient       → injected via _yf_client field (no singleton)
- TickerMappingCache   → real instance (in-memory, no network)

No real network calls, no time.sleep paths hit.  YFinanceTickerMapper unit
tests live in ``test_t212_mapper.py`` (split for the 500-line cap).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from app.services.universe.trading212.builder import (
    BuildProgress,
    BuildResult,
    UniverseBuilder,
)
from app.services.universe.trading212.cache.ticker_cache import TickerMappingCache
from app.services.universe.trading212.config import UniverseBuilderConfig
from app.services.universe.trading212.ticker_mapper import YFinanceTickerMapper

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

CFG = UniverseBuilderConfig()

_NYSE_EXCHANGE = {
    "name": "NYSE",
    "workingSchedules": [{"id": 1}],
}
_AAPL_INSTRUMENT = {
    "ticker": "AAPL_US",
    "type": "STOCK",
    "shortName": "AAPL",
    "isin": "US0378331005",
    "currencyCode": "USD",
    "name": "Apple Inc.",
    "workingScheduleId": 1,
}


# ---------------------------------------------------------------------------
# Fake repository (avoids MagicMock optional-method ambiguity)
# ---------------------------------------------------------------------------


class _FakeRepo:
    """Minimal UniverseRepository stub for builder tests."""

    def __init__(self) -> None:
        self._exchange_id = "ex-1"
        self.save_exchange_calls: list[dict[str, Any]] = []
        self.save_batch_calls: list[tuple[list[dict[str, Any]], Any]] = []

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

    def get_active_tickers(self, exchange_id: Any) -> set[str]:
        return set()

    def mark_delisted(self, **kwargs: Any) -> bool:
        return False

    def clear_all(self) -> tuple[int, int]:
        return 0, 0

    def get_instrument_count(self) -> int:
        return 0

    def get_exchange_count(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# UniverseBuilder.build() — skip_filters=True (happy path)
# ---------------------------------------------------------------------------


class TestUniverseBuilderBuildSkipFilters:
    def _builder(
        self,
        ticker_mapper: Any,
        repo: _FakeRepo,
        progress_cb: MagicMock | None = None,
    ) -> UniverseBuilder:
        return UniverseBuilder(
            config=CFG,
            api_client=_make_api_client(),
            ticker_mapper=ticker_mapper,
            filter_pipeline=_make_filter_pipeline(),
            repository=repo,
            skip_filters=True,
            max_workers=1,
            progress_callback=progress_cb,
        )

    def test_when_discovery_succeeds_then_exchange_saved_once(self) -> None:
        repo = _FakeRepo()
        mapper = MagicMock()
        mapper.discover.return_value = "AAPL"
        builder = self._builder(mapper, repo)

        result = builder.build()

        assert len(repo.save_exchange_calls) == 1
        assert repo.save_exchange_calls[0]["name"] == "NYSE"
        assert result.exchanges_saved == 1

    def test_when_discovery_succeeds_then_instrument_saved_with_yfinance_ticker(
        self,
    ) -> None:
        repo = _FakeRepo()
        mapper = MagicMock()
        mapper.discover.return_value = "AAPL"
        builder = self._builder(mapper, repo)

        result = builder.build()

        assert result.instruments_saved == 1
        assert len(repo.save_batch_calls) == 1
        saved_instrument = repo.save_batch_calls[0][0][0]
        assert saved_instrument["yfinanceTicker"] == "AAPL"
        assert saved_instrument["ticker"] == "AAPL_US"

    def test_when_discovery_succeeds_then_progress_callback_fired(self) -> None:
        repo = _FakeRepo()
        mapper = MagicMock()
        mapper.discover.return_value = "AAPL"
        progress_cb = MagicMock()
        builder = self._builder(mapper, repo, progress_cb)

        builder.build()

        assert progress_cb.call_count >= 1
        first_call_arg = progress_cb.call_args_list[0][0][0]
        assert isinstance(first_call_arg, BuildProgress)

    def test_when_discovery_succeeds_then_result_is_build_result(self) -> None:
        repo = _FakeRepo()
        mapper = MagicMock()
        mapper.discover.return_value = "AAPL"
        builder = self._builder(mapper, repo)

        result = builder.build()

        assert isinstance(result, BuildResult)

    def test_api_client_get_exchanges_and_get_instruments_both_called(self) -> None:
        repo = _FakeRepo()
        mapper = MagicMock()
        mapper.discover.return_value = "AAPL"
        api_client = _make_api_client()
        builder = UniverseBuilder(
            config=CFG,
            api_client=api_client,
            ticker_mapper=mapper,
            filter_pipeline=_make_filter_pipeline(),
            repository=repo,
            skip_filters=True,
            max_workers=1,
        )

        builder.build()

        api_client.get_exchanges.assert_called_once()
        api_client.get_instruments.assert_called_once()


# ---------------------------------------------------------------------------
# UniverseBuilder.build() — discover returns None
# ---------------------------------------------------------------------------


class TestUniverseBuilderDiscoverNone:
    def test_when_discover_returns_none_then_save_batch_not_called(self) -> None:
        repo = _FakeRepo()
        mapper = MagicMock()
        mapper.discover.return_value = None

        builder = UniverseBuilder(
            config=CFG,
            api_client=_make_api_client(),
            ticker_mapper=mapper,
            filter_pipeline=_make_filter_pipeline(),
            repository=repo,
            skip_filters=True,
            max_workers=1,
        )
        result = builder.build()

        assert len(repo.save_batch_calls) == 0
        assert result.instruments_saved == 0

    def test_when_discover_returns_none_then_exchange_is_still_saved(self) -> None:
        repo = _FakeRepo()
        mapper = MagicMock()
        mapper.discover.return_value = None

        builder = UniverseBuilder(
            config=CFG,
            api_client=_make_api_client(),
            ticker_mapper=mapper,
            filter_pipeline=_make_filter_pipeline(),
            repository=repo,
            skip_filters=True,
            max_workers=1,
        )
        result = builder.build()

        assert result.exchanges_saved == 1


# ---------------------------------------------------------------------------
# UniverseBuilder.build() — filter path (skip_filters=False)
# ---------------------------------------------------------------------------


class TestUniverseBuilderFilterPath:
    def test_when_filter_rejects_then_save_batch_not_called(self) -> None:
        repo = _FakeRepo()

        # Use a real YFinanceTickerMapper with mocked out network methods
        mock_yf_client = MagicMock()
        mapper = YFinanceTickerMapper(
            config=CFG,
            cache=TickerMappingCache(),
            _yf_client=mock_yf_client,
        )
        mapper.discover = MagicMock(return_value="AAPL")  # type: ignore[method-assign]
        mapper.fetch_basic_data = MagicMock(  # type: ignore[method-assign]
            return_value={"marketCap": 1}
        )

        pipeline = _make_filter_pipeline(apply_return=(False, "too small"))

        builder = UniverseBuilder(
            config=CFG,
            api_client=_make_api_client(),
            ticker_mapper=mapper,
            filter_pipeline=pipeline,
            repository=repo,
            skip_filters=False,
            max_workers=1,
        )
        result = builder.build()

        pipeline.apply.assert_called_once()
        assert len(repo.save_batch_calls) == 0
        assert result.instruments_saved == 0

    def test_when_filter_passes_then_instrument_saved(self) -> None:
        repo = _FakeRepo()

        mock_yf_client = MagicMock()
        mapper = YFinanceTickerMapper(
            config=CFG,
            cache=TickerMappingCache(),
            _yf_client=mock_yf_client,
        )
        mapper.discover = MagicMock(return_value="AAPL")  # type: ignore[method-assign]
        mapper.fetch_basic_data = MagicMock(  # type: ignore[method-assign]
            return_value={"marketCap": 5_000_000_000, "currentPrice": 150.0}
        )

        pipeline = _make_filter_pipeline(apply_return=(True, "ok"))

        builder = UniverseBuilder(
            config=CFG,
            api_client=_make_api_client(),
            ticker_mapper=mapper,
            filter_pipeline=pipeline,
            repository=repo,
            skip_filters=False,
            max_workers=1,
        )
        result = builder.build()

        pipeline.apply.assert_called_once()
        assert result.instruments_saved == 1

    def test_fetch_basic_data_called_instead_of_yfinance_singleton(self) -> None:
        """When ticker_mapper is YFinanceTickerMapper, fetch_basic_data is used.

        The YFinanceClient singleton import inside _fetch_filter_data is a
        *local* import (inside the else branch), so it is never reached when
        the isinstance check succeeds.  We verify this by asserting that the
        injected mock_yf_client.fetch_info is never called directly by the
        builder (only fetch_basic_data is), and that fetch_basic_data itself
        was called with the discovered ticker.
        """
        repo = _FakeRepo()

        mock_yf_client = MagicMock()
        mapper = YFinanceTickerMapper(
            config=CFG,
            cache=TickerMappingCache(),
            _yf_client=mock_yf_client,
        )
        mapper.discover = MagicMock(return_value="AAPL")  # type: ignore[method-assign]
        fetch_spy = MagicMock(return_value={"marketCap": 5_000_000_000})
        mapper.fetch_basic_data = fetch_spy  # type: ignore[method-assign]

        pipeline = _make_filter_pipeline(apply_return=(True, "ok"))

        builder = UniverseBuilder(
            config=CFG,
            api_client=_make_api_client(),
            ticker_mapper=mapper,
            filter_pipeline=pipeline,
            repository=repo,
            skip_filters=False,
            max_workers=1,
        )
        builder.build()

        # fetch_basic_data (not the singleton) was called with the right ticker
        fetch_spy.assert_called_once_with("AAPL")
        # The injected yf client's fetch_info is never called by the builder path
        mock_yf_client.fetch_info.assert_not_called()

    def test_when_fetch_basic_data_returns_none_then_instrument_not_saved(
        self,
    ) -> None:
        repo = _FakeRepo()

        mock_yf_client = MagicMock()
        mapper = YFinanceTickerMapper(
            config=CFG,
            cache=TickerMappingCache(),
            _yf_client=mock_yf_client,
        )
        mapper.discover = MagicMock(return_value="AAPL")  # type: ignore[method-assign]
        mapper.fetch_basic_data = MagicMock(return_value=None)  # type: ignore[method-assign]

        pipeline = _make_filter_pipeline()

        builder = UniverseBuilder(
            config=CFG,
            api_client=_make_api_client(),
            ticker_mapper=mapper,
            filter_pipeline=pipeline,
            repository=repo,
            skip_filters=False,
            max_workers=1,
        )
        result = builder.build()

        pipeline.apply.assert_not_called()
        assert result.instruments_saved == 0


# ---------------------------------------------------------------------------
# UniverseBuilder — exchange filtering
# ---------------------------------------------------------------------------


class TestUniverseBuilderExchangeFiltering:
    def test_unknown_exchange_is_skipped(self) -> None:
        repo = _FakeRepo()
        mapper = MagicMock()
        mapper.discover.return_value = "SYM"

        unknown_exchange = {
            "name": "SomeObscureMarket",
            "workingSchedules": [{"id": 99}],
        }
        instrument = {
            "ticker": "SYM_X",
            "type": "STOCK",
            "shortName": "SYM",
            "workingScheduleId": 99,
        }
        api_client = _make_api_client(
            exchanges=[unknown_exchange], instruments=[instrument]
        )

        builder = UniverseBuilder(
            config=CFG,
            api_client=api_client,
            ticker_mapper=mapper,
            filter_pipeline=_make_filter_pipeline(),
            repository=repo,
            skip_filters=True,
            max_workers=1,
        )
        result = builder.build()

        assert result.exchanges_saved == 0
        assert result.instruments_saved == 0

    def test_only_exchanges_override_filters_to_listed_exchange(self) -> None:
        repo = _FakeRepo()
        mapper = MagicMock()
        mapper.discover.return_value = "AAPL"
        api_client = _make_api_client()

        builder = UniverseBuilder(
            config=CFG,
            api_client=api_client,
            ticker_mapper=mapper,
            filter_pipeline=_make_filter_pipeline(),
            repository=repo,
            skip_filters=True,
            only_exchanges=["NYSE"],
            max_workers=1,
        )
        result = builder.build()

        assert result.exchanges_saved == 1

    def test_non_stock_type_instruments_are_excluded(self) -> None:
        repo = _FakeRepo()
        mapper = MagicMock()
        mapper.discover.return_value = "ETF"

        etf_instrument = {
            "ticker": "SPY_US",
            "type": "ETF",
            "shortName": "SPY",
            "workingScheduleId": 1,
        }
        api_client = _make_api_client(instruments=[etf_instrument])

        builder = UniverseBuilder(
            config=CFG,
            api_client=api_client,
            ticker_mapper=mapper,
            filter_pipeline=_make_filter_pipeline(),
            repository=repo,
            skip_filters=True,
            max_workers=1,
        )
        result = builder.build()

        # ETF excluded → no exchange_stocks → exchanges_saved=0
        assert result.instruments_saved == 0
