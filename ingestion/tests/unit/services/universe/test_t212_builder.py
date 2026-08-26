"""Unit tests for UniverseBuilder.build().

All external I/O is mocked:
- Trading212ApiClient  → MagicMock
- UniverseRepository   → plain fake class (avoids optional-method edge-cases)
- TickerMapper         → MagicMock

No real network calls. There is no investability filtering: every classified +
mapped STOCK/ETF is admitted. YFinanceTickerMapper unit tests live in
``test_t212_mapper.py``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from app.services.universe.trading212.builder import (
    BuildProgress,
    BuildResult,
    UniverseBuilder,
)
from app.services.universe.trading212.config import UniverseBuilderConfig

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

    def get_active_tickers(
        self, exchange_id: Any, instrument_type: str | None = None
    ) -> set[str]:
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


def _make_builder(
    *,
    api_client: Any | None = None,
    ticker_mapper: Any,
    repo: _FakeRepo,
    progress_cb: Any | None = None,
    only_exchanges: list[str] | None = None,
) -> UniverseBuilder:
    return UniverseBuilder(
        config=CFG,
        api_client=api_client or _make_api_client(),
        ticker_mapper=ticker_mapper,
        repository=repo,
        max_workers=1,
        only_exchanges=only_exchanges,
        progress_callback=progress_cb,
    )


# ---------------------------------------------------------------------------
# UniverseBuilder.build() — no-filter admission (happy path)
# ---------------------------------------------------------------------------


class TestUniverseBuilderBuild:
    def test_when_discovery_succeeds_then_exchange_saved_once(self) -> None:
        repo = _FakeRepo()
        mapper = MagicMock()
        mapper.discover.return_value = "AAPL"
        builder = _make_builder(ticker_mapper=mapper, repo=repo)

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
        builder = _make_builder(ticker_mapper=mapper, repo=repo)

        result = builder.build()

        assert result.instruments_saved == 1
        assert len(repo.save_batch_calls) == 1
        saved_instrument = repo.save_batch_calls[0][0][0]
        assert saved_instrument["yfinanceTicker"] == "AAPL"
        assert saved_instrument["ticker"] == "AAPL_US"

    def test_admits_instrument_without_any_filtering(self) -> None:
        """No .info fetch, no pipeline: a classified + mapped instrument is
        admitted regardless of size/price/liquidity/history. Pre-refactor this
        path fetched fundamentals and applied the filter pipeline."""
        repo = _FakeRepo()
        mapper = MagicMock()
        mapper.discover.return_value = "AAPL"
        # A mapper with no fetch_basic_data at all still yields a saved row —
        # the build never fetches fundamentals for filtering.
        del mapper.fetch_basic_data
        builder = _make_builder(ticker_mapper=mapper, repo=repo)

        result = builder.build()

        assert result.instruments_saved == 1
        assert result.filter_stats == {}

    def test_when_discovery_succeeds_then_progress_callback_fired(self) -> None:
        repo = _FakeRepo()
        mapper = MagicMock()
        mapper.discover.return_value = "AAPL"
        progress_cb = MagicMock()
        builder = _make_builder(
            ticker_mapper=mapper, repo=repo, progress_cb=progress_cb
        )

        builder.build()

        assert progress_cb.call_count >= 1
        first_call_arg = progress_cb.call_args_list[0][0][0]
        assert isinstance(first_call_arg, BuildProgress)

    def test_when_discovery_succeeds_then_result_is_build_result(self) -> None:
        repo = _FakeRepo()
        mapper = MagicMock()
        mapper.discover.return_value = "AAPL"
        builder = _make_builder(ticker_mapper=mapper, repo=repo)

        result = builder.build()

        assert isinstance(result, BuildResult)

    def test_api_client_get_exchanges_and_get_instruments_both_called(self) -> None:
        repo = _FakeRepo()
        mapper = MagicMock()
        mapper.discover.return_value = "AAPL"
        api_client = _make_api_client()
        builder = _make_builder(api_client=api_client, ticker_mapper=mapper, repo=repo)

        builder.build()

        api_client.get_exchanges.assert_called_once()
        api_client.get_instruments.assert_called_once()


# ---------------------------------------------------------------------------
# UniverseBuilder.build() — discover returns None (T1: still dropped; T4 flips)
# ---------------------------------------------------------------------------


class TestUniverseBuilderDiscoverNone:
    def test_when_discover_returns_none_then_save_batch_not_called(self) -> None:
        repo = _FakeRepo()
        mapper = MagicMock()
        mapper.discover.return_value = None
        builder = _make_builder(ticker_mapper=mapper, repo=repo)

        result = builder.build()

        assert len(repo.save_batch_calls) == 0
        assert result.instruments_saved == 0

    def test_when_discover_returns_none_then_exchange_is_still_saved(self) -> None:
        repo = _FakeRepo()
        mapper = MagicMock()
        mapper.discover.return_value = None
        builder = _make_builder(ticker_mapper=mapper, repo=repo)

        result = builder.build()

        assert result.exchanges_saved == 1


# ---------------------------------------------------------------------------
# UniverseBuilder — exchange scoping (T1: still restricted; T3 opens it up)
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
        builder = _make_builder(api_client=api_client, ticker_mapper=mapper, repo=repo)

        result = builder.build()

        assert result.exchanges_saved == 0
        assert result.instruments_saved == 0

    def test_only_exchanges_override_filters_to_listed_exchange(self) -> None:
        repo = _FakeRepo()
        mapper = MagicMock()
        mapper.discover.return_value = "AAPL"
        builder = _make_builder(
            ticker_mapper=mapper, repo=repo, only_exchanges=["NYSE"]
        )

        result = builder.build()

        assert result.exchanges_saved == 1

    def test_equity_etf_is_rejected_by_classification(self) -> None:
        repo = _FakeRepo()
        mapper = MagicMock()
        mapper.discover.return_value = "SPY"

        # SPY is an equity ETF → classify_instrument returns None → dropped
        # (classification integrity, not investability filtering).
        etf_instrument = {
            "ticker": "SPY_US",
            "type": "ETF",
            "shortName": "SPY",
            "name": "SPDR S&P 500 ETF Trust",
            "workingScheduleId": 1,
        }
        api_client = _make_api_client(instruments=[etf_instrument])
        builder = _make_builder(api_client=api_client, ticker_mapper=mapper, repo=repo)

        result = builder.build()

        assert result.instruments_saved == 0
