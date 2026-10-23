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

import threading
import time
from concurrent.futures import CancelledError
from typing import Any
from unittest.mock import MagicMock

import pytest

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
# UniverseBuilder.build() — discover returns None
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
# UniverseBuilder — delisting reconciliation vs. incomplete enumeration
# ---------------------------------------------------------------------------


class _DelistingRepo(_FakeRepo):
    def __init__(self, active: set[str]) -> None:
        super().__init__()
        self._active = active
        self.delisted: list[str] = []

    def get_active_tickers(
        self, exchange_id: Any, instrument_type: str | None = None
    ) -> set[str]:
        return set(self._active)

    def mark_delisted(self, **kwargs: Any) -> bool:
        self.delisted.append(kwargs["ticker"])
        return True


class TestUniverseBuilderIncompleteEnumeration:
    def _build(self, incomplete: frozenset[str]) -> tuple[_DelistingRepo, BuildResult]:
        repo = _DelistingRepo(active={"GONE_US"})
        mapper = MagicMock()
        mapper.discover.return_value = "AAPL"
        api_client = _make_api_client()
        api_client.incomplete_instrument_types = incomplete
        builder = _make_builder(api_client=api_client, ticker_mapper=mapper, repo=repo)
        return repo, builder.build()

    def test_absent_ticker_is_delisted_when_enumeration_completed(self) -> None:
        repo, _ = self._build(frozenset())
        assert repo.delisted == ["GONE_US"]

    def test_no_ticker_is_delisted_when_stock_enumeration_was_incomplete(
        self,
    ) -> None:
        repo, _ = self._build(frozenset({"STOCK"}))
        assert repo.delisted == []

    def test_incomplete_enumeration_is_reported_in_build_errors(self) -> None:
        _, result = self._build(frozenset({"STOCK"}))
        assert any("STOCK" in e and "incomplete" in e for e in result.errors)

    def test_other_instrument_type_still_reconciles(self) -> None:
        repo, _ = self._build(frozenset({"ETF"}))
        assert repo.delisted == ["GONE_US"]


# ---------------------------------------------------------------------------
# UniverseBuilder — exchange scoping
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

    def test_stock_on_newly_mapped_exchange_is_admitted(self) -> None:
        """Scope = suffix-mapped exchanges: Toronto (.TO) is now in scope."""
        repo = _FakeRepo()
        mapper = MagicMock()
        mapper.discover.return_value = "SHOP.TO"
        toronto = {
            "name": "Toronto Stock Exchange",
            "workingSchedules": [{"id": 7}],
        }
        inst = {
            "ticker": "SHOP_CA",
            "type": "STOCK",
            "shortName": "SHOP",
            "name": "Shopify",
            "workingScheduleId": 7,
        }
        api = _make_api_client(exchanges=[toronto], instruments=[inst])
        builder = _make_builder(api_client=api, ticker_mapper=mapper, repo=repo)

        result = builder.build()

        assert result.instruments_saved == 1

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


class TestProcessInstrumentsCancellation:
    """A reaped build must stop at its next progress call, not drain the pool."""

    def test_when_progress_raises_cancelled_then_it_propagates_and_queue_is_dropped(
        self,
    ) -> None:
        processed: list[str] = []
        lock = threading.Lock()

        def slow_single(instrument: dict[str, Any], exchange_name: str) -> Any:
            time.sleep(0.02)
            with lock:
                processed.append(instrument["shortName"])
            return None, "failed", "skipped for test"

        def cancelled(_progress: BuildProgress) -> None:
            raise CancelledError("job reaped")

        builder = UniverseBuilder(
            config=CFG,
            api_client=_make_api_client(),
            ticker_mapper=MagicMock(),
            repository=_FakeRepo(),
            max_workers=2,
            progress_callback=cancelled,
        )
        builder._process_single_instrument = slow_single  # type: ignore[method-assign]
        instruments = [{"shortName": f"S{i}"} for i in range(60)]

        with pytest.raises(CancelledError):
            builder._process_instruments(instruments, "NYSE", len(instruments), 0)

        assert len(processed) < len(instruments) // 2, (
            f"{len(processed)} of {len(instruments)} queued instruments still ran"
        )
        assert builder._errors == [], "cancellation was recorded as a build error"


# ---------------------------------------------------------------------------
# Delisting reconciliation vs. source-reported incompleteness
# ---------------------------------------------------------------------------


class _DelistingRepo(_FakeRepo):
    def __init__(self, active: set[str]) -> None:
        super().__init__()
        self._active = active
        self.delisted: list[str] = []

    def get_active_tickers(
        self, exchange_id: Any, instrument_type: str | None = None
    ) -> set[str]:
        return set(self._active) if instrument_type == "STOCK" else set()

    def mark_delisted(self, **kwargs: Any) -> bool:
        self.delisted.append(kwargs["ticker"])
        return True


class TestDelistingSkippedForIncompleteExchange:
    def _build(self, incomplete_types: list[str]) -> tuple[_DelistingRepo, Any]:
        repo = _DelistingRepo(active={"AAPL_US", "GONE_US"})
        mapper = MagicMock()
        mapper.discover.return_value = "AAPL"
        exchange = {**_NYSE_EXCHANGE, "incompleteTypes": incomplete_types}
        builder = _make_builder(
            api_client=_make_api_client(exchanges=[exchange]),
            ticker_mapper=mapper,
            repo=repo,
        )
        return repo, builder.build()

    def test_incomplete_type_does_not_delist_absent_tickers(self) -> None:
        repo, result = self._build(["STOCK"])

        assert repo.delisted == []
        assert any("incomplete" in e for e in result.errors)

    def test_other_type_incomplete_still_reconciles_this_type(self) -> None:
        repo, _ = self._build(["ETF"])

        assert repo.delisted == ["GONE_US"]


# ---------------------------------------------------------------------------
# Delisting reconciliation: partial / failed source answers
# ---------------------------------------------------------------------------


def _stock(i: int) -> dict[str, Any]:
    return {
        "ticker": f"T{i}_US",
        "type": "STOCK",
        "shortName": f"T{i}",
        "name": f"Company {i}",
        "currencyCode": "USD",
        "workingScheduleId": 1,
    }


def _delisting_build(
    *, active: int, returned: int, discover: Any = lambda short, ex: short
) -> tuple[_DelistingRepo, Any]:
    repo = _DelistingRepo(active={f"T{i}_US" for i in range(active)})
    mapper = MagicMock()
    mapper.discover.side_effect = discover
    builder = _make_builder(
        api_client=_make_api_client(instruments=[_stock(i) for i in range(returned)]),
        ticker_mapper=mapper,
        repo=repo,
    )
    return repo, builder.build()


class TestDelistingSanityGuard:
    def test_mass_absence_looks_like_a_truncated_source_and_delists_nothing(
        self,
    ) -> None:
        # 60 of 100 active tickers missing in one weekly run is a cut-off
        # answer, not 60 real delistings.
        repo, result = _delisting_build(active=100, returned=40)

        assert repo.delisted == []
        assert any("delisting" in e for e in result.errors)

    def test_ordinary_churn_is_still_delisted(self) -> None:
        repo, result = _delisting_build(active=100, returned=98)

        assert sorted(repo.delisted) == ["T98_US", "T99_US"]
        assert result.errors == []

    def test_small_venue_is_not_judged_by_fraction(self) -> None:
        repo, _ = _delisting_build(active=10, returned=5)

        assert len(repo.delisted) == 5


class TestMappingMissIsNotDelisting:
    def test_ticker_the_source_returned_but_mapper_missed_stays_active(self) -> None:
        repo, _ = _delisting_build(
            active=3,
            returned=3,
            discover=lambda short, ex: None if short == "T1" else short,
        )

        assert repo.delisted == []

    def test_ticker_whose_processing_raised_stays_active(self) -> None:
        def discover(short: str, exchange: str) -> str:
            if short == "T2":
                raise RuntimeError("lookup failed")
            return short

        repo, _ = _delisting_build(active=3, returned=3, discover=discover)

        assert repo.delisted == []


class TestExchangesSavedCount:
    def test_exchange_shared_by_stock_and_etf_passes_is_counted_once(self) -> None:
        nasdaq = {"name": "NASDAQ", "workingSchedules": [{"id": 1}]}
        bond_etf = {
            "ticker": "AGG_US",
            "type": "ETF",
            "shortName": "AGG",
            "name": "iShares Core Aggregate Bond",
            "isin": "US0000000001",
            "workingScheduleId": 1,
        }
        repo = _FakeRepo()
        mapper = MagicMock()
        mapper.discover.side_effect = lambda short, ex: short
        builder = _make_builder(
            api_client=_make_api_client(
                exchanges=[nasdaq], instruments=[_AAPL_INSTRUMENT, bond_etf]
            ),
            ticker_mapper=mapper,
            repo=repo,
        )

        result = builder.build()

        assert result.instruments_saved == 2
        assert result.exchanges_saved == 1
