"""Branch coverage for UniverseBuilder residual paths.

Targets app/services/universe/builder/builder.py. All network I/O is mocked.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from app.services.universe.builder.builder import UniverseBuilder
from app.services.universe.builder.config import UniverseBuilderConfig

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
            repository=repo,
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
            repository=repo,
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
            repository=repo,
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
            repository=repo,  # type: ignore[arg-type]
            max_workers=1,
        )
        # Must not raise even though repo has no mark_delisted
        result = builder.build()
        assert result.instruments_saved == 1


class TestBuilderDelistingDetection:
    """Covers: builder.py lines 206-216 — dropped tickers are marked as delisted."""

    def test_ticker_dropped_from_the_source_response_is_marked_delisted(self) -> None:
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
            repository=repo,
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
            repository=repo,
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
            repository=repo,
            max_workers=1,
        )
        # Call directly to inspect return value
        result_data, status, reason = builder._process_single_instrument(
            _AAPL_INSTRUMENT, "NYSE"
        )
        assert result_data is None
        assert status == "error"
        assert "bad symbol" in reason
