"""Branch-coverage tests for app.api.v1.universe.trading212 (missed lines 68-128, 151, 159, 256).

Targets:
- _run_build: success path (completed + result), skip_filters=True branch, exception path.
- get_trading212_client: success path (api key present → returns client).
- get_ticker_cache: always returns a TickerMappingCache instance.
- get_build_status result branch (line 256): job dict with truthy result →
  BuildResultResponse constructed.

Does NOT duplicate the 202/409/200/404 matrices in test_trading212_routes.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tests._fixtures.job_service_mock import mock_job_service

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

_MOD = "app.api.v1.universe.trading212"
_UPDATE_JOB = f"{_MOD}._job_service.update_job"
_DB_CTX = f"{_MOD}.database_manager.get_session"
_JOB = f"{_MOD}._job_service"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _db_ctx_mock(session: MagicMock | None = None) -> MagicMock:
    """Return a MagicMock configured as a context manager yielding *session*."""
    sess = session or MagicMock()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=sess)
    cm.__exit__ = MagicMock(return_value=False)
    parent = MagicMock(return_value=cm)
    return parent


def _build_result_mock(
    exchanges_saved: int = 1,
    instruments_saved: int = 2,
    total_processed: int = 3,
) -> MagicMock:
    r = MagicMock()
    r.exchanges_saved = exchanges_saved
    r.instruments_saved = instruments_saved
    r.total_processed = total_processed
    r.filter_stats = {}
    r.errors = []
    return r


def _make_request(skip_filters: bool = False) -> MagicMock:
    req = MagicMock()
    req.skip_filters = skip_filters
    req.max_workers = 2
    req.exchanges = None
    return req


# ---------------------------------------------------------------------------
# _run_build — success path (filters applied)
# ---------------------------------------------------------------------------


class TestRunBuildSuccess:
    """Lines 68-124: happy path sets completed with result fields."""

    def test_completed_result_has_instruments_saved(self) -> None:
        from app.api.v1.universe.trading212 import _run_build

        build_result = _build_result_mock(instruments_saved=5)

        with (
            patch(_UPDATE_JOB) as mock_update,
            patch(_DB_CTX, new=_db_ctx_mock()),
            patch(f"{_MOD}.UniverseRepository"),
            patch(f"{_MOD}.YFinanceTickerMapper"),
            patch(f"{_MOD}.FilterPipelineImpl"),
            patch(f"{_MOD}.MarketCapFilter"),
            patch(f"{_MOD}.PriceFilter"),
            patch(f"{_MOD}.LiquidityFilter"),
            patch(f"{_MOD}.DataCoverageFilter"),
            patch(f"{_MOD}.HistoricalDataFilter"),
            patch(f"{_MOD}.UniverseBuilder") as MockBuilder,
        ):
            MockBuilder.return_value.build.return_value = build_result
            _run_build("bid-ok", _make_request(), MagicMock(), MagicMock(), MagicMock())

        completed = next(
            c for c in mock_update.call_args_list if c.kwargs.get("status") == "completed"
        )
        assert completed.kwargs["result"]["instruments_saved"] == 5

    def test_status_running_set_before_build(self) -> None:
        from app.api.v1.universe.trading212 import _run_build

        build_result = _build_result_mock()

        with (
            patch(_UPDATE_JOB) as mock_update,
            patch(_DB_CTX, new=_db_ctx_mock()),
            patch(f"{_MOD}.UniverseRepository"),
            patch(f"{_MOD}.YFinanceTickerMapper"),
            patch(f"{_MOD}.FilterPipelineImpl"),
            patch(f"{_MOD}.MarketCapFilter"),
            patch(f"{_MOD}.PriceFilter"),
            patch(f"{_MOD}.LiquidityFilter"),
            patch(f"{_MOD}.DataCoverageFilter"),
            patch(f"{_MOD}.HistoricalDataFilter"),
            patch(f"{_MOD}.UniverseBuilder") as MockBuilder,
        ):
            MockBuilder.return_value.build.return_value = build_result
            _run_build("bid-run", _make_request(), MagicMock(), MagicMock(), MagicMock())

        assert mock_update.call_args_list[0].kwargs.get("status") == "running"


# ---------------------------------------------------------------------------
# _run_build — skip_filters=True branch (line 76)
# ---------------------------------------------------------------------------


class TestRunBuildSkipFilters:
    """Line 76: when skip_filters=True, no filters are added to the pipeline."""

    def test_when_skip_filters_then_no_filter_added(self) -> None:
        from app.api.v1.universe.trading212 import _run_build

        build_result = _build_result_mock()

        with (
            patch(_UPDATE_JOB),
            patch(_DB_CTX, new=_db_ctx_mock()),
            patch(f"{_MOD}.UniverseRepository"),
            patch(f"{_MOD}.YFinanceTickerMapper"),
            patch(f"{_MOD}.FilterPipelineImpl"),
            patch(f"{_MOD}.MarketCapFilter") as MockMC,
            patch(f"{_MOD}.PriceFilter") as MockPF,
            patch(f"{_MOD}.LiquidityFilter") as MockLF,
            patch(f"{_MOD}.DataCoverageFilter") as MockDC,
            patch(f"{_MOD}.HistoricalDataFilter") as MockHD,
            patch(f"{_MOD}.UniverseBuilder") as MockBuilder,
        ):
            MockBuilder.return_value.build.return_value = build_result
            _run_build(
                "bid-skip",
                _make_request(skip_filters=True),
                MagicMock(),
                MagicMock(),
                MagicMock(),
            )

        # None of the filter constructors should have been called
        for mock_filter in (MockMC, MockPF, MockLF, MockDC, MockHD):
            mock_filter.assert_not_called()


# ---------------------------------------------------------------------------
# _run_build — exception path (lines 126-133)
# ---------------------------------------------------------------------------


class TestRunBuildFailure:
    """Lines 126-133: build() raises → update_job(status="failed", error=...)."""

    def test_exception_sets_failed_status(self) -> None:
        from app.api.v1.universe.trading212 import _run_build

        with (
            patch(_UPDATE_JOB) as mock_update,
            patch(_DB_CTX, new=_db_ctx_mock()),
            patch(f"{_MOD}.UniverseRepository"),
            patch(f"{_MOD}.YFinanceTickerMapper"),
            patch(f"{_MOD}.FilterPipelineImpl"),
            patch(f"{_MOD}.MarketCapFilter"),
            patch(f"{_MOD}.PriceFilter"),
            patch(f"{_MOD}.LiquidityFilter"),
            patch(f"{_MOD}.DataCoverageFilter"),
            patch(f"{_MOD}.HistoricalDataFilter"),
            patch(f"{_MOD}.UniverseBuilder") as MockBuilder,
        ):
            MockBuilder.return_value.build.side_effect = Exception("network error")
            _run_build("bid-err", _make_request(), MagicMock(), MagicMock(), MagicMock())

        failed = next(
            c for c in mock_update.call_args_list if c.kwargs.get("status") == "failed"
        )
        assert failed.kwargs["status"] == "failed"
        assert failed.kwargs["error"] == "network error"


# ---------------------------------------------------------------------------
# get_trading212_client — success path (line 151)
# ---------------------------------------------------------------------------


class TestGetTrading212ClientSuccess:
    """Line 151: api key truthy → returns Trading212Client instance."""

    def test_when_api_key_set_then_client_returned(self) -> None:
        from app.api.v1.universe.trading212 import get_trading212_client

        mock_client = MagicMock()

        with (
            patch(f"{_MOD}.settings") as mock_settings,
            patch(f"{_MOD}.Trading212Client", return_value=mock_client),
        ):
            mock_settings.trading_212_api_key = "test-key-123"
            mock_settings.trading_212_secret_key = "test-secret"  # noqa: S105
            mock_settings.trading_212_mode = "demo"

            result = get_trading212_client()

        assert result is mock_client


# ---------------------------------------------------------------------------
# get_ticker_cache — always returns instance (line 159)
# ---------------------------------------------------------------------------


class TestGetTickerCache:
    """Line 159: get_ticker_cache() returns a TickerMappingCache."""

    def test_returns_ticker_mapping_cache_instance(self) -> None:
        from app.api.v1.universe.trading212 import get_ticker_cache
        from app.services.universe.trading212.cache.ticker_cache import (
            TickerMappingCache,
        )

        result = get_ticker_cache()

        assert isinstance(result, TickerMappingCache)


# ---------------------------------------------------------------------------
# get_build_status — result branch (line 256)
# ---------------------------------------------------------------------------


class TestBuildStatusResultBranch:
    """Line 256: job dict with truthy result → BuildResultResponse constructed."""

    def test_when_result_present_then_result_populated_in_response(
        self, client
    ) -> None:
        result_dict = {
            "exchanges_saved": 3,
            "instruments_saved": 42,
            "total_processed": 50,
            "filter_stats": {},
            "errors": [],
        }

        with mock_job_service(
            _JOB,
            job_id="bid-done",
            initial_status="completed",
            extra={
                "result": result_dict,
                "current_exchange": "",
                "current_stock": "",
            },
        ):
            resp = client.get("/api/v1/universe/build/bid-done")

        assert resp.status_code == 200
        body = resp.json()
        assert body["result"] is not None
        assert body["result"]["instruments_saved"] == 42
        assert body["result"]["exchanges_saved"] == 3
