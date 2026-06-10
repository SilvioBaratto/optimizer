"""Branch-coverage tests for market_data route wrappers (missed lines).

yfinance_data.py:   lines 74-84 (_run_bulk_fetch success + exception), line 98 (poll result).
reference_indices.py: lines 62-72 (_run_seed success + exception), line 86 (poll result).

Does NOT duplicate POST 202/409 or GET 200/404 matrices already covered by
test_yfinance_data_routes.py / test_reference_indices_routes.py.
"""

from __future__ import annotations

from concurrent.futures import CancelledError
from unittest.mock import MagicMock, patch

from tests._fixtures.job_service_mock import mock_job_service

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

_YF_MOD = "app.api.v1.market_data.yfinance_data"
_YF_UPDATE = f"{_YF_MOD}._job_service.update_job"
_YF_JOB = f"{_YF_MOD}._job_service"
_BULK_FN = f"{_YF_MOD}.run_bulk_yfinance_fetch"
_MAKE_PROG = f"{_YF_MOD}.make_cancellable_progress"

_RI_MOD = "app.api.v1.market_data.reference_indices"
_RI_UPDATE = f"{_RI_MOD}._job_service.update_job"
_RI_JOB = f"{_RI_MOD}._job_service"
_SEED_FN = f"{_RI_MOD}.seed_reference_indices"
_RI_MAKE_PROG = f"{_RI_MOD}.make_cancellable_progress"


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _noop_progress(*_args, **_kwargs):
    """Stand-in for make_cancellable_progress — returns a no-op callable."""
    return lambda **kw: None


# ---------------------------------------------------------------------------
# _run_bulk_fetch — success path (lines 74-84)
# ---------------------------------------------------------------------------


class TestRunBulkFetchSuccess:
    """Lines 74-84: wrapper sets running, calls run_bulk_yfinance_fetch, no error."""

    def test_status_running_set_before_fetch(self) -> None:
        from app.api.v1.market_data.yfinance_data import _run_bulk_fetch
        from app.schemas.market_data.yfinance_data import YFinanceFetchRequest

        request = YFinanceFetchRequest()

        with (
            patch(_YF_UPDATE) as mock_update,
            patch(_MAKE_PROG, side_effect=_noop_progress),
            patch(_BULK_FN),
        ):
            _run_bulk_fetch("job-yf-1", request, MagicMock())

        assert mock_update.call_args_list[0].kwargs.get("status") == "running"

    def test_bulk_fetch_function_called_once(self) -> None:
        from app.api.v1.market_data.yfinance_data import _run_bulk_fetch
        from app.schemas.market_data.yfinance_data import YFinanceFetchRequest

        request = YFinanceFetchRequest()

        with (
            patch(_YF_UPDATE),
            patch(_MAKE_PROG, side_effect=_noop_progress),
            patch(_BULK_FN) as mock_bulk,
        ):
            _run_bulk_fetch("job-yf-2", request, MagicMock())

        mock_bulk.assert_called_once()


# ---------------------------------------------------------------------------
# _run_bulk_fetch — exception path (lines 82-89)
# ---------------------------------------------------------------------------


class TestRunBulkFetchFailure:
    """Lines 82-89: service raises generic Exception → update_job(failed, error=...)."""

    def test_exception_sets_failed_with_error_message(self) -> None:
        from app.api.v1.market_data.yfinance_data import _run_bulk_fetch
        from app.schemas.market_data.yfinance_data import YFinanceFetchRequest

        request = YFinanceFetchRequest()

        with (
            patch(_YF_UPDATE) as mock_update,
            patch(_MAKE_PROG, side_effect=_noop_progress),
            patch(_BULK_FN, side_effect=Exception("network down")),
        ):
            _run_bulk_fetch("job-yf-fail", request, MagicMock())

        failed = next(
            c for c in mock_update.call_args_list if c.kwargs.get("status") == "failed"
        )
        assert failed.kwargs["error"] == "network down"

    def test_cancelled_error_returns_early_without_failed_update(self) -> None:
        """CancelledError: reaper owns terminal status — wrapper must not write failed."""
        from app.api.v1.market_data.yfinance_data import _run_bulk_fetch
        from app.schemas.market_data.yfinance_data import YFinanceFetchRequest

        request = YFinanceFetchRequest()

        with (
            patch(_YF_UPDATE) as mock_update,
            patch(_MAKE_PROG, side_effect=_noop_progress),
            patch(_BULK_FN, side_effect=CancelledError()),
        ):
            _run_bulk_fetch("job-yf-cancel", request, MagicMock())

        statuses = [c.kwargs.get("status") for c in mock_update.call_args_list]
        assert "failed" not in statuses


# ---------------------------------------------------------------------------
# _run_bulk_fetch — poll result branch (line 98 in yfinance_data.py)
# ---------------------------------------------------------------------------


class TestYFinancePollResultBranch:
    """Line 98: GET /fetch/{job_id} with result dict → result field populated."""

    def test_when_job_has_result_then_result_in_response(self, client) -> None:
        result_payload = {"tickers_processed": 10, "errors": 0}

        with mock_job_service(
            _YF_JOB,
            job_id="yf-done",
            initial_status="completed",
            extra={"result": result_payload, "current_ticker": "AAPL"},
        ):
            resp = client.get("/api/v1/yfinance-data/fetch/yf-done")

        assert resp.status_code == 200
        assert resp.json()["result"] == result_payload


# ---------------------------------------------------------------------------
# _run_seed — success path (lines 62-72 reference_indices.py)
# ---------------------------------------------------------------------------


class TestRunSeedSuccess:
    """Lines 62-72: wrapper sets running, calls seed_reference_indices, no error."""

    def test_status_running_set_before_seed(self) -> None:
        from app.api.v1.market_data.reference_indices import _run_seed
        from app.schemas.market_data.reference_index import ReferenceIndexSeedRequest

        request = ReferenceIndexSeedRequest(tickers=["SPY"])

        with (
            patch(_RI_UPDATE) as mock_update,
            patch(_RI_MAKE_PROG, side_effect=_noop_progress),
            patch(_SEED_FN),
        ):
            _run_seed("job-ri-1", request, MagicMock())

        assert mock_update.call_args_list[0].kwargs.get("status") == "running"

    def test_seed_function_called_with_tickers(self) -> None:
        from app.api.v1.market_data.reference_indices import _run_seed
        from app.schemas.market_data.reference_index import ReferenceIndexSeedRequest

        request = ReferenceIndexSeedRequest(tickers=["SPY", "QQQ"])

        with (
            patch(_RI_UPDATE),
            patch(_RI_MAKE_PROG, side_effect=_noop_progress),
            patch(_SEED_FN) as mock_seed,
        ):
            _run_seed("job-ri-2", request, MagicMock())

        mock_seed.assert_called_once()
        call_tickers = mock_seed.call_args.args[0]
        assert call_tickers == ["SPY", "QQQ"]


# ---------------------------------------------------------------------------
# _run_seed — exception path (lines 70-77)
# ---------------------------------------------------------------------------


class TestRunSeedFailure:
    """Lines 70-77: seeder raises generic Exception → update_job(failed, error=...)."""

    def test_exception_sets_failed_with_error_message(self) -> None:
        from app.api.v1.market_data.reference_indices import _run_seed
        from app.schemas.market_data.reference_index import ReferenceIndexSeedRequest

        request = ReferenceIndexSeedRequest(tickers=["SPY"])

        with (
            patch(_RI_UPDATE) as mock_update,
            patch(_RI_MAKE_PROG, side_effect=_noop_progress),
            patch(_SEED_FN, side_effect=Exception("yfinance timeout")),
        ):
            _run_seed("job-ri-fail", request, MagicMock())

        failed = next(
            c for c in mock_update.call_args_list if c.kwargs.get("status") == "failed"
        )
        assert failed.kwargs["error"] == "yfinance timeout"

    def test_cancelled_error_returns_without_failed_update(self) -> None:
        """CancelledError: reaper owns terminal status — seed wrapper must not write failed."""
        from app.api.v1.market_data.reference_indices import _run_seed
        from app.schemas.market_data.reference_index import ReferenceIndexSeedRequest

        request = ReferenceIndexSeedRequest(tickers=["SPY"])

        with (
            patch(_RI_UPDATE) as mock_update,
            patch(_RI_MAKE_PROG, side_effect=_noop_progress),
            patch(_SEED_FN, side_effect=CancelledError()),
        ):
            _run_seed("job-ri-cancel", request, MagicMock())

        statuses = [c.kwargs.get("status") for c in mock_update.call_args_list]
        assert "failed" not in statuses


# ---------------------------------------------------------------------------
# _run_seed — poll result branch (line 86 in reference_indices.py)
# ---------------------------------------------------------------------------


class TestRefIndexPollResultBranch:
    """Line 86 (get_seed_status): job dict with truthy result → result in response."""

    def test_when_job_has_result_then_result_in_response(self, client) -> None:
        result_payload = {"tickers_seeded": 5}

        with mock_job_service(
            _RI_JOB,
            job_id="ri-done",
            initial_status="completed",
            extra={"result": result_payload, "current_ticker": "SPY"},
        ):
            resp = client.get("/api/v1/reference-indices/seed/ri-done")

        assert resp.status_code == 200
        assert resp.json()["result"] == result_payload
