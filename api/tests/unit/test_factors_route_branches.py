"""Branch coverage for _run_factor_compute_bg in app.api.v1.factors.factors.

Tests the background worker's lifecycle branches directly (no HTTP client,
no real DB).  The 202/409/200/404 route matrix is already covered by
tests/unit/test_factor_compute_routes.py.
"""

from __future__ import annotations

import threading
from concurrent.futures import CancelledError
from unittest.mock import MagicMock, patch

_MODULE = "app.api.v1.factors.factors"
_RUN_FN = f"{_MODULE}.run_factor_compute"
_DB_MGR = f"{_MODULE}.database_manager"
_UPDATE = f"{_MODULE}._factor_job_service.update_job"

_JOB_ID = "fjob-1"


def _make_request() -> MagicMock:
    return MagicMock()


def _import_worker():
    from app.api.v1.factors.factors import _run_factor_compute_bg

    return _run_factor_compute_bg


def _make_db_mock() -> MagicMock:
    """Return a database_manager mock whose get_session context manager works."""
    dm = MagicMock()
    dm.get_session.return_value.__enter__.return_value = MagicMock()
    dm.get_session.return_value.__exit__.return_value = False
    return dm


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _run_worker(
    run_fn_return=42,
    run_fn_side_effect=None,
    cancel_event: threading.Event | None = None,
):
    """Invoke the worker with controlled patches; return (update_calls, exc)."""
    worker = _import_worker()
    request = _make_request()
    dm = _make_db_mock()

    with (
        patch(_RUN_FN, return_value=run_fn_return, side_effect=run_fn_side_effect) as mock_run,
        patch(_DB_MGR, dm),
        patch(_UPDATE) as mock_update,
    ):
        worker(_JOB_ID, request, cancel_event=cancel_event)

    return mock_update, mock_run


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunFactorComputeBg:
    def test_success_sets_running_then_completed(self):
        mock_update, _ = _run_worker(run_fn_return=42)

        calls = mock_update.call_args_list
        statuses = [c.kwargs.get("status") or c.args[1] for c in calls]
        assert "running" in statuses
        assert "completed" in statuses

    def test_success_result_contains_rows_written(self):
        mock_update, _ = _run_worker(run_fn_return=42)

        completed_call = next(
            c for c in mock_update.call_args_list
            if (c.kwargs.get("status") or c.args[1] if len(c.args) > 1 else None) == "completed"
            or c.kwargs.get("status") == "completed"
        )
        result = completed_call.kwargs.get("result", {})
        assert result.get("rows_written") == 42

    def test_cancelled_error_does_not_call_completed_or_failed(self):
        mock_update, _ = _run_worker(run_fn_side_effect=CancelledError())

        statuses = [
            c.kwargs.get("status") or (c.args[1] if len(c.args) > 1 else "")
            for c in mock_update.call_args_list
        ]
        assert "completed" not in statuses
        assert "failed" not in statuses

    def test_exception_sets_failed_with_error(self):
        mock_update, _ = _run_worker(run_fn_side_effect=Exception("boom"))

        failed_calls = [
            c for c in mock_update.call_args_list
            if c.kwargs.get("status") == "failed"
        ]
        assert len(failed_calls) == 1
        assert failed_calls[0].kwargs.get("error") == "boom"

    def test_default_cancel_event_branch_covered(self):
        """Omitting cancel_event triggers the ``if cancel_event is None`` branch."""
        worker = _import_worker()
        request = _make_request()
        dm = _make_db_mock()

        with (
            patch(_RUN_FN, return_value=7),
            patch(_DB_MGR, dm),
            patch(_UPDATE) as mock_update,
        ):
            worker(_JOB_ID, request)  # cancel_event omitted — uses default None

        statuses = [
            c.kwargs.get("status") or (c.args[1] if len(c.args) > 1 else "")
            for c in mock_update.call_args_list
        ]
        assert "completed" in statuses

    def test_explicit_cancel_event_provided(self):
        """Passing a threading.Event covers the ``cancel_event is not None`` branch."""
        event = threading.Event()
        mock_update, _ = _run_worker(run_fn_return=99, cancel_event=event)

        statuses = [
            c.kwargs.get("status") or (c.args[1] if len(c.args) > 1 else "")
            for c in mock_update.call_args_list
        ]
        assert "completed" in statuses
