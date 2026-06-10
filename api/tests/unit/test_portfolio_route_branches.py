"""Pure-unit branch coverage for _run_sync and get_t212_client in portfolio route."""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import CancelledError
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

_MOD = "app.api.v1.portfolio.portfolio"

_T212 = f"{_MOD}.Trading212Client"
_SYNC = f"{_MOD}.sync_portfolio"
_DB_MGR = f"{_MOD}.database_manager"
_JOB_UPDATE = f"{_MOD}._sync_job_service.update_job"
_SETTINGS = f"{_MOD}.settings"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sync_result() -> MagicMock:
    r = MagicMock()
    r.positions_synced = 1
    r.positions_removed = 0
    r.account_synced = True
    r.orders_fetched = 0
    r.dividends_fetched = 0
    r.errors = []
    return r


def _make_db_manager() -> MagicMock:
    dm = MagicMock()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=MagicMock())
    cm.__exit__ = MagicMock(return_value=False)
    dm.get_session.return_value = cm
    return dm


def _has_status_call(mock_update: MagicMock, status: str) -> bool:
    for c in mock_update.call_args_list:
        if c.kwargs.get("status") == status or (len(c.args) > 1 and c.args[1] == status):
            return True
    return False


# ---------------------------------------------------------------------------
# _run_sync tests
# ---------------------------------------------------------------------------


class TestRunSync:
    def _run(self, job_id: str, portfolio_id: str, *, cancel_event=None, **kwargs):
        from app.api.v1.portfolio.portfolio import _run_sync

        if cancel_event is not None:
            _run_sync(job_id, portfolio_id, "live", cancel_event=cancel_event, **kwargs)
        else:
            _run_sync(job_id, portfolio_id, "live", **kwargs)

    def test_success_updates_running_then_completed(self):
        job_id = "job-1"
        portfolio_id = str(uuid.uuid4())
        with ExitStack() as stack:
            stack.enter_context(patch(_T212))
            stack.enter_context(patch(_SYNC, return_value=_make_sync_result()))
            stack.enter_context(patch(_DB_MGR, _make_db_manager()))
            mock_update = stack.enter_context(patch(_JOB_UPDATE))
            self._run(job_id, portfolio_id)

        assert _has_status_call(mock_update, "running"), "expected running call"
        assert _has_status_call(mock_update, "completed"), "expected completed call"
        completed_calls = [
            c for c in mock_update.call_args_list if c.kwargs.get("status") == "completed"
        ]
        result = completed_calls[0].kwargs.get("result")
        assert result is not None
        assert result["positions_synced"] == 1

    def test_cancelled_error_returns_silently_no_terminal_status(self):
        job_id = "job-2"
        portfolio_id = str(uuid.uuid4())
        with ExitStack() as stack:
            stack.enter_context(patch(_T212))
            stack.enter_context(patch(_SYNC, side_effect=CancelledError()))
            stack.enter_context(patch(_DB_MGR, _make_db_manager()))
            mock_update = stack.enter_context(patch(_JOB_UPDATE))
            self._run(job_id, portfolio_id)

        terminal_statuses = {"completed", "failed"}
        assert not any(_has_status_call(mock_update, s) for s in terminal_statuses)

    def test_generic_exception_updates_failed_with_error_message(self):
        job_id = "job-3"
        portfolio_id = str(uuid.uuid4())
        with ExitStack() as stack:
            stack.enter_context(patch(_T212))
            stack.enter_context(patch(_SYNC, side_effect=Exception("boom")))
            stack.enter_context(patch(_DB_MGR, _make_db_manager()))
            mock_update = stack.enter_context(patch(_JOB_UPDATE))
            self._run(job_id, portfolio_id)

        assert _has_status_call(mock_update, "failed"), "expected failed call"
        failed_calls = [
            c for c in mock_update.call_args_list if c.kwargs.get("status") == "failed"
        ]
        assert failed_calls[0].kwargs.get("error") == "boom"

    def test_cancel_event_provided_covers_not_none_branch(self):
        job_id = "job-4"
        portfolio_id = str(uuid.uuid4())
        cancel_event = threading.Event()
        with ExitStack() as stack:
            stack.enter_context(patch(_T212))
            stack.enter_context(patch(_SYNC, return_value=_make_sync_result()))
            stack.enter_context(patch(_DB_MGR, _make_db_manager()))
            mock_update = stack.enter_context(patch(_JOB_UPDATE))
            self._run(job_id, portfolio_id, cancel_event=cancel_event)

        assert _has_status_call(mock_update, "completed")


# ---------------------------------------------------------------------------
# get_t212_client success path
# ---------------------------------------------------------------------------


class TestGetT212ClientSuccess:
    def test_returns_trading212_client_when_api_key_present(self):
        mock_settings = MagicMock()
        mock_settings.trading_212_api_key = "valid-key"
        mock_settings.trading_212_secret_key = "secret"  # noqa: S105 (test dummy)
        mock_settings.trading_212_mode = "live"

        mock_client_instance = MagicMock()
        mock_client_cls = MagicMock(return_value=mock_client_instance)

        with ExitStack() as stack:
            stack.enter_context(patch(_SETTINGS, mock_settings))
            stack.enter_context(patch(_T212, mock_client_cls))
            from app.api.v1.portfolio.portfolio import get_t212_client

            result = get_t212_client()

        assert result is mock_client_instance
