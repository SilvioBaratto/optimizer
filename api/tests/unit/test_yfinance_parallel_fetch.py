"""Tests for the parallel bulk yfinance fetch path (issue #575).

Patches ``database_manager.get_session`` at module level so workers receive
distinct mock sessions. Never hits Postgres.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

from app.services import yfinance_data_service as svc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_instrument(ticker: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid4(),
        yfinance_ticker=ticker,
        exchange_name="NASDAQ",
        currency_code="USD",
    )


def _ok_result() -> dict[str, Any]:
    return {"counts": {"profile": 1}, "errors": [], "skipped": []}


class _SessionTracker:
    """Records every session opened, indexed by the thread that opened it."""

    def __init__(self) -> None:
        self.sessions_by_thread: dict[int, list[MagicMock]] = {}
        self.exited: list[MagicMock] = []
        self._lock = threading.Lock()

    @contextmanager
    def get_session(self):
        session = MagicMock(name="session")
        tid = threading.get_ident()
        with self._lock:
            self.sessions_by_thread.setdefault(tid, []).append(session)
        try:
            yield session
        finally:
            with self._lock:
                self.exited.append(session)


def _patch_module(
    instruments: list[SimpleNamespace],
    tracker: _SessionTracker,
    fetch_side_effect: Any,
):
    """Patch the module-level dependencies used by the parallel fetcher."""
    repo_mock = MagicMock()
    repo_mock.get_instruments_with_yfinance_ticker.return_value = instruments

    service_mock = MagicMock()
    service_mock.fetch_and_store.side_effect = fetch_side_effect

    db_mgr = SimpleNamespace(get_session=tracker.get_session)
    return (
        patch.object(svc, "YFinanceRepository", return_value=repo_mock),
        patch.object(svc, "YFinanceDataService", return_value=service_mock),
        patch("app.database.database_manager", db_mgr),
    )


# ---------------------------------------------------------------------------
# Acceptance criteria
# ---------------------------------------------------------------------------


class TestParallelDispatch:
    def test_when_workers_is_one_then_parallel_function_not_called(self):
        request = SimpleNamespace(workers=1, period="5y", mode="incremental")
        with patch.object(svc, "_run_bulk_yfinance_fetch_parallel") as parallel, \
             patch("app.database.database_manager") as db_mgr:
            db_mgr.get_session.return_value.__enter__.return_value = MagicMock(
                **{"execute.return_value.scalars.return_value.unique.return_value.all.return_value": []}
            )
            db_mgr.get_session.return_value.__exit__.return_value = False
            repo = MagicMock()
            repo.get_instruments_with_yfinance_ticker.return_value = []
            with patch.object(svc, "YFinanceRepository", return_value=repo):
                svc.run_bulk_yfinance_fetch(request, MagicMock())
        parallel.assert_not_called()

    def test_when_workers_above_one_then_parallel_function_invoked(self):
        request = SimpleNamespace(workers=4, period="5y", mode="incremental")
        with patch.object(
            svc, "_run_bulk_yfinance_fetch_parallel", return_value={"ok": True}
        ) as parallel:
            result = svc.run_bulk_yfinance_fetch(request, MagicMock())
        assert result == {"ok": True}
        parallel.assert_called_once()
        kwargs = parallel.call_args.kwargs
        assert kwargs["workers"] == 4


class TestParallelExecution:
    def test_when_twenty_tickers_run_then_counter_is_atomic_with_no_double_writes(self):
        instruments = [_make_instrument(f"T{i}") for i in range(20)]
        tracker = _SessionTracker()
        progress_calls: list[dict[str, Any]] = []
        progress_lock = threading.Lock()

        def progress(**kwargs):
            with progress_lock:
                progress_calls.append(kwargs)

        request = SimpleNamespace(workers=8, period="5y", mode="incremental")
        repo_p, service_p, db_p = _patch_module(
            instruments, tracker, lambda **_: _ok_result()
        )

        with repo_p, service_p, db_p:
            result = svc._run_bulk_yfinance_fetch_parallel(
                request, MagicMock(), on_progress=progress, workers=8
            )

        progress_currents = [c["current"] for c in progress_calls if "current" in c]
        assert sorted(progress_currents) == list(range(1, 21))
        assert result["tickers_processed"] == 20
        assert result["error_count"] == 0

    def test_when_workers_run_then_each_uses_its_own_session(self):
        instruments = [_make_instrument(f"T{i}") for i in range(8)]
        tracker = _SessionTracker()
        request = SimpleNamespace(workers=4, period="5y", mode="incremental")

        repo_p, service_p, db_p = _patch_module(
            instruments, tracker, lambda **_: _ok_result()
        )
        with repo_p, service_p, db_p:
            svc._run_bulk_yfinance_fetch_parallel(
                request, MagicMock(), on_progress=lambda **_: None, workers=4
            )

        worker_sessions = [
            s for sessions in tracker.sessions_by_thread.values() for s in sessions
        ]
        assert len(worker_sessions) >= 8
        assert len({id(s) for s in worker_sessions}) == len(worker_sessions)
        assert all(s in tracker.exited for s in worker_sessions)

    def test_when_one_ticker_raises_then_error_isolated_and_sessions_close(self):
        instruments = [_make_instrument(f"T{i}") for i in range(5)]
        tracker = _SessionTracker()

        def fetch_side_effect(**kwargs):
            if kwargs["yfinance_ticker"] == "T2":
                raise RuntimeError("yahoo 500")
            return _ok_result()

        request = SimpleNamespace(workers=3, period="5y", mode="incremental")
        repo_p, service_p, db_p = _patch_module(
            instruments, tracker, fetch_side_effect
        )

        with repo_p, service_p, db_p:
            result = svc._run_bulk_yfinance_fetch_parallel(
                request, MagicMock(), on_progress=lambda **_: None, workers=3
            )

        assert result["tickers_processed"] == 5
        assert result["error_count"] == 1
        assert len(tracker.exited) == len(
            [s for sessions in tracker.sessions_by_thread.values() for s in sessions]
        )
