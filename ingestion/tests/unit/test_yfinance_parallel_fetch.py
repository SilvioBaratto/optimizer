"""Tests for the parallel bulk yfinance fetch path.

Patches ``database_manager.get_session`` at module level so workers receive
distinct mock sessions. Never hits Postgres.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import CancelledError
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from app.services.market_data import yfinance_data_service as svc

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
        with (
            patch.object(svc, "_run_bulk_yfinance_fetch_parallel") as parallel,
            patch("app.database.database_manager") as db_mgr,
        ):
            db_mgr.get_session.return_value.__enter__.return_value = MagicMock(
                **{
                    "execute.return_value.scalars.return_value.unique.return_value.all.return_value": []
                }
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
        repo_p, service_p, db_p = _patch_module(instruments, tracker, fetch_side_effect)

        with repo_p, service_p, db_p:
            result = svc._run_bulk_yfinance_fetch_parallel(
                request, MagicMock(), on_progress=lambda **_: None, workers=3
            )

        assert result["tickers_processed"] == 5
        assert result["error_count"] == 1
        assert len(tracker.exited) == len(
            [s for sessions in tracker.sessions_by_thread.values() for s in sessions]
        )


class TestSharedEmptyFetchMemory:
    """Bulk sweeps share one empty-answer memory so it outlives a single run."""

    def test_serial_sweep_injects_the_shared_memory(self):
        """Pass the process-wide memory to the serial sweep's service."""
        tracker = _SessionTracker()
        request = SimpleNamespace(workers=1, period="5y", mode="incremental")
        instruments = [_make_instrument("AAA")]
        repo_p, service_p, db_p = _patch_module(
            instruments, tracker, lambda **_: _ok_result()
        )

        with (
            repo_p,
            service_p as service_cls,
            db_p,
            patch.object(svc, "_backfill_instrument_isin_from_profiles"),
            patch("app.config.settings") as settings,
        ):
            settings.yfinance_request_timeout_seconds = 30
            svc.run_bulk_yfinance_fetch(request, MagicMock())

        assert service_cls.call_args.kwargs["empty_fetches"] is svc.SHARED_EMPTY_FETCHES

    def test_parallel_worker_injects_the_shared_memory(self):
        """Pass the process-wide memory to every worker's service."""
        tracker = _SessionTracker()
        request = SimpleNamespace(workers=2, period="5y", mode="incremental")
        spec = svc._InstrumentSpec(
            instrument_id=uuid4(),
            yfinance_ticker="AAA",
            exchange_name="NASDAQ",
            currency_code="USD",
        )
        _, service_p, db_p = _patch_module([], tracker, lambda **_: _ok_result())

        with service_p as service_cls, db_p:
            from app.database import database_manager

            svc._fetch_one_ticker(spec, request, MagicMock(), database_manager)

        assert service_cls.call_args.kwargs["empty_fetches"] is svc.SHARED_EMPTY_FETCHES


class TestParallelCancellation:
    """A reaped step must not keep fetching its queued tickers in the pool."""

    def test_when_progress_raises_cancelled_then_queued_tickers_are_not_fetched(
        self,
    ):
        instruments = [_make_instrument(f"T{i}") for i in range(60)]
        tracker = _SessionTracker()
        fetched: list[str] = []
        fetched_lock = threading.Lock()

        def fetch_side_effect(**kwargs):
            time.sleep(0.02)
            with fetched_lock:
                fetched.append(kwargs["yfinance_ticker"])
            return _ok_result()

        def progress(**kwargs):
            if "current" in kwargs:
                raise CancelledError("job reaped")

        request = SimpleNamespace(workers=2, period="5y", mode="incremental")
        repo_p, service_p, db_p = _patch_module(instruments, tracker, fetch_side_effect)

        with repo_p, service_p, db_p, pytest.raises(CancelledError):
            svc._run_bulk_yfinance_fetch_parallel(
                request, MagicMock(), on_progress=progress, workers=2
            )

        assert len(fetched) < len(instruments) // 2, (
            f"{len(fetched)} of {len(instruments)} queued tickers still ran"
        )


class TestParallelOutageDetection:
    """A pool of workers that Yahoo never answered must not report a clean job."""

    @staticmethod
    def _sweep(
        results: dict[str, dict[str, Any]],
    ) -> tuple[Any, list[dict[str, Any]]]:
        instruments = [_make_instrument(t) for t in results]
        tracker = _SessionTracker()
        progress: list[dict[str, Any]] = []
        progress_lock = threading.Lock()

        def on_progress(**kwargs: Any) -> None:
            with progress_lock:
                progress.append(kwargs)

        request = SimpleNamespace(workers=3, period="5y", mode="incremental")
        repo_p, service_p, db_p = _patch_module(
            instruments,
            tracker,
            lambda **kw: results[kw["yfinance_ticker"]],
        )
        with repo_p, service_p, db_p:
            try:
                outcome: Any = svc._run_bulk_yfinance_fetch_parallel(
                    request, MagicMock(), on_progress=on_progress, workers=3
                )
            except Exception as exc:
                outcome = exc
        return outcome, progress

    @staticmethod
    def _counted(failed: int, answered: int) -> dict[str, Any]:
        return {
            "counts": {},
            "errors": [],
            "skipped": [],
            "transport_failures": failed,
            "requests_answered": answered,
        }

    def test_a_sweep_yahoo_never_answered_raises_instead_of_completing(self) -> None:
        outcome, progress = self._sweep(
            {f"T{i}": self._counted(20, 0) for i in range(6)}
        )

        assert isinstance(outcome, svc.YahooUnreachableError)
        assert "120 of 120" in str(outcome)
        assert not any(
            call.get("status") in ("completed", "failed") for call in progress
        )
        assert progress[-1]["result"]["transport_failures"] == 120

    def test_a_few_failed_requests_leave_the_job_completed(self) -> None:
        outcome, progress = self._sweep(
            {f"T{i}": self._counted(1, 19) for i in range(6)}
        )

        assert outcome["transport_failures"] == 6
        assert outcome["requests_answered"] == 114
        assert progress[-1]["status"] == "completed"

    def test_results_without_request_tallies_are_not_an_outage(self) -> None:
        outcome, progress = self._sweep({"T0": _ok_result(), "T1": _ok_result()})

        assert outcome["tickers_processed"] == 2
        assert progress[-1]["status"] == "completed"


class TestParallelProgressFailure:
    def test_when_progress_write_is_cancelled_then_queued_tickers_never_start(self):
        instruments = [_make_instrument(f"T{i}") for i in range(20)]
        tracker = _SessionTracker()
        started: list[str] = []
        started_lock = threading.Lock()

        def slow_fetch(**kwargs):
            with started_lock:
                started.append(kwargs["yfinance_ticker"])
            time.sleep(0.05)
            return _ok_result()

        def cancelling_progress(**kwargs):
            if "current" in kwargs:
                raise CancelledError("job cancelled")

        request = SimpleNamespace(workers=2, period="5y", mode="incremental")
        repo_p, service_p, db_p = _patch_module(instruments, tracker, slow_fetch)

        with repo_p, service_p, db_p, pytest.raises(CancelledError):
            svc._run_bulk_yfinance_fetch_parallel(
                request, MagicMock(), on_progress=cancelling_progress, workers=2
            )

        assert len(started) <= 6

    def test_when_progress_write_fails_transiently_then_sweep_still_completes(self):
        instruments = [_make_instrument(f"T{i}") for i in range(8)]
        tracker = _SessionTracker()
        calls = {"n": 0}
        calls_lock = threading.Lock()

        def flaky_progress(**kwargs):
            if "current" not in kwargs:
                return
            with calls_lock:
                calls["n"] += 1
                first = calls["n"] == 1
            if first:
                raise RuntimeError("QueuePool limit reached")

        request = SimpleNamespace(workers=3, period="5y", mode="incremental")
        repo_p, service_p, db_p = _patch_module(
            instruments, tracker, lambda **_: _ok_result()
        )

        with repo_p, service_p, db_p:
            result = svc._run_bulk_yfinance_fetch_parallel(
                request, MagicMock(), on_progress=flaky_progress, workers=3
            )

        assert result["tickers_processed"] == 8
        assert result["counts"] == {"profile": 8}
        assert result["error_count"] == 0
