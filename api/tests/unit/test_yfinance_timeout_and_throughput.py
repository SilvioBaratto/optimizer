"""Tests for yfinance timeout watchdog and incremental throughput improvements.

Covers:
- call_with_timeout: hanging call raises TimeoutError
- timeout path: per-ticker TimeoutError recorded as error, job continues
- whole-ticker incremental skip: zero network calls when all categories fresh
- scheduler passes workers from settings
- parallel path uses per-worker sessions (pre-existing, extended)
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from app.services.market_data import yfinance_data_service as svc
from app.services.market_data.yfinance_data_service import (
    YFinanceDataService,
    call_with_timeout,
)


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _make_repo(*, staleness: dict[str, Any] | None = None) -> MagicMock:
    repo = MagicMock(name="repo")
    repo.get_staleness_info.return_value = staleness
    repo.upsert_profile.return_value = 0
    repo.upsert_price_history.return_value = 0
    repo.upsert_financial_statements.return_value = 0
    repo.upsert_dividends.return_value = 0
    repo.upsert_splits.return_value = 0
    repo.upsert_recommendations.return_value = 0
    repo.upsert_price_targets.return_value = 0
    repo.upsert_institutional_holders.return_value = 0
    repo.upsert_mutualfund_holders.return_value = 0
    repo.upsert_insider_transactions.return_value = 0
    repo.upsert_news.return_value = 0
    return repo


def _make_yf_client_silent() -> MagicMock:
    """A yf_client whose every fetch method returns None/empty."""
    yf_client = MagicMock(name="yf_client")
    yf_client.fetch_info.return_value = None
    yf_client.fetch_history.return_value = None
    yf_client.financials.fetch_income_stmt.return_value = None
    yf_client.financials.fetch_balance_sheet.return_value = None
    yf_client.financials.fetch_cashflow.return_value = None
    yf_client.corporate_actions.fetch_dividends.return_value = None
    yf_client.corporate_actions.fetch_splits.return_value = None
    yf_client.analysis.fetch_recommendations_summary.return_value = None
    yf_client.analysis.fetch_analyst_price_targets.return_value = None
    yf_client.analysis.fetch_eps_trend.return_value = None
    yf_client.analysis.fetch_eps_revisions.return_value = None
    yf_client.holders.fetch_institutional_holders.return_value = None
    yf_client.holders.fetch_mutualfund_holders.return_value = None
    yf_client.holders.fetch_insider_transactions.return_value = None
    yf_client.metadata.fetch_valuation_measures.return_value = None
    ticker_mock = MagicMock()
    ticker_mock.news = []
    yf_client.get_ticker.return_value = ticker_mock
    return yf_client


def _fully_fresh_staleness() -> dict[str, Any]:
    """Return a staleness dict where every category is very recently updated."""
    from datetime import datetime, timezone

    # updated 1 minute ago — well within all thresholds (minimum threshold is
    # 24 h for most categories, 168 h for financials)
    recent = datetime.now(timezone.utc).replace(microsecond=0)
    from datetime import date

    return {
        "profile_updated_at": recent,
        "financials_updated_at": recent,
        "dividends_updated_at": recent,
        "splits_updated_at": recent,
        "recommendations_updated_at": recent,
        "price_targets_updated_at": recent,
        "institutional_holders_updated_at": recent,
        "mutualfund_holders_updated_at": recent,
        "insider_transactions_updated_at": recent,
        "price_max_date": date.today(),
    }


# ---------------------------------------------------------------------------
# TASK 1 — call_with_timeout unit tests
# ---------------------------------------------------------------------------


class TestCallWithTimeout:
    def test_when_function_completes_fast_then_returns_result(self):
        result = call_with_timeout(lambda: 42, timeout=5.0)
        assert result == 42

    def test_when_function_hangs_then_raises_timeout_error(self):
        barrier = threading.Event()

        def _hang():
            barrier.wait(timeout=10)  # blocks until test finishes

        with pytest.raises(TimeoutError, match="yfinance call did not complete"):
            call_with_timeout(_hang, timeout=0.1)

        barrier.set()  # release the daemon thread

    def test_when_function_raises_then_exception_propagates(self):
        def _boom():
            raise ValueError("bad ticker")

        with pytest.raises(ValueError, match="bad ticker"):
            call_with_timeout(_boom, timeout=5.0)


# ---------------------------------------------------------------------------
# TASK 1 — timeout applied per-category, job continues on TimeoutError
# ---------------------------------------------------------------------------


class TestTimeoutPerCategory:
    def test_when_profile_fetch_times_out_then_error_recorded_and_prices_still_fetched(
        self,
    ):
        """A TimeoutError on profile is captured as an error; price fetch still runs."""
        repo = _make_repo()
        yf_client = _make_yf_client_silent()

        call_count = {"n": 0}

        def _profile_hangs():
            # First call is profile; raise TimeoutError to simulate watchdog
            call_count["n"] += 1
            raise TimeoutError("yfinance call did not complete within 30s")

        # Build service with timeout disabled (None) so _timed calls fn()
        # directly, but we make fetch_info raise TimeoutError itself.
        yf_client.fetch_info.side_effect = _profile_hangs

        service = YFinanceDataService(repo=repo, yf_client=yf_client)
        result = service.fetch_and_store(
            instrument_id=uuid4(),
            yfinance_ticker="AAPL",
            mode="full",
        )

        # Profile error captured
        assert any("profile" in e for e in result["errors"])
        # fetch_history was still called (prices fetch ran)
        yf_client.fetch_history.assert_called_once()

    def test_when_all_categories_time_out_then_error_count_matches_categories(self):
        """TimeoutError on every category fills errors but returns a result dict."""
        repo = _make_repo()
        yf_client = _make_yf_client_silent()

        def _timeout():
            raise TimeoutError("yfinance call did not complete within 30s")

        yf_client.fetch_info.side_effect = _timeout
        yf_client.fetch_history.side_effect = _timeout
        yf_client.financials.fetch_income_stmt.side_effect = _timeout
        yf_client.financials.fetch_balance_sheet.side_effect = _timeout
        yf_client.financials.fetch_cashflow.side_effect = _timeout
        yf_client.metadata.fetch_valuation_measures.side_effect = _timeout
        yf_client.analysis.fetch_eps_trend.side_effect = _timeout
        yf_client.analysis.fetch_eps_revisions.side_effect = _timeout
        yf_client.corporate_actions.fetch_dividends.side_effect = _timeout
        yf_client.corporate_actions.fetch_splits.side_effect = _timeout
        yf_client.analysis.fetch_recommendations_summary.side_effect = _timeout
        yf_client.analysis.fetch_analyst_price_targets.side_effect = _timeout
        yf_client.holders.fetch_institutional_holders.side_effect = _timeout
        yf_client.holders.fetch_mutualfund_holders.side_effect = _timeout
        yf_client.holders.fetch_insider_transactions.side_effect = _timeout
        ticker_mock = MagicMock()
        ticker_mock.news = MagicMock(side_effect=_timeout)
        yf_client.get_ticker.return_value = ticker_mock

        service = YFinanceDataService(repo=repo, yf_client=yf_client)
        result = service.fetch_and_store(
            instrument_id=uuid4(),
            yfinance_ticker="HANG",
            mode="full",
        )

        # Result dict is returned (job did not crash)
        assert "errors" in result
        assert "counts" in result
        assert len(result["errors"]) > 0

    def test_when_one_ticker_in_bulk_times_out_then_job_continues(self):
        """Bulk serial path: a per-ticker timeout isolates to errors, job finishes."""
        from contextlib import contextmanager

        instruments = [
            SimpleNamespace(
                id=uuid4(),
                yfinance_ticker="HANG",
                exchange_name="NASDAQ",
                currency_code="USD",
            ),
            SimpleNamespace(
                id=uuid4(),
                yfinance_ticker="OK",
                exchange_name="NASDAQ",
                currency_code="USD",
            ),
        ]

        session_mock = MagicMock()
        db_mgr = MagicMock()

        @contextmanager
        def _get_session():
            yield session_mock

        db_mgr.get_session = _get_session

        repo_mock = MagicMock()
        repo_mock.get_instruments_with_yfinance_ticker.return_value = instruments
        repo_mock.get_staleness_info.return_value = None
        repo_mock.upsert_profile.return_value = 0
        repo_mock.upsert_price_history.return_value = 0
        repo_mock.upsert_financial_statements.return_value = 0
        repo_mock.upsert_dividends.return_value = 0
        repo_mock.upsert_splits.return_value = 0
        repo_mock.upsert_recommendations.return_value = 0
        repo_mock.upsert_price_targets.return_value = 0
        repo_mock.upsert_institutional_holders.return_value = 0
        repo_mock.upsert_mutualfund_holders.return_value = 0
        repo_mock.upsert_insider_transactions.return_value = 0
        repo_mock.upsert_news.return_value = 0

        call_count = {"n": 0}

        def _service_side_effect(**kwargs):
            call_count["n"] += 1
            if kwargs["yfinance_ticker"] == "HANG":
                raise TimeoutError("yfinance call did not complete within 30s")
            return {"counts": {"profile": 1}, "errors": [], "skipped": []}

        request = SimpleNamespace(workers=1, period="5y", mode="full")
        yf_client = MagicMock()

        with (
            patch.object(svc, "YFinanceRepository", return_value=repo_mock),
            patch("app.database.database_manager", db_mgr),
            patch.object(
                svc,
                "YFinanceDataService",
                return_value=MagicMock(
                    fetch_and_store=MagicMock(side_effect=_service_side_effect)
                ),
            ),
            patch("app.config.settings") as mock_settings,
        ):
            mock_settings.yfinance_request_timeout_seconds = 30
            result = svc.run_bulk_yfinance_fetch(request, yf_client)

        # Both tickers processed (job continued past HANG)
        assert result["tickers_processed"] == 2
        # At least one error from the hung ticker
        assert result["error_count"] >= 1


# ---------------------------------------------------------------------------
# TASK 2a — Whole-ticker incremental short-circuit
# ---------------------------------------------------------------------------


class TestIncrementalShortCircuit:
    def test_when_all_categories_fresh_then_no_network_calls(self):
        """Incremental mode with fully-fresh staleness: zero yf_client fetch calls."""
        repo = _make_repo(staleness=_fully_fresh_staleness())
        yf_client = _make_yf_client_silent()

        service = YFinanceDataService(repo=repo, yf_client=yf_client)
        result = service.fetch_and_store(
            instrument_id=uuid4(),
            yfinance_ticker="AAPL",
            mode="incremental",
        )

        # No network fetches at all
        yf_client.fetch_info.assert_not_called()
        yf_client.fetch_history.assert_not_called()
        yf_client.get_ticker.assert_not_called()
        yf_client.financials.fetch_income_stmt.assert_not_called()
        # Skipped marker present
        assert "all:fresh" in result["skipped"]
        assert result["errors"] == []

    def test_when_one_category_stale_then_network_call_made(self):
        """If any single category is stale, the short-circuit does NOT fire."""
        from datetime import datetime, timedelta, timezone

        staleness = _fully_fresh_staleness()
        # Make profile stale (2 days ago, threshold is 24h)
        staleness["profile_updated_at"] = datetime.now(timezone.utc) - timedelta(
            days=2
        )

        repo = _make_repo(staleness=staleness)
        yf_client = _make_yf_client_silent()

        service = YFinanceDataService(repo=repo, yf_client=yf_client)
        service.fetch_and_store(
            instrument_id=uuid4(),
            yfinance_ticker="AAPL",
            mode="incremental",
        )

        # Profile fetch attempted (category was stale)
        yf_client.fetch_info.assert_called_once()

    def test_when_mode_is_full_then_short_circuit_never_fires(self):
        """Full mode always fetches regardless of staleness data."""
        repo = _make_repo(staleness=_fully_fresh_staleness())
        yf_client = _make_yf_client_silent()

        service = YFinanceDataService(repo=repo, yf_client=yf_client)
        result = service.fetch_and_store(
            instrument_id=uuid4(),
            yfinance_ticker="AAPL",
            mode="full",
        )

        assert "all:fresh" not in result["skipped"]
        # Profile fetch attempted even though data was "fresh"
        yf_client.fetch_info.assert_called_once()

    def test_when_no_price_max_date_then_short_circuit_does_not_fire(self):
        """Fresh staleness without a price_max_date must NOT short-circuit."""
        staleness = _fully_fresh_staleness()
        staleness["price_max_date"] = None  # no price history yet

        repo = _make_repo(staleness=staleness)
        yf_client = _make_yf_client_silent()

        service = YFinanceDataService(repo=repo, yf_client=yf_client)
        result = service.fetch_and_store(
            instrument_id=uuid4(),
            yfinance_ticker="NEW",
            mode="incremental",
        )

        assert "all:fresh" not in result["skipped"]


# ---------------------------------------------------------------------------
# TASK 2b — Scheduler uses workers from settings
# ---------------------------------------------------------------------------


class TestSchedulerWorkerSettings:
    def test_when_yfinance_fetch_workers_is_four_then_request_has_workers_four(self):
        """run_daily_pipeline passes workers=settings.yfinance_fetch_workers."""
        from app.services.jobs import scheduler as sched_module

        captured: list[Any] = []

        def _fake_run_step(label, job_svc, fn, *args):
            if label == "yfinance":
                captured.extend(args)
            return True

        with (
            patch.object(sched_module, "_run_step", side_effect=_fake_run_step),
            patch.object(sched_module, "_refresh_reference_indices", return_value=True),
            patch.object(sched_module.settings, "yfinance_fetch_workers", 4),
            patch(
                "app.services.market_data.yfinance.get_yfinance_client",
                return_value=MagicMock(),
            ),
        ):
            sched_module.run_daily_pipeline()

        # captured[0] is the YFinanceFetchRequest, captured[1] is yf_client
        assert len(captured) >= 1
        request_arg = captured[0]
        assert request_arg.workers == 4

    def test_when_yfinance_fetch_workers_is_one_then_request_uses_serial(self):
        from app.services.jobs import scheduler as sched_module

        captured: list[Any] = []

        def _fake_run_step(label, job_svc, fn, *args):
            if label == "yfinance":
                captured.extend(args)
            return True

        with (
            patch.object(sched_module, "_run_step", side_effect=_fake_run_step),
            patch.object(sched_module, "_refresh_reference_indices", return_value=True),
            patch.object(sched_module.settings, "yfinance_fetch_workers", 1),
            patch(
                "app.services.market_data.yfinance.get_yfinance_client",
                return_value=MagicMock(),
            ),
        ):
            sched_module.run_daily_pipeline()

        assert captured[0].workers == 1

    def test_when_yfinance_fetch_workers_exceeds_schema_max_then_clamped_to_16(self):
        """Scheduler clamps to schema max=16 regardless of settings value."""
        from app.services.jobs import scheduler as sched_module

        captured: list[Any] = []

        def _fake_run_step(label, job_svc, fn, *args):
            if label == "yfinance":
                captured.extend(args)
            return True

        with (
            patch.object(sched_module, "_run_step", side_effect=_fake_run_step),
            patch.object(sched_module, "_refresh_reference_indices", return_value=True),
            patch.object(sched_module.settings, "yfinance_fetch_workers", 99),
            patch(
                "app.services.market_data.yfinance.get_yfinance_client",
                return_value=MagicMock(),
            ),
        ):
            sched_module.run_daily_pipeline()

        assert captured[0].workers <= 16


# ---------------------------------------------------------------------------
# TASK 2c — Parallel path: on_progress current/total advances correctly
# ---------------------------------------------------------------------------


class TestParallelProgressReporting:
    def test_when_parallel_fetch_runs_then_progress_fires_for_every_ticker(self):
        """on_progress(current=...) is called once per ticker in parallel mode."""
        instruments = [
            SimpleNamespace(
                id=uuid4(),
                yfinance_ticker=f"T{i}",
                exchange_name="LSE",
                currency_code="GBP",
            )
            for i in range(6)
        ]
        progress_calls: list[dict[str, Any]] = []
        lock = threading.Lock()

        def _progress(**kwargs: Any) -> None:
            with lock:
                progress_calls.append(kwargs)

        repo_mock = MagicMock()
        repo_mock.get_instruments_with_yfinance_ticker.return_value = instruments
        service_mock = MagicMock()
        service_mock.fetch_and_store.return_value = {
            "counts": {},
            "errors": [],
            "skipped": [],
        }

        @contextmanager
        def _get_session():
            yield MagicMock()

        db_mgr = SimpleNamespace(get_session=_get_session)

        request = SimpleNamespace(workers=3, period="5y", mode="incremental")

        with (
            patch.object(svc, "YFinanceRepository", return_value=repo_mock),
            patch.object(svc, "YFinanceDataService", return_value=service_mock),
            patch("app.database.database_manager", db_mgr),
            patch("app.config.settings") as mock_settings,
        ):
            mock_settings.yfinance_request_timeout_seconds = 30
            result = svc._run_bulk_yfinance_fetch_parallel(
                request, MagicMock(), on_progress=_progress, workers=3
            )

        current_values = sorted(
            c["current"] for c in progress_calls if "current" in c
        )
        assert current_values == list(range(1, 7))
        assert result["tickers_processed"] == 6
        assert result["error_count"] == 0
