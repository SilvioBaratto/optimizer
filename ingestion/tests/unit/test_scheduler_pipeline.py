"""Unit tests for pipeline-orchestration functions in app.services.jobs.scheduler.

Pure-unit: no HTTP client, no real DB, no real threads, no real network.
All external collaborators are patched with MagicMock.
"""

from __future__ import annotations

import threading
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from app.config import settings
from app.services.jobs.background_job import JobAlreadyRunningError

# Module under test (imported as a string to avoid side-effects at collection time)
M = "app.services.jobs.scheduler"

# A valid UUID string required because _run_step calls uuid.UUID(job_id)
_VALID_JOB_ID = "12345678-1234-1234-1234-123456789abc"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job_svc(*, get_job_return=None):
    """Return a MagicMock BackgroundJobService with sensible defaults."""
    svc = MagicMock()
    svc.create_job.return_value = _VALID_JOB_ID
    svc._heartbeat_cadence = 30
    svc.get_job.return_value = get_job_return or {"status": "completed"}
    return svc


# ---------------------------------------------------------------------------
# TestRunStep
# ---------------------------------------------------------------------------


class TestRunStep:
    def _call(self, job_svc, fn, **kwargs):
        from app.services.jobs.scheduler import _run_step

        return _run_step("test_label", job_svc, fn)

    def test_when_already_running_then_returns_false_and_fn_not_called(self):
        job_svc = _make_job_svc()
        job_svc.create_job.side_effect = JobAlreadyRunningError("existing-id")
        fn = MagicMock()

        with patch(f"{M}.threading.Thread") as mock_thread:
            result = self._call(job_svc, fn)

        assert result is False
        fn.assert_not_called()
        mock_thread.assert_not_called()

    def test_when_fn_succeeds_and_already_completed_then_returns_true_no_defensive_update(
        self,
    ):
        job_svc = _make_job_svc(get_job_return={"status": "completed"})
        fn = MagicMock()

        with patch(f"{M}.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            result = self._call(job_svc, fn)

        assert result is True
        fn.assert_called_once()
        _, kwargs = fn.call_args
        assert "on_progress" in kwargs
        # Defensive update must NOT have been called with status="completed"
        for c in job_svc.update_job.call_args_list:
            assert c.kwargs.get("status") != "completed" or c.args[1:] != ()

    def test_when_fn_succeeds_but_status_not_completed_then_defensive_update_called(
        self,
    ):
        job_svc = _make_job_svc(get_job_return={"status": "running"})
        fn = MagicMock()

        with patch(f"{M}.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            result = self._call(job_svc, fn)

        assert result is True
        # Defensive path: update_job must have been called with status="completed"
        found = any(
            kw.get("status") == "completed"
            for _, kw in [(c.args, c.kwargs) for c in job_svc.update_job.call_args_list]
        )
        assert found

    def test_when_fn_raises_then_returns_false_and_status_failed(self):
        job_svc = _make_job_svc()
        fn = MagicMock(side_effect=Exception("boom"))

        with patch(f"{M}.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            result = self._call(job_svc, fn)

        assert result is False
        failed_calls = [
            c
            for c in job_svc.update_job.call_args_list
            if c.kwargs.get("status") == "failed"
        ]
        assert failed_calls


# ---------------------------------------------------------------------------
# TestRunDailyPipeline
# ---------------------------------------------------------------------------


def _make_step_side_effect(returns: dict[str, bool]):
    """Return a side_effect callable for _run_step that maps label → bool."""

    def _effect(label, *args, **kwargs):
        return returns.get(label, False)

    return _effect


@pytest.mark.parametrize(
    "yf_ok,news_ok,summarize_ok,expected,absent",
    [
        (
            True,
            True,
            True,
            {"yfinance", "macro", "news", "summarize", "calibrate"},
            set(),
        ),
        (
            False,
            False,
            False,
            {"yfinance", "macro"},
            {"news", "summarize", "calibrate"},
        ),
        (
            True,
            False,
            False,
            {"yfinance", "macro", "news"},
            {"summarize", "calibrate"},
        ),
        (
            True,
            True,
            False,
            {"yfinance", "macro", "news", "summarize"},
            {"calibrate"},
        ),
    ],
)
class TestRunDailyPipeline:
    def test_step_gating(self, yf_ok, news_ok, summarize_ok, expected, absent):
        returns = {
            "yfinance": yf_ok,
            "macro": True,
            "news": news_ok,
            "summarize": summarize_ok,
            "calibrate": True,
        }

        with ExitStack() as stack:
            mock_step = stack.enter_context(patch(f"{M}._run_step"))
            mock_refresh = stack.enter_context(patch(f"{M}.refresh_reference_indices"))
            stack.enter_context(
                patch(
                    "app.services.market_data.yfinance.get_yfinance_client",
                    return_value=MagicMock(),
                )
            )
            # Patch schema imports so they don't fail
            stack.enter_context(
                patch(
                    "app.schemas.macro.macro_regime.MacroFetchRequest",
                    return_value=MagicMock(),
                )
            )
            stack.enter_context(
                patch(
                    "app.schemas.macro.macro_regime.MacroNewsFetchRequest",
                    return_value=MagicMock(),
                )
            )
            stack.enter_context(
                patch(
                    "app.schemas.macro.macro_regime.MacroNewsSummarizeRequest",
                    return_value=MagicMock(),
                )
            )
            stack.enter_context(
                patch(
                    "app.schemas.market_data.yfinance_data.YFinanceFetchRequest",
                    return_value=MagicMock(),
                )
            )

            mock_step.side_effect = _make_step_side_effect(returns)

            from app.services.jobs.scheduler import run_daily_pipeline

            run_daily_pipeline()

        called_labels = {c.args[0] for c in mock_step.call_args_list}

        assert expected.issubset(called_labels), (
            f"Expected labels {expected} not all present in {called_labels}"
        )
        for label in absent:
            assert label not in called_labels, (
                f"Label '{label}' should be absent but was called"
            )

        mock_refresh.assert_called_once()


# ---------------------------------------------------------------------------
# TestRunMiddayNewsRefresh
# ---------------------------------------------------------------------------


class TestRunMiddayNewsRefresh:
    def _run(self, step_returns: dict[str, bool]):
        with ExitStack() as stack:
            mock_step = stack.enter_context(patch(f"{M}._run_step"))
            stack.enter_context(
                patch(
                    "app.schemas.macro.macro_regime.MacroNewsFetchRequest",
                    return_value=MagicMock(),
                )
            )
            stack.enter_context(
                patch(
                    "app.schemas.macro.macro_regime.MacroNewsSummarizeRequest",
                    return_value=MagicMock(),
                )
            )
            mock_step.side_effect = _make_step_side_effect(step_returns)

            from app.services.jobs.scheduler import run_midday_news_refresh

            run_midday_news_refresh()

        return mock_step

    def test_when_news_ok_then_summarize_called(self):
        mock_step = self._run({"news": True, "summarize": True})
        labels = {c.args[0] for c in mock_step.call_args_list}
        assert "news" in labels
        assert "summarize" in labels

    def test_when_news_fails_then_summarize_not_called(self):
        mock_step = self._run({"news": False})
        labels = {c.args[0] for c in mock_step.call_args_list}
        assert "news" in labels
        assert "summarize" not in labels


# ---------------------------------------------------------------------------
# TestRunWeeklyRefetch
# ---------------------------------------------------------------------------


class TestRunWeeklyRefetch:
    def test_when_called_then_yfinance_and_macro_run_and_refresh_called_once(self):
        with ExitStack() as stack:
            mock_step = stack.enter_context(patch(f"{M}._run_step"))
            mock_refresh = stack.enter_context(patch(f"{M}.refresh_reference_indices"))
            stack.enter_context(
                patch(
                    "app.services.market_data.yfinance.get_yfinance_client",
                    return_value=MagicMock(),
                )
            )
            stack.enter_context(
                patch(
                    "app.schemas.macro.macro_regime.MacroFetchRequest",
                    return_value=MagicMock(),
                )
            )
            stack.enter_context(
                patch(
                    "app.schemas.market_data.yfinance_data.YFinanceFetchRequest",
                    return_value=MagicMock(),
                )
            )
            mock_step.return_value = True

            from app.services.jobs.scheduler import run_weekly_refetch

            run_weekly_refetch()

        labels = {c.args[0] for c in mock_step.call_args_list}
        assert "yfinance" in labels
        assert "macro" in labels
        mock_refresh.assert_called_once()


# ---------------------------------------------------------------------------
# TestRunFredMonthly
# ---------------------------------------------------------------------------


class TestRunFredMonthly:
    def test_when_called_then_fred_step_runs(self):
        with ExitStack() as stack:
            mock_step = stack.enter_context(patch(f"{M}._run_step"))
            stack.enter_context(
                patch(
                    "app.schemas.macro.macro_regime.FredFetchRequest",
                    return_value=MagicMock(),
                )
            )
            mock_step.return_value = True

            from app.services.jobs.scheduler import run_fred_monthly

            run_fred_monthly()

        labels = {c.args[0] for c in mock_step.call_args_list}
        assert "fred" in labels


# ---------------------------------------------------------------------------
# TestRefreshReferenceIndices
# ---------------------------------------------------------------------------


class TestRefreshReferenceIndices:
    def _patch_all(self, stack, *, tickers=None, seed_side_effect=None):
        """Enter all necessary patches; returns (mock_resolve, mock_seed, mock_ref_jobs).

        Pass ``tickers=[]`` explicitly to test the no-tickers branch; omit to
        default to ``["SPY"]``.
        """
        resolved = ["SPY"] if tickers is None else tickers
        mock_resolve = stack.enter_context(
            patch(f"{M}._resolve_benchmark_tickers", return_value=resolved)
        )
        mock_seed = stack.enter_context(
            patch(
                "app.services.market_data.reference_index_seeder.seed_reference_indices",
                side_effect=seed_side_effect,
            )
        )
        stack.enter_context(
            patch(
                "app.services.market_data.yfinance.get_yfinance_client",
                return_value=MagicMock(),
            )
        )
        mock_ref_jobs = MagicMock()
        # Must be a real UUID string: the heartbeat companion parses it.
        mock_ref_jobs.create_job.return_value = "33333333-3333-3333-3333-333333333333"
        mock_ref_jobs.get_job.return_value = {"status": "completed"}
        mock_ref_jobs._heartbeat_cadence = 0.01
        stack.enter_context(patch(f"{M}._ref_index_jobs", mock_ref_jobs))
        return mock_resolve, mock_seed, mock_ref_jobs

    def test_when_no_tickers_then_returns_false_and_create_job_not_called(self):
        with ExitStack() as stack:
            mock_resolve, _, mock_ref_jobs = self._patch_all(stack, tickers=[])

            from app.services.jobs.scheduler import refresh_reference_indices

            result = refresh_reference_indices("test")

        assert result is False
        mock_ref_jobs.create_job.assert_not_called()

    def test_when_already_running_then_returns_false(self):
        with ExitStack() as stack:
            mock_resolve, _, mock_ref_jobs = self._patch_all(stack)
            err = JobAlreadyRunningError("j9")
            err.existing_job_id = "j9"
            mock_ref_jobs.create_job.side_effect = err

            from app.services.jobs.scheduler import refresh_reference_indices

            result = refresh_reference_indices("test")

        assert result is False

    def test_when_success_then_returns_true(self):
        with ExitStack() as stack:
            _, mock_seed, mock_ref_jobs = self._patch_all(stack)
            mock_ref_jobs.get_job.return_value = {"status": "completed"}

            from app.services.jobs.scheduler import refresh_reference_indices

            result = refresh_reference_indices("test")

        assert result is True
        mock_seed.assert_called_once()

    def test_when_success_but_status_not_completed_then_defensive_update_called(self):
        with ExitStack() as stack:
            _, _, mock_ref_jobs = self._patch_all(stack)
            mock_ref_jobs.get_job.return_value = {"status": "running"}

            from app.services.jobs.scheduler import refresh_reference_indices

            result = refresh_reference_indices("test")

        assert result is True
        found = any(
            c.kwargs.get("status") == "completed"
            for c in mock_ref_jobs.update_job.call_args_list
        )
        assert found

    def test_when_seed_raises_then_returns_false_and_status_failed(self):
        with ExitStack() as stack:
            _, _, mock_ref_jobs = self._patch_all(
                stack, seed_side_effect=Exception("network error")
            )

            from app.services.jobs.scheduler import refresh_reference_indices

            result = refresh_reference_indices("test")

        assert result is False
        failed_calls = [
            c
            for c in mock_ref_jobs.update_job.call_args_list
            if c.kwargs.get("status") == "failed"
        ]
        assert failed_calls


# ---------------------------------------------------------------------------
# TestResolveBenchmarkTickers
# ---------------------------------------------------------------------------


class TestResolveBenchmarkTickers:
    """Benchmarks come from settings alone — the resolver touches no database."""

    def test_returns_configured_tickers_sorted_and_deduped(self):
        from app.services.jobs.scheduler import _resolve_benchmark_tickers

        result = _resolve_benchmark_tickers()

        assert result == sorted(set(settings.benchmark_tickers))
        assert result == sorted(result)
        assert len(result) == len(set(result))

    def test_reflects_overridden_settings(self, monkeypatch):
        from app.services.jobs import scheduler as sched

        monkeypatch.setattr(
            sched.settings, "benchmark_tickers", ["QQQ", "SPY", "SPY"], raising=False
        )

        assert sched._resolve_benchmark_tickers() == ["QQQ", "SPY"]


# ---------------------------------------------------------------------------
# TestRunNewsRefresh
# ---------------------------------------------------------------------------


class TestRunNewsRefresh:
    def _patch_all(
        self,
        stack,
        *,
        morning_complete=True,
        countries=None,
        pruned=0,
        db_side_effect=None,
    ):
        mock_dm = MagicMock()
        mock_session = MagicMock()
        mock_repo = MagicMock()
        mock_repo.delete_old_news_summaries.return_value = pruned

        if db_side_effect:
            mock_dm.get_session.side_effect = db_side_effect
        else:
            mock_dm.get_session.return_value.__enter__ = MagicMock(
                return_value=mock_session
            )
            mock_dm.get_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_repo_cls = MagicMock(return_value=mock_repo)

        stack.enter_context(patch(f"{M}.database_manager", mock_dm))
        stack.enter_context(
            patch(
                "app.repositories.macro.macro_regime_repository.MacroRegimeRepository",
                mock_repo_cls,
            )
        )
        mock_morning = stack.enter_context(
            patch(
                "app.services.macro.macro_news_summary._is_morning_pipeline_complete",
                return_value=morning_complete,
            )
        )
        mock_find = stack.enter_context(
            patch(
                "app.services.macro.macro_news_summary._find_countries_with_new_articles",
                return_value=countries or [],
            )
        )
        mock_summarize = stack.enter_context(
            patch("app.services.macro.macro_news_summary._summarize_country_safe")
        )
        stack.enter_context(
            patch(
                f"{M}._get_last_refresh_time",
                return_value=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
        )
        return mock_morning, mock_find, mock_summarize, mock_session, mock_repo

    def test_when_morning_incomplete_then_early_return(self):
        with ExitStack() as stack:
            mock_morning, mock_find, mock_summarize, _, _ = self._patch_all(
                stack, morning_complete=False
            )

            from app.services.jobs.scheduler import run_news_refresh

            run_news_refresh()

        mock_find.assert_not_called()

    def test_when_no_new_articles_then_summarize_not_called_but_prune_runs(self):
        with ExitStack() as stack:
            _, mock_find, mock_summarize, mock_session, mock_repo = self._patch_all(
                stack, morning_complete=True, countries=[]
            )

            from app.services.jobs.scheduler import run_news_refresh

            run_news_refresh()

        mock_summarize.assert_not_called()
        mock_repo.delete_old_news_summaries.assert_called_once()
        mock_session.commit.assert_called_once()

    def test_when_countries_found_then_summarize_called_per_country_and_prune_runs(
        self,
    ):
        with ExitStack() as stack:
            _, _, mock_summarize, mock_session, mock_repo = self._patch_all(
                stack, morning_complete=True, countries=["USA", "UK"], pruned=3
            )

            from app.services.jobs.scheduler import run_news_refresh

            run_news_refresh()

        assert mock_summarize.call_count == 2
        mock_session.commit.assert_called_once()

    def test_when_db_raises_then_exception_is_swallowed(self):
        with ExitStack() as stack:
            self._patch_all(stack, db_side_effect=Exception("boom"))

            from app.services.jobs.scheduler import run_news_refresh

            # Must not raise
            run_news_refresh()


# ---------------------------------------------------------------------------
# TestGetLastRefreshTime
# ---------------------------------------------------------------------------


class TestGetLastRefreshTime:
    def test_when_db_has_value_then_returns_that_datetime(self):
        from app.services.jobs.scheduler import _get_last_refresh_time

        expected = datetime(2026, 1, 1, tzinfo=timezone.utc)
        mock_session = MagicMock()
        mock_session.execute.return_value.scalar_one_or_none.return_value = expected

        result = _get_last_refresh_time(mock_session)

        assert result == expected

    def test_when_db_returns_none_then_returns_past_datetime(self):
        from app.services.jobs.scheduler import _get_last_refresh_time

        mock_session = MagicMock()
        mock_session.execute.return_value.scalar_one_or_none.return_value = None

        now = datetime.now(timezone.utc)
        result = _get_last_refresh_time(mock_session)

        assert isinstance(result, datetime)
        assert result < now


# ---------------------------------------------------------------------------
# TestRunOrphanReaper
# ---------------------------------------------------------------------------


class TestRunOrphanReaper:
    def _run(self, stack, *, reaped=None, db_side_effect=None):
        mock_dm = MagicMock()
        mock_session = MagicMock()
        mock_repo = MagicMock()
        mock_repo.reap_orphans.return_value = reaped or []

        if db_side_effect:
            mock_dm.get_session.side_effect = db_side_effect
        else:
            mock_dm.get_session.return_value.__enter__ = MagicMock(
                return_value=mock_session
            )
            mock_dm.get_session.return_value.__exit__ = MagicMock(return_value=False)

        mock_repo_cls = MagicMock(return_value=mock_repo)

        stack.enter_context(patch(f"{M}.database_manager", mock_dm))
        stack.enter_context(
            patch(
                "app.repositories.jobs.background_job_repository.BackgroundJobRepository",
                mock_repo_cls,
            )
        )
        # Default strategy is 'fail'; stub metrics so a reaped list doesn't touch
        # the real Prometheus collectors.
        stack.enter_context(patch(f"{M}._emit_reap_metrics"))

        from app.services.jobs.scheduler import run_orphan_reaper

        run_orphan_reaper()

        return mock_session, mock_repo

    def test_when_orphans_reaped_then_commit_called(self):
        with ExitStack() as stack:
            mock_session, mock_repo = self._run(
                stack, reaped=[{"job_type": "macro_fetch", "attempt": 0}]
            )

        mock_repo.reap_orphans.assert_called_once()
        mock_session.commit.assert_called_once()

    def test_when_no_orphans_then_commit_still_called(self):
        with ExitStack() as stack:
            mock_session, mock_repo = self._run(stack, reaped=[])

        mock_repo.reap_orphans.assert_called_once()
        mock_session.commit.assert_called_once()

    def test_when_db_raises_then_exception_is_swallowed(self):
        with ExitStack() as stack:
            # Must not raise
            self._run(stack, db_side_effect=Exception("db gone"))


class TestRunUniverseStep:
    """The universe build heads the pipeline — every later step iterates `instruments`."""

    _SVC = "app.services.universe.universe_build_service"

    def test_when_api_key_missing_then_skips_without_claiming_a_job(self):
        """No key is a config state, not a failure — claiming a slot would wedge the type."""
        from app.services.universe.universe_build_service import (
            Trading212NotConfiguredError,
        )

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    f"{self._SVC}.build_trading212_client",
                    side_effect=Trading212NotConfiguredError("no key"),
                )
            )
            run_step = stack.enter_context(patch(f"{M}._run_step"))

            from app.services.jobs.scheduler import run_universe_step

            assert run_universe_step() is False

        run_step.assert_not_called()

    def test_when_api_key_present_then_runs_the_build_step(self):
        client = MagicMock()

        with ExitStack() as stack:
            stack.enter_context(
                patch(f"{self._SVC}.build_trading212_client", return_value=client)
            )
            run_step = stack.enter_context(patch(f"{M}._run_step", return_value=True))

            from app.services.jobs.scheduler import run_universe_step

            assert run_universe_step() is True

        args = run_step.call_args.args
        assert args[0] == "universe"
        assert args[-1] is client

    def test_run_universe_build_job_delegates_to_the_step(self):
        with patch(f"{M}.run_universe_step", return_value=True) as step:
            from app.services.jobs.scheduler import run_universe_build

            run_universe_build()

        step.assert_called_once()


class TestHeartbeatCompanion:
    """Synchronous steps must heartbeat, or the orphan reaper eats them mid-run.

    ``_run_step`` and ``refresh_reference_indices`` execute in the scheduler
    thread rather than through ``start_background``, so no heartbeat thread
    comes for free. Only the heartbeat stamps ``last_heartbeat_at`` —
    ``update_job`` never does — so a step outliving the reaper timeout (300s)
    gets flipped to ``failed`` while it is still working. Both the yfinance
    fetch and the reference-index seed routinely exceed that. Observed in a
    live run: reference_index_seed reaped at 5/12 tickers.
    """

    @staticmethod
    def _blocking_heartbeat(_job_id, stop_event, _cadence):
        """Stand-in for the real loop: stays alive until the caller stops it."""
        stop_event.wait(timeout=5)

    def _job_svc(self, job_id: str) -> MagicMock:
        svc = MagicMock()
        svc.create_job.return_value = job_id
        svc.get_job.return_value = {"status": "completed"}
        svc._heartbeat_cadence = 0.01
        svc._run_heartbeat.side_effect = self._blocking_heartbeat
        return svc

    @staticmethod
    def _hb_threads() -> list[str]:
        return [t.name for t in threading.enumerate() if t.name.startswith("hb:sched:")]

    def test_run_step_heartbeats_for_the_duration_of_the_step(self):
        job_svc = self._job_svc("11111111-1111-1111-1111-111111111111")
        seen: list[list[str]] = []

        def fn(*_args, on_progress=None):
            seen.append(self._hb_threads())

        from app.services.jobs.scheduler import _run_step

        assert _run_step("yfinance", job_svc, fn) is True

        assert seen and seen[0], "no heartbeat thread ran alongside the step"
        job_svc._run_heartbeat.assert_called_once()
        assert not self._hb_threads(), "heartbeat outlived the step"

    def test_refresh_reference_indices_heartbeats_for_the_duration_of_the_seed(self):
        seen: list[list[str]] = []

        def _seed(_tickers, _client, on_progress=None):
            seen.append(self._hb_threads())

        jobs = self._job_svc("22222222-2222-2222-2222-222222222222")

        with ExitStack() as stack:
            stack.enter_context(patch(f"{M}._ref_index_jobs", jobs))
            stack.enter_context(
                patch(
                    "app.services.market_data.reference_index_seeder"
                    ".seed_reference_indices",
                    _seed,
                )
            )
            stack.enter_context(
                patch("app.services.market_data.yfinance.get_yfinance_client")
            )

            from app.services.jobs.scheduler import refresh_reference_indices

            assert refresh_reference_indices("test") is True

        assert seen and seen[0], "reference-index seed ran with no heartbeat"
        jobs._run_heartbeat.assert_called_once()
        assert not self._hb_threads(), "heartbeat outlived the seed"


# ---------------------------------------------------------------------------
# TestOrphanReaperStrategy — R3/§5.3 fail vs reclaim + re-dispatch cap
# ---------------------------------------------------------------------------


class TestOrphanReaperStrategy:
    """run_orphan_reaper reaps via reap_orphans; reclaim re-dispatches + metrics."""

    _REPO = "app.repositories.jobs.background_job_repository.BackgroundJobRepository"

    def _run_reaper_with(self, strategy, reaped):
        from app.services.jobs import scheduler as sched

        @contextmanager
        def _sess():
            yield MagicMock()

        repo = MagicMock()
        repo.reap_orphans.return_value = reaped
        with (
            patch.object(sched.settings, "scheduler_orphan_strategy", strategy),
            patch.object(sched.database_manager, "get_session", side_effect=_sess),
            patch(self._REPO, return_value=repo),
            patch.object(sched, "_redispatch_reclaimed") as redispatch,
            patch.object(sched, "_emit_reap_metrics") as emit,
        ):
            sched.run_orphan_reaper()
        return repo, redispatch, emit

    def test_fail_strategy_reaps_and_emits_but_does_not_redispatch(self):
        reaped = [{"job_type": "yfinance_fetch", "attempt": 0}]
        repo, redispatch, emit = self._run_reaper_with("fail", reaped)

        repo.reap_orphans.assert_called_once()
        emit.assert_called_once_with(reaped)
        redispatch.assert_not_called()

    def test_reclaim_strategy_reaps_emits_and_redispatches(self):
        reaped = [{"job_type": "yfinance_fetch", "attempt": 0}]
        repo, redispatch, emit = self._run_reaper_with("reclaim", reaped)

        repo.reap_orphans.assert_called_once()
        emit.assert_called_once_with(reaped)
        redispatch.assert_called_once_with(reaped)

    def test_no_orphans_skips_emit_and_redispatch(self):
        repo, redispatch, emit = self._run_reaper_with("reclaim", [])

        repo.reap_orphans.assert_called_once()
        emit.assert_not_called()
        redispatch.assert_not_called()

    # --- _redispatch_reclaimed submits capped scheduler one-shot jobs ---

    def test_redispatch_below_cap_submits_job_with_next_attempt(self):
        from app.services.jobs import scheduler as sched

        with (
            patch.object(sched.settings, "scheduler_orphan_max_reclaim_attempts", 3),
            patch.object(sched, "_scheduler", MagicMock()) as sch,
        ):
            sched._redispatch_reclaimed([{"job_type": "yfinance_fetch", "attempt": 0}])

        sch.add_job.assert_called_once()
        kw = sch.add_job.call_args.kwargs
        assert kw["kwargs"] == {"job_type": "yfinance_fetch", "attempt": 1}

    def test_redispatch_skips_when_over_cap(self):
        from app.services.jobs import scheduler as sched

        with (
            patch.object(sched.settings, "scheduler_orphan_max_reclaim_attempts", 2),
            patch.object(sched, "_scheduler", MagicMock()) as sch,
        ):
            # attempt 2 -> next 3 > max 2 -> skip
            sched._redispatch_reclaimed([{"job_type": "yfinance_fetch", "attempt": 2}])

        sch.add_job.assert_not_called()

    def test_redispatch_skips_unknown_job_type(self):
        from app.services.jobs import scheduler as sched

        with patch.object(sched, "_scheduler", MagicMock()) as sch:
            sched._redispatch_reclaimed([{"job_type": "mystery", "attempt": 0}])

        sch.add_job.assert_not_called()

    def test_redispatch_without_bound_scheduler_is_noop(self):
        from app.services.jobs import scheduler as sched

        with patch.object(sched, "_scheduler", None):
            # No scheduler bound (e.g. before create_scheduler) — must not raise.
            sched._redispatch_reclaimed([{"job_type": "yfinance_fetch", "attempt": 0}])

    # --- _run_reclaim maps job_type -> step ---

    def test_run_reclaim_dispatches_normal_step(self):
        from app.services.jobs import scheduler as sched

        # _run_reclaim resolves the step via the _RECLAIM_STEP dict (references
        # captured at import), so patch the dict entry, not the module attr.
        step = MagicMock(return_value=True)
        with patch.dict(sched._RECLAIM_STEP, {"yfinance_fetch": step}):
            sched._run_reclaim("yfinance_fetch", 2)

        step.assert_called_once_with(attempt=2)

    def test_run_reclaim_reference_index_uses_refresh(self):
        from app.services.jobs import scheduler as sched

        with patch.object(sched, "refresh_reference_indices") as refresh:
            sched._run_reclaim("reference_index_seed", 1)

        refresh.assert_called_once_with("orphan_reclaim", attempt=1)

    # --- _emit_reap_metrics records the terminal transition ---

    def test_emit_reap_metrics_decrements_gauge_and_increments_failures(self):
        from app.services.jobs import scheduler as sched

        gauge = MagicMock()
        counter = MagicMock()
        with (
            patch.object(sched.settings, "enable_metrics", True),
            patch("app.metrics.jobs_in_progress", gauge),
            patch("app.metrics.jobs_failed_total", counter),
        ):
            sched._emit_reap_metrics([{"job_type": "macro_fetch", "attempt": 0}])

        gauge.labels.assert_called_once_with(domain="macro_fetch")
        gauge.labels.return_value.dec.assert_called_once()
        counter.labels.assert_called_once_with(domain="macro_fetch")
        counter.labels.return_value.inc.assert_called_once()
