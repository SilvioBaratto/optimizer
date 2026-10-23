"""Unit tests for pipeline-orchestration functions in app.services.jobs.scheduler.

Pure-unit: no HTTP client, no real DB, no real threads, no real network.
All external collaborators are patched with MagicMock.
"""

from __future__ import annotations

import threading
import time
from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy.exc import OperationalError

from app.services.jobs.background_job import JobAlreadyRunningError

# Module under test (imported as a string to avoid side-effects at collection time)
M = "app.services.jobs.scheduler"

# A valid UUID string required because _run_step calls uuid.UUID(job_id)
_VALID_JOB_ID = "12345678-1234-1234-1234-123456789abc"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _db_error() -> OperationalError:
    return OperationalError("UPDATE background_jobs", {}, Exception("db down"))


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

    def test_when_fn_reports_failed_status_then_returns_false_and_row_stays_failed(
        self,
    ):
        # run_bulk_macro_fetch reports a partial failure via
        # on_progress(status="failed") and then returns normally; the step
        # runner must treat that as final instead of defensively completing it.
        job_svc = _make_job_svc(get_job_return={"status": "failed"})
        fn = MagicMock()

        with patch(f"{M}.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            result = self._call(job_svc, fn)

        assert result is False
        completed_writes = [
            c
            for c in job_svc.update_job.call_args_list
            if c.kwargs.get("status") == "completed"
        ]
        assert completed_writes == []

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

    def test_when_fn_raises_then_failure_reason_is_recorded_on_the_row(self):
        job_svc = _make_job_svc()
        fn = MagicMock(side_effect=ValueError("yahoo schema drifted"))

        with patch(f"{M}.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            self._call(job_svc, fn)

        failed_calls = [
            c
            for c in job_svc.update_job.call_args_list
            if c.kwargs.get("status") == "failed"
        ]
        assert len(failed_calls) == 1
        error = failed_calls[0].kwargs.get("error")
        assert error == "ValueError: yahoo schema drifted"

    @staticmethod
    def _failed_writes(job_svc):
        return [
            c.kwargs
            for c in job_svc.update_job.call_args_list
            if c.kwargs.get("status") == "failed"
        ]

    def test_when_fn_raises_then_failed_row_carries_the_error_text(self):
        job_svc = _make_job_svc()
        fn = MagicMock(side_effect=RuntimeError("boom"))

        with patch(f"{M}.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            self._call(job_svc, fn)

        (failed,) = self._failed_writes(job_svc)
        assert failed["error"] == "RuntimeError: boom"

    def test_when_fn_is_interrupted_then_failed_row_names_the_interrupt(self):
        job_svc = _make_job_svc()
        fn = MagicMock(side_effect=KeyboardInterrupt())

        with patch(f"{M}.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            with pytest.raises(KeyboardInterrupt):
                self._call(job_svc, fn)

        (failed,) = self._failed_writes(job_svc)
        assert failed["error"] == "KeyboardInterrupt"

    def test_when_marking_running_hits_db_error_then_failed_row_carries_it(self):
        job_svc = _make_job_svc()
        job_svc.update_job.side_effect = [_db_error(), None]

        with patch(f"{M}.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            self._call(job_svc, MagicMock())

        (failed,) = self._failed_writes(job_svc)
        assert failed["error"].startswith("OperationalError:")
        assert "db down" in failed["error"]

    def test_when_fn_raises_then_failure_webhook_receives_the_error_text(self):
        from app.services.jobs.background_job import BackgroundJobService

        job_svc = BackgroundJobService(job_type="t", session_factory=MagicMock())
        fn = MagicMock(side_effect=RuntimeError("boom"))

        with (
            patch.object(job_svc, "create_job", return_value=_VALID_JOB_ID),
            patch(f"{M}.threading.Thread") as mock_thread,
            patch(
                "app.services.jobs.background_job.BackgroundJobRepository"
            ) as mock_repo,
            patch(
                "app.services.jobs.background_job.settings.notification_webhook_url",
                "https://hooks.example/x",
            ),
            patch("app.services._shared.notify_failure") as notify,
        ):
            mock_thread.return_value = MagicMock()
            mock_repo.return_value.update.return_value = 1
            self._call(job_svc, fn)

        notify.assert_called_once()
        assert notify.call_args.args[3] == "RuntimeError: boom"

    def test_when_claim_hits_db_error_then_returns_false_without_raising(self):
        job_svc = _make_job_svc()
        job_svc.create_job.side_effect = _db_error()
        fn = MagicMock()

        result = self._call(job_svc, fn)

        assert result is False
        fn.assert_not_called()

    def test_when_marking_running_hits_db_error_then_row_failed_and_fn_skipped(self):
        job_svc = _make_job_svc()
        job_svc.update_job.side_effect = [_db_error(), None]
        fn = MagicMock()

        with patch(f"{M}.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            result = self._call(job_svc, fn)

        assert result is False
        fn.assert_not_called()
        statuses = [c.kwargs.get("status") for c in job_svc.update_job.call_args_list]
        assert statuses == ["running", "failed"]

    def test_when_failed_status_write_also_errors_then_still_returns_false(self):
        job_svc = _make_job_svc()
        job_svc.update_job.side_effect = _db_error()
        fn = MagicMock(side_effect=RuntimeError("boom"))

        with patch(f"{M}.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            result = self._call(job_svc, fn)

        assert result is False

    @pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
    def test_when_fn_is_interrupted_then_row_failed_and_interrupt_propagates(
        self, interrupt
    ):
        # Ctrl-C / SIGTERM-driven exit in a CLI run bypasses ``except
        # Exception``; the row must still leave the active state so an
        # immediate re-run is not refused as "already running".
        job_svc = _make_job_svc()
        fn = MagicMock(side_effect=interrupt())

        with patch(f"{M}.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            with pytest.raises(interrupt):
                self._call(job_svc, fn)

        statuses = [c.kwargs.get("status") for c in job_svc.update_job.call_args_list]
        assert statuses == ["running", "failed"]

    def test_when_interrupted_and_failed_write_errors_then_interrupt_still_propagates(
        self,
    ):
        job_svc = _make_job_svc()
        job_svc.update_job.side_effect = [None, _db_error()]
        fn = MagicMock(side_effect=KeyboardInterrupt())

        with patch(f"{M}.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            with pytest.raises(KeyboardInterrupt):
                self._call(job_svc, fn)


class TestStepFailureIsolation:
    def test_when_yfinance_claim_errors_then_daily_pipeline_runs_independent_steps(
        self,
    ):
        from app.services.jobs import scheduler

        broken = _make_job_svc()
        broken.create_job.side_effect = _db_error()

        with (
            patch.object(scheduler, "_yfinance_jobs", broken),
            patch(
                "app.services.market_data.yfinance.get_yfinance_client",
                return_value=MagicMock(),
            ),
            patch.object(scheduler, "run_fx_step") as fx,
            patch.object(scheduler, "run_macro_step") as macro,
            patch.object(scheduler, "run_news_step") as news,
            patch.object(scheduler, "run_daily_events_step") as events,
        ):
            scheduler.run_daily_pipeline()

        fx.assert_called_once()
        macro.assert_called_once()
        events.assert_called_once()
        news.assert_not_called()


# ---------------------------------------------------------------------------
# TestShutdownDrain
# ---------------------------------------------------------------------------


@contextmanager
def _running_scheduler():
    """Bind a started ``SupervisedScheduler`` as the module's live scheduler."""
    from app.services.jobs.scheduler import SupervisedScheduler

    sched = SupervisedScheduler()
    sched.start()
    try:
        with patch(f"{M}._scheduler", sched):
            yield sched
    finally:
        if sched.running:
            sched.shutdown(wait=False)


class TestShutdownDrain:
    """After SIGTERM pauses the scheduler, a composite pipeline must finish its
    in-flight step but not claim a slot for the next one: the forced exit at the
    drain deadline would leave that fresh row 'running' and block the step on the
    restarted daemon until the heartbeat lease expires."""

    def _call(self, job_svc, fn):
        from app.services.jobs.scheduler import _run_step

        return _run_step("test_label", job_svc, fn)

    def test_when_scheduler_paused_then_step_claims_no_slot(self):
        job_svc = _make_job_svc()
        fn = MagicMock()

        with _running_scheduler() as sched:
            sched.pause()
            result = self._call(job_svc, fn)

        assert result is False
        job_svc.create_job.assert_not_called()
        fn.assert_not_called()

    def test_when_scheduler_shut_down_then_step_claims_no_slot(self):
        # The drain calls pause() and then shutdown(wait=True) straight away, so
        # most of the drain window is spent in the STOPPED state, not PAUSED.
        job_svc = _make_job_svc()
        fn = MagicMock()

        with _running_scheduler() as sched:
            sched.pause()
            sched.shutdown(wait=False)
            result = self._call(job_svc, fn)

        assert result is False
        job_svc.create_job.assert_not_called()
        fn.assert_not_called()

    def test_when_scheduler_never_started_then_step_claims_a_slot(self):
        # create_scheduler() binds the module scheduler before start() is called.
        from app.services.jobs.scheduler import SupervisedScheduler

        job_svc = _make_job_svc()

        with patch(f"{M}._scheduler", SupervisedScheduler()):
            result = self._call(job_svc, MagicMock())

        assert result is True
        job_svc.create_job.assert_called_once()

    def test_when_no_scheduler_bound_then_step_claims_a_slot(self):
        # The CLI composes the same steps without ever creating a scheduler.
        job_svc = _make_job_svc()

        with patch(f"{M}._scheduler", None):
            result = self._call(job_svc, MagicMock())

        assert result is True
        job_svc.create_job.assert_called_once()

    def test_when_scheduler_running_then_step_still_claims_a_slot(self):
        job_svc = _make_job_svc()
        fn = MagicMock()

        with _running_scheduler():
            result = self._call(job_svc, fn)

        assert result is True
        job_svc.create_job.assert_called_once()

    def test_when_resumed_then_steps_claim_slots_again(self):
        job_svc = _make_job_svc()

        with _running_scheduler() as sched:
            sched.pause()
            sched.resume()
            result = self._call(job_svc, MagicMock())

        assert result is True

    def test_when_paused_mid_pipeline_then_later_steps_claim_no_slot(self):
        from app.services.jobs import scheduler as scheduler_module

        labels = ["market_structure", "calendars", "market_summary", "options"]
        svcs = {label: _make_job_svc() for label in labels}
        fns = {label: MagicMock() for label in labels}

        with _running_scheduler() as sched:
            # SIGTERM lands while the first step is still running.
            fns["market_structure"].side_effect = lambda **_kw: sched.pause()

            def _step(label):
                return lambda **_kw: scheduler_module._run_step(
                    label, svcs[label], fns[label]
                )

            with ExitStack() as stack:
                for label in labels:
                    stack.enter_context(
                        patch(f"{M}.run_{label}_step", side_effect=_step(label))
                    )
                scheduler_module.run_weekly_market_wide()

        svcs["market_structure"].create_job.assert_called_once()
        fns["market_structure"].assert_called_once()
        for label in labels[1:]:
            svcs[label].create_job.assert_not_called()
            fns[label].assert_not_called()


class _StatefulJobService:
    """Minimal in-memory job service that keeps the last written status."""

    _heartbeat_cadence = 30

    def __init__(self) -> None:
        self.status = "pending"
        self.updates: list[dict] = []

    def create_job(self, attempt: int = 0) -> str:
        return _VALID_JOB_ID

    def _run_heartbeat(self, *args) -> None:
        return None

    def get_job(self, job_id: str) -> dict:
        return {"status": self.status}

    def update_job(self, job_id: str, **kwargs) -> None:
        self.updates.append(kwargs)
        if "status" in kwargs:
            self.status = kwargs["status"]


class TestRunStepHonoursFailedStatus:
    """A step that reports ``failed`` through ``on_progress`` has failed."""

    def _run(self, fn) -> tuple[bool, _StatefulJobService]:
        from app.services.jobs.scheduler import _run_step

        svc = _StatefulJobService()
        with patch(f"{M}.threading.Thread", return_value=MagicMock()):
            ok = _run_step("macro", svc, fn)  # type: ignore[arg-type]
        return ok, svc

    def test_when_fn_reports_failed_then_status_stays_failed(self):
        def fn(*, on_progress):
            on_progress(status="failed", errors=["Germany: boom"])

        _, svc = self._run(fn)

        assert svc.status == "failed"
        assert [u for u in svc.updates if u.get("status") == "completed"] == []

    def test_when_fn_reports_failed_then_step_returns_false(self):
        def fn(*, on_progress):
            on_progress(status="failed", errors=["Germany: boom"])

        ok, _ = self._run(fn)

        assert ok is False

    def test_when_fn_never_reports_a_status_then_it_is_still_completed(self):
        ok, svc = self._run(lambda *, on_progress: None)

        assert ok is True
        assert svc.status == "completed"


# ---------------------------------------------------------------------------
# TestRunDailyPipeline
# ---------------------------------------------------------------------------


def _make_step_side_effect(returns: dict[str, bool]):
    """Return a side_effect callable for _run_step that maps label → bool."""

    def _effect(label, *args, **kwargs):
        return returns.get(label, False)

    return _effect


@pytest.mark.parametrize(
    "yf_ok,expected,absent",
    [
        (
            True,
            {"yfinance", "macro", "news", "daily_events"},
            set(),
        ),
        (
            False,
            {"yfinance", "macro", "daily_events"},
            {"news"},
        ),
    ],
)
class TestRunDailyPipeline:
    def test_step_gating(self, yf_ok, expected, absent):
        returns = {
            "yfinance": yf_ok,
            "macro": True,
            "news": True,
            "daily_events": True,
        }

        with ExitStack() as stack:
            mock_step = stack.enter_context(patch(f"{M}._run_step"))
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


class TestRunDailyPipelineFredRefresh:
    """Daily-frequency FRED series must not wait for the monthly job.

    ``fund.attribution`` forward-fills DGS10 / BAA10Y / DTWEXBGS onto the NAV
    grid, so a month-old tail becomes a run of exactly-zero factor returns.
    """

    _DAILY = {
        "DGS10",
        "DGS2",
        "DGS3MO",
        "BAA10Y",
        "T10Y2Y",
        "VIXCLS",
        "DTWEXBGS",
    }
    _SLOW = {
        "USALOLITOAASTSAM",
        "RECPROUSM156N",
        "JHGDPBRINDX",
        "USREC",
        "A191RL1Q225SBEA",
    }

    def _run(self, returns: dict[str, bool]):
        with patch(f"{M}._run_step") as mock_step:
            mock_step.side_effect = _make_step_side_effect(returns)

            from app.services.jobs.scheduler import run_daily_pipeline

            run_daily_pipeline()
        return mock_step

    def test_daily_pipeline_incrementally_refreshes_only_daily_fred_series(self):
        mock_step = self._run({"yfinance": True, "daily_events": True})

        fred_calls = [c for c in mock_step.call_args_list if c.args[0] == "fred"]
        assert len(fred_calls) == 1, "daily pipeline never refreshes FRED"
        request = fred_calls[0].args[3]
        assert request.incremental is True
        assert request.series_ids is not None
        assert set(request.series_ids) >= self._DAILY
        assert not (self._SLOW & set(request.series_ids))

    def test_fred_refresh_precedes_the_digest_that_reads_it(self):
        mock_step = self._run({"yfinance": True})

        labels = [c.args[0] for c in mock_step.call_args_list]
        assert labels.index("fred") < labels.index("daily_events")

    def test_failed_fred_refresh_does_not_skip_news_or_digest(self):
        mock_step = self._run({"yfinance": True, "fred": False})

        labels = {c.args[0] for c in mock_step.call_args_list}
        assert {"fred", "news", "daily_events"} <= labels

    def test_fred_refresh_runs_even_when_yfinance_fails(self):
        mock_step = self._run({"yfinance": False})

        labels = [c.args[0] for c in mock_step.call_args_list]
        assert "fred" in labels
        assert labels.index("fred") < labels.index("daily_events")


class TestDailyPipelineFxOrdering:
    """The fund NAV mark runs at 08:00 UTC and reads ``fx_rates``; the FX feed is
    short and independent of prices, so it must land ahead of the long yfinance
    sweep rather than queue behind it."""

    @staticmethod
    def _run(yf_ok: bool) -> list[str]:
        order: list[str] = []

        def _effect(label, *_args, **_kwargs):
            order.append(label)
            return yf_ok if label == "yfinance" else True

        with ExitStack() as stack:
            stack.enter_context(patch(f"{M}._run_step", side_effect=_effect))
            stack.enter_context(
                patch(
                    "app.services.market_data.yfinance.get_yfinance_client",
                    return_value=MagicMock(),
                )
            )

            from app.services.jobs.scheduler import run_daily_pipeline

            run_daily_pipeline()
        return order

    @pytest.mark.parametrize("yf_ok", [True, False])
    def test_daily_pipeline_fetches_fx_before_the_yfinance_sweep(self, yf_ok):
        order = self._run(yf_ok)

        assert "fx" in order
        assert order.index("fx") < order.index("yfinance")


class TestDailyPipelineRefreshesFred:
    """FRED holds daily series (VIXCLS, DGS10, BAA10Y, DTWEXBGS), so the daily
    pipeline must refresh it rather than leaving it to the monthly job."""

    def _run_pipeline(self, *, yf_ok: bool) -> list[tuple]:
        calls: list[tuple] = []

        def _fake_run_step(label, job_svc, fn, *args, **kwargs):
            calls.append((label, args))
            return yf_ok if label == "yfinance" else True

        with (
            patch(f"{M}._run_step", side_effect=_fake_run_step),
            patch(
                "app.services.market_data.yfinance.get_yfinance_client",
                return_value=MagicMock(),
            ),
        ):
            from app.services.jobs.scheduler import run_daily_pipeline

            run_daily_pipeline()
        return calls

    def test_when_daily_pipeline_runs_then_fred_is_fetched_incrementally(self):
        calls = self._run_pipeline(yf_ok=True)

        fred_calls = [args for label, args in calls if label == "fred"]
        assert len(fred_calls) == 1
        assert fred_calls[0][0].incremental is True

    def test_when_yfinance_fails_then_fred_still_runs(self):
        calls = self._run_pipeline(yf_ok=False)

        assert "fred" in [label for label, _ in calls]

    def test_when_daily_pipeline_runs_then_fred_lands_before_the_digest(self):
        labels = [label for label, _ in self._run_pipeline(yf_ok=True)]

        assert labels.index("fred") < labels.index("daily_events")

    def test_macro_fetch_request_declares_no_unused_fred_flag(self):
        from app.schemas.macro.macro_regime import MacroFetchRequest

        assert "include_fred" not in MacroFetchRequest.model_fields


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
            mock_step.side_effect = _make_step_side_effect(step_returns)

            from app.services.jobs.scheduler import run_midday_news_refresh

            run_midday_news_refresh()

        return mock_step

    def test_runs_news_step(self):
        mock_step = self._run({"news": True})
        labels = {c.args[0] for c in mock_step.call_args_list}
        assert labels == {"news"}


# ---------------------------------------------------------------------------
# TestRunWeeklyRefetch
# ---------------------------------------------------------------------------


class TestRunWeeklyRefetch:
    def test_when_called_then_yfinance_and_macro_run(self):
        with ExitStack() as stack:
            mock_step = stack.enter_context(patch(f"{M}._run_step"))
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


class TestRunWeeklyMarketWide:
    def test_runs_all_four_market_wide_steps(self):
        with ExitStack() as stack:
            mock_step = stack.enter_context(patch(f"{M}._run_step"))
            stack.enter_context(
                patch(
                    "app.services.market_data.yfinance.get_yfinance_client",
                    return_value=MagicMock(),
                )
            )
            mock_step.return_value = True

            from app.services.jobs.scheduler import run_weekly_market_wide

            run_weekly_market_wide()

        labels = {c.args[0] for c in mock_step.call_args_list}
        assert labels == {
            "market_structure",
            "calendars",
            "market_summary",
            "options",
        }

    def test_registry_includes_weekly_market_wide(self):
        from app.services.jobs.scheduler import _build_schedule_registry

        ids = {d.job_id for d in _build_schedule_registry()}
        assert "weekly_market_wide" in ids

    @pytest.mark.parametrize(
        "job_type",
        [
            "market_structure_fetch",
            "calendars_fetch",
            "market_summary_fetch",
            "options_fetch",
        ],
    )
    def test_market_wide_job_types_are_reclaim_dispatchable(self, job_type):
        from app.services.jobs.scheduler import _reclaim_dispatchable

        assert _reclaim_dispatchable(job_type) is True


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


class TestRunOptionsStep:
    def test_composes_run_step_with_options_label(self):
        with ExitStack() as stack:
            mock_step = stack.enter_context(patch(f"{M}._run_step", return_value=True))
            stack.enter_context(
                patch(
                    "app.services.market_data.yfinance.get_yfinance_client",
                    return_value=MagicMock(),
                )
            )

            from app.services.jobs.scheduler import _options_jobs, run_options_step

            assert run_options_step() is True

        assert mock_step.call_count == 1
        args = mock_step.call_args.args
        assert args[0] == "options"
        assert args[1] is _options_jobs


class TestRunDailyEventsStep:
    """The daily digest is a heartbeat-wrapped step composed via _run_step."""

    def test_composes_run_step_with_daily_events_label(self):
        """It delegates to _run_step with its label, job slot, and the builder fn."""
        with patch(f"{M}._run_step", return_value=True) as mock_step:
            from app.services.jobs.scheduler import (
                _daily_events_jobs,
                run_daily_events_step,
            )
            from app.services.market_data.market_journal_service import (
                run_build_market_journal,
            )

            assert run_daily_events_step() is True

        assert mock_step.call_count == 1
        args = mock_step.call_args.args
        assert args[0] == "daily_events"
        assert args[1] is _daily_events_jobs
        assert args[2] is run_build_market_journal

    def test_daily_events_job_type_is_reclaim_dispatchable(self):
        """The orphan reaper can re-dispatch a wedged daily_events run."""
        from app.services.jobs.scheduler import _reclaim_dispatchable

        assert _reclaim_dispatchable("daily_events") is True

    def test_heartbeats_for_the_duration_of_the_step(self):
        """A heartbeat thread runs while the digest builds, so the reaper can't eat it."""
        from app.services.jobs import scheduler as sched

        job_id = "22222222-2222-2222-2222-222222222222"
        svc = MagicMock()
        svc.create_job.return_value = job_id
        svc.get_job.return_value = {"status": "completed"}
        svc._heartbeat_cadence = 0.01
        svc._run_heartbeat.side_effect = lambda _j, stop, _c, **_kw: stop.wait(
            timeout=5
        )

        seen: list[list[str]] = []

        def _fake_build(_request, *, on_progress):
            seen.append(
                [
                    t.name
                    for t in threading.enumerate()
                    if t.name.startswith("hb:sched:daily_events")
                ]
            )

        with (
            patch.object(sched, "_daily_events_jobs", svc),
            patch(
                "app.services.market_data.market_journal_service.run_build_market_journal",
                _fake_build,
            ),
        ):
            assert sched.run_daily_events_step() is True

        assert seen and seen[0], (
            "no daily_events heartbeat thread ran alongside the step"
        )
        svc._run_heartbeat.assert_called_once()


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

    def test_runs_regardless_of_trading212(self):
        """yfinance now sources the universe — the step runs with no T212 key, no skip."""
        with ExitStack() as stack:
            run_step = stack.enter_context(patch(f"{M}._run_step", return_value=True))

            from app.services.jobs.scheduler import run_universe_step

            assert run_universe_step() is True

        run_step.assert_called_once()

    def test_delegates_to_build_without_a_trading212_client(self):
        from app.schemas.universe.trading212 import UniverseBuildRequest
        from app.services.universe.universe_build_service import run_universe_build

        with ExitStack() as stack:
            run_step = stack.enter_context(patch(f"{M}._run_step", return_value=True))

            from app.services.jobs.scheduler import run_universe_step

            assert run_universe_step() is True

        args = run_step.call_args.args
        assert args[0] == "universe"
        assert args[2] is run_universe_build  # the build fn
        assert isinstance(args[-1], UniverseBuildRequest)  # only the request; no client

    def test_run_universe_build_job_runs_build_then_annotate(self):
        with (
            patch(f"{M}.run_universe_step", return_value=True) as step,
            patch(f"{M}.run_t212_annotate_step", return_value=False) as annotate,
        ):
            from app.services.jobs.scheduler import run_universe_build

            run_universe_build()

        step.assert_called_once()
        annotate.assert_called_once()


class TestT212AnnotateStep:
    """T212 mapping is a follow-on — it skips without a slot when unconfigured."""

    _SVC = "app.services.universe.universe_build_service"

    def test_skips_without_claiming_when_not_configured(self):
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

            from app.services.jobs.scheduler import run_t212_annotate_step

            assert run_t212_annotate_step() is False

        run_step.assert_not_called()

    def test_runs_the_annotate_step_when_configured(self):
        from app.services.universe import t212_annotate

        with ExitStack() as stack:
            stack.enter_context(
                patch(f"{self._SVC}.build_trading212_client", return_value=MagicMock())
            )
            run_step = stack.enter_context(patch(f"{M}._run_step", return_value=True))

            from app.services.jobs.scheduler import run_t212_annotate_step

            assert run_t212_annotate_step() is True

        args = run_step.call_args.args
        assert args[0] == "t212_annotate"
        assert args[2] is t212_annotate.run_t212_annotate


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
    def _blocking_heartbeat(_job_id, stop_event, _cadence, **_kwargs):
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


class TestStalledStepLease:
    """A step that stops reporting progress must stop renewing its lease.

    The heartbeat thread is independent of the worker, so without a progress
    gate a step wedged on a lock keeps a fresh lease forever and the periodic
    orphan reaper can never free its job slot.
    """

    def test_lease_renewal_pauses_while_stalled_and_resumes_on_progress(self):
        import time

        from app.services.jobs import scheduler as sched

        job_id = "33333333-3333-3333-3333-333333333333"
        svc = MagicMock()
        svc.create_job.return_value = job_id
        svc.get_job.return_value = {"status": "completed"}
        svc._heartbeat_cadence = 0.01

        gates: list = []
        gate_ready = threading.Event()

        def _hb(_job, stop, _cadence, should_renew=None):
            gates.append(should_renew)
            gate_ready.set()
            stop.wait(timeout=5)

        svc._run_heartbeat.side_effect = _hb
        observed: list[bool] = []

        def fn(*_args, on_progress):
            assert gate_ready.wait(timeout=2)
            gate = gates[0]
            assert gate is not None, "heartbeat has no progress gate"
            observed.append(gate())
            time.sleep(0.3)
            observed.append(gate())
            on_progress(current=1)
            observed.append(gate())

        with patch.object(sched.settings, "scheduler_step_stall_timeout_seconds", 0.2):
            assert sched._run_step("stall", svc, fn) is True

        assert observed == [True, False, True]


class TestReapedStepIsFenced:
    """A step whose row the reaper failed must stop instead of running on.

    The heartbeat thread learns the row was reaped (``update_heartbeat`` is
    False) and signals through the event it was handed. The step's progress
    callback has to observe that event, otherwise the zombie keeps fetching
    while the freed job slot lets a second run start.
    """

    def test_when_heartbeat_reports_reaped_then_next_progress_call_aborts_step(
        self,
    ):
        job_svc = MagicMock()
        job_svc.create_job.return_value = _VALID_JOB_ID
        job_svc.get_job.return_value = {"status": "failed"}
        job_svc._heartbeat_cadence = 0.01
        job_svc._run_heartbeat.side_effect = lambda _j, stop, _c, **_kw: stop.set()
        ticks: list[int] = []

        def fn(*, on_progress):
            for i in range(500):
                time.sleep(0.01)
                on_progress(current=i)
                ticks.append(i)

        from app.services.jobs.scheduler import _run_step

        assert _run_step("yfinance", job_svc, fn) is False

        assert len(ticks) < 500, "step kept running after its row was reaped"
        statuses = [c.kwargs.get("status") for c in job_svc.update_job.call_args_list]
        assert "completed" not in statuses


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
