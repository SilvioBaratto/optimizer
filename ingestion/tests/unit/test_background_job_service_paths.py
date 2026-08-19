"""Branch coverage for ``BackgroundJobService`` — the job lifecycle itself.

Every scheduled step and every CLI command claims a slot through this service.
The behaviours locked here are the ones that keep a wedged run from poisoning
the pipeline:

- a bad job_id string is ignored rather than raising inside a worker thread
- the heartbeat keeps ticking through a transient DB error, but stops and
  signals cancel once the reaper has taken the row
- ``start_background`` always sets the cancel event, even when the target
  raises, so the heartbeat companion cannot outlive its worker
- metrics and the failure webhook fire on the right transitions and stay off
  when disabled

The repository is mocked throughout — no DB.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from app.services.jobs.background_job import BackgroundJobService

M = "app.services.jobs.background_job"
JOB_ID = "11111111-1111-1111-1111-111111111111"


@pytest.fixture
def session() -> MagicMock:
    s = MagicMock()
    s.execute = MagicMock()  # marks it as a session, not a context manager
    return s


@pytest.fixture
def svc(session: MagicMock) -> BackgroundJobService:
    @contextmanager
    def _factory():
        yield session

    return BackgroundJobService(
        job_type="test_domain",
        session_factory=_factory,
        heartbeat_cadence_seconds=0.01,
    )


@contextmanager
def _repo() -> Iterator[MagicMock]:
    with patch(f"{M}.BackgroundJobRepository") as cls:
        yield cls.return_value


class TestMalformedJobId:
    def test_get_job_returns_none(self, svc: BackgroundJobService) -> None:
        assert svc.get_job("not-a-uuid") is None

    def test_update_job_is_a_noop(self, svc: BackgroundJobService) -> None:
        with _repo() as repo:
            svc.update_job("not-a-uuid", status="running")

        repo.update.assert_not_called()


class TestGetJob:
    def test_when_row_absent_then_none(self, svc: BackgroundJobService) -> None:
        with _repo() as repo:
            repo.get.return_value = None
            assert svc.get_job(JOB_ID) is None

    def test_row_is_flattened_with_extra_merged_in(
        self, svc: BackgroundJobService
    ) -> None:
        row = MagicMock()
        row.id = uuid.UUID(JOB_ID)
        row.status = "running"
        row.current = 3
        row.total = 10
        row.errors = None
        row.result = None
        row.error = None
        row.started_at = None
        row.finished_at = None
        row.extra = {"current_country": "Italy"}

        with _repo() as repo:
            repo.get.return_value = row
            out = svc.get_job(JOB_ID)

        assert out is not None
        assert out["job_id"] == JOB_ID
        assert out["errors"] == []
        assert out["current_country"] == "Italy"


class TestUpdateJob:
    def test_when_no_row_matched_and_status_non_terminal_then_warns(
        self, svc: BackgroundJobService, caplog
    ) -> None:
        """The reaper won the race — a progress write landing after it is a no-op."""
        import logging

        with _repo() as repo:
            repo.update.return_value = 0
            with caplog.at_level(logging.WARNING, logger=M):
                svc.update_job(JOB_ID, status="running")

        assert any("no-op" in r.getMessage() for r in caplog.records)

    def test_when_no_row_matched_but_status_terminal_then_silent(
        self, svc: BackgroundJobService, caplog
    ) -> None:
        import logging

        with _repo() as repo:
            repo.update.return_value = 0
            with caplog.at_level(logging.WARNING, logger=M):
                svc.update_job(JOB_ID, status="failed")

        assert not any("no-op" in r.getMessage() for r in caplog.records)

    def test_commits(self, svc: BackgroundJobService, session: MagicMock) -> None:
        with _repo() as repo:
            repo.update.return_value = 1
            svc.update_job(JOB_ID, current=5)

        session.commit.assert_called_once()


class TestIsAnyRunning:
    def test_delegates_to_the_repository_with_the_job_type(
        self, svc: BackgroundJobService
    ) -> None:
        with _repo() as repo:
            repo.is_any_running.return_value = (True, JOB_ID)
            assert svc.is_any_running() == (True, JOB_ID)

        repo.is_any_running.assert_called_once_with("test_domain")


class TestHeartbeatTick:
    def test_healthy_tick_continues_and_commits(
        self, svc: BackgroundJobService, session: MagicMock
    ) -> None:
        stop = threading.Event()

        with _repo() as repo:
            repo.update_heartbeat.return_value = True
            assert svc._tick_heartbeat(uuid.UUID(JOB_ID), stop) is True

        session.commit.assert_called_once()
        assert not stop.is_set()

    def test_when_row_reaped_then_signals_cancel_and_stops(
        self, svc: BackgroundJobService
    ) -> None:
        """The reaper failed the row; the worker must learn it is orphaned."""
        stop = threading.Event()

        with _repo() as repo:
            repo.update_heartbeat.return_value = False
            assert svc._tick_heartbeat(uuid.UUID(JOB_ID), stop) is False

        assert stop.is_set()

    def test_transient_db_error_keeps_the_loop_alive(
        self, svc: BackgroundJobService
    ) -> None:
        """A DB hiccup must not kill an otherwise-healthy multi-hour fetch."""
        stop = threading.Event()

        with _repo() as repo:
            repo.update_heartbeat.side_effect = RuntimeError("db blip")
            assert svc._tick_heartbeat(uuid.UUID(JOB_ID), stop) is True

        assert not stop.is_set()

    def test_run_heartbeat_exits_once_stop_is_set(
        self, svc: BackgroundJobService
    ) -> None:
        stop = threading.Event()
        stop.set()

        with _repo() as repo:
            svc._run_heartbeat(uuid.UUID(JOB_ID), stop, cadence=0.01)

        repo.update_heartbeat.assert_not_called()

    def test_run_heartbeat_stops_when_a_tick_reports_reaped(
        self, svc: BackgroundJobService
    ) -> None:
        stop = threading.Event()

        with patch.object(svc, "_tick_heartbeat", return_value=False) as tick:
            svc._run_heartbeat(uuid.UUID(JOB_ID), stop, cadence=0.001)

        tick.assert_called_once()


class TestExtractJobId:
    @pytest.mark.parametrize("args", [(), (42,), (None,)])
    def test_rejects_args_without_a_leading_job_id_string(self, args) -> None:
        with pytest.raises(ValueError, match="job_id string"):
            BackgroundJobService._extract_job_id(args)

    def test_returns_the_first_positional_arg(self) -> None:
        assert BackgroundJobService._extract_job_id((JOB_ID, "x")) == JOB_ID


class TestStartBackground:
    def test_runs_the_target_and_sets_the_cancel_event(
        self, svc: BackgroundJobService
    ) -> None:
        seen: list[tuple] = []

        def target(job_id: str, *, cancel_event: threading.Event) -> None:
            seen.append((job_id, cancel_event))

        with _repo() as repo:
            repo.update_heartbeat.return_value = True
            worker, cancel = svc.start_background(target, (JOB_ID,))
            worker.join(timeout=5)

        assert seen and seen[0][0] == JOB_ID
        assert cancel.is_set()

    def test_when_target_raises_then_cancel_is_still_set(
        self, svc: BackgroundJobService
    ) -> None:
        """Otherwise the heartbeat thread outlives its dead worker forever."""

        def target(job_id: str, *, cancel_event: threading.Event) -> None:
            raise RuntimeError("scraper exploded")

        with _repo() as repo:
            repo.update_heartbeat.return_value = True
            worker, cancel = svc.start_background(target, (JOB_ID,))
            worker.join(timeout=5)

        assert not worker.is_alive()
        assert cancel.is_set()

    def test_an_external_cancel_event_is_shared_with_the_target(
        self, svc: BackgroundJobService
    ) -> None:
        """``make_cancellable_progress`` builds the event before the thread exists."""
        external = threading.Event()
        seen: list[threading.Event] = []

        def target(job_id: str, *, cancel_event: threading.Event) -> None:
            seen.append(cancel_event)

        with _repo() as repo:
            repo.update_heartbeat.return_value = True
            worker, cancel = svc.start_background(
                target, (JOB_ID,), cancel_event=external
            )
            worker.join(timeout=5)

        assert cancel is external
        assert seen == [external]


class TestMetricsEmission:
    def _svc(self, session: MagicMock) -> BackgroundJobService:
        @contextmanager
        def _factory():
            yield session

        return BackgroundJobService(job_type="metrics_domain", session_factory=_factory)

    def test_running_then_completed_emits_duration_and_counter(
        self, session: MagicMock
    ) -> None:
        svc = self._svc(session)

        with _repo() as repo, patch(f"{M}.settings") as st, patch("app.metrics") as met:
            repo.update.return_value = 1
            st.enable_metrics = True
            st.notification_webhook_url = None

            svc.update_job(JOB_ID, status="running")
            svc.update_job(JOB_ID, status="completed")

        met.job_duration_seconds.labels.assert_called_with(domain="metrics_domain")
        met.jobs_completed_total.labels.assert_called_with(domain="metrics_domain")
        met.jobs_in_progress.labels.assert_called_with(domain="metrics_domain")

    def test_failed_emits_the_failure_counter(self, session: MagicMock) -> None:
        svc = self._svc(session)

        with _repo() as repo, patch(f"{M}.settings") as st, patch("app.metrics") as met:
            repo.update.return_value = 1
            st.enable_metrics = True
            st.notification_webhook_url = None

            svc.update_job(JOB_ID, status="failed")

        met.jobs_failed_total.labels.assert_called_with(domain="metrics_domain")

    def test_non_lifecycle_update_emits_nothing(self, session: MagicMock) -> None:
        svc = self._svc(session)

        with _repo() as repo, patch(f"{M}.settings") as st, patch("app.metrics") as met:
            repo.update.return_value = 1
            st.enable_metrics = True
            st.notification_webhook_url = None

            svc.update_job(JOB_ID, current=7)

        met.jobs_completed_total.labels.assert_not_called()
        met.jobs_failed_total.labels.assert_not_called()

    def test_when_metrics_disabled_then_nothing_is_emitted(
        self, session: MagicMock
    ) -> None:
        svc = self._svc(session)

        with _repo() as repo, patch(f"{M}.settings") as st, patch("app.metrics") as met:
            repo.update.return_value = 1
            st.enable_metrics = False
            st.notification_webhook_url = None

            svc.update_job(JOB_ID, status="completed")

        met.jobs_completed_total.labels.assert_not_called()


class TestFailureWebhook:
    def _svc(self, session: MagicMock) -> BackgroundJobService:
        @contextmanager
        def _factory():
            yield session

        return BackgroundJobService(job_type="hook_domain", session_factory=_factory)

    def test_failure_posts_the_webhook_with_the_error_text(
        self, session: MagicMock
    ) -> None:
        svc = self._svc(session)

        with (
            _repo() as repo,
            patch(f"{M}.settings") as st,
            patch("app.services._shared.notify_failure") as notify,
        ):
            repo.update.return_value = 1
            st.enable_metrics = False
            st.notification_webhook_url = "https://hooks.example/x"

            svc.update_job(JOB_ID, status="failed", error="fred timed out")

        notify.assert_called_once_with(
            "https://hooks.example/x", "hook_domain", JOB_ID, "fred timed out"
        )

    def test_no_webhook_configured_then_no_post(self, session: MagicMock) -> None:
        svc = self._svc(session)

        with (
            _repo() as repo,
            patch(f"{M}.settings") as st,
            patch("app.services._shared.notify_failure") as notify,
        ):
            repo.update.return_value = 1
            st.enable_metrics = False
            st.notification_webhook_url = None

            svc.update_job(JOB_ID, status="failed", error="boom")

        notify.assert_not_called()

    def test_completion_does_not_post(self, session: MagicMock) -> None:
        svc = self._svc(session)

        with (
            _repo() as repo,
            patch(f"{M}.settings") as st,
            patch("app.services._shared.notify_failure") as notify,
        ):
            repo.update.return_value = 1
            st.enable_metrics = False
            st.notification_webhook_url = "https://hooks.example/x"

            svc.update_job(JOB_ID, status="completed")

        notify.assert_not_called()


class TestJsonbSafe:
    @pytest.mark.parametrize("value", ["s", 1, 1.5, True, None, [1], {"a": 1}])
    def test_json_native_values_pass_through(self, value) -> None:
        assert BackgroundJobService._jsonb_safe(value) == value

    def test_non_native_values_are_stringified(self) -> None:
        uid = uuid.UUID(JOB_ID)
        assert BackgroundJobService._jsonb_safe(uid) == JOB_ID
