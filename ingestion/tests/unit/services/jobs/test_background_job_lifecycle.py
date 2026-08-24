"""Lifecycle tests for BackgroundJobService + progress + orphan reaper (issue #824).

Fills the gaps left by ``test_background_job_service_heartbeat.py`` (which
covers heartbeat ticks, the start_background return shape, and the cancellable
progress closure): the plain ``make_progress`` forwarder, ``create_job``
conflict, thread-level cancel via ``cancel_event.set()``, and the periodic
``run_orphan_reaper``.  Repository + session are mocked — never a real session
(CLAUDE.md gotcha).
"""

from __future__ import annotations

import threading
import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.config import settings
from app.services._shared._progress import make_cancellable_progress, make_progress
from app.services.jobs.background_job import (
    BackgroundJobService,
    JobAlreadyRunningError,
)

_REPO = "app.services.jobs.background_job.BackgroundJobRepository"


# ---------------------------------------------------------------------------
# make_progress / make_cancellable_progress
# ---------------------------------------------------------------------------


class TestMakeProgress:
    def test_when_called_then_forwards_kwargs_to_update_job(self) -> None:
        svc = MagicMock()
        cb = make_progress("job-1", svc)
        cb(current=1, total=2)
        svc.update_job.assert_called_once_with("job-1", current=1, total=2)

    def test_cancellable_when_event_set_then_raises(self) -> None:
        from concurrent.futures import CancelledError

        event = threading.Event()
        event.set()
        cb = make_cancellable_progress("job-1", MagicMock(), event)
        with pytest.raises(CancelledError):
            cb(current=1)

    def test_cancellable_when_event_clear_then_forwards(self) -> None:
        svc = MagicMock()
        cb = make_cancellable_progress("job-1", svc, threading.Event())
        cb(current=3, total=5)
        svc.update_job.assert_called_once_with("job-1", current=3, total=5)


# ---------------------------------------------------------------------------
# create_job conflict
# ---------------------------------------------------------------------------


class TestCreateJob:
    def _service(self) -> BackgroundJobService:
        # A MagicMock has both __enter__ and execute, so the service treats it
        # as a live session directly (no nested context-manager unwrapping).
        return BackgroundJobService(job_type="t", session_factory=MagicMock())

    def test_when_same_type_running_then_raises_already_running(self) -> None:
        with patch(_REPO) as MockRepo:
            repo = MockRepo.return_value
            repo.claim_or_create.return_value = None  # slot taken
            repo.is_any_running.return_value = (True, "existing-id-123")
            with pytest.raises(JobAlreadyRunningError) as exc:
                self._service().create_job()

        assert exc.value.existing_job_id == "existing-id-123"

    def test_when_no_job_running_then_returns_new_id(self) -> None:
        new_id = uuid.uuid4()
        with patch(_REPO) as MockRepo:
            MockRepo.return_value.claim_or_create.return_value = new_id
            result = self._service().create_job(current_country="")

        assert result == str(new_id)


# ---------------------------------------------------------------------------
# start_background — cancel via event
# ---------------------------------------------------------------------------


class TestStartBackgroundCancel:
    def test_when_event_set_then_worker_stops_and_event_remains_set(self) -> None:
        svc = BackgroundJobService(
            job_type="t",
            session_factory=MagicMock(),
            heartbeat_cadence_seconds=0.05,
        )
        observed: dict[str, threading.Event] = {}

        def target(_job_id: str, *, cancel_event: threading.Event) -> None:
            observed["event"] = cancel_event
            cancel_event.wait(timeout=2.0)  # blocks until cancelled

        with patch(_REPO) as MockRepo:
            MockRepo.return_value.update_heartbeat.return_value = True
            thread, event = svc.start_background(target, args=(str(uuid.uuid4()),))
            event.set()  # request cancellation
            thread.join(timeout=2.0)

        assert not thread.is_alive()
        assert event.is_set()
        assert observed["event"] is event  # same fence token reached the target


# ---------------------------------------------------------------------------
# Orphan reaper
# ---------------------------------------------------------------------------

_SCHED_DM = "app.services.jobs.scheduler.database_manager"
_REAPER_REPO = "app.repositories.jobs.background_job_repository.BackgroundJobRepository"


class TestOrphanReaper:
    def _patch_session(self, dm: MagicMock) -> MagicMock:
        session = MagicMock()
        dm.get_session.return_value.__enter__.return_value = session
        return session

    def test_when_run_then_reap_orphans_invoked_and_committed(self) -> None:
        from app.services.jobs import scheduler as sched

        with (
            patch(_SCHED_DM) as dm,
            patch(_REAPER_REPO) as MockRepo,
            patch.object(sched, "_emit_reap_metrics"),
        ):
            session = self._patch_session(dm)
            MockRepo.return_value.reap_orphans.return_value = [
                {"job_type": "macro_fetch", "attempt": 0}
            ]
            sched.run_orphan_reaper()

        call = MockRepo.return_value.reap_orphans.call_args
        assert "heartbeat stale" in call.args[0]
        assert call.kwargs["heartbeat_timeout_seconds"] == (
            settings.scheduler_orphan_heartbeat_timeout_seconds
        )
        session.commit.assert_called_once()

    def test_when_reap_raises_then_swallowed_non_fatal(self) -> None:
        from app.services.jobs.scheduler import run_orphan_reaper

        with patch(_SCHED_DM) as dm, patch(_REAPER_REPO) as MockRepo:
            self._patch_session(dm)
            MockRepo.return_value.reap_orphans.side_effect = RuntimeError("boom")
            run_orphan_reaper()  # must NOT raise
