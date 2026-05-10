"""Cooperative cancellation integration test (issue #589).

Exercises the worker-vs-reaper race closure end-to-end:
1. Seed a running job row with stale heartbeat.
2. ``reconcile_orphans`` flips it to ``failed`` (the reaper closes the door).
3. A simulated wrapper invokes ``make_cancellable_progress`` whose closure
   hits the now-set cancel_event and raises ``CancelledError``.
4. The wrapper catches it and exits early — no terminal status overwrite.
"""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import CancelledError
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session

from app.repositories.jobs.background_job_repository import BackgroundJobRepository
from app.services._shared import make_cancellable_progress
from app.services.jobs.background_job import BackgroundJobService


@pytest.fixture()
def stale_running_job(db_session: Session) -> uuid.UUID:
    """Create a running job with a heartbeat past the orphan timeout."""
    repo = BackgroundJobRepository(db_session)
    jid = repo.claim_or_create("cancel_test")
    assert jid is not None
    repo.update(jid, status="running")
    repo.update(
        jid,
        last_heartbeat_at=datetime.now(timezone.utc) - timedelta(minutes=10),
    )
    db_session.flush()
    return jid


def _service_factory_yielding(session: Session):
    """Wrap a single Session in a context-manager so BackgroundJobService can use it."""

    @contextmanager
    def _factory():
        yield session

    return _factory


class TestCooperativeCancellation:
    def test_when_reaper_flips_row_then_progress_callback_raises_cancelled(
        self,
        db_session: Session,
        stale_running_job: uuid.UUID,
    ) -> None:
        repo = BackgroundJobRepository(db_session)
        reaped = repo.reconcile_orphans("orphan")
        assert reaped == 1
        row = repo.get(stale_running_job)
        assert row is not None and row.status == "failed"

        svc = BackgroundJobService(
            "cancel_test",
            session_factory=_service_factory_yielding(db_session),
        )
        cancel_event = threading.Event()
        cancel_event.set()  # simulate heartbeat-thread signalling reap

        on_progress = make_cancellable_progress(
            str(stale_running_job),
            svc,
            cancel_event,
        )

        with pytest.raises(CancelledError):
            on_progress(current=1, total=10)

    def test_when_wrapper_observes_cancel_then_no_terminal_status_write(
        self,
        db_session: Session,
        stale_running_job: uuid.UUID,
    ) -> None:
        repo = BackgroundJobRepository(db_session)
        repo.reconcile_orphans("orphan")  # row -> failed

        svc = MagicMock()
        cancel_event = threading.Event()
        cancel_event.set()
        on_progress = make_cancellable_progress(
            str(stale_running_job),
            svc,
            cancel_event,
        )

        # Simulated wrapper body — fake "service function" that calls on_progress
        def fake_service(checkpoints: int) -> None:
            for i in range(checkpoints):
                on_progress(current=i + 1, total=checkpoints)

        terminal_writes = 0

        def _wrapper() -> None:
            nonlocal terminal_writes
            try:
                fake_service(checkpoints=5)
            except CancelledError:
                # Reaper owns terminal status — no terminal write here.
                return
            except Exception:
                terminal_writes += 1
                svc.update_job(
                    str(stale_running_job),
                    status="failed",
                    error="exception path",
                )

        _wrapper()

        assert terminal_writes == 0
        # update_job not called because make_cancellable_progress raised
        # before forwarding kwargs.
        svc.update_job.assert_not_called()

    def test_when_event_clear_then_progress_forwards_to_update_job(
        self,
        db_session: Session,
        stale_running_job: uuid.UUID,
    ) -> None:
        svc = MagicMock()
        cancel_event = threading.Event()  # not set
        on_progress = make_cancellable_progress(
            str(stale_running_job),
            svc,
            cancel_event,
        )

        on_progress(current=3, total=10)

        svc.update_job.assert_called_once_with(
            str(stale_running_job),
            current=3,
            total=10,
        )


class TestStartBackgroundAcceptsExternalCancelEvent:
    def test_when_external_event_passed_then_returned_event_is_same(self) -> None:
        @contextmanager
        def _factory():
            yield MagicMock()

        svc = BackgroundJobService(
            "ext_event_svc",
            session_factory=_factory,
            heartbeat_cadence_seconds=0.05,
        )
        external = threading.Event()
        done = threading.Event()

        def _target(
            _job_id: str, *, cancel_event: threading.Event | None = None
        ) -> None:
            done.set()

        thread, returned = svc.start_background(
            target=_target,
            args=(str(uuid.uuid4()),),
            cancel_event=external,
        )
        try:
            assert returned is external
            done.wait(timeout=1.0)
        finally:
            external.set()
            thread.join(timeout=1.0)
