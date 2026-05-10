"""Heartbeat thread + cancellation event tests for BackgroundJobService (issue #587)."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Generator
from concurrent.futures import CancelledError
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.services._shared import make_cancellable_progress
from app.services.jobs.background_job import BackgroundJobService

# ---------------------------------------------------------------------------
# Fixtures — autouse teardown to set every stop-event created in a test
# ---------------------------------------------------------------------------


_LIVE_EVENTS: list[threading.Event] = []
_LIVE_THREADS: list[threading.Thread] = []


@pytest.fixture(autouse=True)
def _teardown_heartbeat_threads() -> Generator[None, None, None]:
    """Set every stop-event created in this test and join its thread."""
    _LIVE_EVENTS.clear()
    _LIVE_THREADS.clear()
    try:
        yield
    finally:
        for ev in _LIVE_EVENTS:
            ev.set()
        for th in _LIVE_THREADS:
            th.join(timeout=0.5)
        _LIVE_EVENTS.clear()
        _LIVE_THREADS.clear()


def _track(event: threading.Event, thread: threading.Thread | None = None) -> None:
    _LIVE_EVENTS.append(event)
    if thread is not None:
        _LIVE_THREADS.append(thread)


def _make_session_factory(repo_mock: MagicMock):
    """Return a callable that yields a context-manager wrapping repo_mock."""
    session = MagicMock()
    session.commit.return_value = None

    @contextmanager
    def _factory():
        yield session

    repo_factory_patch = patch(
        "app.services.jobs.background_job.BackgroundJobRepository",
        return_value=repo_mock,
    )
    return _factory, repo_factory_patch


# ---------------------------------------------------------------------------
# Heartbeat behaviour
# ---------------------------------------------------------------------------


class TestHeartbeatLoop:
    def test_when_cadence_elapses_repeatedly_then_heartbeat_writes_every_cadence(
        self,
    ) -> None:
        repo = MagicMock()
        repo.update_heartbeat.return_value = True
        factory, repo_patch = _make_session_factory(repo)
        with repo_patch:
            svc = BackgroundJobService(
                "hb_cadence",
                session_factory=factory,
                heartbeat_cadence_seconds=0.05,
            )
            stop = threading.Event()
            th = threading.Thread(
                target=svc._run_heartbeat,
                args=(uuid.uuid4(), stop, 0.05),
                daemon=True,
            )
            _track(stop, th)
            th.start()
            time.sleep(0.2)
            stop.set()
            th.join(timeout=0.5)

        assert repo.update_heartbeat.call_count >= 3

    def test_when_stop_event_set_then_heartbeat_thread_joins_quickly(self) -> None:
        repo = MagicMock()
        repo.update_heartbeat.return_value = True
        factory, repo_patch = _make_session_factory(repo)
        with repo_patch:
            svc = BackgroundJobService(
                "hb_stop",
                session_factory=factory,
                heartbeat_cadence_seconds=5.0,
            )
            stop = threading.Event()
            th = threading.Thread(
                target=svc._run_heartbeat,
                args=(uuid.uuid4(), stop, 5.0),
                daemon=True,
            )
            _track(stop, th)
            th.start()
            stop.set()
            th.join(timeout=1.0)

        assert not th.is_alive()

    def test_when_update_heartbeat_raises_then_thread_keeps_running(self) -> None:
        repo = MagicMock()
        repo.update_heartbeat.side_effect = RuntimeError("db boom")
        factory, repo_patch = _make_session_factory(repo)
        with repo_patch:
            svc = BackgroundJobService(
                "hb_swallow",
                session_factory=factory,
                heartbeat_cadence_seconds=0.05,
            )
            stop = threading.Event()
            th = threading.Thread(
                target=svc._run_heartbeat,
                args=(uuid.uuid4(), stop, 0.05),
                daemon=True,
            )
            _track(stop, th)
            th.start()
            time.sleep(0.15)
            assert th.is_alive()
            stop.set()
            th.join(timeout=0.5)

        assert repo.update_heartbeat.call_count >= 2

    def test_when_update_heartbeat_returns_false_then_stop_event_set(self) -> None:
        repo = MagicMock()
        repo.update_heartbeat.return_value = False
        factory, repo_patch = _make_session_factory(repo)
        with repo_patch:
            svc = BackgroundJobService(
                "hb_reaped",
                session_factory=factory,
                heartbeat_cadence_seconds=0.05,
            )
            stop = threading.Event()
            th = threading.Thread(
                target=svc._run_heartbeat,
                args=(uuid.uuid4(), stop, 0.05),
                daemon=True,
            )
            _track(stop, th)
            th.start()
            th.join(timeout=0.5)

        assert stop.is_set()


# ---------------------------------------------------------------------------
# start_background return shape
# ---------------------------------------------------------------------------


class TestStartBackgroundReturn:
    def test_when_start_background_called_then_returns_thread_and_event(self) -> None:
        repo = MagicMock()
        repo.update_heartbeat.return_value = True
        factory, repo_patch = _make_session_factory(repo)
        with repo_patch:
            svc = BackgroundJobService(
                "sb_return",
                session_factory=factory,
                heartbeat_cadence_seconds=0.05,
            )
            done = threading.Event()

            def _target(
                _job_id: str, *, cancel_event: threading.Event | None = None
            ) -> None:
                done.set()

            result = svc.start_background(target=_target, args=(str(uuid.uuid4()),))

        assert isinstance(result, tuple)
        thread, cancel_event = result
        assert isinstance(thread, threading.Thread)
        assert isinstance(cancel_event, threading.Event)
        _track(cancel_event, thread)
        done.wait(timeout=1.0)
        thread.join(timeout=1.0)


# ---------------------------------------------------------------------------
# update_job — rowcount==0 WARNING surface
# ---------------------------------------------------------------------------


class TestUpdateJobNoopWarning:
    def test_when_repo_update_returns_zero_then_warning_emitted(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        repo = MagicMock()
        repo.update.return_value = 0
        factory, repo_patch = _make_session_factory(repo)
        with (
            repo_patch,
            caplog.at_level(logging.WARNING, logger="app.services.jobs.background_job"),
        ):
            svc = BackgroundJobService("noop", session_factory=factory)
            svc.update_job(str(uuid.uuid4()), current=42, status="running")

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "expected a WARNING about reaped/missing row"


# ---------------------------------------------------------------------------
# make_cancellable_progress
# ---------------------------------------------------------------------------


class TestMakeCancellableProgress:
    def test_when_event_set_then_closure_raises_cancelled_error(self) -> None:
        svc = MagicMock()
        cancel = threading.Event()
        cb = make_cancellable_progress("job-1", svc, cancel)
        cancel.set()

        with pytest.raises(CancelledError):
            cb(current=10)

        svc.update_job.assert_not_called()

    def test_when_event_clear_then_closure_forwards_kwargs(self) -> None:
        svc = MagicMock()
        cancel = threading.Event()
        cb = make_cancellable_progress("job-2", svc, cancel)

        cb(current=5, total=10)

        svc.update_job.assert_called_once_with("job-2", current=5, total=10)


# ---------------------------------------------------------------------------
# Suppress unused-import warning (stub kept for the heartbeat join contract)
# ---------------------------------------------------------------------------

_: Any = None
