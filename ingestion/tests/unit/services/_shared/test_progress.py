"""Unit tests for ``_shared/_progress`` progress-callback factories."""

from __future__ import annotations

from concurrent.futures import CancelledError
from threading import Event
from unittest.mock import MagicMock

import pytest

from app.services._shared._progress import (
    _noop,
    make_cancellable_progress,
    make_progress,
)


class TestNoop:
    def test_when_called_with_arbitrary_kwargs_then_returns_none(self) -> None:
        result = _noop(current=5, total=10, label="step")
        assert result is None

    def test_when_called_with_no_args_then_returns_none(self) -> None:
        assert _noop() is None


class TestMakeProgress:
    def test_when_invoked_then_forwards_kwargs_to_update_job(self) -> None:
        job_svc = MagicMock()
        cb = make_progress("job-123", job_svc)

        cb(current=3, total=10)

        job_svc.update_job.assert_called_once_with("job-123", current=3, total=10)

    def test_when_invoked_multiple_times_then_each_call_forwarded(self) -> None:
        job_svc = MagicMock()
        cb = make_progress("job-abc", job_svc)

        cb(current=1, total=5)
        cb(current=5, total=5)

        assert job_svc.update_job.call_count == 2


class TestMakeCancellableProgress:
    def test_when_event_not_set_then_forwards_to_update_job(self) -> None:
        job_svc = MagicMock()
        event = Event()
        cb = make_cancellable_progress("job-xyz", job_svc, event)

        cb(current=2, total=8)

        job_svc.update_job.assert_called_once_with("job-xyz", current=2, total=8)

    def test_when_event_is_set_then_raises_cancelled_error(self) -> None:
        job_svc = MagicMock()
        event = Event()
        event.set()
        cb = make_cancellable_progress("job-xyz", job_svc, event)

        with pytest.raises(CancelledError, match="job job-xyz cancelled"):
            cb(current=2, total=8)

    def test_when_event_is_set_then_update_job_not_called(self) -> None:
        job_svc = MagicMock()
        event = Event()
        event.set()
        cb = make_cancellable_progress("job-xyz", job_svc, event)

        with pytest.raises(CancelledError):
            cb(current=1, total=1)

        job_svc.update_job.assert_not_called()
