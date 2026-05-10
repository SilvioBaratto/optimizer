"""Shared progress-callback helpers for background job workers."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import CancelledError
from threading import Event
from typing import Any

ProgressCallback = Callable[..., None]


def _noop(**kwargs: Any) -> None:
    """No-op progress callback for use when progress tracking is not needed."""


def make_progress(job_id: str, job_svc: Any) -> ProgressCallback:
    """Return a closure that forwards kwargs to job_svc.update_job(job_id, ...)."""

    def _cb(**kwargs: Any) -> None:
        job_svc.update_job(job_id, **kwargs)

    return _cb


def make_cancellable_progress(
    job_id: str, job_svc: Any, cancel_event: Event,
) -> ProgressCallback:
    """Return a progress closure that aborts when *cancel_event* is set.

    The closure raises ``concurrent.futures.CancelledError`` at the next
    invocation after the heartbeat thread (or any other producer) sets
    the event. Otherwise it forwards kwargs to ``job_svc.update_job``.
    """

    def _cb(**kwargs: Any) -> None:
        if cancel_event.is_set():
            raise CancelledError(f"job {job_id} cancelled")
        job_svc.update_job(job_id, **kwargs)

    return _cb
