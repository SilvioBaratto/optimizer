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
    """Return a progress callback bound to a specific job.

    Args:
        job_id: UUID string of the background job to update.
        job_svc: Service object exposing ``update_job(job_id, **kwargs)``.

    Returns:
        A callable that forwards all keyword arguments to
        ``job_svc.update_job(job_id, ...)``.
    """

    def _cb(**kwargs: Any) -> None:
        job_svc.update_job(job_id, **kwargs)

    return _cb


def make_cancellable_progress(
    job_id: str,
    job_svc: Any,
    cancel_event: Event,
) -> ProgressCallback:
    """Return a progress callback that raises on cancellation.

    The closure raises ``concurrent.futures.CancelledError`` at the next
    invocation after the heartbeat thread (or any other producer) sets
    the event. Otherwise it forwards kwargs to ``job_svc.update_job``.

    Args:
        job_id: UUID string of the background job to update.
        job_svc: Service object exposing ``update_job(job_id, **kwargs)``.
        cancel_event: Shared event; when set, the next callback invocation
            aborts the job by raising ``CancelledError``.

    Returns:
        A callable compatible with ``ProgressCallback``.
    """

    def _cb(**kwargs: Any) -> None:
        if cancel_event.is_set():
            raise CancelledError(f"job {job_id} cancelled")
        job_svc.update_job(job_id, **kwargs)

    return _cb
