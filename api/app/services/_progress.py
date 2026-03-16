"""Shared progress-callback helpers for background job workers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

ProgressCallback = Callable[..., None]


def _noop(**kwargs: Any) -> None:
    """No-op progress callback for use when progress tracking is not needed."""


def make_progress(job_id: str, job_svc: Any) -> ProgressCallback:
    """Return a closure that forwards kwargs to job_svc.update_job(job_id, ...)."""

    def _cb(**kwargs: Any) -> None:
        job_svc.update_job(job_id, **kwargs)

    return _cb
