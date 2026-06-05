"""Reusable mock for module-level ``BackgroundJobService`` instances (issue #806).

``BackgroundJobService`` is instantiated at module level in route files
(e.g. ``_job_service``, ``_factor_job_service``) and uses
``database_manager.get_session`` — bypassing FastAPI DI and the SQLite test
session.  Tests must therefore patch the **live instance's methods**, not the
class.  This helper patches ``create_job`` / ``get_job`` / ``start_background``
on a dotted instance path so route tests stop re-deriving patch targets.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Any, NamedTuple
from unittest.mock import MagicMock, patch

from app.services.jobs.background_job import JobAlreadyRunningError


class JobServiceStub(NamedTuple):
    create_job: MagicMock
    get_job: MagicMock
    start_background: MagicMock


def _default_job_dict(
    job_id: str, status: str, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Complete ``get_job`` dict mirroring ``BackgroundJobService._row_to_dict``.

    Includes ``errors`` (a list) — routes read ``job.get("errors", [])`` and
    the real row dict always carries it, so the stub stays contract-faithful.
    ``extra`` mirrors ``_row_to_dict``'s ``d.update(row.extra)`` merge so
    callers can assert domain progress keys (e.g. ``current_country``).
    """
    job = {
        "job_id": job_id,
        "status": status,
        "current": 0,
        "total": 0,
        "errors": [],
        "result": None,
        "error": None,
        "started_at": None,
        "finished_at": None,
    }
    if extra:
        job.update(extra)
    return job


def _build_create_mock(job_id: str, raise_already_running: bool) -> MagicMock:
    if raise_already_running:
        return MagicMock(side_effect=JobAlreadyRunningError(job_id))
    return MagicMock(return_value=job_id)


@contextmanager
def mock_job_service(
    module_path: str,
    *,
    job_id: str = "test-job-id",
    initial_status: str = "pending",
    raise_already_running: bool = False,
    extra: dict[str, Any] | None = None,
):
    """Patch ``create_job``/``get_job``/``start_background`` on a live instance.

    ``module_path`` is the dotted path to the module-level instance, e.g.
    ``"app.api.v1.optimization.optimize._job_service"``.  ``extra`` merges
    domain progress keys into the ``get_job`` dict.  Yields a
    ``JobServiceStub`` of the three ``MagicMock``s for assertions.
    """
    create = _build_create_mock(job_id, raise_already_running)
    get = MagicMock(return_value=_default_job_dict(job_id, initial_status, extra))
    start = MagicMock(return_value=(MagicMock(), MagicMock()))
    with ExitStack() as stack:
        stack.enter_context(patch(f"{module_path}.create_job", create))
        stack.enter_context(patch(f"{module_path}.get_job", get))
        stack.enter_context(patch(f"{module_path}.start_background", start))
        yield JobServiceStub(create_job=create, get_job=get, start_background=start)
