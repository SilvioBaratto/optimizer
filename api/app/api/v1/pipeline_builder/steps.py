"""Steps router for ``/api/v1/pipeline-builder/sessions/{id}/steps`` (issue #714).

Fire-and-poll dispatch for the 13 Cycle-2 step handlers.

``POST  /sessions/{sid}/steps/{step_id}`` → 202 + job_id, background dispatch.
``GET   /sessions/{sid}/steps/{step_id}`` → current step state.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import CancelledError
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ValidationError

from app.schemas.pipeline_builder import (
    CostStepRequest,
    EmptyStepRequest,
    LoadStepRequest,
    RebalanceDecisionStepRequest,
    RegimeStepRequest,
    ScreenStepRequest,
    StepPollResponse,
    StepRunResponse,
)
from app.services._shared import make_cancellable_progress
from app.services.jobs.background_job import JobAlreadyRunningError
from app.services.pipeline_builder.session_store import (
    get_job_service,
    get_session,
    update_session,
)
from app.services.pipeline_builder.steps import (
    step_build_history,
    step_clean_returns,
    step_cost,
    step_coverage_gate,
    step_load,
    step_optimize,
    step_persist,
    step_rebalance_decision,
    step_regime,
    step_report,
    step_screen,
    step_validate_is,
    step_validate_oos,
)
from optimizer.exceptions import FactorCoverageError

logger = logging.getLogger(__name__)

router = APIRouter()


# step_id → (request_model, handler_fn). Mirrors STEP_IDS exactly.
_STEP_HANDLER_MAP: dict[str, tuple[type[BaseModel], Callable[..., dict]]] = {
    "load": (LoadStepRequest, step_load),
    "screen": (ScreenStepRequest, step_screen),
    "clean_returns": (EmptyStepRequest, step_clean_returns),
    "build_history": (EmptyStepRequest, step_build_history),
    "validate_is": (EmptyStepRequest, step_validate_is),
    "validate_oos": (EmptyStepRequest, step_validate_oos),
    "coverage_gate": (EmptyStepRequest, step_coverage_gate),
    "regime": (RegimeStepRequest, step_regime),
    "optimize": (EmptyStepRequest, step_optimize),
    "rebalance_decision": (RebalanceDecisionStepRequest, step_rebalance_decision),
    "cost": (CostStepRequest, step_cost),
    "report": (EmptyStepRequest, step_report),
    "persist": (EmptyStepRequest, step_persist),
}


# ---------------------------------------------------------------------------
# Foreground routes
# ---------------------------------------------------------------------------


@router.post(
    "/sessions/{session_id}/steps/{step_id}",
    response_model=StepRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def post_step(session_id: str, step_id: str, request: Request) -> StepRunResponse:
    """Claim a job slot and dispatch the handler in a background thread."""
    request_model, _ = _require_step(step_id)
    body = await _parse_body(request, request_model)
    _require_session(session_id)

    svc = get_job_service(session_id)
    try:
        job_id = svc.create_job()
    except JobAlreadyRunningError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc

    _mark_step_running(session_id, step_id, job_id)
    svc.start_background(
        target=_thread_runner, args=(job_id, session_id, step_id, body)
    )

    return StepRunResponse(
        job_id=job_id,
        status="pending",
        message=f"Step '{step_id}' dispatched. Poll GET for status.",
    )


@router.get("/sessions/{session_id}/steps/{step_id}", response_model=StepPollResponse)
def get_step(session_id: str, step_id: str) -> StepPollResponse:
    """Return the current state of the step for *session_id*."""
    _require_step(step_id)
    session = _require_session(session_id)

    entry = session.step_status.get(step_id, "pending")
    if entry == "pending":
        return StepPollResponse(status="not_started", progress={})
    if entry == "completed":
        return StepPollResponse(
            status="completed",
            progress={},
            result=session.step_results.get(step_id),
        )

    job = get_job_service(session_id).get_job(entry)
    if job is None:
        return StepPollResponse(status="not_started", progress={})
    return _poll_response_from_job(job)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


def _thread_runner(
    job_id: str,
    session_id: str,
    step_id: str,
    params: dict[str, Any],
    *,
    cancel_event: threading.Event | None = None,
) -> None:
    """Thin adapter for ``start_background`` — args[0] must be the job_id string."""
    _run_step(session_id, step_id, job_id, params, cancel_event=cancel_event)


def _run_step(
    session_id: str,
    step_id: str,
    job_id: str,
    params: dict[str, Any],
    *,
    cancel_event: threading.Event | None = None,
) -> None:
    """Background worker — call the handler, surface terminal state."""
    if cancel_event is None:
        cancel_event = threading.Event()
    svc = get_job_service(session_id)
    on_progress = make_cancellable_progress(job_id, svc, cancel_event)
    svc.update_job(job_id, status="running")

    session = get_session(session_id)
    if session is None:
        svc.update_job(
            job_id,
            status="failed",
            error=f"Session '{session_id}' disappeared mid-run",
            finished_at=_now_iso(),
        )
        return

    _, handler = _STEP_HANDLER_MAP[step_id]
    try:
        result = handler(session_id, session, params, on_progress)
    except CancelledError:
        return  # reaper owns terminal status
    except FactorCoverageError as exc:
        svc.update_job(job_id, status="failed", error=str(exc), finished_at=_now_iso())
        return
    except RuntimeError as exc:
        svc.update_job(
            job_id,
            status="failed",
            gate_reason=str(exc),
            finished_at=_now_iso(),
        )
        return
    except Exception as exc:
        logger.exception("step %s failed for session %s", step_id, session_id)
        svc.update_job(job_id, status="failed", error=str(exc), finished_at=_now_iso())
        return

    svc.update_job(job_id, status="completed", result=result, finished_at=_now_iso())
    _mark_step_completed(session_id, step_id, result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_step(step_id: str) -> tuple[type[BaseModel], Callable[..., dict]]:
    entry = _STEP_HANDLER_MAP.get(step_id)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown step '{step_id}'",
        )
    return entry


def _require_session(session_id: str):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found",
        )
    return session


async def _parse_body(
    request: Request, request_model: type[BaseModel]
) -> dict[str, Any]:
    """Parse + validate the per-step body using *request_model*."""
    raw = await _safe_json(request)
    try:
        return request_model.model_validate(raw).model_dump()
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=exc.errors(),
        ) from exc


async def _safe_json(request: Request) -> dict[str, Any]:
    body = await request.body()
    if not body:
        return {}
    import json

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid JSON: {exc.msg}",
        ) from exc
    return parsed if isinstance(parsed, dict) else {}


def _mark_step_running(session_id: str, step_id: str, job_id: str) -> None:
    """Write *job_id* into ``step_status[step_id]`` for the poll endpoint."""
    session = get_session(session_id)
    if session is None:
        return
    new_status = dict(session.step_status)
    new_status[step_id] = job_id
    update_session(session_id, step_status=new_status, current_step=step_id)


def _mark_step_completed(
    session_id: str, step_id: str, result: dict[str, Any] | None = None
) -> None:
    session = get_session(session_id)
    if session is None:
        return
    new_status = dict(session.step_status)
    new_status[step_id] = "completed"
    new_results = dict(session.step_results)
    if result is not None:
        new_results[step_id] = result
    update_session(
        session_id, step_status=new_status, step_results=new_results
    )


def _poll_response_from_job(job: dict[str, Any]) -> StepPollResponse:
    return StepPollResponse(
        status=job.get("status", "unknown"),
        progress={
            "current": job.get("current", 0),
            "total": job.get("total", 0),
        },
        result=job.get("result"),
        error=job.get("error"),
        gate_reason=job.get("gate_reason"),
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
