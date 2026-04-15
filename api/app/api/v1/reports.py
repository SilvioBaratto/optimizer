"""FastAPI router for report generation endpoints.

Pattern: background job (POST /generate → 202 + job_id, GET /jobs/{job_id}).
Download: GET /{report_id}/download → streams the PDF file.

Follows macro_regime.py / backtest.py background job pattern.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from app.database import database_manager
from app.schemas.reports import (
    ReportGenerateRequest,
    ReportJobCreateResponse,
    ReportJobProgress,
)
from app.services.background_job import BackgroundJobService, JobAlreadyRunningError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reports", tags=["Reports"])

# Module-level job service — follows macro_regime.py / backtest.py pattern.
_report_job_service = BackgroundJobService(
    job_type="report_generate",
    session_factory=database_manager.get_session,
)

# Default output directory — configurable via env var.
_DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "reports",
)
_REPORT_OUTPUT_DIR = os.environ.get("REPORT_OUTPUT_DIR", _DEFAULT_OUTPUT_DIR)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_report_path(report_id: str) -> str | None:
    """Return the absolute path to the report file, or None if not found."""
    expected = os.path.join(_REPORT_OUTPUT_DIR, f"report_{report_id}.pdf")
    if os.path.isfile(expected):
        return expected
    return None


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


def _run_report_generation(
    job_id: str,
    request: ReportGenerateRequest,
) -> None:
    """Execute report generation in a daemon thread."""
    _report_job_service.update_job(job_id, status="running")
    try:
        from app.services.report_service import ReportService

        with database_manager.get_session() as session:
            service = ReportService(session, output_dir=_REPORT_OUTPUT_DIR)
            report_id = service.generate(
                sections=request.sections,
                branding=request.branding.model_dump(),
                orientation=request.orientation.value,
                portfolio_id=request.portfolio_id,
            )

        download_url = f"/api/v1/reports/{report_id}/download"
        filename = f"report_{report_id}.pdf"
        _report_job_service.update_job(
            job_id,
            status="completed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            result={
                "download_url": download_url,
                "filename": filename,
                "report_id": report_id,
            },
        )
        logger.info("Report job %s completed (report_id=%s)", job_id, report_id)
    except Exception as exc:
        logger.error("Report job %s failed: %s", job_id, exc)
        _report_job_service.update_job(
            job_id,
            status="failed",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/generate",
    response_model=ReportJobCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def generate_report(request: ReportGenerateRequest) -> ReportJobCreateResponse:
    """Start report generation as a background job. Returns 202 + job_id."""
    try:
        job_id = _report_job_service.create_job()
    except JobAlreadyRunningError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    _report_job_service.start_background(
        target=_run_report_generation,
        args=(job_id, request),
    )

    return ReportJobCreateResponse(
        job_id=job_id,
        status="pending",
        message=f"Report generation started. Poll GET /reports/jobs/{job_id}.",
    )


@router.get("/jobs/{job_id}", response_model=ReportJobProgress)
def get_report_job_status(job_id: str) -> ReportJobProgress:
    """Poll a report generation job for status and progress."""
    job = _report_job_service.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report job '{job_id}' not found",
        )

    return ReportJobProgress(
        job_id=job["job_id"],
        status=job["status"],
        current=job.get("current", 0),
        total=job.get("total", 0),
        errors=job.get("errors", []),
        result=job.get("result"),
        error=job.get("error"),
    )


@router.get("/{report_id}/download")
def download_report(report_id: str) -> FileResponse:
    """Stream a generated PDF report for download.

    Returns 404 if the report_id is not found or the file is missing from disk.
    """
    file_path = _find_report_path(report_id)
    if file_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report '{report_id}' not found",
        )

    filename = f"report_{report_id}.pdf"
    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=filename,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )
