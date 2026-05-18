"""Sessions router for ``/api/v1/pipeline-builder/sessions`` (issue #713).

Three endpoints:
  - ``POST   /sessions``                              create
  - ``DELETE /sessions/{session_id}``                 idempotent delete
  - ``GET    /sessions/{session_id}/artifacts/{name}`` stream final artefact
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import FileResponse

from app.schemas.pipeline_builder import CreateSessionResponse, RunLevelConfig
from app.services.pipeline_builder.session_store import (
    PipelineSessionCapacityError,
    create_session,
    delete_session,
    get_session,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# URL name → key inside ``session.report_result["artifact_paths"]``.
# Whitelist is the only allowed input; rejects path traversal before path resolution.
_ARTIFACT_NAME_TO_KEY: Final[dict[str, str]] = {
    "report.md": "report_md",
    "weights.csv": "weights_csv",
    "metrics.json": "metrics_json",
    "checklist.json": "checklist_json",
}

_MEDIA_TYPES: Final[dict[str, str]] = {
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".json": "application/json",
}


@router.post(
    "/sessions",
    response_model=CreateSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
def post_session(body: RunLevelConfig) -> CreateSessionResponse:
    """Create a new pipeline-builder session."""
    try:
        sid = create_session(run_config=body.model_dump(mode="json"))
    except PipelineSessionCapacityError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
        ) from exc
    return CreateSessionResponse(session_id=sid)


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_pipeline_session(session_id: str) -> Response:
    """Delete a session. Idempotent — returns 204 even when missing."""
    delete_session(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/sessions/{session_id}/artifacts/{name}")
def download_artifact(session_id: str, name: str) -> FileResponse:
    """Stream a final artefact file. Requires ``step_persist`` completed."""
    key = _validate_artifact_name(name)
    session = _require_persist_completed(session_id)
    path = _resolve_artifact_path(session.report_result, key)
    return _file_response(path, name)


def _validate_artifact_name(name: str) -> str:
    """Map URL name → artifact_paths key. Reject anything not in the whitelist."""
    key = _ARTIFACT_NAME_TO_KEY.get(name)
    if key is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown artifact '{name}'",
        )
    return key


def _require_persist_completed(session_id: str):
    """Return the session iff ``step_persist`` has completed; else 404."""
    session = get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found",
        )
    if session.step_status.get("persist") != "completed":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Artifacts unavailable: pipeline has not reached step_persist",
        )
    return session


def _resolve_artifact_path(report_result: dict | None, key: str) -> str:
    """Look up the absolute artefact path; 404 if missing on disk."""
    paths = (report_result or {}).get("artifact_paths") or {}
    path = paths.get(key)
    if not path or not Path(path).is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Artifact '{key}' missing on disk",
        )
    return path


def _file_response(path: str, name: str) -> FileResponse:
    """Build the ``FileResponse`` with media-type + ``Content-Disposition``."""
    suffix = Path(name).suffix
    media = _MEDIA_TYPES.get(suffix, "application/octet-stream")
    return FileResponse(
        path=path,
        media_type=media,
        filename=name,
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )
