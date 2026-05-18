"""Public facade for the ``pipeline_builder`` package.

Re-exports the data-model surface and the lock-guarded CRUD functions.
The per-session ``BackgroundJobService`` factory and the TTL eviction
daemon land in later issues (#691-#693) and will be re-exported here
once they exist.
"""

from app.services.pipeline_builder.session_store import (
    STEP_IDS,
    PipelineSessionCapacityError,
    _PipelineSession,
    create_session,
    delete_session,
    get_job_service,
    get_session,
    update_session,
)

__all__ = [
    "STEP_IDS",
    "PipelineSessionCapacityError",
    "_PipelineSession",
    "create_session",
    "delete_session",
    "get_job_service",
    "get_session",
    "update_session",
]
