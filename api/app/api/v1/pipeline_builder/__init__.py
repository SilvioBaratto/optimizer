"""Pipeline-builder v1 router package (issue #713)."""

from fastapi import APIRouter

from app.api.v1.pipeline_builder.sessions import router as _sessions_router
from app.api.v1.pipeline_builder.steps import router as _steps_router

pipeline_builder_router = APIRouter(
    prefix="/pipeline-builder", tags=["Pipeline Builder"]
)
pipeline_builder_router.include_router(_sessions_router)
pipeline_builder_router.include_router(_steps_router)

__all__ = ["pipeline_builder_router"]
