"""Jobs Routes."""

from app.api.v1.jobs.jobs import router as jobs_router
from app.api.v1.jobs.scheduler import router as scheduler_router

__all__ = ["jobs_router", "scheduler_router"]
