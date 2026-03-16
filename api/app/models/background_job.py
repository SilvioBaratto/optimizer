"""SQLAlchemy model for persistent background job tracking."""

from datetime import datetime

from sqlalchemy import JSON, DateTime, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

# Use JSONB on PostgreSQL, plain JSON elsewhere (e.g. SQLite in tests).
_JSON = JSON().with_variant(JSONB, "postgresql")

from app.models.base import BaseModel


class BackgroundJob(BaseModel):
    """Persistent background job record.

    Replaces the in-memory dict in ``BackgroundJobService`` so that job state
    survives API restarts and is visible across processes.
    """

    __tablename__ = "background_jobs"
    __table_args__ = (
        Index("ix_background_jobs_type_status", "job_type", "status"),
    )

    job_type: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True,
    )
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="pending",
    )
    current: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    total: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    extra: Mapped[dict | None] = mapped_column(_JSON, nullable=True)
    result: Mapped[dict | None] = mapped_column(_JSON, nullable=True)
    errors: Mapped[list | None] = mapped_column(_JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
