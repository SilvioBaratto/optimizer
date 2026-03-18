"""SQLAlchemy model for persistent background job tracking."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import BaseModel

# Use JSONB on PostgreSQL, plain JSON elsewhere (e.g. SQLite in tests).
_JSON = JSON().with_variant(JSONB, "postgresql")


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
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    # Relationships
    error_entries: Mapped[list[BackgroundJobError]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="BackgroundJobError.error_index",
    )

    @property
    def errors(self) -> list[str] | None:
        if not self.error_entries:
            return None
        return [e.message for e in self.error_entries]


class BackgroundJobError(BaseModel):
    """Individual error message for a background job."""

    __tablename__ = "background_job_errors"
    __table_args__ = (
        UniqueConstraint("job_id", "error_index", name="uq_bg_job_error_index"),
        Index("ix_background_job_errors_job_id", "job_id"),
    )

    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("background_jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    error_index: Mapped[int] = mapped_column(Integer, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)

    job: Mapped[BackgroundJob] = relationship(back_populates="error_entries")
