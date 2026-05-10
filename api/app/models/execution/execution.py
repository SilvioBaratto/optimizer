"""SQLAlchemy ORM models for execution result persistence.

Stores per-run records produced by the ``/optimize`` and ``/backtest``
endpoints.  Runs outlive portfolio and job deletion (``ondelete="SET NULL"``).
"""

from __future__ import annotations

import uuid

from sqlalchemy import JSON, Float, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models._shared import BaseModel

# JSONB on PostgreSQL, plain JSON elsewhere (e.g. SQLite in tests).
_JSON = JSON().with_variant(JSONB, "postgresql")


class OptimizationRun(BaseModel):
    """Persisted result of a single call to ``POST /api/v1/optimize``.

    ``portfolio_id`` and ``job_id`` are nullable:
    - Ad-hoc runs may not belong to any portfolio.
    - Synchronous runs have no background job.

    ``error_message`` captures failure details for synchronous runs where
    ``job_id`` is ``NULL`` and ``BackgroundJob.error`` cannot be used as a
    fallback.
    """

    __tablename__ = "optimization_runs"
    __table_args__ = (
        Index("ix_optimization_runs_portfolio_id", "portfolio_id"),
        Index("ix_optimization_runs_status", "status"),
        Index("ix_optimization_runs_created_at", "created_at"),
    )

    portfolio_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("portfolios.id", ondelete="SET NULL"),
        nullable=True,
        index=False,  # covered by the explicit composite-friendly index above
    )
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("background_jobs.id", ondelete="SET NULL"),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        server_default="pending",
    )
    optimizer_type: Mapped[str] = mapped_column(String(50), nullable=False)
    universe_tickers: Mapped[list] = mapped_column(_JSON, nullable=False)
    config: Mapped[dict] = mapped_column(_JSON, nullable=False)
    weights: Mapped[dict] = mapped_column(_JSON, nullable=False)
    metrics: Mapped[dict] = mapped_column(_JSON, nullable=False)
    risk_contributions: Mapped[dict] = mapped_column(_JSON, nullable=False)
    efficient_frontier: Mapped[list | None] = mapped_column(_JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    solver_log: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Forward-only relationships — no back_populates on parent models.
    portfolio: Mapped["Portfolio | None"] = relationship(  # type: ignore[name-defined]  # noqa: F821, UP037
        "Portfolio",
        foreign_keys=[portfolio_id],
    )
    job: Mapped["BackgroundJob | None"] = relationship(  # type: ignore[name-defined]  # noqa: F821, UP037
        "BackgroundJob",
        foreign_keys=[job_id],
    )


class BacktestRun(BaseModel):
    """Persisted result of a single call to ``POST /api/v1/backtest``.

    ``portfolio_id`` and ``job_id`` are nullable for the same reasons as
    ``OptimizationRun``.  ``error_message`` stores failure details for
    synchronous (no-job) runs.
    """

    __tablename__ = "backtest_runs"
    __table_args__ = (
        Index("ix_backtest_runs_portfolio_id", "portfolio_id"),
        Index("ix_backtest_runs_status", "status"),
        Index("ix_backtest_runs_created_at", "created_at"),
    )

    portfolio_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("portfolios.id", ondelete="SET NULL"),
        nullable=True,
        index=False,  # covered by the explicit index above
    )
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("background_jobs.id", ondelete="SET NULL"),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        server_default="pending",
    )
    config: Mapped[dict] = mapped_column(_JSON, nullable=False)
    equity_curve: Mapped[dict] = mapped_column(_JSON, nullable=False)
    drawdowns: Mapped[dict] = mapped_column(_JSON, nullable=False)
    monthly_returns: Mapped[dict] = mapped_column(_JSON, nullable=False)
    yearly_returns: Mapped[dict] = mapped_column(_JSON, nullable=False)
    rolling_metrics: Mapped[dict] = mapped_column(_JSON, nullable=False)
    turnover_history: Mapped[dict] = mapped_column(_JSON, nullable=False)
    cv_fold_metrics: Mapped[list | None] = mapped_column(_JSON, nullable=True)
    summary_stats: Mapped[dict] = mapped_column(_JSON, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Forward-only relationships — no back_populates on parent models.
    portfolio: Mapped["Portfolio | None"] = relationship(  # type: ignore[name-defined]  # noqa: F821, UP037
        "Portfolio",
        foreign_keys=[portfolio_id],
    )
    job: Mapped["BackgroundJob | None"] = relationship(  # type: ignore[name-defined]  # noqa: F821, UP037
        "BackgroundJob",
        foreign_keys=[job_id],
    )
