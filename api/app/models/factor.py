"""SQLAlchemy ORM models for factor research persistence."""

from __future__ import annotations

import datetime

from sqlalchemy import JSON, Date, Float, Index, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import BaseModel

# JSONB on PostgreSQL, plain JSON elsewhere (e.g. SQLite in tests).
_JSON = JSON().with_variant(JSONB, "postgresql")


class FactorScore(BaseModel):
    """Per-ticker, per-date factor score from ``compute_all_factors()``.

    Unique constraint on ``(ticker, score_date, factor_type)`` prevents
    duplicate inserts for the same computation run.
    """

    __tablename__ = "factor_scores"
    __table_args__ = (
        UniqueConstraint(
            "ticker",
            "score_date",
            "factor_type",
            name="uq_factor_scores_ticker_date_type",
        ),
        Index("ix_factor_scores_ticker", "ticker"),
        Index("ix_factor_scores_score_date", "score_date"),
        Index("ix_factor_scores_factor_type", "factor_type"),
    )

    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    score_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    factor_type: Mapped[str] = mapped_column(String(50), nullable=False)
    factor_group: Mapped[str] = mapped_column(String(50), nullable=False)
    raw_score: Mapped[float] = mapped_column(Float, nullable=False)
    standardized_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    composite_score: Mapped[float | None] = mapped_column(Float, nullable=True)


class FactorValidationReport(BaseModel):
    """Validation report from ``run_factor_validation()``.

    Also covers ``run_factor_oos_validation()``.

    ``factor_type`` is nullable — null signals an aggregate report covering all factors.

    Unique constraint on ``(report_date, factor_type, validation_type)``.
    Note: both SQLite and PostgreSQL treat each NULL as distinct in unique
    constraints, so multiple aggregate (null factor_type) rows with the same
    ``report_date`` and ``validation_type`` are allowed at the DB level.
    Enforce single-aggregate uniqueness in the service layer if required.
    """

    __tablename__ = "factor_validation_reports"
    __table_args__ = (
        UniqueConstraint(
            "report_date",
            "factor_type",
            "validation_type",
            name="uq_factor_validation_reports_date_type_validation",
        ),
        Index("ix_factor_validation_reports_report_date", "report_date"),
        Index("ix_factor_validation_reports_factor_type", "factor_type"),
    )

    report_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    factor_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    validation_type: Mapped[str] = mapped_column(String(20), nullable=False)
    ic_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    ic_std: Mapped[float | None] = mapped_column(Float, nullable=True)
    icir: Mapped[float | None] = mapped_column(Float, nullable=True)
    t_stat: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    vif: Mapped[float | None] = mapped_column(Float, nullable=True)
    details: Mapped[dict | None] = mapped_column(_JSON, nullable=True)
