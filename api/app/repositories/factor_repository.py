"""Repository for factor research persistence operations."""

from __future__ import annotations

import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.factor import FactorScore, FactorValidationReport
from app.repositories.base import RepositoryBase


class FactorRepository(RepositoryBase):
    """Repository for FactorScore and FactorValidationReport persistence.

    Follows the project's synchronous SQLAlchemy pattern: ``session.flush()``
    inside repository methods, ``session.commit()`` delegated to the caller.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session)

    # ------------------------------------------------------------------
    # FactorScore queries
    # ------------------------------------------------------------------

    def get_scores_by_ticker(
        self,
        ticker: str,
        start_date: datetime.date | None = None,
        end_date: datetime.date | None = None,
    ) -> list[FactorScore]:
        """Return all factor scores for *ticker*, optionally bounded by date range.

        Args:
            ticker: Asset ticker symbol.
            start_date: Inclusive lower bound on ``score_date``.
            end_date: Inclusive upper bound on ``score_date``.

        Returns:
            List of matching :class:`FactorScore` rows ordered by date ascending.
        """
        stmt = select(FactorScore).where(FactorScore.ticker == ticker)

        if start_date is not None:
            stmt = stmt.where(FactorScore.score_date >= start_date)
        if end_date is not None:
            stmt = stmt.where(FactorScore.score_date <= end_date)

        stmt = stmt.order_by(FactorScore.score_date.asc())
        return list(self.session.execute(stmt).scalars().all())

    def get_scores_by_tickers_and_date_range(
        self,
        tickers: list[str],
        start_date: datetime.date,
        end_date: datetime.date,
        factor_type: str | None = None,
    ) -> list[FactorScore]:
        """Return factor scores for given tickers over a date range.

        Args:
            tickers: Asset ticker symbols to filter on.
            start_date: Inclusive lower bound on ``score_date``.
            end_date: Inclusive upper bound on ``score_date``.
            factor_type: Optional filter on factor type.

        Returns:
            List of matching :class:`FactorScore` rows ordered by date ascending.
        """
        stmt = select(FactorScore).where(FactorScore.ticker.in_(tickers))
        stmt = stmt.where(FactorScore.score_date >= start_date)
        stmt = stmt.where(FactorScore.score_date <= end_date)

        if factor_type is not None:
            stmt = stmt.where(FactorScore.factor_type == factor_type)

        stmt = stmt.order_by(FactorScore.score_date.asc(), FactorScore.ticker.asc())
        return list(self.session.execute(stmt).scalars().all())

    def get_scores_by_tickers_at_date(
        self,
        tickers: list[str],
        score_date: datetime.date,
    ) -> list[FactorScore]:
        """Return all factor scores for given tickers on a specific date.

        Args:
            tickers: Asset ticker symbols to filter on.
            score_date: The exact date to filter on.

        Returns:
            List of matching :class:`FactorScore` rows.
        """
        stmt = (
            select(FactorScore)
            .where(FactorScore.ticker.in_(tickers))
            .where(FactorScore.score_date == score_date)
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_scores_by_date(self, score_date: datetime.date) -> list[FactorScore]:
        """Return all factor scores recorded on *score_date*.

        Args:
            score_date: The exact date to filter on.

        Returns:
            List of matching :class:`FactorScore` rows.
        """
        stmt = select(FactorScore).where(FactorScore.score_date == score_date)
        return list(self.session.execute(stmt).scalars().all())

    def bulk_upsert_scores(self, scores: list[dict[str, Any]]) -> int:
        """Upsert a batch of factor scores using the unique constraint.

        Delegates to :meth:`RepositoryBase._upsert` which issues a PostgreSQL
        ``INSERT … ON CONFLICT DO UPDATE``. On SQLite (e.g. in tests) callers
        must mock ``_upsert``.

        Args:
            scores: List of dicts matching :class:`FactorScore` column names.

        Returns:
            Number of rows processed.
        """
        return self._upsert(
            FactorScore,
            scores,
            constraint_name="uq_factor_scores_ticker_date_type",
        )

    # ------------------------------------------------------------------
    # FactorValidationReport queries
    # ------------------------------------------------------------------

    def get_validation_reports(
        self,
        factor_type: str | None = None,
        validation_type: str | None = None,
    ) -> list[FactorValidationReport]:
        """Return validation reports filtered by optional criteria.

        Args:
            factor_type: Filter by factor type string. ``None`` returns all.
            validation_type: Filter by validation type string. ``None`` returns all.

        Returns:
            List of matching :class:`FactorValidationReport` rows.
        """
        stmt = select(FactorValidationReport)

        if factor_type is not None:
            stmt = stmt.where(FactorValidationReport.factor_type == factor_type)
        if validation_type is not None:
            stmt = stmt.where(FactorValidationReport.validation_type == validation_type)

        stmt = stmt.order_by(FactorValidationReport.report_date.desc())
        return list(self.session.execute(stmt).scalars().all())

    def save_validation_report(
        self, report_data: dict[str, Any]
    ) -> FactorValidationReport:
        """Persist a new validation report and flush.

        Args:
            report_data: Column name → value for FactorValidationReport.

        Returns:
            The newly created :class:`FactorValidationReport` with ``id`` populated.
        """
        report = FactorValidationReport(**report_data)
        self.session.add(report)
        self.session.flush()
        self.session.refresh(report)
        return report
