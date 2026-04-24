"""Repository for optimization and backtest execution run persistence."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.execution import BacktestRun, OptimizationRun
from app.repositories.base import RepositoryBase


class ExecutionRepository(RepositoryBase):
    """Repository for :class:`OptimizationRun` and :class:`BacktestRun` persistence.

    Both run models use nullable ``portfolio_id`` and ``job_id`` foreign keys
    so no parent records are required for ad-hoc runs.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session)

    # ------------------------------------------------------------------
    # OptimizationRun
    # ------------------------------------------------------------------

    def get_optimization_run(self, run_id: uuid.UUID) -> OptimizationRun | None:
        """Return a single OptimizationRun by its primary-key UUID, or None."""
        stmt = select(OptimizationRun).where(OptimizationRun.id == run_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def get_optimization_runs(
        self,
        portfolio_id: uuid.UUID | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[OptimizationRun]:
        """Return optimization runs with optional filtering.

        Args:
            portfolio_id: Filter by portfolio UUID. ``None`` returns all portfolios.
            status: Filter by status string. ``None`` returns all statuses.
            limit: Maximum number of rows to return (default 50).

        Returns:
            List of :class:`OptimizationRun` ordered by creation time descending.
        """
        stmt = select(OptimizationRun)

        if portfolio_id is not None:
            stmt = stmt.where(OptimizationRun.portfolio_id == portfolio_id)
        if status is not None:
            stmt = stmt.where(OptimizationRun.status == status)

        stmt = stmt.order_by(OptimizationRun.created_at.desc()).limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def get_latest_optimization(
        self, portfolio_id: uuid.UUID
    ) -> OptimizationRun | None:
        """Return the most recently created optimization run for *portfolio_id*.

        Args:
            portfolio_id: Portfolio UUID to query.

        Returns:
            Most recent :class:`OptimizationRun` or ``None`` if no runs exist.
        """
        stmt = (
            select(OptimizationRun)
            .where(OptimizationRun.portfolio_id == portfolio_id)
            .order_by(OptimizationRun.created_at.desc())
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def create_optimization_run(self, data: dict[str, Any]) -> OptimizationRun:
        """Persist a new optimization run.

        Args:
            data: Dict of column name → value for :class:`OptimizationRun`.

        Returns:
            The newly created :class:`OptimizationRun` with ``id`` populated.
        """
        run = OptimizationRun(**data)
        self.session.add(run)
        self.session.flush()
        self.session.refresh(run)
        return run

    def update_optimization_status(
        self,
        run_id: uuid.UUID,
        status: str,
        result_data: dict[str, Any] | None = None,
    ) -> OptimizationRun:
        """Update the status of an existing optimization run.

        Args:
            run_id: UUID of the :class:`OptimizationRun` to update.
            status: New status string (e.g. ``"completed"``, ``"failed"``).
            result_data: Optional dict of additional attributes to merge into
                the run (only keys that exist as model attributes are applied).

        Returns:
            The updated :class:`OptimizationRun`.

        Raises:
            ValueError: If no run with *run_id* exists.
        """
        stmt = select(OptimizationRun).where(OptimizationRun.id == run_id)
        run = self.session.execute(stmt).scalar_one_or_none()
        if run is None:
            raise ValueError(f"OptimizationRun {run_id!r} not found")

        run.status = status
        if result_data:
            for key, value in result_data.items():
                if hasattr(run, key):
                    setattr(run, key, value)

        self.session.flush()
        self.session.refresh(run)
        return run

    # ------------------------------------------------------------------
    # BacktestRun
    # ------------------------------------------------------------------

    def get_backtest_run(self, run_id: uuid.UUID) -> BacktestRun | None:
        """Return a single BacktestRun by its primary-key UUID, or None."""
        stmt = select(BacktestRun).where(BacktestRun.id == run_id)
        return self.session.execute(stmt).scalar_one_or_none()

    def get_backtest_runs(
        self,
        portfolio_id: uuid.UUID | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[BacktestRun]:
        """Return backtest runs with optional filtering.

        Args:
            portfolio_id: Filter by portfolio UUID. ``None`` returns all portfolios.
            status: Filter by status string. ``None`` returns all statuses.
            limit: Maximum number of rows to return (default 50).

        Returns:
            List of :class:`BacktestRun` ordered by creation time descending.
        """
        stmt = select(BacktestRun)

        if portfolio_id is not None:
            stmt = stmt.where(BacktestRun.portfolio_id == portfolio_id)
        if status is not None:
            stmt = stmt.where(BacktestRun.status == status)

        stmt = stmt.order_by(BacktestRun.created_at.desc()).limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def get_latest_backtest(self, portfolio_id: uuid.UUID) -> BacktestRun | None:
        """Return the most recently created backtest run for *portfolio_id*.

        Args:
            portfolio_id: Portfolio UUID to query.

        Returns:
            Most recent :class:`BacktestRun` or ``None`` if no runs exist.
        """
        stmt = (
            select(BacktestRun)
            .where(BacktestRun.portfolio_id == portfolio_id)
            .order_by(BacktestRun.created_at.desc())
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def create_backtest_run(self, data: dict[str, Any]) -> BacktestRun:
        """Persist a new backtest run.

        Args:
            data: Dict of column name → value for :class:`BacktestRun`.

        Returns:
            The newly created :class:`BacktestRun` with ``id`` populated.
        """
        run = BacktestRun(**data)
        self.session.add(run)
        self.session.flush()
        self.session.refresh(run)
        return run

    def update_backtest_status(
        self,
        run_id: uuid.UUID,
        status: str,
        result_data: dict[str, Any] | None = None,
    ) -> BacktestRun:
        """Update the status of an existing backtest run.

        Args:
            run_id: UUID of the :class:`BacktestRun` to update.
            status: New status string (e.g. ``"completed"``, ``"failed"``).
            result_data: Optional dict of additional attributes to merge into
                the run (only keys that exist as model attributes are applied).

        Returns:
            The updated :class:`BacktestRun`.

        Raises:
            ValueError: If no run with *run_id* exists.
        """
        stmt = select(BacktestRun).where(BacktestRun.id == run_id)
        run = self.session.execute(stmt).scalar_one_or_none()
        if run is None:
            raise ValueError(f"BacktestRun {run_id!r} not found")

        run.status = status
        if result_data:
            for key, value in result_data.items():
                if hasattr(run, key):
                    setattr(run, key, value)

        self.session.flush()
        self.session.refresh(run)
        return run
