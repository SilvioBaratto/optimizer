"""Repository for risk limit persistence operations."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.risk import RiskLimit
from app.repositories.base import RepositoryBase


class RiskRepository(RepositoryBase):
    """Repository for :class:`RiskLimit` persistence.

    Risk limits require a valid ``portfolio_id`` foreign key — callers must
    ensure the referenced portfolio exists before creating limits.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session)

    def get_limits_by_portfolio(self, portfolio_id: uuid.UUID) -> list[RiskLimit]:
        """Return all risk limits for *portfolio_id*.

        Args:
            portfolio_id: UUID of the portfolio to query.

        Returns:
            List of :class:`RiskLimit` rows for the portfolio.
        """
        stmt = select(RiskLimit).where(RiskLimit.portfolio_id == portfolio_id)
        return list(self.session.execute(stmt).scalars().all())

    def get_breached_limits(self, portfolio_id: uuid.UUID) -> list[RiskLimit]:
        """Return only the breached risk limits for *portfolio_id*.

        Args:
            portfolio_id: UUID of the portfolio to query.

        Returns:
            List of :class:`RiskLimit` rows where ``is_breached`` is ``True``.
        """
        stmt = (
            select(RiskLimit)
            .where(RiskLimit.portfolio_id == portfolio_id)
            .where(RiskLimit.is_breached.is_(True))
        )
        return list(self.session.execute(stmt).scalars().all())

    def create_limit(self, data: dict[str, Any]) -> RiskLimit:
        """Persist a new risk limit.

        Args:
            data: Dict of column name → value for :class:`RiskLimit`.

        Returns:
            The newly created :class:`RiskLimit` with ``id`` populated.
        """
        limit = RiskLimit(**data)
        self.session.add(limit)
        self.session.flush()
        self.session.refresh(limit)
        return limit

    def delete_limit(self, limit_id: uuid.UUID) -> bool:
        """Delete a risk limit by its primary key.

        Args:
            limit_id: UUID of the :class:`RiskLimit` to delete.

        Returns:
            ``True`` if the row was found and deleted, ``False`` if not found.
        """
        stmt = select(RiskLimit).where(RiskLimit.id == limit_id)
        limit = self.session.execute(stmt).scalar_one_or_none()
        if limit is None:
            return False
        self.session.delete(limit)
        self.session.flush()
        return True

    def update_limit_value(
        self,
        limit_id: uuid.UUID,
        current_value: float,
        is_breached: bool,
    ) -> RiskLimit:
        """Update the current observed value and breach flag for a risk limit.

        Args:
            limit_id: UUID of the :class:`RiskLimit` to update.
            current_value: The latest measured value for the risk metric.
            is_breached: Whether the limit is currently breached.

        Returns:
            The updated :class:`RiskLimit`.

        Raises:
            ValueError: If no limit with *limit_id* exists.
        """
        stmt = select(RiskLimit).where(RiskLimit.id == limit_id)
        limit = self.session.execute(stmt).scalar_one_or_none()
        if limit is None:
            raise ValueError(f"RiskLimit {limit_id!r} not found")

        limit.current_value = current_value
        limit.is_breached = is_breached

        self.session.flush()
        self.session.refresh(limit)
        return limit
