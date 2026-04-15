"""Repository for rebalancing policy persistence operations."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.rebalancing import RebalancingPolicy
from app.repositories.base import RepositoryBase


class RebalancingRepository(RepositoryBase):
    """Repository for :class:`RebalancingPolicy` persistence.

    Rebalancing policies require a valid ``portfolio_id`` foreign key.
    The ``activate_policy`` method atomically deactivates all other policies
    for the portfolio within the same flush, ensuring a single active policy
    invariant without a separate unique index.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session)

    def get_policies_by_portfolio(
        self, portfolio_id: uuid.UUID
    ) -> list[RebalancingPolicy]:
        """Return all rebalancing policies for *portfolio_id*.

        Args:
            portfolio_id: UUID of the portfolio to query.

        Returns:
            List of :class:`RebalancingPolicy` rows for the portfolio.
        """
        stmt = select(RebalancingPolicy).where(
            RebalancingPolicy.portfolio_id == portfolio_id
        )
        return list(self.session.execute(stmt).scalars().all())

    def get_active_policy(self, portfolio_id: uuid.UUID) -> RebalancingPolicy | None:
        """Return the single active rebalancing policy for *portfolio_id*.

        Args:
            portfolio_id: UUID of the portfolio to query.

        Returns:
            The active :class:`RebalancingPolicy` or ``None`` if none is active.
        """
        stmt = (
            select(RebalancingPolicy)
            .where(RebalancingPolicy.portfolio_id == portfolio_id)
            .where(RebalancingPolicy.is_active.is_(True))
            .limit(1)
        )
        return self.session.execute(stmt).scalar_one_or_none()

    def create_policy(self, data: dict[str, Any]) -> RebalancingPolicy:
        """Persist a new rebalancing policy.

        Args:
            data: Dict of column name → value for :class:`RebalancingPolicy`.

        Returns:
            The newly created :class:`RebalancingPolicy` with ``id`` populated.
        """
        policy = RebalancingPolicy(**data)
        self.session.add(policy)
        self.session.flush()
        self.session.refresh(policy)
        return policy

    def activate_policy(self, policy_id: uuid.UUID, portfolio_id: uuid.UUID) -> None:
        """Atomically activate *policy_id* and deactivate all other policies.

        All deactivations and the activation are issued in a single flush so
        the database never sees an intermediate state with multiple active
        policies or no active policy.

        Args:
            policy_id: UUID of the :class:`RebalancingPolicy` to activate.
            portfolio_id: UUID of the owning portfolio — used to scope the
                deactivation of sibling policies.

        Raises:
            ValueError: If no policy with *policy_id* exists.
        """
        target_stmt = select(RebalancingPolicy).where(RebalancingPolicy.id == policy_id)
        target = self.session.execute(target_stmt).scalar_one_or_none()
        if target is None:
            raise ValueError(f"RebalancingPolicy {policy_id!r} not found")

        # Deactivate all other policies for this portfolio in one pass.
        siblings_stmt = select(RebalancingPolicy).where(
            RebalancingPolicy.portfolio_id == portfolio_id,
            RebalancingPolicy.id != policy_id,
            RebalancingPolicy.is_active.is_(True),
        )
        for sibling in self.session.execute(siblings_stmt).scalars().all():
            sibling.is_active = False

        target.is_active = True
        self.session.flush()
