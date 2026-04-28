"""Unit tests for RebalancingRepository.

RebalancingPolicy has a NOT NULL FK to portfolios — each test that inserts
a policy must first create a Portfolio.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.orm import Session

from app.models.portfolio import Portfolio
from app.models.rebalancing import RebalancingPolicy
from app.repositories.rebalancing_repository import RebalancingRepository

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_portfolio(db_session: Session) -> Portfolio:
    portfolio = Portfolio(name=f"test-{uuid.uuid4().hex[:8]}")
    db_session.add(portfolio)
    db_session.flush()
    return portfolio


def _policy_data(
    portfolio_id: uuid.UUID,
    name: str = "default",
    policy_type: str = "calendar",
    is_active: bool = False,
    **overrides: object,
) -> dict:
    base: dict = {
        "portfolio_id": portfolio_id,
        "name": name,
        "policy_type": policy_type,
        "config": {"interval_days": 63},
        "is_active": is_active,
    }
    return {**base, **overrides}


# ---------------------------------------------------------------------------
# get_policies_by_portfolio
# ---------------------------------------------------------------------------


class TestGetPoliciesByPortfolio:
    def test_returns_empty_list_when_no_data(self, db_session: Session) -> None:
        repo = RebalancingRepository(db_session)

        result = repo.get_policies_by_portfolio(uuid.uuid4())

        assert result == []

    def test_returns_policies_for_portfolio(self, db_session: Session) -> None:
        portfolio = _create_portfolio(db_session)
        other = _create_portfolio(db_session)
        repo = RebalancingRepository(db_session)
        repo.create_policy(_policy_data(portfolio.id, name="monthly"))
        repo.create_policy(_policy_data(portfolio.id, name="quarterly"))
        repo.create_policy(_policy_data(other.id, name="monthly"))

        result = repo.get_policies_by_portfolio(portfolio.id)

        assert len(result) == 2
        assert all(r.portfolio_id == portfolio.id for r in result)


# ---------------------------------------------------------------------------
# get_active_policy
# ---------------------------------------------------------------------------


class TestGetActivePolicy:
    def test_returns_none_when_no_active_policy(self, db_session: Session) -> None:
        portfolio = _create_portfolio(db_session)
        repo = RebalancingRepository(db_session)
        repo.create_policy(_policy_data(portfolio.id, is_active=False))

        result = repo.get_active_policy(portfolio.id)

        assert result is None

    def test_returns_active_policy(self, db_session: Session) -> None:
        portfolio = _create_portfolio(db_session)
        repo = RebalancingRepository(db_session)
        repo.create_policy(_policy_data(portfolio.id, name="inactive", is_active=False))
        active = repo.create_policy(_policy_data(portfolio.id, name="active", is_active=True))

        result = repo.get_active_policy(portfolio.id)

        assert result is not None
        assert result.id == active.id
        assert result.is_active is True


# ---------------------------------------------------------------------------
# create_policy
# ---------------------------------------------------------------------------


class TestCreatePolicy:
    def test_creates_and_returns_policy(self, db_session: Session) -> None:
        portfolio = _create_portfolio(db_session)
        repo = RebalancingRepository(db_session)
        data = _policy_data(portfolio.id, name="threshold-5pct", policy_type="threshold")

        policy = repo.create_policy(data)

        assert policy.id is not None
        assert policy.portfolio_id == portfolio.id
        assert policy.name == "threshold-5pct"
        assert policy.policy_type == "threshold"


# ---------------------------------------------------------------------------
# activate_policy
# ---------------------------------------------------------------------------


class TestActivatePolicy:
    def test_activates_target_and_deactivates_others(self, db_session: Session) -> None:
        portfolio = _create_portfolio(db_session)
        repo = RebalancingRepository(db_session)
        first = repo.create_policy(_policy_data(portfolio.id, name="first", is_active=True))
        second = repo.create_policy(_policy_data(portfolio.id, name="second", is_active=False))

        repo.activate_policy(second.id, portfolio.id)

        db_session.expire_all()
        updated_first = db_session.get(RebalancingPolicy, first.id)
        updated_second = db_session.get(RebalancingPolicy, second.id)
        assert updated_first is not None
        assert updated_second is not None
        assert updated_first.is_active is False
        assert updated_second.is_active is True

    def test_raises_value_error_when_policy_not_found(self, db_session: Session) -> None:
        portfolio = _create_portfolio(db_session)
        repo = RebalancingRepository(db_session)

        with pytest.raises(ValueError, match="not found"):
            repo.activate_policy(uuid.uuid4(), portfolio.id)
