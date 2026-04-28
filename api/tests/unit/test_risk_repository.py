"""Unit tests for RiskRepository.

RiskLimit has a NOT NULL FK to portfolios — each test that inserts a
RiskLimit must first create a Portfolio.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.orm import Session

from app.models.portfolio import Portfolio
from app.repositories.risk_repository import RiskRepository

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_portfolio(db_session: Session) -> Portfolio:
    portfolio = Portfolio(name=f"test-{uuid.uuid4().hex[:8]}")
    db_session.add(portfolio)
    db_session.flush()
    return portfolio


def _limit_data(portfolio_id: uuid.UUID, **overrides: object) -> dict:
    base: dict = {
        "portfolio_id": portfolio_id,
        "metric": "volatility",
        "limit_type": "max",
        "threshold": 0.20,
        "is_breached": False,
    }
    return {**base, **overrides}


# ---------------------------------------------------------------------------
# get_limits_by_portfolio
# ---------------------------------------------------------------------------


class TestGetLimitsByPortfolio:
    def test_returns_empty_list_when_no_data(self, db_session: Session) -> None:
        repo = RiskRepository(db_session)

        result = repo.get_limits_by_portfolio(uuid.uuid4())

        assert result == []

    def test_returns_limits_for_portfolio(self, db_session: Session) -> None:
        portfolio = _create_portfolio(db_session)
        other = _create_portfolio(db_session)
        repo = RiskRepository(db_session)
        repo.create_limit(_limit_data(portfolio.id, metric="volatility"))
        repo.create_limit(_limit_data(portfolio.id, metric="drawdown", limit_type="min"))
        repo.create_limit(_limit_data(other.id, metric="volatility"))

        result = repo.get_limits_by_portfolio(portfolio.id)

        assert len(result) == 2
        assert all(r.portfolio_id == portfolio.id for r in result)


# ---------------------------------------------------------------------------
# get_breached_limits
# ---------------------------------------------------------------------------


class TestGetBreachedLimits:
    def test_returns_only_breached_limits(self, db_session: Session) -> None:
        portfolio = _create_portfolio(db_session)
        repo = RiskRepository(db_session)
        repo.create_limit(_limit_data(portfolio.id, metric="volatility", is_breached=False))
        repo.create_limit(_limit_data(portfolio.id, metric="drawdown", limit_type="min", is_breached=True))

        result = repo.get_breached_limits(portfolio.id)

        assert len(result) == 1
        assert result[0].is_breached is True
        assert result[0].metric == "drawdown"

    def test_returns_empty_when_no_breaches(self, db_session: Session) -> None:
        portfolio = _create_portfolio(db_session)
        repo = RiskRepository(db_session)
        repo.create_limit(_limit_data(portfolio.id, is_breached=False))

        result = repo.get_breached_limits(portfolio.id)

        assert result == []


# ---------------------------------------------------------------------------
# create_limit
# ---------------------------------------------------------------------------


class TestCreateLimit:
    def test_creates_and_returns_limit(self, db_session: Session) -> None:
        portfolio = _create_portfolio(db_session)
        repo = RiskRepository(db_session)
        data = _limit_data(portfolio.id, metric="cvar", threshold=0.05)

        limit = repo.create_limit(data)

        assert limit.id is not None
        assert limit.portfolio_id == portfolio.id
        assert limit.metric == "cvar"
        assert limit.threshold == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# delete_limit
# ---------------------------------------------------------------------------


class TestDeleteLimit:
    def test_returns_true_when_found_and_deleted(self, db_session: Session) -> None:
        portfolio = _create_portfolio(db_session)
        repo = RiskRepository(db_session)
        limit = repo.create_limit(_limit_data(portfolio.id))

        result = repo.delete_limit(limit.id)

        assert result is True

    def test_returns_false_when_not_found(self, db_session: Session) -> None:
        repo = RiskRepository(db_session)

        result = repo.delete_limit(uuid.uuid4())

        assert result is False


# ---------------------------------------------------------------------------
# update_limit_value
# ---------------------------------------------------------------------------


class TestUpdateLimitValue:
    def test_updates_current_value_and_is_breached(self, db_session: Session) -> None:
        portfolio = _create_portfolio(db_session)
        repo = RiskRepository(db_session)
        limit = repo.create_limit(_limit_data(portfolio.id, is_breached=False))

        updated = repo.update_limit_value(limit.id, current_value=0.25, is_breached=True)

        assert updated.current_value == pytest.approx(0.25)
        assert updated.is_breached is True

    def test_raises_value_error_when_limit_not_found(self, db_session: Session) -> None:
        repo = RiskRepository(db_session)

        with pytest.raises(ValueError, match="not found"):
            repo.update_limit_value(uuid.uuid4(), current_value=0.1, is_breached=False)
