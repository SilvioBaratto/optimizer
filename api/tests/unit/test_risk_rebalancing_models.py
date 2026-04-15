"""Tests for RiskLimit and RebalancingPolicy SQLAlchemy models.

Covers:
- Table creation via Base.metadata.create_all()
- Successful row insertion with required fields
- Nullable fields on both models
- Default values (is_breached, is_active)
- Unique constraint enforcement raises IntegrityError on duplicate insert
- ORM relationship() attribute exists pointing to Portfolio
- Indexes on portfolio_id for both tables
"""

from __future__ import annotations

import uuid
from collections.abc import Generator

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.models.base import Base
from app.models.rebalancing import RebalancingPolicy
from app.models.risk import RiskLimit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=eng)
    yield eng
    Base.metadata.drop_all(bind=eng)


@pytest.fixture
def session(engine) -> Generator[Session, None, None]:
    connection = engine.connect()
    transaction = connection.begin()
    SessionLocal = sessionmaker(bind=connection)
    sess = SessionLocal()
    sess.begin_nested()
    yield sess
    sess.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def portfolio_id() -> uuid.UUID:
    return uuid.uuid4()


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------


class TestTableCreation:
    """Base.metadata.create_all() must succeed for both tables."""

    def test_risk_limits_table_exists(self, engine) -> None:
        table_names = inspect(engine).get_table_names()
        assert "risk_limits" in table_names

    def test_rebalancing_policies_table_exists(self, engine) -> None:
        table_names = inspect(engine).get_table_names()
        assert "rebalancing_policies" in table_names


# ---------------------------------------------------------------------------
# RiskLimit — insertion
# ---------------------------------------------------------------------------


class TestRiskLimitInsertion:
    """RiskLimit rows can be inserted with required and optional fields."""

    def test_inserts_row_with_required_fields(
        self, session: Session, portfolio_id: uuid.UUID
    ) -> None:
        limit = RiskLimit(
            portfolio_id=portfolio_id,
            metric="var_95",
            limit_type="upper",
            threshold=0.05,
        )
        session.add(limit)
        session.flush()

        result = session.query(RiskLimit).filter_by(portfolio_id=portfolio_id).first()
        assert result is not None
        assert result.metric == "var_95"
        assert result.limit_type == "upper"
        assert result.threshold == pytest.approx(0.05)

    def test_uuid_pk_assigned_automatically(
        self, session: Session, portfolio_id: uuid.UUID
    ) -> None:
        limit = RiskLimit(
            portfolio_id=portfolio_id,
            metric="cvar_99",
            limit_type="upper",
            threshold=0.08,
        )
        session.add(limit)
        session.flush()

        assert limit.id is not None
        assert isinstance(limit.id, uuid.UUID)

    def test_current_value_is_nullable(
        self, session: Session, portfolio_id: uuid.UUID
    ) -> None:
        limit = RiskLimit(
            portfolio_id=portfolio_id,
            metric="max_drawdown",
            limit_type="lower",
            threshold=-0.20,
        )
        session.add(limit)
        session.flush()

        assert limit.current_value is None

    def test_current_value_stored_when_provided(
        self, session: Session, portfolio_id: uuid.UUID
    ) -> None:
        limit = RiskLimit(
            portfolio_id=portfolio_id,
            metric="volatility",
            limit_type="upper",
            threshold=0.15,
            current_value=0.12,
        )
        session.add(limit)
        session.flush()

        result = session.query(RiskLimit).filter_by(id=limit.id).one()
        assert result.current_value == pytest.approx(0.12)

    def test_is_breached_defaults_to_false(
        self, session: Session, portfolio_id: uuid.UUID
    ) -> None:
        limit = RiskLimit(
            portfolio_id=portfolio_id,
            metric="concentration_hhi",
            limit_type="upper",
            threshold=0.25,
        )
        session.add(limit)
        session.flush()

        result = session.query(RiskLimit).filter_by(id=limit.id).one()
        assert result.is_breached is False

    def test_last_checked_at_is_nullable(
        self, session: Session, portfolio_id: uuid.UUID
    ) -> None:
        limit = RiskLimit(
            portfolio_id=portfolio_id,
            metric="var_95",
            limit_type="lower",
            threshold=0.01,
        )
        session.add(limit)
        session.flush()

        assert limit.last_checked_at is None


# ---------------------------------------------------------------------------
# RiskLimit — unique constraint
# ---------------------------------------------------------------------------


class TestRiskLimitUniqueConstraint:
    """Duplicate (portfolio_id, metric, limit_type) must raise IntegrityError."""

    def test_raises_integrity_error_on_duplicate_portfolio_metric_limit_type(
        self, session: Session, portfolio_id: uuid.UUID
    ) -> None:
        limit_a = RiskLimit(
            portfolio_id=portfolio_id,
            metric="var_95",
            limit_type="upper",
            threshold=0.05,
        )
        limit_b = RiskLimit(
            portfolio_id=portfolio_id,
            metric="var_95",
            limit_type="upper",
            threshold=0.07,
        )
        session.add(limit_a)
        session.flush()

        session.add(limit_b)
        with pytest.raises(IntegrityError):
            session.flush()

    def test_allows_same_metric_different_limit_type(
        self, session: Session, portfolio_id: uuid.UUID
    ) -> None:
        limit_a = RiskLimit(
            portfolio_id=portfolio_id,
            metric="cvar_99",
            limit_type="upper",
            threshold=0.10,
        )
        limit_b = RiskLimit(
            portfolio_id=portfolio_id,
            metric="cvar_99",
            limit_type="lower",
            threshold=0.02,
        )
        session.add_all([limit_a, limit_b])
        session.flush()  # must not raise

        count = session.query(RiskLimit).filter_by(
            portfolio_id=portfolio_id, metric="cvar_99"
        ).count()
        assert count == 2

    def test_allows_same_metric_and_limit_type_different_portfolio(
        self, session: Session
    ) -> None:
        pid_a = uuid.uuid4()
        pid_b = uuid.uuid4()
        limit_a = RiskLimit(
            portfolio_id=pid_a,
            metric="volatility",
            limit_type="upper",
            threshold=0.15,
        )
        limit_b = RiskLimit(
            portfolio_id=pid_b,
            metric="volatility",
            limit_type="upper",
            threshold=0.20,
        )
        session.add_all([limit_a, limit_b])
        session.flush()  # must not raise

        count = session.query(RiskLimit).filter_by(metric="volatility").count()
        assert count == 2


# ---------------------------------------------------------------------------
# RiskLimit — ORM relationship
# ---------------------------------------------------------------------------


class TestRiskLimitRelationship:
    """RiskLimit exposes a portfolio relationship attribute."""

    def test_portfolio_relationship_attribute_exists(
        self, session: Session, portfolio_id: uuid.UUID
    ) -> None:
        limit = RiskLimit(
            portfolio_id=portfolio_id,
            metric="var_95",
            limit_type="upper",
            threshold=0.05,
        )
        session.add(limit)
        session.flush()

        # No Portfolio row inserted — relationship resolves to None
        assert limit.portfolio is None


# ---------------------------------------------------------------------------
# RiskLimit — indexes
# ---------------------------------------------------------------------------


class TestRiskLimitIndexes:
    """risk_limits table has an index on portfolio_id."""

    def test_has_index_on_portfolio_id(self, engine) -> None:
        inspector = inspect(engine)
        indexes = {idx["name"] for idx in inspector.get_indexes("risk_limits")}
        assert any("portfolio_id" in name for name in indexes)


# ---------------------------------------------------------------------------
# RebalancingPolicy — insertion
# ---------------------------------------------------------------------------


class TestRebalancingPolicyInsertion:
    """RebalancingPolicy rows can be inserted with required and optional fields."""

    def test_inserts_row_with_required_fields(
        self, session: Session, portfolio_id: uuid.UUID
    ) -> None:
        policy = RebalancingPolicy(
            portfolio_id=portfolio_id,
            name="quarterly-calendar",
            policy_type="calendar",
            config={"frequency_days": 63},
        )
        session.add(policy)
        session.flush()

        result = session.query(RebalancingPolicy).filter_by(
            portfolio_id=portfolio_id
        ).first()
        assert result is not None
        assert result.name == "quarterly-calendar"
        assert result.policy_type == "calendar"

    def test_uuid_pk_assigned_automatically(
        self, session: Session, portfolio_id: uuid.UUID
    ) -> None:
        policy = RebalancingPolicy(
            portfolio_id=portfolio_id,
            name="threshold-5pct",
            policy_type="threshold",
            config={"threshold": 0.05, "relative": True},
        )
        session.add(policy)
        session.flush()

        assert policy.id is not None
        assert isinstance(policy.id, uuid.UUID)

    def test_is_active_defaults_to_false(
        self, session: Session, portfolio_id: uuid.UUID
    ) -> None:
        policy = RebalancingPolicy(
            portfolio_id=portfolio_id,
            name="hybrid-monthly",
            policy_type="hybrid",
            config={"frequency_days": 21, "threshold": 0.10},
        )
        session.add(policy)
        session.flush()

        result = session.query(RebalancingPolicy).filter_by(id=policy.id).one()
        assert result.is_active is False

    def test_config_stores_dict_payload(
        self, session: Session, portfolio_id: uuid.UUID
    ) -> None:
        cfg = {"frequency_days": 252, "relative": False, "cost_bps": 10}
        policy = RebalancingPolicy(
            portfolio_id=portfolio_id,
            name="annual-calendar",
            policy_type="calendar",
            config=cfg,
        )
        session.add(policy)
        session.flush()

        result = session.query(RebalancingPolicy).filter_by(id=policy.id).one()
        assert result.config == cfg


# ---------------------------------------------------------------------------
# RebalancingPolicy — unique constraint
# ---------------------------------------------------------------------------


class TestRebalancingPolicyUniqueConstraint:
    """Duplicate (portfolio_id, name) must raise IntegrityError."""

    def test_raises_integrity_error_on_duplicate_portfolio_name(
        self, session: Session, portfolio_id: uuid.UUID
    ) -> None:
        policy_a = RebalancingPolicy(
            portfolio_id=portfolio_id,
            name="quarterly-calendar",
            policy_type="calendar",
            config={"frequency_days": 63},
        )
        policy_b = RebalancingPolicy(
            portfolio_id=portfolio_id,
            name="quarterly-calendar",
            policy_type="threshold",
            config={"threshold": 0.05},
        )
        session.add(policy_a)
        session.flush()

        session.add(policy_b)
        with pytest.raises(IntegrityError):
            session.flush()

    def test_allows_same_name_different_portfolio(
        self, session: Session
    ) -> None:
        pid_a = uuid.uuid4()
        pid_b = uuid.uuid4()
        policy_a = RebalancingPolicy(
            portfolio_id=pid_a,
            name="default",
            policy_type="calendar",
            config={"frequency_days": 63},
        )
        policy_b = RebalancingPolicy(
            portfolio_id=pid_b,
            name="default",
            policy_type="calendar",
            config={"frequency_days": 63},
        )
        session.add_all([policy_a, policy_b])
        session.flush()  # must not raise

        count = session.query(RebalancingPolicy).filter_by(name="default").count()
        assert count == 2

    def test_allows_same_portfolio_different_name(
        self, session: Session, portfolio_id: uuid.UUID
    ) -> None:
        policy_a = RebalancingPolicy(
            portfolio_id=portfolio_id,
            name="policy-alpha",
            policy_type="calendar",
            config={"frequency_days": 21},
        )
        policy_b = RebalancingPolicy(
            portfolio_id=portfolio_id,
            name="policy-beta",
            policy_type="threshold",
            config={"threshold": 0.05},
        )
        session.add_all([policy_a, policy_b])
        session.flush()  # must not raise

        count = session.query(RebalancingPolicy).filter_by(
            portfolio_id=portfolio_id
        ).count()
        assert count == 2


# ---------------------------------------------------------------------------
# RebalancingPolicy — ORM relationship
# ---------------------------------------------------------------------------


class TestRebalancingPolicyRelationship:
    """RebalancingPolicy exposes a portfolio relationship attribute."""

    def test_portfolio_relationship_attribute_exists(
        self, session: Session, portfolio_id: uuid.UUID
    ) -> None:
        policy = RebalancingPolicy(
            portfolio_id=portfolio_id,
            name="forward-only-check",
            policy_type="calendar",
            config={"frequency_days": 63},
        )
        session.add(policy)
        session.flush()

        # No Portfolio row inserted — relationship resolves to None
        assert policy.portfolio is None


# ---------------------------------------------------------------------------
# RebalancingPolicy — indexes
# ---------------------------------------------------------------------------


class TestRebalancingPolicyIndexes:
    """rebalancing_policies table has an index on portfolio_id."""

    def test_has_index_on_portfolio_id(self, engine) -> None:
        inspector = inspect(engine)
        indexes = {idx["name"] for idx in inspector.get_indexes("rebalancing_policies")}
        assert any("portfolio_id" in name for name in indexes)
