"""Tests for OptimizationRun and BacktestRun SQLAlchemy models.

Covers:
- Table creation via Base.metadata.create_all()
- UUID PK and timestamp columns from BaseModel
- Successful row insertion with required fields
- Default status value ("pending")
- Nullable fields: portfolio_id, job_id, error_message, duration_seconds
- JSONB columns accept dict/list payloads
- ORM relationship() accessors to Portfolio and BackgroundJob
- Indexes on portfolio_id, status, created_at
"""

from __future__ import annotations

import uuid
from collections.abc import Generator

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.models.base import Base
from app.models.execution import BacktestRun, OptimizationRun


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


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------


class TestTableCreation:
    """Base.metadata.create_all() must succeed for both tables."""

    def test_optimization_runs_table_exists(self, engine) -> None:
        table_names = inspect(engine).get_table_names()
        assert "optimization_runs" in table_names

    def test_backtest_runs_table_exists(self, engine) -> None:
        table_names = inspect(engine).get_table_names()
        assert "backtest_runs" in table_names


# ---------------------------------------------------------------------------
# OptimizationRun — basic insertion
# ---------------------------------------------------------------------------


class TestOptimizationRunInsertion:
    """OptimizationRun rows can be inserted with required and optional fields."""

    def test_inserts_row_with_required_fields(self, session: Session) -> None:
        run = OptimizationRun(
            optimizer_type="mean_risk",
            universe_tickers=["AAPL", "MSFT"],
            config={"risk_measure": "variance"},
            weights={"AAPL": 0.6, "MSFT": 0.4},
            metrics={"expected_return": 0.12, "volatility": 0.15},
            risk_contributions={"AAPL": 0.5, "MSFT": 0.5},
        )
        session.add(run)
        session.flush()

        result = session.query(OptimizationRun).filter_by(
            optimizer_type="mean_risk"
        ).first()
        assert result is not None
        assert result.optimizer_type == "mean_risk"
        assert result.weights == {"AAPL": 0.6, "MSFT": 0.4}

    def test_uuid_pk_is_assigned_automatically(self, session: Session) -> None:
        run = OptimizationRun(
            optimizer_type="hrp",
            universe_tickers=["AAPL"],
            config={},
            weights={"AAPL": 1.0},
            metrics={},
            risk_contributions={"AAPL": 1.0},
        )
        session.add(run)
        session.flush()

        assert run.id is not None
        assert isinstance(run.id, uuid.UUID)

    def test_timestamps_are_set(self, session: Session) -> None:
        run = OptimizationRun(
            optimizer_type="robust_mean_risk",
            universe_tickers=["AAPL"],
            config={},
            weights={"AAPL": 1.0},
            metrics={},
            risk_contributions={"AAPL": 1.0},
        )
        session.add(run)
        session.flush()

        assert run.created_at is not None
        assert run.updated_at is not None

    def test_default_status_is_pending(self, session: Session) -> None:
        run = OptimizationRun(
            optimizer_type="dr_cvar",
            universe_tickers=["AAPL"],
            config={},
            weights={"AAPL": 1.0},
            metrics={},
            risk_contributions={"AAPL": 1.0},
        )
        session.add(run)
        session.flush()

        result = session.query(OptimizationRun).filter_by(id=run.id).one()
        assert result.status == "pending"

    def test_portfolio_id_is_nullable(self, session: Session) -> None:
        run = OptimizationRun(
            optimizer_type="mean_risk",
            universe_tickers=["TSLA"],
            config={},
            weights={"TSLA": 1.0},
            metrics={},
            risk_contributions={"TSLA": 1.0},
        )
        session.add(run)
        session.flush()

        assert run.portfolio_id is None

    def test_job_id_is_nullable(self, session: Session) -> None:
        run = OptimizationRun(
            optimizer_type="mean_risk",
            universe_tickers=["NVDA"],
            config={},
            weights={"NVDA": 1.0},
            metrics={},
            risk_contributions={"NVDA": 1.0},
        )
        session.add(run)
        session.flush()

        assert run.job_id is None

    def test_error_message_is_nullable(self, session: Session) -> None:
        run = OptimizationRun(
            optimizer_type="mean_risk",
            universe_tickers=["GOOGL"],
            config={},
            weights={"GOOGL": 1.0},
            metrics={},
            risk_contributions={"GOOGL": 1.0},
        )
        session.add(run)
        session.flush()

        assert run.error_message is None

    def test_error_message_stores_text(self, session: Session) -> None:
        run = OptimizationRun(
            optimizer_type="mean_risk",
            universe_tickers=["META"],
            config={},
            weights={},
            metrics={},
            risk_contributions={},
            status="failed",
            error_message="Solver failed to converge after 1000 iterations.",
        )
        session.add(run)
        session.flush()

        result = session.query(OptimizationRun).filter_by(id=run.id).one()
        assert result.error_message == "Solver failed to converge after 1000 iterations."

    def test_optional_jsonb_columns_are_nullable(self, session: Session) -> None:
        run = OptimizationRun(
            optimizer_type="hrp",
            universe_tickers=["AMZN"],
            config={},
            weights={"AMZN": 1.0},
            metrics={},
            risk_contributions={"AMZN": 1.0},
        )
        session.add(run)
        session.flush()

        assert run.efficient_frontier is None
        assert run.solver_log is None
        assert run.duration_seconds is None

    def test_efficient_frontier_and_solver_log_stored(self, session: Session) -> None:
        run = OptimizationRun(
            optimizer_type="mean_risk",
            universe_tickers=["AAPL", "MSFT"],
            config={"risk_measure": "cvar"},
            weights={"AAPL": 0.5, "MSFT": 0.5},
            metrics={"sharpe": 1.2},
            risk_contributions={"AAPL": 0.5, "MSFT": 0.5},
            efficient_frontier=[{"return": 0.1, "risk": 0.12}],
            solver_log="Optimal solution found.",
            duration_seconds=3.14,
        )
        session.add(run)
        session.flush()

        result = session.query(OptimizationRun).filter_by(id=run.id).one()
        assert result.efficient_frontier == [{"return": 0.1, "risk": 0.12}]
        assert result.solver_log == "Optimal solution found."
        assert result.duration_seconds == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# OptimizationRun — ORM relationships
# ---------------------------------------------------------------------------


class TestOptimizationRunRelationships:
    """OptimizationRun exposes portfolio and job relationship attributes."""

    def test_portfolio_relationship_attribute_exists(self, session: Session) -> None:
        run = OptimizationRun(
            optimizer_type="mean_risk",
            universe_tickers=["AAPL"],
            config={},
            weights={"AAPL": 1.0},
            metrics={},
            risk_contributions={"AAPL": 1.0},
        )
        session.add(run)
        session.flush()

        # With no portfolio set, the relationship returns None
        assert run.portfolio is None

    def test_job_relationship_attribute_exists(self, session: Session) -> None:
        run = OptimizationRun(
            optimizer_type="mean_risk",
            universe_tickers=["AAPL"],
            config={},
            weights={"AAPL": 1.0},
            metrics={},
            risk_contributions={"AAPL": 1.0},
        )
        session.add(run)
        session.flush()

        # With no job set, the relationship returns None
        assert run.job is None


# ---------------------------------------------------------------------------
# OptimizationRun — indexes
# ---------------------------------------------------------------------------


class TestOptimizationRunIndexes:
    """Verify that expected indexes are defined on the optimization_runs table."""

    def test_has_index_on_portfolio_id(self, engine) -> None:
        inspector = inspect(engine)
        indexes = {idx["name"] for idx in inspector.get_indexes("optimization_runs")}
        assert any("portfolio_id" in name for name in indexes)

    def test_has_index_on_status(self, engine) -> None:
        inspector = inspect(engine)
        indexes = {idx["name"] for idx in inspector.get_indexes("optimization_runs")}
        assert any("status" in name for name in indexes)


# ---------------------------------------------------------------------------
# BacktestRun — basic insertion
# ---------------------------------------------------------------------------


class TestBacktestRunInsertion:
    """BacktestRun rows can be inserted with required and optional fields."""

    def test_inserts_row_with_required_fields(self, session: Session) -> None:
        run = BacktestRun(
            config={"optimizer": "mean_risk"},
            equity_curve={"2024-01-01": 1.0, "2024-12-31": 1.15},
            drawdowns={"2024-01-01": 0.0},
            monthly_returns={"2024-01": 0.01},
            yearly_returns={"2024": 0.15},
            rolling_metrics={"sharpe": [1.1, 1.2]},
            turnover_history={"2024-01-01": 0.05},
            summary_stats={"total_return": 0.15, "max_drawdown": -0.08},
        )
        session.add(run)
        session.flush()

        result = session.query(BacktestRun).filter_by(id=run.id).one()
        assert result is not None
        assert result.summary_stats == {"total_return": 0.15, "max_drawdown": -0.08}

    def test_uuid_pk_is_assigned_automatically(self, session: Session) -> None:
        run = BacktestRun(
            config={},
            equity_curve={},
            drawdowns={},
            monthly_returns={},
            yearly_returns={},
            rolling_metrics={},
            turnover_history={},
            summary_stats={},
        )
        session.add(run)
        session.flush()

        assert run.id is not None
        assert isinstance(run.id, uuid.UUID)

    def test_timestamps_are_set(self, session: Session) -> None:
        run = BacktestRun(
            config={},
            equity_curve={},
            drawdowns={},
            monthly_returns={},
            yearly_returns={},
            rolling_metrics={},
            turnover_history={},
            summary_stats={},
        )
        session.add(run)
        session.flush()

        assert run.created_at is not None
        assert run.updated_at is not None

    def test_default_status_is_pending(self, session: Session) -> None:
        run = BacktestRun(
            config={},
            equity_curve={},
            drawdowns={},
            monthly_returns={},
            yearly_returns={},
            rolling_metrics={},
            turnover_history={},
            summary_stats={},
        )
        session.add(run)
        session.flush()

        result = session.query(BacktestRun).filter_by(id=run.id).one()
        assert result.status == "pending"

    def test_portfolio_id_is_nullable(self, session: Session) -> None:
        run = BacktestRun(
            config={},
            equity_curve={},
            drawdowns={},
            monthly_returns={},
            yearly_returns={},
            rolling_metrics={},
            turnover_history={},
            summary_stats={},
        )
        session.add(run)
        session.flush()

        assert run.portfolio_id is None

    def test_job_id_is_nullable(self, session: Session) -> None:
        run = BacktestRun(
            config={},
            equity_curve={},
            drawdowns={},
            monthly_returns={},
            yearly_returns={},
            rolling_metrics={},
            turnover_history={},
            summary_stats={},
        )
        session.add(run)
        session.flush()

        assert run.job_id is None

    def test_error_message_is_nullable(self, session: Session) -> None:
        run = BacktestRun(
            config={},
            equity_curve={},
            drawdowns={},
            monthly_returns={},
            yearly_returns={},
            rolling_metrics={},
            turnover_history={},
            summary_stats={},
        )
        session.add(run)
        session.flush()

        assert run.error_message is None

    def test_error_message_stores_text_on_failure(self, session: Session) -> None:
        run = BacktestRun(
            config={},
            equity_curve={},
            drawdowns={},
            monthly_returns={},
            yearly_returns={},
            rolling_metrics={},
            turnover_history={},
            summary_stats={},
            status="failed",
            error_message="Pipeline raised ValueError: insufficient data.",
        )
        session.add(run)
        session.flush()

        result = session.query(BacktestRun).filter_by(id=run.id).one()
        assert result.error_message == "Pipeline raised ValueError: insufficient data."

    def test_cv_fold_metrics_is_nullable(self, session: Session) -> None:
        run = BacktestRun(
            config={},
            equity_curve={},
            drawdowns={},
            monthly_returns={},
            yearly_returns={},
            rolling_metrics={},
            turnover_history={},
            summary_stats={},
        )
        session.add(run)
        session.flush()

        assert run.cv_fold_metrics is None

    def test_duration_seconds_is_nullable(self, session: Session) -> None:
        run = BacktestRun(
            config={},
            equity_curve={},
            drawdowns={},
            monthly_returns={},
            yearly_returns={},
            rolling_metrics={},
            turnover_history={},
            summary_stats={},
        )
        session.add(run)
        session.flush()

        assert run.duration_seconds is None

    def test_optional_columns_stored_when_provided(self, session: Session) -> None:
        run = BacktestRun(
            config={"cv": "walk_forward"},
            equity_curve={"2024-01-01": 1.0},
            drawdowns={"2024-01-01": 0.0},
            monthly_returns={"2024-01": 0.01},
            yearly_returns={"2024": 0.12},
            rolling_metrics={"sharpe": [1.0]},
            turnover_history={"2024-01-01": 0.02},
            summary_stats={"total_return": 0.12},
            cv_fold_metrics=[{"fold": 1, "sharpe": 0.9}],
            duration_seconds=12.5,
        )
        session.add(run)
        session.flush()

        result = session.query(BacktestRun).filter_by(id=run.id).one()
        assert result.cv_fold_metrics == [{"fold": 1, "sharpe": 0.9}]
        assert result.duration_seconds == pytest.approx(12.5)


# ---------------------------------------------------------------------------
# BacktestRun — ORM relationships
# ---------------------------------------------------------------------------


class TestBacktestRunRelationships:
    """BacktestRun exposes portfolio and job relationship attributes."""

    def test_portfolio_relationship_attribute_exists(self, session: Session) -> None:
        run = BacktestRun(
            config={},
            equity_curve={},
            drawdowns={},
            monthly_returns={},
            yearly_returns={},
            rolling_metrics={},
            turnover_history={},
            summary_stats={},
        )
        session.add(run)
        session.flush()

        assert run.portfolio is None

    def test_job_relationship_attribute_exists(self, session: Session) -> None:
        run = BacktestRun(
            config={},
            equity_curve={},
            drawdowns={},
            monthly_returns={},
            yearly_returns={},
            rolling_metrics={},
            turnover_history={},
            summary_stats={},
        )
        session.add(run)
        session.flush()

        assert run.job is None


# ---------------------------------------------------------------------------
# BacktestRun — indexes
# ---------------------------------------------------------------------------


class TestBacktestRunIndexes:
    """Verify that expected indexes are defined on the backtest_runs table."""

    def test_has_index_on_portfolio_id(self, engine) -> None:
        inspector = inspect(engine)
        indexes = {idx["name"] for idx in inspector.get_indexes("backtest_runs")}
        assert any("portfolio_id" in name for name in indexes)

    def test_has_index_on_status(self, engine) -> None:
        inspector = inspect(engine)
        indexes = {idx["name"] for idx in inspector.get_indexes("backtest_runs")}
        assert any("status" in name for name in indexes)
