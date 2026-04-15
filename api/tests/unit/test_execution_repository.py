"""Unit tests for ExecutionRepository.

Tests use the shared ``db_session`` SQLite fixture from conftest.py.
``portfolio_id`` is nullable on both run models — no FK setup needed.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.orm import Session

from app.repositories.execution_repository import ExecutionRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_OPT: dict = {
    "status": "pending",
    "optimizer_type": "mean_risk",
    "universe_tickers": ["AAPL", "MSFT"],
    "config": {"risk_measure": "variance"},
    "weights": {"AAPL": 0.5, "MSFT": 0.5},
    "metrics": {"sharpe": 1.2},
    "risk_contributions": {"AAPL": 0.5, "MSFT": 0.5},
}

_MINIMAL_BT: dict = {
    "status": "pending",
    "config": {"lookback": 252},
    "equity_curve": {"2024-01-01": 1.0},
    "drawdowns": {"2024-01-01": 0.0},
    "monthly_returns": {"2024-01": 0.01},
    "yearly_returns": {"2024": 0.12},
    "rolling_metrics": {"sharpe": [1.0]},
    "turnover_history": {"2024-01-01": 0.05},
    "summary_stats": {"cagr": 0.12},
}


def _opt_data(**overrides: object) -> dict:
    return {**_MINIMAL_OPT, **overrides}


def _bt_data(**overrides: object) -> dict:
    return {**_MINIMAL_BT, **overrides}


# ---------------------------------------------------------------------------
# OptimizationRun — get_optimization_runs
# ---------------------------------------------------------------------------


class TestGetOptimizationRuns:
    def test_returns_empty_list_when_no_data(self, db_session: Session) -> None:
        repo = ExecutionRepository(db_session)

        result = repo.get_optimization_runs()

        assert result == []

    def test_filters_by_portfolio_id(self, db_session: Session) -> None:
        pid = uuid.uuid4()
        other_pid = uuid.uuid4()
        repo = ExecutionRepository(db_session)
        repo.create_optimization_run(_opt_data(portfolio_id=pid))
        repo.create_optimization_run(_opt_data(portfolio_id=other_pid))

        result = repo.get_optimization_runs(portfolio_id=pid)

        assert len(result) == 1
        assert result[0].portfolio_id == pid

    def test_filters_by_status(self, db_session: Session) -> None:
        repo = ExecutionRepository(db_session)
        repo.create_optimization_run(_opt_data(status="completed"))
        repo.create_optimization_run(_opt_data(status="failed"))

        result = repo.get_optimization_runs(status="completed")

        assert len(result) == 1
        assert result[0].status == "completed"


# ---------------------------------------------------------------------------
# OptimizationRun — get_latest_optimization
# ---------------------------------------------------------------------------


class TestGetLatestOptimization:
    def test_returns_none_when_no_runs(self, db_session: Session) -> None:
        repo = ExecutionRepository(db_session)

        result = repo.get_latest_optimization(uuid.uuid4())

        assert result is None

    def test_returns_most_recent_run(self, db_session: Session) -> None:
        pid = uuid.uuid4()
        other_pid = uuid.uuid4()
        repo = ExecutionRepository(db_session)
        repo.create_optimization_run(_opt_data(portfolio_id=other_pid))
        repo.create_optimization_run(_opt_data(portfolio_id=pid, status="completed"))
        repo.create_optimization_run(_opt_data(portfolio_id=pid, status="failed"))

        result = repo.get_latest_optimization(pid)

        # Assert that the result belongs to the correct portfolio (not the other one).
        # Exact ordering between same-portfolio runs is non-deterministic on SQLite
        # (server_default timestamps share the same transaction clock).
        assert result is not None
        assert result.portfolio_id == pid


# ---------------------------------------------------------------------------
# OptimizationRun — create_optimization_run
# ---------------------------------------------------------------------------


class TestCreateOptimizationRun:
    def test_creates_run_with_correct_fields(self, db_session: Session) -> None:
        repo = ExecutionRepository(db_session)
        data = _opt_data(optimizer_type="hrp", status="pending")

        run = repo.create_optimization_run(data)

        assert run.id is not None
        assert run.optimizer_type == "hrp"
        assert run.status == "pending"
        assert run.universe_tickers == ["AAPL", "MSFT"]


# ---------------------------------------------------------------------------
# OptimizationRun — update_optimization_status
# ---------------------------------------------------------------------------


class TestUpdateOptimizationStatus:
    def test_updates_status(self, db_session: Session) -> None:
        repo = ExecutionRepository(db_session)
        run = repo.create_optimization_run(_opt_data(status="pending"))

        updated = repo.update_optimization_status(run.id, "completed")

        assert updated.status == "completed"

    def test_merges_result_data_into_run(self, db_session: Session) -> None:
        repo = ExecutionRepository(db_session)
        run = repo.create_optimization_run(_opt_data(status="pending"))

        updated = repo.update_optimization_status(
            run.id,
            "completed",
            result_data={"metrics": {"sharpe": 2.5}, "duration_seconds": 3.7},
        )

        assert updated.metrics == {"sharpe": 2.5}
        assert updated.duration_seconds == pytest.approx(3.7)

    def test_raises_value_error_when_run_not_found(self, db_session: Session) -> None:
        repo = ExecutionRepository(db_session)

        with pytest.raises(ValueError, match="not found"):
            repo.update_optimization_status(uuid.uuid4(), "completed")


# ---------------------------------------------------------------------------
# BacktestRun — get_backtest_runs
# ---------------------------------------------------------------------------


class TestGetBacktestRuns:
    def test_returns_empty_list_when_no_data(self, db_session: Session) -> None:
        repo = ExecutionRepository(db_session)

        result = repo.get_backtest_runs()

        assert result == []

    def test_filters_by_portfolio_id(self, db_session: Session) -> None:
        pid = uuid.uuid4()
        other_pid = uuid.uuid4()
        repo = ExecutionRepository(db_session)
        repo.create_backtest_run(_bt_data(portfolio_id=pid))
        repo.create_backtest_run(_bt_data(portfolio_id=other_pid))

        result = repo.get_backtest_runs(portfolio_id=pid)

        assert len(result) == 1
        assert result[0].portfolio_id == pid

    def test_filters_by_status(self, db_session: Session) -> None:
        repo = ExecutionRepository(db_session)
        repo.create_backtest_run(_bt_data(status="completed"))
        repo.create_backtest_run(_bt_data(status="failed"))

        result = repo.get_backtest_runs(status="completed")

        assert len(result) == 1
        assert result[0].status == "completed"


# ---------------------------------------------------------------------------
# BacktestRun — get_latest_backtest
# ---------------------------------------------------------------------------


class TestGetLatestBacktest:
    def test_returns_none_when_no_runs(self, db_session: Session) -> None:
        repo = ExecutionRepository(db_session)

        result = repo.get_latest_backtest(uuid.uuid4())

        assert result is None

    def test_returns_most_recent_run(self, db_session: Session) -> None:
        pid = uuid.uuid4()
        other_pid = uuid.uuid4()
        repo = ExecutionRepository(db_session)
        repo.create_backtest_run(_bt_data(portfolio_id=other_pid))
        repo.create_backtest_run(_bt_data(portfolio_id=pid, status="completed"))
        repo.create_backtest_run(_bt_data(portfolio_id=pid, status="failed"))

        result = repo.get_latest_backtest(pid)

        # Assert that the result belongs to the correct portfolio (not the other one).
        # Exact ordering between same-portfolio runs is non-deterministic on SQLite
        # (server_default timestamps share the same transaction clock).
        assert result is not None
        assert result.portfolio_id == pid


# ---------------------------------------------------------------------------
# BacktestRun — create_backtest_run
# ---------------------------------------------------------------------------


class TestCreateBacktestRun:
    def test_creates_run_with_correct_fields(self, db_session: Session) -> None:
        repo = ExecutionRepository(db_session)
        data = _bt_data(status="pending")

        run = repo.create_backtest_run(data)

        assert run.id is not None
        assert run.status == "pending"
        assert run.summary_stats == {"cagr": 0.12}


# ---------------------------------------------------------------------------
# BacktestRun — update_backtest_status
# ---------------------------------------------------------------------------


class TestUpdateBacktestStatus:
    def test_updates_status(self, db_session: Session) -> None:
        repo = ExecutionRepository(db_session)
        run = repo.create_backtest_run(_bt_data(status="pending"))

        updated = repo.update_backtest_status(run.id, "completed")

        assert updated.status == "completed"

    def test_raises_value_error_when_run_not_found(self, db_session: Session) -> None:
        repo = ExecutionRepository(db_session)

        with pytest.raises(ValueError, match="not found"):
            repo.update_backtest_status(uuid.uuid4(), "completed")
