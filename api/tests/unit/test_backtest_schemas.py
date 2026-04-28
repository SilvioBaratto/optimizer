"""Tests for api/app/schemas/backtest.py — backtest run schemas.

Covers (TDD Red → Green → Refactor):
- Round-trip serialization
- ORM deserialization
- camelCase alias output
- Validation rejection
- Optional/nullable field handling
"""

from __future__ import annotations

import datetime
import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError


class TestBacktestRequest:
    """BacktestRequest validates POST /backtest request bodies."""

    def test_importable(self) -> None:
        from app.schemas.backtest import BacktestRequest
        _ = BacktestRequest

    def test_valid_request(self) -> None:
        from app.schemas.backtest import BacktestRequest

        req = BacktestRequest(
            tickers=["AAPL", "MSFT"],
            start_date=datetime.date(2018, 1, 1),
            end_date=datetime.date(2024, 1, 1),
            pipeline_config={"optimizer_type": "hrp"},
        )
        assert req.tickers == ["AAPL", "MSFT"]
        assert req.pipeline_config == {"optimizer_type": "hrp"}

    def test_empty_tickers_raises(self) -> None:
        from app.schemas.backtest import BacktestRequest

        with pytest.raises(ValidationError):
            BacktestRequest(
                tickers=[],
                start_date=datetime.date(2018, 1, 1),
                end_date=datetime.date(2024, 1, 1),
                pipeline_config={},
            )

    def test_optional_pipeline_config(self) -> None:
        from app.schemas.backtest import BacktestRequest

        req = BacktestRequest(
            tickers=["AAPL"],
            start_date=datetime.date(2018, 1, 1),
            end_date=datetime.date(2024, 1, 1),
        )
        assert req.pipeline_config == {}


class TestBacktestRunResponse:
    """BacktestRunResponse serialises ORM rows to camelCase JSON."""

    def _make_orm(self, **overrides: Any) -> SimpleNamespace:
        defaults = {
            "id": uuid.uuid4(),
            "portfolio_id": None,
            "job_id": None,
            "status": "completed",
            "config": {"optimizer_type": "hrp"},
            "equity_curve": {"2024-01-01": 1.0, "2024-06-01": 1.1},
            "drawdowns": {"max_drawdown": -0.05},
            "monthly_returns": {"2024-01": 0.02},
            "yearly_returns": {"2024": 0.12},
            "rolling_metrics": {"sharpe_1y": 1.3},
            "turnover_history": {"2024-01-01": 0.1},
            "cv_fold_metrics": None,
            "summary_stats": {"sharpe": 1.2, "cagr": 0.14},
            "error_message": None,
            "duration_seconds": 15.3,
            "created_at": datetime.datetime(2024, 1, 16, 0, 0, 0),
            "updated_at": datetime.datetime(2024, 1, 16, 0, 0, 0),
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_importable(self) -> None:
        from app.schemas.backtest import BacktestRunResponse
        _ = BacktestRunResponse

    def test_orm_deserialization_succeeds(self) -> None:
        from app.schemas.backtest import BacktestRunResponse

        orm_obj = self._make_orm()
        response = BacktestRunResponse.model_validate(orm_obj)
        assert response.status == "completed"

    def test_camel_case_output(self) -> None:
        from app.schemas.backtest import BacktestRunResponse

        data = BacktestRunResponse.model_validate(
            self._make_orm()
        ).model_dump(by_alias=True)
        assert "equityCurve" in data
        assert "monthlyReturns" in data
        assert "yearlyReturns" in data
        assert "rollingMetrics" in data
        assert "turnoverHistory" in data
        assert "cvFoldMetrics" in data
        assert "summaryStats" in data
        assert "errorMessage" in data
        assert "durationSeconds" in data
        assert "createdAt" in data

    def test_id_serializes_as_string(self) -> None:
        from app.schemas.backtest import BacktestRunResponse

        uid = uuid.uuid4()
        response = BacktestRunResponse.model_validate(self._make_orm(id=uid))
        assert response.id == str(uid)

    def test_nullable_portfolio_id(self) -> None:
        from app.schemas.backtest import BacktestRunResponse

        response = BacktestRunResponse.model_validate(
            self._make_orm(portfolio_id=None)
        )
        assert response.portfolio_id is None

    def test_optional_cv_fold_metrics_none(self) -> None:
        from app.schemas.backtest import BacktestRunResponse

        response = BacktestRunResponse.model_validate(
            self._make_orm(cv_fold_metrics=None)
        )
        assert response.cv_fold_metrics is None

    def test_round_trip_serialization(self) -> None:
        from app.schemas.backtest import BacktestRunResponse

        uid = uuid.uuid4()
        orm_obj = self._make_orm(id=uid)
        data = BacktestRunResponse.model_validate(orm_obj).model_dump()
        reconstructed = BacktestRunResponse(**data)
        assert reconstructed.id == str(uid)


class TestBacktestRunListResponse:
    """BacktestRunListResponse wraps paginated results."""

    def test_importable(self) -> None:
        from app.schemas.backtest import BacktestRunListResponse
        _ = BacktestRunListResponse

    def test_empty_list(self) -> None:
        from app.schemas.backtest import BacktestRunListResponse

        resp = BacktestRunListResponse(items=[], total=0)
        assert resp.total == 0

    def test_camel_case_keys(self) -> None:
        from app.schemas.backtest import BacktestRunListResponse

        resp = BacktestRunListResponse(items=[], total=0)
        data = resp.model_dump(by_alias=True)
        assert "items" in data
        assert "total" in data

    def test_total_must_be_non_negative(self) -> None:
        from app.schemas.backtest import BacktestRunListResponse

        with pytest.raises(ValidationError):
            BacktestRunListResponse(items=[], total=-1)
