"""Tests for api/app/schemas/optimization.py — optimization run schemas.

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


class TestOptimizeRequest:
    """OptimizeRequest validates POST /optimize request bodies."""

    def test_importable(self) -> None:
        from app.schemas.optimization.optimization import OptimizeRequest

        _ = OptimizeRequest

    def test_valid_request(self) -> None:
        from app.schemas.optimization.optimization import OptimizeRequest

        req = OptimizeRequest(
            tickers=["AAPL", "MSFT", "GOOGL"],
            start_date=datetime.date(2020, 1, 1),
            end_date=datetime.date(2024, 1, 1),
            optimizer_type="mean_risk",
            config={"risk_measure": "variance"},
        )
        assert req.optimizer_type == "mean_risk"
        assert req.config == {"risk_measure": "variance"}

    def test_empty_tickers_raises(self) -> None:
        from app.schemas.optimization.optimization import OptimizeRequest

        with pytest.raises(ValidationError):
            OptimizeRequest(
                tickers=[],
                start_date=datetime.date(2020, 1, 1),
                end_date=datetime.date(2024, 1, 1),
                optimizer_type="mean_risk",
                config={},
            )

    def test_optional_config_defaults_to_empty(self) -> None:
        from app.schemas.optimization.optimization import OptimizeRequest

        req = OptimizeRequest(
            tickers=["AAPL"],
            start_date=datetime.date(2020, 1, 1),
            end_date=datetime.date(2024, 1, 1),
            optimizer_type="mean_risk",
        )
        assert req.config == {}

    def test_supported_optimizer_type_parses_successfully(self) -> None:
        from app.schemas.optimization.optimization import OptimizeRequest

        req = OptimizeRequest(
            tickers=["AAPL"],
            start_date=datetime.date(2020, 1, 1),
            end_date=datetime.date(2024, 1, 1),
            optimizer_type="mean_risk",
        )
        assert req.optimizer_type == "mean_risk"

    @pytest.mark.parametrize(
        "bad_type",
        [
            "regime_blended",
            "hrp",
            "herc",
            "nco",
            "risk_budgeting",
            "max_diversification",
            "equal_weighted",
            "inverse_volatility",
            "stacking",
            "robust_mean_risk",
            "dr_cvar",
            "regime_risk_budgeting",
        ],
    )
    def test_removed_optimizer_types_raise_validation_error(
        self, bad_type: str
    ) -> None:
        from app.schemas.optimization.optimization import OptimizeRequest

        with pytest.raises(ValidationError):
            OptimizeRequest(
                tickers=["AAPL"],
                start_date=datetime.date(2020, 1, 1),
                end_date=datetime.date(2024, 1, 1),
                optimizer_type=bad_type,  # type: ignore[arg-type]
            )

    def test_optional_constraints(self) -> None:
        from app.schemas.optimization.optimization import OptimizeRequest

        req = OptimizeRequest(
            tickers=["AAPL", "MSFT"],
            start_date=datetime.date(2020, 1, 1),
            end_date=datetime.date(2024, 1, 1),
            optimizer_type="mean_risk",
            constraints={"max_weight": 0.3},
        )
        assert req.constraints == {"max_weight": 0.3}


class TestOptimizationRunResponse:
    """OptimizationRunResponse serialises ORM rows to camelCase JSON."""

    def _make_orm(self, **overrides: Any) -> SimpleNamespace:
        defaults = {
            "id": uuid.uuid4(),
            "portfolio_id": None,
            "job_id": None,
            "status": "completed",
            "optimizer_type": "mean_risk",
            "universe_tickers": ["AAPL", "MSFT"],
            "config": {"risk_measure": "variance"},
            "weights": {"AAPL": 0.6, "MSFT": 0.4},
            "metrics": {"sharpe": 1.2, "volatility": 0.15},
            "risk_contributions": {"AAPL": 0.55, "MSFT": 0.45},
            "efficient_frontier": None,
            "error_message": None,
            "solver_log": None,
            "duration_seconds": 2.5,
            "created_at": datetime.datetime(2024, 1, 16, 0, 0, 0),
            "updated_at": datetime.datetime(2024, 1, 16, 0, 0, 0),
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_importable(self) -> None:
        from app.schemas.optimization.optimization import OptimizationRunResponse

        _ = OptimizationRunResponse

    def test_orm_deserialization_succeeds(self) -> None:
        from app.schemas.optimization.optimization import OptimizationRunResponse

        orm_obj = self._make_orm()
        response = OptimizationRunResponse.model_validate(orm_obj)
        assert response.status == "completed"
        assert response.optimizer_type == "mean_risk"

    def test_camel_case_output(self) -> None:
        from app.schemas.optimization.optimization import OptimizationRunResponse

        data = OptimizationRunResponse.model_validate(self._make_orm()).model_dump(
            by_alias=True
        )
        assert "optimizerType" in data
        assert "universeTickers" in data
        assert "riskContributions" in data
        assert "efficientFrontier" in data
        assert "errorMessage" in data
        assert "durationSeconds" in data
        assert "createdAt" in data

    def test_id_serializes_as_string(self) -> None:
        from app.schemas.optimization.optimization import OptimizationRunResponse

        uid = uuid.uuid4()
        response = OptimizationRunResponse.model_validate(self._make_orm(id=uid))
        assert response.id == str(uid)

    def test_nullable_portfolio_id(self) -> None:
        from app.schemas.optimization.optimization import OptimizationRunResponse

        response = OptimizationRunResponse.model_validate(
            self._make_orm(portfolio_id=None)
        )
        assert response.portfolio_id is None

    def test_portfolio_id_as_uuid_coerced_to_string(self) -> None:
        from app.schemas.optimization.optimization import OptimizationRunResponse

        pid = uuid.uuid4()
        response = OptimizationRunResponse.model_validate(
            self._make_orm(portfolio_id=pid)
        )
        assert response.portfolio_id == str(pid)

    def test_optional_fields_none(self) -> None:
        from app.schemas.optimization.optimization import OptimizationRunResponse

        response = OptimizationRunResponse.model_validate(
            self._make_orm(
                efficient_frontier=None,
                error_message=None,
                solver_log=None,
                duration_seconds=None,
            )
        )
        assert response.efficient_frontier is None
        assert response.error_message is None
        assert response.duration_seconds is None

    def test_round_trip_serialization(self) -> None:
        from app.schemas.optimization.optimization import OptimizationRunResponse

        uid = uuid.uuid4()
        orm_obj = self._make_orm(id=uid)
        data = OptimizationRunResponse.model_validate(orm_obj).model_dump()
        reconstructed = OptimizationRunResponse(**data)
        assert reconstructed.id == str(uid)
        assert reconstructed.status == "completed"


class TestOptimizationRunListResponse:
    """OptimizationRunListResponse wraps paginated results."""

    def test_importable(self) -> None:
        from app.schemas.optimization.optimization import OptimizationRunListResponse

        _ = OptimizationRunListResponse

    def test_empty_list(self) -> None:
        from app.schemas.optimization.optimization import OptimizationRunListResponse

        resp = OptimizationRunListResponse(items=[], total=0)
        assert resp.total == 0

    def test_camel_case_keys(self) -> None:
        from app.schemas.optimization.optimization import OptimizationRunListResponse

        resp = OptimizationRunListResponse(items=[], total=0)
        data = resp.model_dump(by_alias=True)
        assert "items" in data
        assert "total" in data

    def test_total_must_be_non_negative(self) -> None:
        from app.schemas.optimization.optimization import OptimizationRunListResponse

        with pytest.raises(ValidationError):
            OptimizationRunListResponse(items=[], total=-1)
