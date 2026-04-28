"""Tests for api/app/schemas/risk.py — risk limit schemas.

Covers (TDD Red → Green → Refactor):
- Round-trip serialization
- ORM deserialization
- camelCase alias output
- Validation rejection (negative threshold, empty metric)
- Optional/nullable field handling
"""

from __future__ import annotations

import datetime
import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError


class TestRiskLimitCreate:
    """RiskLimitCreate validates POST request bodies for risk limits."""

    def test_importable(self) -> None:
        from app.schemas.risk import RiskLimitCreate
        _ = RiskLimitCreate

    def test_valid_create(self) -> None:
        from app.schemas.risk import RiskLimitCreate

        create = RiskLimitCreate(
            metric="max_drawdown",
            limit_type="upper",
            threshold=0.15,
        )
        assert create.metric == "max_drawdown"
        assert create.threshold == pytest.approx(0.15)

    def test_negative_threshold_raises(self) -> None:
        from app.schemas.risk import RiskLimitCreate

        with pytest.raises(ValidationError):
            RiskLimitCreate(
                metric="volatility",
                limit_type="upper",
                threshold=-0.05,
            )

    def test_zero_threshold_raises(self) -> None:
        from app.schemas.risk import RiskLimitCreate

        with pytest.raises(ValidationError):
            RiskLimitCreate(
                metric="volatility",
                limit_type="upper",
                threshold=0.0,
            )

    def test_empty_metric_raises(self) -> None:
        from app.schemas.risk import RiskLimitCreate

        with pytest.raises(ValidationError):
            RiskLimitCreate(
                metric="",
                limit_type="upper",
                threshold=0.1,
            )


class TestRiskLimitResponse:
    """RiskLimitResponse serialises ORM rows to camelCase JSON."""

    def _make_orm(self, **overrides: Any) -> SimpleNamespace:
        defaults = {
            "id": uuid.uuid4(),
            "portfolio_id": uuid.uuid4(),
            "metric": "max_drawdown",
            "limit_type": "upper",
            "threshold": 0.15,
            "current_value": 0.08,
            "is_breached": False,
            "last_checked_at": datetime.datetime(2024, 1, 15, 10, 0, 0),
            "created_at": datetime.datetime(2024, 1, 1, 0, 0, 0),
            "updated_at": datetime.datetime(2024, 1, 15, 10, 0, 0),
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_importable(self) -> None:
        from app.schemas.risk import RiskLimitResponse
        _ = RiskLimitResponse

    def test_orm_deserialization_succeeds(self) -> None:
        from app.schemas.risk import RiskLimitResponse

        orm_obj = self._make_orm()
        response = RiskLimitResponse.model_validate(orm_obj)
        assert response.metric == "max_drawdown"
        assert response.is_breached is False

    def test_camel_case_output(self) -> None:
        from app.schemas.risk import RiskLimitResponse

        data = RiskLimitResponse.model_validate(
            self._make_orm()
        ).model_dump(by_alias=True)
        assert "portfolioId" in data
        assert "limitType" in data
        assert "currentValue" in data
        assert "isBreached" in data
        assert "lastCheckedAt" in data
        assert "createdAt" in data

    def test_id_and_portfolio_id_as_strings(self) -> None:
        from app.schemas.risk import RiskLimitResponse

        uid = uuid.uuid4()
        pid = uuid.uuid4()
        response = RiskLimitResponse.model_validate(
            self._make_orm(id=uid, portfolio_id=pid)
        )
        assert response.id == str(uid)
        assert response.portfolio_id == str(pid)

    def test_current_value_none(self) -> None:
        from app.schemas.risk import RiskLimitResponse

        response = RiskLimitResponse.model_validate(
            self._make_orm(current_value=None)
        )
        assert response.current_value is None

    def test_last_checked_at_none(self) -> None:
        from app.schemas.risk import RiskLimitResponse

        response = RiskLimitResponse.model_validate(
            self._make_orm(last_checked_at=None)
        )
        assert response.last_checked_at is None

    def test_round_trip_serialization(self) -> None:
        from app.schemas.risk import RiskLimitResponse

        uid = uuid.uuid4()
        orm_obj = self._make_orm(id=uid)
        data = RiskLimitResponse.model_validate(orm_obj).model_dump()
        reconstructed = RiskLimitResponse(**data)
        assert reconstructed.metric == "max_drawdown"
        assert reconstructed.id == str(uid)


class TestRiskLimitListResponse:
    """RiskLimitListResponse wraps items with breach count."""

    def test_importable(self) -> None:
        from app.schemas.risk import RiskLimitListResponse
        _ = RiskLimitListResponse

    def test_empty_list(self) -> None:
        from app.schemas.risk import RiskLimitListResponse

        resp = RiskLimitListResponse(items=[], breach_count=0)
        assert resp.breach_count == 0

    def test_camel_case_keys(self) -> None:
        from app.schemas.risk import RiskLimitListResponse

        resp = RiskLimitListResponse(items=[], breach_count=2)
        data = resp.model_dump(by_alias=True)
        assert "items" in data
        assert "breachCount" in data

    def test_breach_count_non_negative(self) -> None:
        from app.schemas.risk import RiskLimitListResponse

        with pytest.raises(ValidationError):
            RiskLimitListResponse(items=[], breach_count=-1)
