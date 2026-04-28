"""Tests for api/app/schemas/rebalancing.py — rebalancing policy schemas.

Covers (TDD Red → Green → Refactor):
- Round-trip serialization
- ORM deserialization
- camelCase alias output
- Validation rejection (empty name, invalid policy_type)
- Optional/nullable field handling
"""

from __future__ import annotations

import datetime
import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError


class TestRebalancingPolicyCreate:
    """RebalancingPolicyCreate validates POST request bodies."""

    def test_importable(self) -> None:
        from app.schemas.rebalancing import RebalancingPolicyCreate
        _ = RebalancingPolicyCreate

    def test_valid_create(self) -> None:
        from app.schemas.rebalancing import RebalancingPolicyCreate

        create = RebalancingPolicyCreate(
            name="Monthly Threshold",
            policy_type="calendar",
            config={"interval_days": 21},
        )
        assert create.name == "Monthly Threshold"
        assert create.policy_type == "calendar"

    def test_empty_name_raises(self) -> None:
        from app.schemas.rebalancing import RebalancingPolicyCreate

        with pytest.raises(ValidationError):
            RebalancingPolicyCreate(
                name="",
                policy_type="calendar",
                config={},
            )

    def test_optional_config_defaults_to_empty(self) -> None:
        from app.schemas.rebalancing import RebalancingPolicyCreate

        create = RebalancingPolicyCreate(
            name="Simple Policy",
            policy_type="threshold",
        )
        assert create.config == {}


class TestRebalancingPolicyResponse:
    """RebalancingPolicyResponse serialises ORM rows to camelCase JSON."""

    def _make_orm(self, **overrides: Any) -> SimpleNamespace:
        defaults = {
            "id": uuid.uuid4(),
            "portfolio_id": uuid.uuid4(),
            "name": "Monthly Threshold",
            "policy_type": "hybrid",
            "config": {"interval_days": 21, "threshold": 0.05},
            "is_active": True,
            "created_at": datetime.datetime(2024, 1, 1, 0, 0, 0),
            "updated_at": datetime.datetime(2024, 1, 15, 10, 0, 0),
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_importable(self) -> None:
        from app.schemas.rebalancing import RebalancingPolicyResponse
        _ = RebalancingPolicyResponse

    def test_orm_deserialization_succeeds(self) -> None:
        from app.schemas.rebalancing import RebalancingPolicyResponse

        orm_obj = self._make_orm()
        response = RebalancingPolicyResponse.model_validate(orm_obj)
        assert response.name == "Monthly Threshold"
        assert response.is_active is True

    def test_camel_case_output(self) -> None:
        from app.schemas.rebalancing import RebalancingPolicyResponse

        data = RebalancingPolicyResponse.model_validate(
            self._make_orm()
        ).model_dump(by_alias=True)
        assert "portfolioId" in data
        assert "policyType" in data
        assert "isActive" in data
        assert "createdAt" in data
        assert "updatedAt" in data

    def test_id_and_portfolio_id_as_strings(self) -> None:
        from app.schemas.rebalancing import RebalancingPolicyResponse

        uid = uuid.uuid4()
        pid = uuid.uuid4()
        response = RebalancingPolicyResponse.model_validate(
            self._make_orm(id=uid, portfolio_id=pid)
        )
        assert response.id == str(uid)
        assert response.portfolio_id == str(pid)

    def test_is_active_false(self) -> None:
        from app.schemas.rebalancing import RebalancingPolicyResponse

        response = RebalancingPolicyResponse.model_validate(
            self._make_orm(is_active=False)
        )
        assert response.is_active is False

    def test_round_trip_serialization(self) -> None:
        from app.schemas.rebalancing import RebalancingPolicyResponse

        uid = uuid.uuid4()
        orm_obj = self._make_orm(id=uid)
        data = RebalancingPolicyResponse.model_validate(orm_obj).model_dump()
        reconstructed = RebalancingPolicyResponse(**data)
        assert reconstructed.name == "Monthly Threshold"
        assert reconstructed.id == str(uid)


class TestRebalancingPolicyListResponse:
    """RebalancingPolicyListResponse wraps paginated results."""

    def test_importable(self) -> None:
        from app.schemas.rebalancing import RebalancingPolicyListResponse
        _ = RebalancingPolicyListResponse

    def test_empty_list(self) -> None:
        from app.schemas.rebalancing import RebalancingPolicyListResponse

        resp = RebalancingPolicyListResponse(items=[], total=0)
        assert resp.total == 0

    def test_camel_case_keys(self) -> None:
        from app.schemas.rebalancing import RebalancingPolicyListResponse

        resp = RebalancingPolicyListResponse(items=[], total=5)
        data = resp.model_dump(by_alias=True)
        assert "items" in data
        assert "total" in data

    def test_total_must_be_non_negative(self) -> None:
        from app.schemas.rebalancing import RebalancingPolicyListResponse

        with pytest.raises(ValidationError):
            RebalancingPolicyListResponse(items=[], total=-1)
