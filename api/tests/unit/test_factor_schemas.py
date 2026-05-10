"""Tests for api/app/schemas/factors.py — factor score and validation schemas.

Covers (TDD Red → Green → Refactor):
- Round-trip serialization: dict → model → dict
- ORM deserialization: mock ORM object → model_validate(orm_obj)
- camelCase alias output: model_dump(by_alias=True) keys are camelCase
- Validation rejection: invalid inputs raise ValidationError
- Optional/nullable field handling
"""

from __future__ import annotations

import datetime
import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# FactorScoreResponse
# ---------------------------------------------------------------------------


class TestFactorScoreResponse:
    """FactorScoreResponse serialises ORM rows to camelCase JSON."""

    def _make_orm(self, **overrides: Any) -> SimpleNamespace:
        defaults = {
            "id": uuid.uuid4(),
            "ticker": "AAPL",
            "score_date": datetime.date(2024, 1, 15),
            "factor_type": "momentum",
            "factor_group": "momentum",
            "raw_score": 1.23,
            "standardized_score": 0.82,
            "composite_score": 0.79,
            "created_at": datetime.datetime(2024, 1, 16, 0, 0, 0),
            "updated_at": datetime.datetime(2024, 1, 16, 0, 0, 0),
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_importable(self) -> None:
        from app.schemas.factors.factors import FactorScoreResponse

        _ = FactorScoreResponse

    def test_orm_deserialization_succeeds(self) -> None:
        from app.schemas.factors.factors import FactorScoreResponse

        orm_obj = self._make_orm()
        response = FactorScoreResponse.model_validate(orm_obj)
        assert response.ticker == "AAPL"
        assert response.factor_type == "momentum"
        assert response.raw_score == pytest.approx(1.23)

    def test_camel_case_output(self) -> None:
        from app.schemas.factors.factors import FactorScoreResponse

        orm_obj = self._make_orm()
        data = FactorScoreResponse.model_validate(orm_obj).model_dump(by_alias=True)
        assert "factorType" in data
        assert "factorGroup" in data
        assert "rawScore" in data
        assert "standardizedScore" in data
        assert "compositeScore" in data
        assert "scoreDate" in data
        assert "createdAt" in data

    def test_id_serializes_as_string(self) -> None:
        from app.schemas.factors.factors import FactorScoreResponse

        uid = uuid.uuid4()
        orm_obj = self._make_orm(id=uid)
        response = FactorScoreResponse.model_validate(orm_obj)
        assert response.id == str(uid)

    def test_optional_scores_none(self) -> None:
        from app.schemas.factors.factors import FactorScoreResponse

        orm_obj = self._make_orm(standardized_score=None, composite_score=None)
        response = FactorScoreResponse.model_validate(orm_obj)
        assert response.standardized_score is None
        assert response.composite_score is None

    def test_round_trip_serialization(self) -> None:
        from app.schemas.factors.factors import FactorScoreResponse

        uid = uuid.uuid4()
        orm_obj = self._make_orm(id=uid)
        data = FactorScoreResponse.model_validate(orm_obj).model_dump()
        reconstructed = FactorScoreResponse(**data)
        assert reconstructed.ticker == "AAPL"
        assert reconstructed.id == str(uid)


# ---------------------------------------------------------------------------
# FactorScoreListResponse
# ---------------------------------------------------------------------------


class TestFactorScoreListResponse:
    """FactorScoreListResponse wraps a list of items with a total count."""

    def test_importable(self) -> None:
        from app.schemas.factors.factors import FactorScoreListResponse

        _ = FactorScoreListResponse

    def test_empty_list(self) -> None:
        from app.schemas.factors.factors import FactorScoreListResponse

        resp = FactorScoreListResponse(items=[], total=0)
        assert resp.total == 0
        assert resp.items == []

    def test_camel_case_keys(self) -> None:
        from app.schemas.factors.factors import FactorScoreListResponse

        resp = FactorScoreListResponse(items=[], total=0)
        data = resp.model_dump(by_alias=True)
        assert "items" in data
        assert "total" in data

    def test_total_must_be_non_negative(self) -> None:
        from app.schemas.factors.factors import FactorScoreListResponse

        with pytest.raises(ValidationError):
            FactorScoreListResponse(items=[], total=-1)


# ---------------------------------------------------------------------------
# FactorValidationReportResponse
# ---------------------------------------------------------------------------


class TestFactorValidationReportResponse:
    """FactorValidationReportResponse serialises validation reports."""

    def _make_orm(self, **overrides: Any) -> SimpleNamespace:
        defaults = {
            "id": uuid.uuid4(),
            "report_date": datetime.date(2024, 3, 1),
            "factor_type": "momentum",
            "validation_type": "in_sample",
            "ic_mean": 0.05,
            "ic_std": 0.02,
            "icir": 2.5,
            "t_stat": 3.1,
            "p_value": 0.002,
            "vif": 1.1,
            "details": {"observations": 252},
            "created_at": datetime.datetime(2024, 3, 2, 0, 0, 0),
            "updated_at": datetime.datetime(2024, 3, 2, 0, 0, 0),
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_importable(self) -> None:
        from app.schemas.factors.factors import FactorValidationReportResponse

        _ = FactorValidationReportResponse

    def test_orm_deserialization_succeeds(self) -> None:
        from app.schemas.factors.factors import FactorValidationReportResponse

        orm_obj = self._make_orm()
        response = FactorValidationReportResponse.model_validate(orm_obj)
        assert response.validation_type == "in_sample"
        assert response.ic_mean == pytest.approx(0.05)

    def test_camel_case_output(self) -> None:
        from app.schemas.factors.factors import FactorValidationReportResponse

        data = FactorValidationReportResponse.model_validate(
            self._make_orm()
        ).model_dump(by_alias=True)
        assert "reportDate" in data
        assert "factorType" in data
        assert "validationType" in data
        assert "icMean" in data
        assert "icStd" in data
        assert "tStat" in data
        assert "pValue" in data

    def test_all_optional_metrics_can_be_none(self) -> None:
        from app.schemas.factors.factors import FactorValidationReportResponse

        orm_obj = self._make_orm(
            factor_type=None,
            ic_mean=None,
            ic_std=None,
            icir=None,
            t_stat=None,
            p_value=None,
            vif=None,
            details=None,
        )
        response = FactorValidationReportResponse.model_validate(orm_obj)
        assert response.factor_type is None
        assert response.ic_mean is None
        assert response.details is None


# ---------------------------------------------------------------------------
# FactorComputeRequest
# ---------------------------------------------------------------------------


class TestFactorComputeRequest:
    """FactorComputeRequest validates incoming compute requests."""

    def test_importable(self) -> None:
        from app.schemas.factors.factors import FactorComputeRequest

        _ = FactorComputeRequest

    def test_valid_request(self) -> None:
        from app.schemas.factors.factors import FactorComputeRequest

        req = FactorComputeRequest(
            tickers=["AAPL", "MSFT"],
            start_date=datetime.date(2020, 1, 1),
            end_date=datetime.date(2024, 1, 1),
        )
        assert req.tickers == ["AAPL", "MSFT"]

    def test_empty_tickers_raises(self) -> None:
        from app.schemas.factors.factors import FactorComputeRequest

        with pytest.raises(ValidationError):
            FactorComputeRequest(
                tickers=[],
                start_date=datetime.date(2020, 1, 1),
                end_date=datetime.date(2024, 1, 1),
            )

    def test_optional_factor_config(self) -> None:
        from app.schemas.factors.factors import FactorComputeRequest

        req = FactorComputeRequest(
            tickers=["AAPL"],
            start_date=datetime.date(2020, 1, 1),
            end_date=datetime.date(2024, 1, 1),
            factor_config={"method": "ic_weighted"},
        )
        assert req.factor_config == {"method": "ic_weighted"}


# ---------------------------------------------------------------------------
# FactorValidateRequest
# ---------------------------------------------------------------------------


class TestFactorValidateRequest:
    """FactorValidateRequest validates incoming validation requests."""

    def test_importable(self) -> None:
        from app.schemas.factors.factors import FactorValidateRequest

        _ = FactorValidateRequest

    def test_valid_request(self) -> None:
        import datetime

        from app.schemas.factors.factors import FactorValidateRequest

        req = FactorValidateRequest(
            tickers=["AAPL", "MSFT"],
            start_date=datetime.date(2020, 1, 1),
            end_date=datetime.date(2024, 1, 1),
            factor_type="momentum",
            validation_type="in_sample",
        )
        assert req.factor_type == "momentum"

    def test_default_validation_type_is_in_sample(self) -> None:
        import datetime

        from app.schemas.factors.factors import FactorValidateRequest

        req = FactorValidateRequest(
            tickers=["AAPL"],
            start_date=datetime.date(2020, 1, 1),
            end_date=datetime.date(2024, 1, 1),
            factor_type="quality",
        )
        assert req.validation_type == "in_sample"
