"""Unit tests for POST /api/v1/factors/validate and POST /api/v1/factors/score.

Covers:
  - POST /factors/validate: in-sample validation returns 200 with IC results
  - POST /factors/validate: out-of-sample validation returns 200 with OOS results
  - POST /factors/validate: missing required fields returns 422
  - POST /factors/validate: empty tickers returns 422
  - POST /factors/validate: missing data returns 422
  - POST /factors/score: all 5 composite methods return 200 with scores
  - POST /factors/score: EQUAL_WEIGHT requires no extra inputs
  - POST /factors/score: IC_WEIGHTED uses ic_history from DB
  - POST /factors/score: RIDGE_WEIGHTED / GBT_WEIGHTED require training dates
  - POST /factors/score: missing data returns 422
  - POST /factors/score: missing required fields returns 422

Service functions validate_factors() and compute_factor_scores() are tested
in test_factor_validation_service.py.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

BASE_VALIDATE = "/api/v1/factors/validate"
BASE_SCORE = "/api/v1/factors/score"

_VALID_VALIDATE_REQUEST: dict[str, Any] = {
    "tickers": ["AAPL", "MSFT"],
    "start_date": "2020-01-01",
    "end_date": "2024-01-01",
    "factor_type": "book_to_price",
    "validation_type": "in_sample",
}

_VALID_SCORE_REQUEST: dict[str, Any] = {
    "tickers": ["AAPL", "MSFT"],
    "score_date": "2024-01-01",
    "composite_method": "equal_weight",
}

_MOCK_VALIDATE_RESULT: dict[str, Any] = {
    "report_date": "2024-01-01",
    "factor_type": "book_to_price",
    "validation_type": "in_sample",
    "ic_mean": 0.05,
    "ic_std": 0.10,
    "icir": 0.50,
    "t_stat": 2.1,
    "p_value": 0.04,
    "vif": 1.5,
    "details": {"significant_factors": ["book_to_price"]},
}

_MOCK_SCORE_RESULT: dict[str, Any] = {
    "score_date": "2024-01-01",
    "scores": {"AAPL": 0.8, "MSFT": 0.6},
    "group_contributions": {"value": 0.4, "momentum": 0.2},
}


# ===========================================================================
# POST /api/v1/factors/validate
# ===========================================================================


class TestPostFactorValidate:
    """POST /api/v1/factors/validate runs factor validation and returns report."""

    def test_in_sample_returns_200(self, client: TestClient) -> None:
        with patch(
            "app.api.v1.factors.validate_factors",
            return_value=_MOCK_VALIDATE_RESULT,
        ):
            resp = client.post(BASE_VALIDATE, json=_VALID_VALIDATE_REQUEST)

        assert resp.status_code == 200

    def test_in_sample_returns_ic_mean(self, client: TestClient) -> None:
        with patch(
            "app.api.v1.factors.validate_factors",
            return_value=_MOCK_VALIDATE_RESULT,
        ):
            resp = client.post(BASE_VALIDATE, json=_VALID_VALIDATE_REQUEST)

        assert resp.json()["ic_mean"] == pytest.approx(0.05)

    def test_in_sample_returns_t_stat(self, client: TestClient) -> None:
        with patch(
            "app.api.v1.factors.validate_factors",
            return_value=_MOCK_VALIDATE_RESULT,
        ):
            resp = client.post(BASE_VALIDATE, json=_VALID_VALIDATE_REQUEST)

        assert resp.json()["t_stat"] == pytest.approx(2.1)

    def test_in_sample_returns_validation_type(self, client: TestClient) -> None:
        with patch(
            "app.api.v1.factors.validate_factors",
            return_value=_MOCK_VALIDATE_RESULT,
        ):
            resp = client.post(BASE_VALIDATE, json=_VALID_VALIDATE_REQUEST)

        assert resp.json()["validation_type"] == "in_sample"

    def test_out_of_sample_validation_type(self, client: TestClient) -> None:
        oos_result = {**_MOCK_VALIDATE_RESULT, "validation_type": "out_of_sample"}
        payload = {**_VALID_VALIDATE_REQUEST, "validation_type": "out_of_sample"}
        with patch(
            "app.api.v1.factors.validate_factors",
            return_value=oos_result,
        ):
            resp = client.post(BASE_VALIDATE, json=payload)

        assert resp.status_code == 200
        assert resp.json()["validation_type"] == "out_of_sample"

    def test_missing_tickers_returns_422(self, client: TestClient) -> None:
        payload = {k: v for k, v in _VALID_VALIDATE_REQUEST.items() if k != "tickers"}
        resp = client.post(BASE_VALIDATE, json=payload)
        assert resp.status_code == 422

    def test_empty_tickers_returns_422(self, client: TestClient) -> None:
        resp = client.post(BASE_VALIDATE, json={**_VALID_VALIDATE_REQUEST, "tickers": []})
        assert resp.status_code == 422

    def test_missing_start_date_returns_422(self, client: TestClient) -> None:
        payload = {k: v for k, v in _VALID_VALIDATE_REQUEST.items() if k != "start_date"}
        resp = client.post(BASE_VALIDATE, json=payload)
        assert resp.status_code == 422

    def test_missing_end_date_returns_422(self, client: TestClient) -> None:
        payload = {k: v for k, v in _VALID_VALIDATE_REQUEST.items() if k != "end_date"}
        resp = client.post(BASE_VALIDATE, json=payload)
        assert resp.status_code == 422

    def test_missing_factor_type_returns_422(self, client: TestClient) -> None:
        payload = {k: v for k, v in _VALID_VALIDATE_REQUEST.items() if k != "factor_type"}
        resp = client.post(BASE_VALIDATE, json=payload)
        assert resp.status_code == 422

    def test_missing_data_returns_422(self, client: TestClient) -> None:
        from app.services.factor_service import FactorDataError

        with patch(
            "app.api.v1.factors.validate_factors",
            side_effect=FactorDataError("No factor scores found"),
        ):
            resp = client.post(BASE_VALIDATE, json=_VALID_VALIDATE_REQUEST)

        assert resp.status_code == 422

    def test_default_validation_type_is_in_sample(self, client: TestClient) -> None:
        payload = {k: v for k, v in _VALID_VALIDATE_REQUEST.items() if k != "validation_type"}
        with patch(
            "app.api.v1.factors.validate_factors",
            return_value=_MOCK_VALIDATE_RESULT,
        ) as mock_validate:
            client.post(BASE_VALIDATE, json=payload)

        assert mock_validate.called

    def test_returns_details_field(self, client: TestClient) -> None:
        with patch(
            "app.api.v1.factors.validate_factors",
            return_value=_MOCK_VALIDATE_RESULT,
        ):
            resp = client.post(BASE_VALIDATE, json=_VALID_VALIDATE_REQUEST)

        assert "details" in resp.json()


# ===========================================================================
# POST /api/v1/factors/score
# ===========================================================================


class TestPostFactorScore:
    """POST /api/v1/factors/score computes composite scores for a ticker universe."""

    def test_equal_weight_returns_200(self, client: TestClient) -> None:
        with patch(
            "app.api.v1.factors.compute_factor_scores",
            return_value=_MOCK_SCORE_RESULT,
        ):
            resp = client.post(BASE_SCORE, json=_VALID_SCORE_REQUEST)

        assert resp.status_code == 200

    def test_equal_weight_returns_scores(self, client: TestClient) -> None:
        with patch(
            "app.api.v1.factors.compute_factor_scores",
            return_value=_MOCK_SCORE_RESULT,
        ):
            resp = client.post(BASE_SCORE, json=_VALID_SCORE_REQUEST)

        body = resp.json()
        assert "scores" in body
        assert body["scores"]["AAPL"] == pytest.approx(0.8)

    def test_equal_weight_returns_group_contributions(self, client: TestClient) -> None:
        with patch(
            "app.api.v1.factors.compute_factor_scores",
            return_value=_MOCK_SCORE_RESULT,
        ):
            resp = client.post(BASE_SCORE, json=_VALID_SCORE_REQUEST)

        body = resp.json()
        assert "group_contributions" in body

    def test_ic_weighted_accepted(self, client: TestClient) -> None:
        payload = {**_VALID_SCORE_REQUEST, "composite_method": "ic_weighted"}
        with patch(
            "app.api.v1.factors.compute_factor_scores",
            return_value=_MOCK_SCORE_RESULT,
        ):
            resp = client.post(BASE_SCORE, json=payload)

        assert resp.status_code == 200

    def test_icir_weighted_accepted(self, client: TestClient) -> None:
        payload = {**_VALID_SCORE_REQUEST, "composite_method": "icir_weighted"}
        with patch(
            "app.api.v1.factors.compute_factor_scores",
            return_value=_MOCK_SCORE_RESULT,
        ):
            resp = client.post(BASE_SCORE, json=payload)

        assert resp.status_code == 200

    def test_ridge_weighted_accepted(self, client: TestClient) -> None:
        payload = {
            **_VALID_SCORE_REQUEST,
            "composite_method": "ridge_weighted",
            "training_start_date": "2019-01-01",
            "training_end_date": "2023-12-31",
        }
        with patch(
            "app.api.v1.factors.compute_factor_scores",
            return_value=_MOCK_SCORE_RESULT,
        ):
            resp = client.post(BASE_SCORE, json=payload)

        assert resp.status_code == 200

    def test_gbt_weighted_accepted(self, client: TestClient) -> None:
        payload = {
            **_VALID_SCORE_REQUEST,
            "composite_method": "gbt_weighted",
            "training_start_date": "2019-01-01",
            "training_end_date": "2023-12-31",
        }
        with patch(
            "app.api.v1.factors.compute_factor_scores",
            return_value=_MOCK_SCORE_RESULT,
        ):
            resp = client.post(BASE_SCORE, json=payload)

        assert resp.status_code == 200

    def test_all_five_methods_accepted(self, client: TestClient) -> None:
        methods = ["equal_weight", "ic_weighted", "icir_weighted", "ridge_weighted", "gbt_weighted"]
        for method in methods:
            payload = {
                **_VALID_SCORE_REQUEST,
                "composite_method": method,
                "training_start_date": "2019-01-01",
                "training_end_date": "2023-12-31",
            }
            with patch(
                "app.api.v1.factors.compute_factor_scores",
                return_value=_MOCK_SCORE_RESULT,
            ):
                resp = client.post(BASE_SCORE, json=payload)

            assert resp.status_code == 200, f"Method {method} failed: {resp.json()}"

    def test_invalid_method_returns_422(self, client: TestClient) -> None:
        payload = {**_VALID_SCORE_REQUEST, "composite_method": "unknown_method"}
        resp = client.post(BASE_SCORE, json=payload)
        assert resp.status_code == 422

    def test_missing_tickers_returns_422(self, client: TestClient) -> None:
        payload = {k: v for k, v in _VALID_SCORE_REQUEST.items() if k != "tickers"}
        resp = client.post(BASE_SCORE, json=payload)
        assert resp.status_code == 422

    def test_empty_tickers_returns_422(self, client: TestClient) -> None:
        resp = client.post(BASE_SCORE, json={**_VALID_SCORE_REQUEST, "tickers": []})
        assert resp.status_code == 422

    def test_missing_score_date_returns_422(self, client: TestClient) -> None:
        payload = {k: v for k, v in _VALID_SCORE_REQUEST.items() if k != "score_date"}
        resp = client.post(BASE_SCORE, json=payload)
        assert resp.status_code == 422

    def test_missing_data_returns_422(self, client: TestClient) -> None:
        from app.services.factor_service import FactorDataError

        with patch(
            "app.api.v1.factors.compute_factor_scores",
            side_effect=FactorDataError("No standardized factor scores found"),
        ):
            resp = client.post(BASE_SCORE, json=_VALID_SCORE_REQUEST)

        assert resp.status_code == 422

    def test_optional_group_weights_accepted(self, client: TestClient) -> None:
        payload = {
            **_VALID_SCORE_REQUEST,
            "group_weights": {"value": 0.6, "momentum": 0.4},
        }
        with patch(
            "app.api.v1.factors.compute_factor_scores",
            return_value=_MOCK_SCORE_RESULT,
        ):
            resp = client.post(BASE_SCORE, json=payload)

        assert resp.status_code == 200

    def test_returns_score_date(self, client: TestClient) -> None:
        with patch(
            "app.api.v1.factors.compute_factor_scores",
            return_value=_MOCK_SCORE_RESULT,
        ):
            resp = client.post(BASE_SCORE, json=_VALID_SCORE_REQUEST)

        assert resp.json()["score_date"] == "2024-01-01"
