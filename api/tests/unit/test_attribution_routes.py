"""Unit tests for attribution route endpoints.

Covers:
  - POST /api/v1/attribution/brinson — happy path, validation errors, 404/422
  - POST /api/v1/attribution/factor — happy path, missing factor scores, invalid weights

Service layer is fully mocked so tests are DB-independent and fast.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

BASE_BRINSON = "/api/v1/attribution/brinson"
BASE_FACTOR = "/api/v1/attribution/factor"

_SVC_BRINSON = "app.api.v1.attribution.run_brinson_attribution"
_SVC_FACTOR = "app.api.v1.attribution.run_factor_attribution"

# ---------------------------------------------------------------------------
# Shared valid payloads
# ---------------------------------------------------------------------------

_VALID_BRINSON: dict[str, Any] = {
    "portfolio_weights": {"AAPL": 0.6, "JPM": 0.4},
    "benchmark_weights": {"AAPL": 0.5, "JPM": 0.5},
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
}

_VALID_FACTOR: dict[str, Any] = {
    "portfolio_weights": {"AAPL": 0.6, "MSFT": 0.4},
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
}

_MOCK_BRINSON_RESULT: dict[str, Any] = {
    "sectors": [
        {
            "sector": "Technology",
            "portfolio_weight": 0.6,
            "benchmark_weight": 0.5,
            "portfolio_return": 0.12,
            "benchmark_return": 0.10,
            "allocation_effect": 0.0025,
            "selection_effect": 0.01,
            "interaction_effect": 0.002,
            "total_effect": 0.0145,
        },
        {
            "sector": "Financials",
            "portfolio_weight": 0.4,
            "benchmark_weight": 0.5,
            "portfolio_return": 0.04,
            "benchmark_return": 0.05,
            "allocation_effect": 0.0025,
            "selection_effect": -0.005,
            "interaction_effect": 0.001,
            "total_effect": -0.0015,
        },
    ],
    "total_allocation": 0.005,
    "total_selection": 0.005,
    "total_interaction": 0.003,
    "total_active_return": 0.013,
    "portfolio_return": 0.088,
    "benchmark_return": 0.075,
}

_MOCK_FACTOR_RESULT: dict[str, Any] = {
    "factors": [
        {
            "factor_name": "momentum",
            "exposure": 0.4,
            "factor_return": 0.05,
            "contribution": 0.02,
        }
    ],
    "portfolio_return": 0.08,
    "explained_return": 0.02,
    "residual": 0.06,
}


# ===========================================================================
# POST /api/v1/attribution/brinson
# ===========================================================================


class TestBrinsonEndpoint:
    """POST /api/v1/attribution/brinson."""

    def test_valid_request_returns_200(self, client: TestClient) -> None:
        with patch(_SVC_BRINSON, return_value=_MOCK_BRINSON_RESULT):
            resp = client.post(BASE_BRINSON, json=_VALID_BRINSON)
        assert resp.status_code == 200

    def test_response_contains_sectors(self, client: TestClient) -> None:
        with patch(_SVC_BRINSON, return_value=_MOCK_BRINSON_RESULT):
            resp = client.post(BASE_BRINSON, json=_VALID_BRINSON)
        assert "sectors" in resp.json()

    def test_response_contains_total_fields(self, client: TestClient) -> None:
        with patch(_SVC_BRINSON, return_value=_MOCK_BRINSON_RESULT):
            resp = client.post(BASE_BRINSON, json=_VALID_BRINSON)
        body = resp.json()
        assert "totalAllocation" in body or "total_allocation" in body

    def test_weights_not_summing_to_one_returns_422(self, client: TestClient) -> None:
        bad_payload = {**_VALID_BRINSON, "portfolio_weights": {"AAPL": 0.3, "JPM": 0.3}}
        resp = client.post(BASE_BRINSON, json=bad_payload)
        assert resp.status_code == 422

    def test_start_date_after_end_date_returns_422(self, client: TestClient) -> None:
        bad_payload = {**_VALID_BRINSON, "start_date": "2024-01-01", "end_date": "2023-01-01"}
        resp = client.post(BASE_BRINSON, json=bad_payload)
        assert resp.status_code == 422

    def test_missing_portfolio_weights_returns_422(self, client: TestClient) -> None:
        payload = {k: v for k, v in _VALID_BRINSON.items() if k != "portfolio_weights"}
        resp = client.post(BASE_BRINSON, json=payload)
        assert resp.status_code == 422

    def test_missing_benchmark_returns_404(self, client: TestClient) -> None:
        from app.services.attribution_service import MissingDataError
        with patch(_SVC_BRINSON, side_effect=MissingDataError("benchmark not found")):
            resp = client.post(BASE_BRINSON, json=_VALID_BRINSON)
        assert resp.status_code == 404

    def test_insufficient_sector_coverage_returns_422(self, client: TestClient) -> None:
        from app.services.attribution_service import InsufficientSectorCoverageError
        with patch(
            _SVC_BRINSON,
            side_effect=InsufficientSectorCoverageError("sector coverage < 80%"),
        ):
            resp = client.post(BASE_BRINSON, json=_VALID_BRINSON)
        assert resp.status_code == 422

    def test_empty_portfolio_weights_returns_422(self, client: TestClient) -> None:
        bad_payload = {**_VALID_BRINSON, "portfolio_weights": {}}
        resp = client.post(BASE_BRINSON, json=bad_payload)
        assert resp.status_code == 422

    def test_benchmark_weights_not_summing_to_one_returns_422(
        self, client: TestClient
    ) -> None:
        bad_payload = {**_VALID_BRINSON, "benchmark_weights": {"AAPL": 0.2, "JPM": 0.2}}
        resp = client.post(BASE_BRINSON, json=bad_payload)
        assert resp.status_code == 422


# ===========================================================================
# POST /api/v1/attribution/factor
# ===========================================================================


class TestFactorAttributionEndpoint:
    """POST /api/v1/attribution/factor."""

    def test_valid_request_returns_200(self, client: TestClient) -> None:
        with patch(_SVC_FACTOR, return_value=_MOCK_FACTOR_RESULT):
            resp = client.post(BASE_FACTOR, json=_VALID_FACTOR)
        assert resp.status_code == 200

    def test_response_contains_factors(self, client: TestClient) -> None:
        with patch(_SVC_FACTOR, return_value=_MOCK_FACTOR_RESULT):
            resp = client.post(BASE_FACTOR, json=_VALID_FACTOR)
        assert "factors" in resp.json()

    def test_response_contains_residual(self, client: TestClient) -> None:
        with patch(_SVC_FACTOR, return_value=_MOCK_FACTOR_RESULT):
            resp = client.post(BASE_FACTOR, json=_VALID_FACTOR)
        assert "residual" in resp.json()

    def test_missing_factor_scores_returns_404(self, client: TestClient) -> None:
        from app.services.attribution_service import MissingDataError
        with patch(_SVC_FACTOR, side_effect=MissingDataError("no factor scores")):
            resp = client.post(BASE_FACTOR, json=_VALID_FACTOR)
        assert resp.status_code == 404

    def test_weights_not_summing_to_one_returns_422(self, client: TestClient) -> None:
        bad_payload = {**_VALID_FACTOR, "portfolio_weights": {"AAPL": 0.3, "MSFT": 0.3}}
        resp = client.post(BASE_FACTOR, json=bad_payload)
        assert resp.status_code == 422

    def test_missing_portfolio_weights_returns_422(self, client: TestClient) -> None:
        payload = {k: v for k, v in _VALID_FACTOR.items() if k != "portfolio_weights"}
        resp = client.post(BASE_FACTOR, json=payload)
        assert resp.status_code == 422

    def test_empty_portfolio_weights_returns_422(self, client: TestClient) -> None:
        bad_payload = {**_VALID_FACTOR, "portfolio_weights": {}}
        resp = client.post(BASE_FACTOR, json=bad_payload)
        assert resp.status_code == 422

    def test_start_date_after_end_date_returns_422(self, client: TestClient) -> None:
        bad_payload = {**_VALID_FACTOR, "start_date": "2024-01-01", "end_date": "2023-01-01"}
        resp = client.post(BASE_FACTOR, json=bad_payload)
        assert resp.status_code == 422
