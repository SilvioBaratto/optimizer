"""Unit tests for GET /api/v1/market/snapshot.

Mocks DashboardRepository at the router's import path.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

_DASHBOARD_REPO = "app.api.v1.dashboard.DashboardRepository"

BASE_URL = "/api/v1/market"


class TestGetMarketSnapshot:
    def _mock_repo(self, MockDashRepo):
        """Configure mock with valid default data."""
        repo = MockDashRepo.return_value
        repo.get_latest_fred_observations.return_value = {
            "VIXCLS": (16.8, 18.0),
            "DTWEXBGS": (103.4, 103.68),
        }
        repo.get_spy_prices.return_value = [449.30, 457.48]
        repo.get_ten_year_yield_usa.return_value = (4.22, -0.03)
        repo.get_latest_fred_observation_dates.return_value = date(2026, 3, 18)
        repo.get_spy_latest_date.return_value = date(2026, 3, 18)
        repo.get_ten_year_yield_reference_date.return_value = date(2026, 3, 17)
        return repo

    def test_success(self, client: TestClient):
        with patch(_DASHBOARD_REPO) as MockDashRepo:
            self._mock_repo(MockDashRepo)
            resp = client.get(f"{BASE_URL}/snapshot")

        assert resp.status_code == 200
        body = resp.json()
        assert "vix" in body
        assert "sp500Return" in body
        assert "tenYearYield" in body
        assert "usdIndex" in body
        assert "asOf" in body

    def test_camel_case_serialization(self, client: TestClient):
        with patch(_DASHBOARD_REPO) as MockDashRepo:
            self._mock_repo(MockDashRepo)
            resp = client.get(f"{BASE_URL}/snapshot")

        assert resp.status_code == 200
        body = resp.json()
        # camelCase keys present
        assert "vixChange" in body
        assert "sp500Return" in body
        assert "tenYearYield" in body
        assert "yieldChange" in body
        assert "usdIndex" in body
        assert "usdChange" in body
        assert "asOf" in body
        # snake_case keys absent
        assert "vix_change" not in body
        assert "sp500_return" not in body
        assert "ten_year_yield" not in body

    def test_values_correct(self, client: TestClient):
        with patch(_DASHBOARD_REPO) as MockDashRepo:
            self._mock_repo(MockDashRepo)
            resp = client.get(f"{BASE_URL}/snapshot")

        body = resp.json()
        assert body["vix"] == pytest.approx(16.8, abs=0.01)
        assert body["vixChange"] == pytest.approx(-1.2, abs=0.01)
        assert body["tenYearYield"] == pytest.approx(4.22, abs=0.01)
        assert body["yieldChange"] == pytest.approx(-0.03, abs=0.01)
        assert body["usdIndex"] == pytest.approx(103.4, abs=0.01)

    def test_missing_fred_data_returns_503(self, client: TestClient):
        with patch(_DASHBOARD_REPO) as MockDashRepo:
            repo = MockDashRepo.return_value
            repo.get_latest_fred_observations.return_value = {}

            resp = client.get(f"{BASE_URL}/snapshot")

        assert resp.status_code == 503

    def test_missing_vixcls_returns_503(self, client: TestClient):
        with patch(_DASHBOARD_REPO) as MockDashRepo:
            repo = MockDashRepo.return_value
            repo.get_latest_fred_observations.return_value = {
                "DTWEXBGS": (103.4, 103.68),
            }

            resp = client.get(f"{BASE_URL}/snapshot")

        assert resp.status_code == 503

    def test_missing_spy_prices_returns_503(self, client: TestClient):
        with patch(_DASHBOARD_REPO) as MockDashRepo:
            repo = self._mock_repo(MockDashRepo)
            repo.get_spy_prices.return_value = []

            resp = client.get(f"{BASE_URL}/snapshot")

        assert resp.status_code == 503

    def test_missing_bond_yield_returns_503(self, client: TestClient):
        with patch(_DASHBOARD_REPO) as MockDashRepo:
            repo = self._mock_repo(MockDashRepo)
            repo.get_ten_year_yield_usa.return_value = None

            resp = client.get(f"{BASE_URL}/snapshot")

        assert resp.status_code == 503

    def test_spy_single_price_returns_503(self, client: TestClient):
        with patch(_DASHBOARD_REPO) as MockDashRepo:
            repo = self._mock_repo(MockDashRepo)
            repo.get_spy_prices.return_value = [449.30]

            resp = client.get(f"{BASE_URL}/snapshot")

        assert resp.status_code == 503
