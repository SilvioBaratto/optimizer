"""Unit tests for the asset-class-returns endpoint.

Mocks PortfolioRepository and DashboardRepository at the router's import
path so no real database is touched.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

_PORTFOLIO_REPO = "app.api.v1.dashboard.PortfolioRepository"
_DASHBOARD_REPO = "app.api.v1.dashboard.DashboardRepository"

BASE_URL = "/api/v1/portfolio-analytics"

_WEIGHTS = {"AAPL": 0.3, "MSFT": 0.2, "XOM": 0.3, "CVX": 0.2}
_SECTOR_MAP = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "XOM": "Energy",
    "CVX": "Energy",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_portfolio(name: str = "test") -> MagicMock:
    p = MagicMock()
    p.id = uuid.uuid4()
    p.name = name
    return p


def _make_snapshot(
    weights: dict[str, float] | None = None,
    sector_mapping: dict[str, str] | None = "default",
) -> MagicMock:
    s = MagicMock()
    s.weights = weights or _WEIGHTS
    if sector_mapping == "default":
        s.sector_mapping = _SECTOR_MAP
    else:
        s.sector_mapping = sector_mapping
    return s


def _make_prices(tickers: list[str], n_days: int = 60) -> pd.DataFrame:
    """Deterministic prices starting well before Jan 1 of current year."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2025-11-01", periods=n_days)
    data = {}
    for t in tickers:
        data[t] = (1 + rng.normal(0.0003, 0.012, n_days)).cumprod() * 100
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetAssetClassReturns:
    def test_success(self, client: TestClient):
        portfolio = _make_portfolio("myport")
        snapshot = _make_snapshot()
        prices = _make_prices(["AAPL", "MSFT", "XOM", "CVX"])

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
        ):
            repo = MockPortRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot

            MockDashRepo.return_value.get_multi_ticker_prices.return_value = prices

            resp = client.get(f"{BASE_URL}/myport/asset-class-returns")

        assert resp.status_code == 200
        body = resp.json()
        assert "returns" in body
        assert "asOf" in body
        assert len(body["returns"]) > 0

        row = body["returns"][0]
        assert "name" in row
        assert "1D" in row
        assert "1W" in row
        assert "1M" in row
        assert "YTD" in row

    def test_portfolio_not_found(self, client: TestClient):
        with patch(_PORTFOLIO_REPO) as MockPortRepo:
            MockPortRepo.return_value.get_by_name.return_value = None

            resp = client.get(f"{BASE_URL}/missing/asset-class-returns")

        assert resp.status_code == 404
        msg = resp.json()["error"]["message"]
        assert "not found" in msg.lower()

    def test_no_snapshot(self, client: TestClient):
        portfolio = _make_portfolio()

        with patch(_PORTFOLIO_REPO) as MockPortRepo:
            repo = MockPortRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = None

            resp = client.get(f"{BASE_URL}/test/asset-class-returns")

        assert resp.status_code == 404
        msg = resp.json()["error"]["message"]
        assert "snapshot" in msg.lower()

    def test_no_sector_mapping(self, client: TestClient):
        portfolio = _make_portfolio()
        snapshot = _make_snapshot(sector_mapping=None)

        with patch(_PORTFOLIO_REPO) as MockPortRepo:
            repo = MockPortRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot

            resp = client.get(f"{BASE_URL}/test/asset-class-returns")

        assert resp.status_code == 422
        msg = resp.json()["error"]["message"]
        assert "sector" in msg.lower()

    def test_no_price_data(self, client: TestClient):
        portfolio = _make_portfolio()
        snapshot = _make_snapshot()

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
        ):
            repo = MockPortRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot

            MockDashRepo.return_value.get_multi_ticker_prices.return_value = (
                pd.DataFrame()
            )

            resp = client.get(f"{BASE_URL}/test/asset-class-returns")

        assert resp.status_code == 422
        msg = resp.json()["error"]["message"]
        assert "price data" in msg.lower()

    def test_camel_case_serialization(self, client: TestClient):
        portfolio = _make_portfolio()
        snapshot = _make_snapshot()
        prices = _make_prices(["AAPL", "MSFT", "XOM", "CVX"])

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
        ):
            repo = MockPortRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot

            MockDashRepo.return_value.get_multi_ticker_prices.return_value = prices

            resp = client.get(f"{BASE_URL}/test/asset-class-returns")

        assert resp.status_code == 200
        body = resp.json()
        # camelCase key present
        assert "asOf" in body
        # snake_case key absent
        assert "as_of" not in body

    def test_sectors_ordered_by_weight_descending(self, client: TestClient):
        portfolio = _make_portfolio()
        snapshot = _make_snapshot(
            weights={"AAPL": 0.1, "XOM": 0.9},
            sector_mapping={"AAPL": "Technology", "XOM": "Energy"},
        )
        prices = _make_prices(["AAPL", "XOM"])

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
        ):
            repo = MockPortRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot

            MockDashRepo.return_value.get_multi_ticker_prices.return_value = prices

            resp = client.get(f"{BASE_URL}/test/asset-class-returns")

        assert resp.status_code == 200
        rows = resp.json()["returns"]
        # Energy (0.9) should be first
        assert rows[0]["name"] == "Energy"
