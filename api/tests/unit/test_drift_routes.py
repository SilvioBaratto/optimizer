"""Unit tests for the drift analysis route endpoint.

Mocks PortfolioRepository and DashboardRepository at the router's import
path so no real database is touched.
"""

from __future__ import annotations

import uuid
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

_PORTFOLIO_REPO = "app.api.v1.dashboard.PortfolioRepository"
_DASHBOARD_REPO = "app.api.v1.dashboard.DashboardRepository"

BASE_URL = "/api/v1/portfolio-analytics"


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
    snapshot_date: date | None = None,
) -> MagicMock:
    s = MagicMock()
    s.weights = weights or {"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2}
    s.snapshot_date = snapshot_date or date(2024, 1, 2)
    return s


def _make_position(
    ticker: str,
    yfinance_ticker: str | None,
    name: str | None,
    quantity: float,
    current_price: float | None,
) -> MagicMock:
    p = MagicMock()
    p.ticker = ticker
    p.yfinance_ticker = yfinance_ticker
    p.name = name
    p.quantity = quantity
    p.current_price = current_price
    return p


def _make_prices(tickers: list[str], n_days: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    data = {}
    for t in tickers:
        data[t] = (1 + rng.normal(0.0003, 0.012, n_days)).cumprod() * 100
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetDrift:
    def test_success_with_broker_positions(self, client: TestClient):
        portfolio = _make_portfolio("myport")
        snapshot = _make_snapshot()
        positions = [
            _make_position("AAPL_US_EQ", "AAPL", "Apple Inc.", 10, 150.0),
            _make_position("MSFT_US_EQ", "MSFT", "Microsoft", 10, 90.0),
            _make_position("GOOG_US_EQ", "GOOG", "Alphabet", 10, 60.0),
        ]

        with patch(_PORTFOLIO_REPO) as MockPortRepo:
            repo = MockPortRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot
            repo.get_positions.return_value = positions

            resp = client.get(f"{BASE_URL}/myport/drift")

        assert resp.status_code == 200
        body = resp.json()
        assert "entries" in body
        assert "totalDrift" in body
        assert "breachedCount" in body
        assert "threshold" in body
        assert body["threshold"] == 0.05
        assert len(body["entries"]) == 3

    def test_success_fallback_to_prices(self, client: TestClient):
        portfolio = _make_portfolio("myport")
        snapshot = _make_snapshot()
        prices = _make_prices(["AAPL", "MSFT", "GOOG"])

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
        ):
            repo = MockPortRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot
            repo.get_positions.return_value = []  # no broker positions

            MockDashRepo.return_value.get_multi_ticker_prices.return_value = prices

            resp = client.get(f"{BASE_URL}/myport/drift")

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["entries"]) == 3

    def test_portfolio_not_found(self, client: TestClient):
        with patch(_PORTFOLIO_REPO) as MockPortRepo:
            MockPortRepo.return_value.get_by_name.return_value = None

            resp = client.get(f"{BASE_URL}/missing/drift")

        assert resp.status_code == 404
        msg = resp.json()["error"]["message"]
        assert "not found" in msg.lower()

    def test_no_snapshot(self, client: TestClient):
        portfolio = _make_portfolio()

        with patch(_PORTFOLIO_REPO) as MockPortRepo:
            repo = MockPortRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = None

            resp = client.get(f"{BASE_URL}/test/drift")

        assert resp.status_code == 404
        msg = resp.json()["error"]["message"]
        assert "snapshot" in msg.lower()

    def test_no_positions_no_prices_returns_422(self, client: TestClient):
        portfolio = _make_portfolio()
        snapshot = _make_snapshot()

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
        ):
            repo = MockPortRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot
            repo.get_positions.return_value = []

            MockDashRepo.return_value.get_multi_ticker_prices.return_value = (
                pd.DataFrame()
            )

            resp = client.get(f"{BASE_URL}/test/drift")

        assert resp.status_code == 422
        msg = resp.json()["error"]["message"]
        assert "price data" in msg.lower()

    def test_custom_threshold(self, client: TestClient):
        portfolio = _make_portfolio()
        snapshot = _make_snapshot()
        positions = [
            _make_position("AAPL_EQ", "AAPL", "Apple", 10, 100.0),
            _make_position("MSFT_EQ", "MSFT", "Microsoft", 10, 100.0),
            _make_position("GOOG_EQ", "GOOG", "Google", 10, 100.0),
        ]

        with patch(_PORTFOLIO_REPO) as MockPortRepo:
            repo = MockPortRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot
            repo.get_positions.return_value = positions

            resp = client.get(f"{BASE_URL}/test/drift?threshold=0.1")

        assert resp.status_code == 200
        assert resp.json()["threshold"] == 0.1

    def test_camel_case_serialization(self, client: TestClient):
        portfolio = _make_portfolio()
        snapshot = _make_snapshot()
        positions = [
            _make_position("AAPL_EQ", "AAPL", "Apple", 10, 100.0),
            _make_position("MSFT_EQ", "MSFT", "Microsoft", 10, 100.0),
            _make_position("GOOG_EQ", "GOOG", "Google", 10, 100.0),
        ]

        with patch(_PORTFOLIO_REPO) as MockPortRepo:
            repo = MockPortRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot
            repo.get_positions.return_value = positions

            resp = client.get(f"{BASE_URL}/test/drift")

        assert resp.status_code == 200
        body = resp.json()
        # camelCase keys present
        assert "totalDrift" in body
        assert "breachedCount" in body
        # snake_case keys absent
        assert "total_drift" not in body
        assert "breached_count" not in body
