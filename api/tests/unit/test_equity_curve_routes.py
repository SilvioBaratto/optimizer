"""Unit tests for the equity-curve route endpoint.

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

BASE_URL = "/api/v1/portfolio"


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
) -> MagicMock:
    s = MagicMock()
    s.weights = weights or {"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2}
    return s


def _make_prices(tickers: list[str], n_days: int = 756) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2021-03-01", periods=n_days)
    data = {}
    for t in tickers:
        data[t] = (1 + rng.normal(0.0003, 0.012, n_days)).cumprod() * 100
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetEquityCurve:
    def test_success(self, client: TestClient):
        portfolio = _make_portfolio("myport")
        snapshot = _make_snapshot()
        prices = _make_prices(["AAPL", "MSFT", "GOOG", "SPY"])

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
        ):
            repo = MockPortRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot

            MockDashRepo.return_value.get_multi_ticker_prices.return_value = prices

            resp = client.get(f"{BASE_URL}/myport/equity-curve")

        assert resp.status_code == 200
        body = resp.json()
        assert "points" in body
        assert "portfolioTotalReturn" in body
        assert "benchmarkTotalReturn" in body
        assert len(body["points"]) > 0

    def test_points_rebased_to_100(self, client: TestClient):
        portfolio = _make_portfolio("myport")
        snapshot = _make_snapshot()
        prices = _make_prices(["AAPL", "MSFT", "GOOG", "SPY"])

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
        ):
            repo = MockPortRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot

            MockDashRepo.return_value.get_multi_ticker_prices.return_value = prices

            resp = client.get(f"{BASE_URL}/myport/equity-curve")

        assert resp.status_code == 200
        first = resp.json()["points"][0]
        assert first["portfolio"] == pytest.approx(100.0)
        assert first["benchmark"] == pytest.approx(100.0)

    @pytest.mark.parametrize("period", ["1Y", "3Y", "5Y", "MAX"])
    def test_all_period_values(self, client: TestClient, period: str):
        portfolio = _make_portfolio("myport")
        snapshot = _make_snapshot()
        prices = _make_prices(["AAPL", "MSFT", "GOOG", "SPY"])

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
        ):
            repo = MockPortRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot

            MockDashRepo.return_value.get_multi_ticker_prices.return_value = prices

            resp = client.get(
                f"{BASE_URL}/myport/equity-curve?period={period}"
            )

        assert resp.status_code == 200

    def test_invalid_period(self, client: TestClient):
        portfolio = _make_portfolio("myport")
        snapshot = _make_snapshot()

        with patch(_PORTFOLIO_REPO) as MockPortRepo:
            repo = MockPortRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot

            resp = client.get(
                f"{BASE_URL}/myport/equity-curve?period=10Y"
            )

        assert resp.status_code == 422

    def test_portfolio_not_found(self, client: TestClient):
        with patch(_PORTFOLIO_REPO) as MockPortRepo:
            MockPortRepo.return_value.get_by_name.return_value = None

            resp = client.get(f"{BASE_URL}/missing/equity-curve")

        assert resp.status_code == 404
        msg = resp.json()["error"]["message"]
        assert "not found" in msg.lower()

    def test_no_snapshot(self, client: TestClient):
        portfolio = _make_portfolio()

        with patch(_PORTFOLIO_REPO) as MockPortRepo:
            repo = MockPortRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = None

            resp = client.get(f"{BASE_URL}/test/equity-curve")

        assert resp.status_code == 404
        msg = resp.json()["error"]["message"]
        assert "snapshot" in msg.lower()

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

            resp = client.get(f"{BASE_URL}/test/equity-curve")

        assert resp.status_code == 422
        msg = resp.json()["error"]["message"]
        assert "price data" in msg.lower()

    def test_benchmark_missing(self, client: TestClient):
        portfolio = _make_portfolio()
        snapshot = _make_snapshot()
        prices = _make_prices(["AAPL", "MSFT", "GOOG"])

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
        ):
            repo = MockPortRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot

            MockDashRepo.return_value.get_multi_ticker_prices.return_value = prices

            resp = client.get(
                f"{BASE_URL}/test/equity-curve?benchmark=SPY"
            )

        assert resp.status_code == 422
        msg = resp.json()["error"]["message"]
        assert "benchmark" in msg.lower()

    def test_custom_benchmark(self, client: TestClient):
        portfolio = _make_portfolio()
        snapshot = _make_snapshot()
        prices = _make_prices(["AAPL", "MSFT", "GOOG", "QQQ"])

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
        ):
            repo = MockPortRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot

            MockDashRepo.return_value.get_multi_ticker_prices.return_value = prices

            resp = client.get(
                f"{BASE_URL}/test/equity-curve?benchmark=QQQ"
            )

        assert resp.status_code == 200
