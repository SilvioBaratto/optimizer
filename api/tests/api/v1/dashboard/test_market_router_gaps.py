"""Coverage gap-fill for dashboard.py service-layer ValueError → 422 branches.

Issue #827: lines 185-186, 261-262, 341-342, 407-408, 617-618 in
``app.api.v1.dashboard.dashboard`` were uncovered.  Each is the ``except
ValueError`` handler around a ``dashboard_service.*`` call.

All repository calls are mocked at the router's import path so no real DB
is touched.  Uses the conftest ``client`` fixture.
"""

from __future__ import annotations

import uuid
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

_PORTFOLIO_REPO = "app.api.v1.dashboard.dashboard.PortfolioRepository"
_DASHBOARD_REPO = "app.api.v1.dashboard.dashboard.DashboardRepository"
_DASHBOARD_SVC = "app.api.v1.dashboard.dashboard.dashboard_service"

BASE_URL = "/api/v1/portfolio-analytics"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_portfolio(name: str = "test") -> MagicMock:
    p = MagicMock()
    p.id = uuid.uuid4()
    p.name = name
    p.currency = "EUR"
    return p


def _make_snapshot(
    weights: dict[str, float] | None = None,
    sector_mapping: dict[str, str] | None = None,
    snapshot_date: date | None = None,
) -> MagicMock:
    s = MagicMock()
    s.weights = weights or {"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2}
    s.sector_mapping = sector_mapping or {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "GOOG": "Technology",
    }
    s.snapshot_date = snapshot_date or date(2024, 1, 2)
    return s


def _make_prices(tickers: list[str], n_days: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    data = {t: (1 + rng.normal(0.0003, 0.012, n_days)).cumprod() * 100 for t in tickers}
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# GET /{name}/performance-metrics — lines 185-186
# ---------------------------------------------------------------------------


class TestPerformanceMetricsServiceError:
    def test_when_compute_performance_metrics_raises_value_error_then_422(
        self, client: TestClient
    ) -> None:
        portfolio = _make_portfolio("myport")
        snapshot = _make_snapshot()
        prices = _make_prices(["AAPL", "MSFT", "GOOG", "SPY"])

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
            patch(
                f"{_DASHBOARD_SVC}.compute_performance_metrics",
                side_effect=ValueError("insufficient history"),
            ),
        ):
            repo = MockPortRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot
            repo.get_latest_account_snapshot.return_value = None

            dash = MockDashRepo.return_value
            dash.get_multi_ticker_prices.return_value = prices
            dash.instrument_exists.return_value = True

            resp = client.get(f"{BASE_URL}/myport/performance-metrics")

        assert resp.status_code == 422
        assert "insufficient history" in resp.json()["error"]["message"]


# ---------------------------------------------------------------------------
# GET /{name}/equity-curve — lines 261-262
# ---------------------------------------------------------------------------


class TestEquityCurveServiceError:
    def test_when_compute_equity_curve_raises_value_error_then_422(
        self, client: TestClient
    ) -> None:
        portfolio = _make_portfolio("myport")
        snapshot = _make_snapshot()
        prices = _make_prices(["AAPL", "MSFT", "GOOG", "SPY"])

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
            patch(
                f"{_DASHBOARD_SVC}.compute_equity_curve",
                side_effect=ValueError("not enough returns"),
            ),
        ):
            repo = MockPortRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot

            dash = MockDashRepo.return_value
            dash.get_multi_ticker_prices.return_value = prices
            dash.instrument_exists.return_value = True

            resp = client.get(f"{BASE_URL}/myport/equity-curve")

        assert resp.status_code == 422
        assert "not enough returns" in resp.json()["error"]["message"]


# ---------------------------------------------------------------------------
# GET /{name}/rolling-metrics — lines 341-342
# ---------------------------------------------------------------------------


class TestRollingMetricsServiceError:
    def test_when_compute_rolling_metrics_raises_value_error_then_422(
        self, client: TestClient
    ) -> None:
        portfolio = _make_portfolio("myport")
        snapshot = _make_snapshot()
        prices = _make_prices(["AAPL", "MSFT", "GOOG", "SPY"])

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
            patch(
                f"{_DASHBOARD_SVC}.compute_rolling_metrics",
                side_effect=ValueError("window too large"),
            ),
        ):
            repo = MockPortRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot

            dash = MockDashRepo.return_value
            dash.get_multi_ticker_prices.return_value = prices
            dash.instrument_exists.return_value = True

            resp = client.get(f"{BASE_URL}/myport/rolling-metrics")

        assert resp.status_code == 422
        assert "window too large" in resp.json()["error"]["message"]


# ---------------------------------------------------------------------------
# GET /{name}/allocation — lines 407-408
# ---------------------------------------------------------------------------


class TestAllocationServiceError:
    def test_when_compute_allocation_raises_value_error_then_422(
        self, client: TestClient
    ) -> None:
        portfolio = _make_portfolio("myport")
        snapshot = _make_snapshot()

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(
                f"{_DASHBOARD_SVC}.compute_allocation",
                side_effect=ValueError("empty weights"),
            ),
        ):
            repo = MockPortRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot

            resp = client.get(f"{BASE_URL}/myport/allocation")

        assert resp.status_code == 422
        assert "empty weights" in resp.json()["error"]["message"]


# ---------------------------------------------------------------------------
# GET /{name}/asset-class-returns — lines 617-618
# ---------------------------------------------------------------------------


class TestAssetClassReturnsServiceError:
    def test_when_compute_asset_class_returns_raises_value_error_then_422(
        self, client: TestClient
    ) -> None:
        portfolio = _make_portfolio("myport")
        snapshot = _make_snapshot(
            weights={"AAPL": 0.5, "XOM": 0.5},
            sector_mapping={"AAPL": "Technology", "XOM": "Energy"},
        )
        prices = _make_prices(["AAPL", "XOM"])

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
            patch(
                f"{_DASHBOARD_SVC}.compute_asset_class_returns",
                side_effect=ValueError("bad period"),
            ),
        ):
            repo = MockPortRepo.return_value
            repo.get_by_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot

            dash = MockDashRepo.return_value
            dash.get_multi_ticker_prices.return_value = prices

            resp = client.get(f"{BASE_URL}/myport/asset-class-returns")

        assert resp.status_code == 422
        assert "bad period" in resp.json()["error"]["message"]
