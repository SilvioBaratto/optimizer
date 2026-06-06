"""Unit tests for dashboard route endpoints.

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

_PORTFOLIO_REPO = "app.api.v1.dashboard.dashboard.PortfolioRepository"
_DASHBOARD_REPO = "app.api.v1.dashboard.dashboard.DashboardRepository"

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
) -> MagicMock:
    s = MagicMock()
    s.weights = weights or {"AAPL": 0.5, "MSFT": 0.3, "GOOG": 0.2}
    return s


def _make_account(total: float = 12500.0) -> MagicMock:
    a = MagicMock()
    a.total = total
    return a


def _make_prices(tickers: list[str], n_days: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    data = {}
    for t in tickers:
        data[t] = (1 + rng.normal(0.0003, 0.012, n_days)).cumprod() * 100
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetPerformanceMetrics:
    def test_success(self, client: TestClient):
        portfolio = _make_portfolio("myport")
        snapshot = _make_snapshot()
        account = _make_account()
        prices = _make_prices(["AAPL", "MSFT", "GOOG", "SPY"])

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
        ):
            repo = MockPortRepo.return_value
            repo.get_by_id_or_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot
            repo.get_latest_account_snapshot.return_value = account

            MockDashRepo.return_value.get_multi_ticker_prices.return_value = prices

            resp = client.get(f"{BASE_URL}/myport/performance-metrics")

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["kpis"]) == 7
        assert "nav" in body
        assert "navChangePct" in body

        # Verify camelCase serialization
        kpi = body["kpis"][0]
        assert "changeLabel" in kpi
        assert "label" in kpi
        assert "sparkline" in kpi

    def test_portfolio_not_found(self, client: TestClient):
        with patch(_PORTFOLIO_REPO) as MockPortRepo:
            MockPortRepo.return_value.get_by_id_or_name.return_value = None

            resp = client.get(f"{BASE_URL}/missing/performance-metrics")

        assert resp.status_code == 404
        msg = resp.json()["error"]["message"]
        assert "not found" in msg.lower()

    def test_no_snapshot(self, client: TestClient):
        portfolio = _make_portfolio()

        with patch(_PORTFOLIO_REPO) as MockPortRepo:
            repo = MockPortRepo.return_value
            repo.get_by_id_or_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = None

            resp = client.get(f"{BASE_URL}/test/performance-metrics")

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
            repo.get_by_id_or_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot
            repo.get_latest_account_snapshot.return_value = None

            MockDashRepo.return_value.get_multi_ticker_prices.return_value = (
                pd.DataFrame()
            )

            resp = client.get(f"{BASE_URL}/test/performance-metrics")

        assert resp.status_code == 422
        msg = resp.json()["error"]["message"]
        assert "price data" in msg.lower()

    def test_benchmark_not_in_instruments_returns_422(self, client: TestClient):
        """Issue #460: benchmark ticker absent from instruments table."""
        portfolio = _make_portfolio()
        snapshot = _make_snapshot()
        # Must provide non-empty prices so the prices.empty check passes
        prices = _make_prices(["AAPL", "MSFT", "GOOG"])

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
        ):
            repo = MockPortRepo.return_value
            repo.get_by_id_or_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot
            repo.get_latest_account_snapshot.return_value = None

            dash = MockDashRepo.return_value
            dash.instrument_exists.return_value = False
            dash.get_multi_ticker_prices.return_value = prices

            resp = client.get(f"{BASE_URL}/test/performance-metrics?benchmark=FAKE")

        assert resp.status_code == 422
        msg = resp.json()["error"]["message"].lower()
        assert "not found" in msg
        assert "instruments" in msg

    def test_benchmark_no_price_data_returns_422(self, client: TestClient):
        """Issue #460: benchmark exists in instruments but has no price rows."""
        portfolio = _make_portfolio()
        snapshot = _make_snapshot()
        # Prices without the benchmark column
        prices = _make_prices(["AAPL", "MSFT", "GOOG"])

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
        ):
            repo = MockPortRepo.return_value
            repo.get_by_id_or_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot
            repo.get_latest_account_snapshot.return_value = None

            dash = MockDashRepo.return_value
            dash.instrument_exists.return_value = True
            dash.get_multi_ticker_prices.return_value = prices

            resp = client.get(f"{BASE_URL}/test/performance-metrics?benchmark=SPY")

        assert resp.status_code == 422
        msg = resp.json()["error"]["message"].lower()
        assert "benchmark" in msg
        assert "price data" in msg

    def test_custom_benchmark(self, client: TestClient):
        portfolio = _make_portfolio()
        snapshot = _make_snapshot()
        prices = _make_prices(["AAPL", "MSFT", "GOOG", "QQQ"])

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
        ):
            repo = MockPortRepo.return_value
            repo.get_by_id_or_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot
            repo.get_latest_account_snapshot.return_value = None

            MockDashRepo.return_value.get_multi_ticker_prices.return_value = prices

            resp = client.get(f"{BASE_URL}/test/performance-metrics?benchmark=QQQ")

        assert resp.status_code == 200

    def test_no_broker_account_uses_normalized_nav(self, client: TestClient):
        portfolio = _make_portfolio()
        snapshot = _make_snapshot()
        prices = _make_prices(["AAPL", "MSFT", "GOOG", "SPY"])

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
        ):
            repo = MockPortRepo.return_value
            repo.get_by_id_or_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot
            repo.get_latest_account_snapshot.return_value = None

            MockDashRepo.return_value.get_multi_ticker_prices.return_value = prices

            resp = client.get(f"{BASE_URL}/test/performance-metrics")

        assert resp.status_code == 200
        # NAV should be a normalised index (not 0 or None)
        assert resp.json()["nav"] > 0

    @pytest.mark.parametrize("period", ["1Y", "3Y", "5Y", "MAX"])
    def test_accepts_period_query_param(
        self,
        client: TestClient,
        period: str,
    ):
        """Issue #433: performance-metrics must honor a ``period`` query param.

        Mirrors the existing equity-curve route contract so the dashboard
        period selector can refresh both endpoints in lockstep.
        """
        portfolio = _make_portfolio("myport")
        snapshot = _make_snapshot()
        prices = _make_prices(["AAPL", "MSFT", "GOOG", "SPY"])

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
        ):
            repo = MockPortRepo.return_value
            repo.get_by_id_or_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot
            repo.get_latest_account_snapshot.return_value = None

            MockDashRepo.return_value.get_multi_ticker_prices.return_value = prices

            resp = client.get(f"{BASE_URL}/myport/performance-metrics?period={period}")

        assert resp.status_code == 200

    def test_rejects_invalid_period(self, client: TestClient):
        portfolio = _make_portfolio("myport")
        snapshot = _make_snapshot()

        with patch(_PORTFOLIO_REPO) as MockPortRepo:
            repo = MockPortRepo.return_value
            repo.get_by_id_or_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot

            resp = client.get(f"{BASE_URL}/myport/performance-metrics?period=10Y")

        assert resp.status_code == 422

    @pytest.mark.parametrize(
        "period,expected_lookback_days",
        [("1Y", 365), ("3Y", 365 * 3), ("5Y", 365 * 5)],
    )
    def test_period_filters_price_window(
        self,
        client: TestClient,
        period: str,
        expected_lookback_days: int,
    ):
        """The ``period`` query param must drive the price lookback window.

        Asserts the repository call uses ``today - period`` as the start date,
        mirroring the equity-curve route at ``dashboard.py:191-194``.
        """
        portfolio = _make_portfolio("myport")
        snapshot = _make_snapshot()
        prices = _make_prices(["AAPL", "MSFT", "GOOG", "SPY"])

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
        ):
            repo = MockPortRepo.return_value
            repo.get_by_id_or_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot
            repo.get_latest_account_snapshot.return_value = None

            dash = MockDashRepo.return_value
            dash.get_multi_ticker_prices.return_value = prices

            resp = client.get(f"{BASE_URL}/myport/performance-metrics?period={period}")

        assert resp.status_code == 200
        call_args = dash.get_multi_ticker_prices.call_args
        # Signature: get_multi_ticker_prices(tickers, start_date, end_date)
        start_date, end_date = call_args.args[1], call_args.args[2]
        actual_lookback = (end_date - start_date).days
        assert actual_lookback == expected_lookback_days


# ---------------------------------------------------------------------------
# Allocation sunburst
# ---------------------------------------------------------------------------

_SAMPLE_WEIGHTS = {"AAPL": 0.3, "MSFT": 0.2, "XOM": 0.3, "CVX": 0.2}
_SAMPLE_SECTOR_MAP = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "XOM": "Energy",
    "CVX": "Energy",
}


def _make_snapshot_with_sectors(
    weights: dict[str, float] | None = None,
    sector_mapping: dict[str, str] | None = None,
) -> MagicMock:
    s = MagicMock()
    s.weights = weights or _SAMPLE_WEIGHTS
    s.sector_mapping = (
        sector_mapping if sector_mapping is not None else _SAMPLE_SECTOR_MAP
    )
    return s


class TestGetAllocation:
    def test_success(self, client: TestClient):
        portfolio = _make_portfolio("myport")
        snapshot = _make_snapshot_with_sectors()

        with patch(_PORTFOLIO_REPO) as MockPortRepo:
            repo = MockPortRepo.return_value
            repo.get_by_id_or_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot

            resp = client.get(f"{BASE_URL}/myport/allocation")

        assert resp.status_code == 200
        body = resp.json()
        assert "nodes" in body
        assert "totalPositions" in body
        assert "totalSectors" in body
        assert len(body["nodes"]) > 0
        assert "children" in body["nodes"][0]

    def test_portfolio_not_found(self, client: TestClient):
        with patch(_PORTFOLIO_REPO) as MockPortRepo:
            MockPortRepo.return_value.get_by_id_or_name.return_value = None

            resp = client.get(f"{BASE_URL}/missing/allocation")

        assert resp.status_code == 404
        msg = resp.json()["error"]["message"]
        assert "not found" in msg.lower()

    def test_no_snapshot(self, client: TestClient):
        portfolio = _make_portfolio()

        with patch(_PORTFOLIO_REPO) as MockPortRepo:
            repo = MockPortRepo.return_value
            repo.get_by_id_or_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = None

            resp = client.get(f"{BASE_URL}/test/allocation")

        assert resp.status_code == 404
        msg = resp.json()["error"]["message"]
        assert "snapshot" in msg.lower()

    def test_no_sector_mapping(self, client: TestClient):
        portfolio = _make_portfolio()
        snapshot = _make_snapshot_with_sectors(sector_mapping=None)
        snapshot.sector_mapping = None

        with patch(_PORTFOLIO_REPO) as MockPortRepo:
            repo = MockPortRepo.return_value
            repo.get_by_id_or_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot

            resp = client.get(f"{BASE_URL}/test/allocation")

        assert resp.status_code == 422
        msg = resp.json()["error"]["message"]
        assert "sector" in msg.lower()

    def test_nodes_sorted_by_weight_descending(self, client: TestClient):
        portfolio = _make_portfolio()
        snapshot = _make_snapshot_with_sectors(
            weights={"AAPL": 0.1, "XOM": 0.9},
            sector_mapping={"AAPL": "Technology", "XOM": "Energy"},
        )

        with patch(_PORTFOLIO_REPO) as MockPortRepo:
            repo = MockPortRepo.return_value
            repo.get_by_id_or_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot

            resp = client.get(f"{BASE_URL}/test/allocation")

        assert resp.status_code == 200
        nodes = resp.json()["nodes"]
        assert nodes[0]["value"] >= nodes[1]["value"]

    def test_total_positions_equals_weight_count(self, client: TestClient):
        portfolio = _make_portfolio()
        snapshot = _make_snapshot_with_sectors()

        with patch(_PORTFOLIO_REPO) as MockPortRepo:
            repo = MockPortRepo.return_value
            repo.get_by_id_or_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot

            resp = client.get(f"{BASE_URL}/test/allocation")

        assert resp.status_code == 200
        assert resp.json()["totalPositions"] == len(_SAMPLE_WEIGHTS)

    def test_total_sectors_equals_sector_count(self, client: TestClient):
        portfolio = _make_portfolio()
        snapshot = _make_snapshot_with_sectors()

        with patch(_PORTFOLIO_REPO) as MockPortRepo:
            repo = MockPortRepo.return_value
            repo.get_by_id_or_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot

            resp = client.get(f"{BASE_URL}/test/allocation")

        assert resp.status_code == 200
        assert resp.json()["totalSectors"] == 2

    def test_camel_case_serialization(self, client: TestClient):
        portfolio = _make_portfolio()
        snapshot = _make_snapshot_with_sectors()

        with patch(_PORTFOLIO_REPO) as MockPortRepo:
            repo = MockPortRepo.return_value
            repo.get_by_id_or_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot

            resp = client.get(f"{BASE_URL}/test/allocation")

        assert resp.status_code == 200
        body = resp.json()
        # camelCase keys present
        assert "totalPositions" in body
        assert "totalSectors" in body
        # snake_case keys absent
        assert "total_positions" not in body
        assert "total_sectors" not in body


class TestGetRollingMetrics:
    def test_success(self, client: TestClient):
        portfolio = _make_portfolio()
        snapshot = _make_snapshot()
        prices = _make_prices(["AAPL", "MSFT", "GOOG", "SPY"])

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
        ):
            repo = MockPortRepo.return_value
            repo.get_by_id_or_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = snapshot

            MockDashRepo.return_value.get_multi_ticker_prices.return_value = prices

            resp = client.get(f"{BASE_URL}/test/rolling-metrics")

        assert resp.status_code == 200
        body = resp.json()
        assert set(body.keys()) == {"window", "sharpe", "volatility", "beta"}
        assert body["window"] == 63
        assert isinstance(body["sharpe"], list)
        assert len(body["sharpe"]) > 0
        # Each point shape: { date, value }
        first = body["sharpe"][0]
        assert set(first.keys()) == {"date", "value"}

    def test_accepts_window_query_param(self, client: TestClient):
        portfolio = _make_portfolio()
        snapshot = _make_snapshot()
        prices = _make_prices(["AAPL", "MSFT", "GOOG", "SPY"])

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
        ):
            MockPortRepo.return_value.get_by_id_or_name.return_value = portfolio
            MockPortRepo.return_value.get_latest_snapshot.return_value = snapshot
            MockDashRepo.return_value.get_multi_ticker_prices.return_value = prices

            resp = client.get(f"{BASE_URL}/test/rolling-metrics?window=21")

        assert resp.status_code == 200
        assert resp.json()["window"] == 21

    def test_rejects_window_below_minimum(self, client: TestClient):
        resp = client.get(f"{BASE_URL}/test/rolling-metrics?window=1")
        assert resp.status_code == 422

    def test_portfolio_not_found(self, client: TestClient):
        with patch(_PORTFOLIO_REPO) as MockPortRepo:
            MockPortRepo.return_value.get_by_id_or_name.return_value = None

            resp = client.get(f"{BASE_URL}/missing/rolling-metrics")

        assert resp.status_code == 404

    def test_no_snapshot(self, client: TestClient):
        portfolio = _make_portfolio()
        with patch(_PORTFOLIO_REPO) as MockPortRepo:
            repo = MockPortRepo.return_value
            repo.get_by_id_or_name.return_value = portfolio
            repo.get_latest_snapshot.return_value = None

            resp = client.get(f"{BASE_URL}/test/rolling-metrics")

        assert resp.status_code == 404

    def test_no_price_data(self, client: TestClient):
        portfolio = _make_portfolio()
        snapshot = _make_snapshot()

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
        ):
            MockPortRepo.return_value.get_by_id_or_name.return_value = portfolio
            MockPortRepo.return_value.get_latest_snapshot.return_value = snapshot
            MockDashRepo.return_value.get_multi_ticker_prices.return_value = (
                pd.DataFrame()
            )

            resp = client.get(f"{BASE_URL}/test/rolling-metrics")

        assert resp.status_code == 422

    def test_benchmark_not_in_instruments_returns_422(self, client: TestClient):
        """Issue #460: benchmark ticker absent from instruments table."""
        portfolio = _make_portfolio()
        snapshot = _make_snapshot()
        prices = _make_prices(["AAPL", "MSFT", "GOOG"])

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
        ):
            MockPortRepo.return_value.get_by_id_or_name.return_value = portfolio
            MockPortRepo.return_value.get_latest_snapshot.return_value = snapshot

            dash = MockDashRepo.return_value
            dash.instrument_exists.return_value = False
            dash.get_multi_ticker_prices.return_value = prices

            resp = client.get(f"{BASE_URL}/test/rolling-metrics?benchmark=FAKE")

        assert resp.status_code == 422
        msg = resp.json()["error"]["message"].lower()
        assert "not found" in msg
        assert "instruments" in msg

    def test_benchmark_no_price_data_returns_422(self, client: TestClient):
        """Issue #460: benchmark exists in instruments but has no price rows."""
        portfolio = _make_portfolio()
        snapshot = _make_snapshot()
        # Prices only contain portfolio tickers, not the benchmark
        prices = _make_prices(["AAPL", "MSFT", "GOOG"])

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
        ):
            MockPortRepo.return_value.get_by_id_or_name.return_value = portfolio
            MockPortRepo.return_value.get_latest_snapshot.return_value = snapshot

            dash = MockDashRepo.return_value
            dash.instrument_exists.return_value = True
            dash.get_multi_ticker_prices.return_value = prices

            resp = client.get(f"{BASE_URL}/test/rolling-metrics?benchmark=URTH")

        assert resp.status_code == 422
        msg = resp.json()["error"]["message"].lower()
        assert "benchmark" in msg
        assert "price data" in msg

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
        ):
            MockPortRepo.return_value.get_by_id_or_name.return_value = portfolio
            MockPortRepo.return_value.get_latest_snapshot.return_value = snapshot
            MockDashRepo.return_value.get_multi_ticker_prices.return_value = prices

            resp = client.get(f"{BASE_URL}/test/rolling-metrics?benchmark=URTH")

        assert resp.status_code == 422
