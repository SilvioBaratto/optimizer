"""Unit tests for GET /api/v1/market/indices.

Mocks DashboardRepository at the router's import path.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

_DASHBOARD_REPO = "app.api.v1.dashboard.DashboardRepository"

BASE_URL = "/api/v1/market"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_etf_instrument(
    ticker: str,
    name: str = "SPDR S&P 500 ETF",
    instrument_type: str = "ETF",
) -> MagicMock:
    inst = MagicMock()
    inst.yfinance_ticker = ticker
    inst.name = name
    inst.instrument_type = instrument_type
    return inst


# ---------------------------------------------------------------------------
# GET /api/v1/market/indices
# ---------------------------------------------------------------------------


class TestGetMarketIndices:
    def test_returns_list(self, client: TestClient):
        instruments = [
            _make_etf_instrument("SPY"),
            _make_etf_instrument("QQQ", "Invesco QQQ Trust"),
        ]

        with patch(_DASHBOARD_REPO) as MockRepo:
            MockRepo.return_value.get_etf_instruments.return_value = instruments
            resp = client.get(f"{BASE_URL}/indices")

        assert resp.status_code == 200
        body = resp.json()
        assert "indices" in body
        assert "total" in body
        assert body["total"] == 2
        assert len(body["indices"]) == 2

    def test_index_item_contains_expected_fields(self, client: TestClient):
        instruments = [_make_etf_instrument("SPY", "SPDR S&P 500 ETF", "ETF")]

        with patch(_DASHBOARD_REPO) as MockRepo:
            MockRepo.return_value.get_etf_instruments.return_value = instruments
            resp = client.get(f"{BASE_URL}/indices")

        item = resp.json()["indices"][0]
        assert "ticker" in item
        assert "name" in item
        assert "instrumentType" in item

    def test_index_values_correct(self, client: TestClient):
        instruments = [_make_etf_instrument("SPY", "SPDR S&P 500 ETF", "ETF")]

        with patch(_DASHBOARD_REPO) as MockRepo:
            MockRepo.return_value.get_etf_instruments.return_value = instruments
            resp = client.get(f"{BASE_URL}/indices")

        item = resp.json()["indices"][0]
        assert item["ticker"] == "SPY"
        assert item["name"] == "SPDR S&P 500 ETF"
        assert item["instrumentType"] == "ETF"

    def test_empty_db_returns_empty_list(self, client: TestClient):
        with patch(_DASHBOARD_REPO) as MockRepo:
            MockRepo.return_value.get_etf_instruments.return_value = []
            resp = client.get(f"{BASE_URL}/indices")

        assert resp.status_code == 200
        body = resp.json()
        assert body["indices"] == []
        assert body["total"] == 0

    def test_camel_case_serialization(self, client: TestClient):
        instruments = [_make_etf_instrument("SPY")]

        with patch(_DASHBOARD_REPO) as MockRepo:
            MockRepo.return_value.get_etf_instruments.return_value = instruments
            resp = client.get(f"{BASE_URL}/indices")

        item = resp.json()["indices"][0]
        assert "instrumentType" in item
        assert "instrument_type" not in item
