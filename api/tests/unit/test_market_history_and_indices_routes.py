"""Unit tests for GET /api/v1/market/regime/history and GET /api/v1/market/indices.

Mocks DashboardRepository at the router's import path.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

_DASHBOARD_REPO = "app.api.v1.dashboard.DashboardRepository"

BASE_URL = "/api/v1/market"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_regime_state(
    state_date: date,
    regime: str = "bull",
    probs: dict[str, float] | None = None,
) -> MagicMock:
    """Build a MagicMock shaped like RegimeState with probability_entries."""
    if probs is None:
        probs = {"bull": 0.70, "bear": 0.10, "sideways": 0.15, "volatile": 0.05}

    state = MagicMock()
    state.state_date = state_date
    state.regime = regime

    entries = []
    for r, p in probs.items():
        entry = MagicMock()
        entry.regime = r
        entry.probability = p
        entries.append(entry)

    state.probability_entries = entries
    return state


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
# GET /api/v1/market/regime/history
# ---------------------------------------------------------------------------


class TestGetRegimeHistory:
    def test_returns_time_series(self, client: TestClient):
        states = [
            _make_regime_state(date(2025, 1, 2), "bull"),
            _make_regime_state(date(2025, 1, 3), "bear"),
        ]

        with patch(_DASHBOARD_REPO) as MockRepo:
            MockRepo.return_value.get_regime_history.return_value = states
            resp = client.get(f"{BASE_URL}/regime/history")

        assert resp.status_code == 200
        body = resp.json()
        assert "points" in body
        assert "total" in body
        assert body["total"] == 2
        assert len(body["points"]) == 2

    def test_points_contain_expected_fields(self, client: TestClient):
        states = [_make_regime_state(date(2025, 1, 2), "bull")]

        with patch(_DASHBOARD_REPO) as MockRepo:
            MockRepo.return_value.get_regime_history.return_value = states
            resp = client.get(f"{BASE_URL}/regime/history")

        point = resp.json()["points"][0]
        assert "date" in point
        assert "bullProb" in point
        assert "bearProb" in point
        assert "sidewaysProb" in point
        assert "volatileProb" in point
        assert "regime" in point

    def test_probabilities_match_entries(self, client: TestClient):
        probs = {"bull": 0.72, "bear": 0.08, "sideways": 0.15, "volatile": 0.05}
        states = [_make_regime_state(date(2025, 1, 2), "bull", probs)]

        with patch(_DASHBOARD_REPO) as MockRepo:
            MockRepo.return_value.get_regime_history.return_value = states
            resp = client.get(f"{BASE_URL}/regime/history")

        point = resp.json()["points"][0]
        assert point["bullProb"] == pytest.approx(0.72)
        assert point["bearProb"] == pytest.approx(0.08)
        assert point["sidewaysProb"] == pytest.approx(0.15)
        assert point["volatileProb"] == pytest.approx(0.05)

    def test_empty_db_returns_empty_list(self, client: TestClient):
        with patch(_DASHBOARD_REPO) as MockRepo:
            MockRepo.return_value.get_regime_history.return_value = []
            resp = client.get(f"{BASE_URL}/regime/history")

        assert resp.status_code == 200
        body = resp.json()
        assert body["points"] == []
        assert body["total"] == 0

    def test_start_date_filter_passed_to_repo(self, client: TestClient):
        with patch(_DASHBOARD_REPO) as MockRepo:
            MockRepo.return_value.get_regime_history.return_value = []
            resp = client.get(
                f"{BASE_URL}/regime/history",
                params={"start_date": "2025-01-01"},
            )

        assert resp.status_code == 200
        MockRepo.return_value.get_regime_history.assert_called_once_with(
            start_date=date(2025, 1, 1),
            end_date=None,
        )

    def test_end_date_filter_passed_to_repo(self, client: TestClient):
        with patch(_DASHBOARD_REPO) as MockRepo:
            MockRepo.return_value.get_regime_history.return_value = []
            resp = client.get(
                f"{BASE_URL}/regime/history",
                params={"end_date": "2025-12-31"},
            )

        assert resp.status_code == 200
        MockRepo.return_value.get_regime_history.assert_called_once_with(
            start_date=None,
            end_date=date(2025, 12, 31),
        )

    def test_date_range_filter_passed_to_repo(self, client: TestClient):
        with patch(_DASHBOARD_REPO) as MockRepo:
            MockRepo.return_value.get_regime_history.return_value = []
            resp = client.get(
                f"{BASE_URL}/regime/history",
                params={"start_date": "2025-01-01", "end_date": "2025-06-30"},
            )

        assert resp.status_code == 200
        MockRepo.return_value.get_regime_history.assert_called_once_with(
            start_date=date(2025, 1, 1),
            end_date=date(2025, 6, 30),
        )

    def test_camel_case_serialization(self, client: TestClient):
        states = [_make_regime_state(date(2025, 1, 2))]

        with patch(_DASHBOARD_REPO) as MockRepo:
            MockRepo.return_value.get_regime_history.return_value = states
            resp = client.get(f"{BASE_URL}/regime/history")

        body = resp.json()
        # camelCase present
        assert "bullProb" in body["points"][0]
        assert "bearProb" in body["points"][0]
        assert "sidewaysProb" in body["points"][0]
        assert "volatileProb" in body["points"][0]
        # snake_case absent
        assert "bull_prob" not in body["points"][0]
        assert "bear_prob" not in body["points"][0]


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
