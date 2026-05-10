"""Unit tests for POST /api/v1/universe/screen (issue #361).

Covers:
  - Screen with default config returns 200, passing_tickers, diagnostics
  - Screen with preset config returns 200
  - Screen with custom thresholds returns 200
  - Empty universe returns empty result (not 500)
  - Preset + custom fields are mutually exclusive → 422
  - Invalid hysteresis (exit > entry) → 422

Service layer is mocked to isolate route logic from DB.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

BASE_URL = "/api/v1/universe/screen"

_SERVICE_PATH = "app.api.v1.universe.universe_screen.run_universe_screen"

_DEFAULT_SERVICE_RESULT: dict[str, Any] = {
    "passing_tickers": ["AAPL", "MSFT"],
    "total_screened": 5,
    "diagnostics": {
        "AAPL": {
            "market_cap": True,
            "mcap_percentile": True,
            "addv_12m": True,
            "addv_3m": True,
            "trading_frequency": True,
            "price": True,
            "trading_history": True,
            "ipo_seasoning": True,
            "financial_statements": True,
        },
        "MSFT": {
            "market_cap": True,
            "mcap_percentile": True,
            "addv_12m": True,
            "addv_3m": True,
            "trading_frequency": True,
            "price": True,
            "trading_history": True,
            "ipo_seasoning": True,
            "financial_statements": True,
        },
    },
}

_EMPTY_SERVICE_RESULT: dict[str, Any] = {
    "passing_tickers": [],
    "total_screened": 0,
    "diagnostics": {},
}


# ===========================================================================
# Default config
# ===========================================================================


class TestScreenDefaultConfig:
    """No preset, no custom fields — uses default thresholds."""

    def test_returns_200(self, client: TestClient) -> None:
        with patch(_SERVICE_PATH, return_value=_DEFAULT_SERVICE_RESULT):
            resp = client.post(BASE_URL, json={})

        assert resp.status_code == 200

    def test_body_contains_passing_tickers(self, client: TestClient) -> None:
        with patch(_SERVICE_PATH, return_value=_DEFAULT_SERVICE_RESULT):
            resp = client.post(BASE_URL, json={})

        body = resp.json()
        assert "passingTickers" in body
        assert set(body["passingTickers"]) == {"AAPL", "MSFT"}

    def test_body_contains_total_screened(self, client: TestClient) -> None:
        with patch(_SERVICE_PATH, return_value=_DEFAULT_SERVICE_RESULT):
            resp = client.post(BASE_URL, json={})

        body = resp.json()
        assert body["totalScreened"] == 5

    def test_body_contains_diagnostics(self, client: TestClient) -> None:
        with patch(_SERVICE_PATH, return_value=_DEFAULT_SERVICE_RESULT):
            resp = client.post(BASE_URL, json={})

        body = resp.json()
        assert "diagnostics" in body
        assert "AAPL" in body["diagnostics"]
        assert body["diagnostics"]["AAPL"]["market_cap"] is True


# ===========================================================================
# Preset config
# ===========================================================================


class TestScreenPreset:
    """When preset is provided, the corresponding factory method is used."""

    @pytest.mark.parametrize(
        "preset",
        ["developed_markets", "broad_universe", "small_cap", "large_cap"],
    )
    def test_valid_preset_returns_200(self, client: TestClient, preset: str) -> None:
        with patch(_SERVICE_PATH, return_value=_DEFAULT_SERVICE_RESULT):
            resp = client.post(BASE_URL, json={"preset": preset})

        assert resp.status_code == 200

    def test_service_receives_request_with_preset(self, client: TestClient) -> None:
        with patch(_SERVICE_PATH, return_value=_DEFAULT_SERVICE_RESULT) as mock_svc:
            client.post(BASE_URL, json={"preset": "broad_universe"})

        call_args = mock_svc.call_args
        request_arg = call_args.args[0]
        assert request_arg.preset is not None
        assert request_arg.preset.value == "broad_universe"


# ===========================================================================
# Custom thresholds
# ===========================================================================


class TestScreenCustomThresholds:
    """When individual config fields are provided, they override defaults."""

    def test_custom_market_cap_returns_200(self, client: TestClient) -> None:
        payload = {
            "market_cap": {"entry": 500_000_000, "exit": 400_000_000},
        }
        with patch(_SERVICE_PATH, return_value=_DEFAULT_SERVICE_RESULT):
            resp = client.post(BASE_URL, json=payload)

        assert resp.status_code == 200

    def test_service_receives_custom_fields(self, client: TestClient) -> None:
        payload = {
            "market_cap": {"entry": 500_000_000, "exit": 400_000_000},
            "min_trading_history": 180,
        }
        with patch(_SERVICE_PATH, return_value=_DEFAULT_SERVICE_RESULT) as mock_svc:
            client.post(BASE_URL, json=payload)

        request_arg = mock_svc.call_args.args[0]
        assert request_arg.market_cap is not None
        assert request_arg.market_cap.entry == 500_000_000
        assert request_arg.min_trading_history == 180

    def test_current_members_forwarded(self, client: TestClient) -> None:
        payload = {"current_members": ["AAPL", "MSFT"]}
        with patch(_SERVICE_PATH, return_value=_DEFAULT_SERVICE_RESULT) as mock_svc:
            client.post(BASE_URL, json=payload)

        request_arg = mock_svc.call_args.args[0]
        assert request_arg.current_members == ["AAPL", "MSFT"]


# ===========================================================================
# Empty universe
# ===========================================================================


class TestScreenEmptyUniverse:
    """Empty instruments table returns empty result, not a 500."""

    def test_empty_universe_returns_200(self, client: TestClient) -> None:
        with patch(_SERVICE_PATH, return_value=_EMPTY_SERVICE_RESULT):
            resp = client.post(BASE_URL, json={})

        assert resp.status_code == 200

    def test_empty_universe_has_empty_lists(self, client: TestClient) -> None:
        with patch(_SERVICE_PATH, return_value=_EMPTY_SERVICE_RESULT):
            resp = client.post(BASE_URL, json={})

        body = resp.json()
        assert body["passingTickers"] == []
        assert body["totalScreened"] == 0
        assert body["diagnostics"] == {}


# ===========================================================================
# Validation errors
# ===========================================================================


class TestScreenValidation:
    """Invalid request bodies return 422 Unprocessable Entity."""

    def test_preset_and_custom_fields_returns_422(self, client: TestClient) -> None:
        payload = {
            "preset": "developed_markets",
            "market_cap": {"entry": 500_000_000, "exit": 400_000_000},
        }
        resp = client.post(BASE_URL, json=payload)

        assert resp.status_code == 422

    def test_invalid_hysteresis_exit_greater_than_entry_returns_422(
        self, client: TestClient
    ) -> None:
        payload = {
            "market_cap": {"entry": 100_000_000, "exit": 200_000_000},
        }
        resp = client.post(BASE_URL, json=payload)

        assert resp.status_code == 422

    def test_invalid_preset_value_returns_422(self, client: TestClient) -> None:
        resp = client.post(BASE_URL, json={"preset": "not_a_preset"})

        assert resp.status_code == 422
