"""Unit tests for GET /api/v1/market/regime.

Mocks PortfolioRepository and DashboardRepository at the router's import path.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

_PORTFOLIO_REPO = "app.api.v1.dashboard.PortfolioRepository"
_DASHBOARD_REPO = "app.api.v1.dashboard.DashboardRepository"
_FIT_HMM = "app.api.v1.dashboard.fit_hmm"

BASE_URL = "/api/v1/market"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spy_prices(n: int = 520) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-02", periods=n)
    prices = (1 + rng.normal(0.0003, 0.01, n)).cumprod() * 450.0
    return pd.DataFrame({"SPY": prices}, index=dates)


def _make_hmm_result(n_states: int = 4) -> MagicMock:
    """Return a MagicMock shaped like HMMResult."""
    dates = pd.bdate_range("2024-01-02", periods=519)
    result = MagicMock()
    # State 0: lowest mean (bear), State 1: low var (sideways),
    # State 2: highest mean (bull), State 3: highest var (volatile)
    result.regime_means = pd.DataFrame(
        np.array([[-0.002], [0.000], [0.003], [-0.001]]),
        index=list(range(n_states)),
        columns=["SPY"],
    )
    result.regime_covariances = np.array(
        [[[0.0004]], [[0.0001]], [[0.0002]], [[0.0009]]]
    )
    probs = np.zeros((519, n_states))
    probs[:, 2] = 0.72  # state 2 = bull (highest mean)
    probs[:, 1] = 0.21
    probs[:, 0] = 0.05
    probs[:, 3] = 0.02
    result.filtered_probs = pd.DataFrame(probs, index=dates)
    result.smoothed_probs = pd.DataFrame(probs, index=dates)
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetMarketRegime:
    def test_success_cache_miss(self, client: TestClient):
        spy_df = _make_spy_prices()
        hmm_mock = _make_hmm_result()

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
            patch(_FIT_HMM, return_value=hmm_mock),
        ):
            MockPortRepo.return_value.get_latest_regime.return_value = None
            MockDashRepo.return_value.get_multi_ticker_prices.return_value = spy_df

            resp = client.get(f"{BASE_URL}/regime")

        assert resp.status_code == 200
        body = resp.json()
        assert "current" in body
        assert "probability" in body
        assert "since" in body
        assert "hmmStates" in body
        assert "modelInfo" in body

    def test_camel_case_serialization(self, client: TestClient):
        spy_df = _make_spy_prices()
        hmm_mock = _make_hmm_result()

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
            patch(_FIT_HMM, return_value=hmm_mock),
        ):
            MockPortRepo.return_value.get_latest_regime.return_value = None
            MockDashRepo.return_value.get_multi_ticker_prices.return_value = spy_df

            resp = client.get(f"{BASE_URL}/regime")

        body = resp.json()
        # camelCase keys present
        assert "hmmStates" in body
        assert "modelInfo" in body
        assert "nStates" in body["modelInfo"]
        assert "lastFitted" in body["modelInfo"]
        # snake_case keys absent
        assert "hmm_states" not in body
        assert "model_info" not in body

    def test_hmm_states_sum_to_one(self, client: TestClient):
        spy_df = _make_spy_prices()
        hmm_mock = _make_hmm_result()

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
            patch(_FIT_HMM, return_value=hmm_mock),
        ):
            MockPortRepo.return_value.get_latest_regime.return_value = None
            MockDashRepo.return_value.get_multi_ticker_prices.return_value = spy_df

            resp = client.get(f"{BASE_URL}/regime")

        states = resp.json()["hmmStates"]
        total = sum(s["probability"] for s in states)
        assert total == pytest.approx(1.0, abs=0.01)

    def test_regime_labels_valid(self, client: TestClient):
        spy_df = _make_spy_prices()
        hmm_mock = _make_hmm_result()

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
            patch(_FIT_HMM, return_value=hmm_mock),
        ):
            MockPortRepo.return_value.get_latest_regime.return_value = None
            MockDashRepo.return_value.get_multi_ticker_prices.return_value = spy_df

            resp = client.get(f"{BASE_URL}/regime")

        valid = {"bull", "bear", "sideways", "volatile"}
        body = resp.json()
        assert body["current"] in valid
        for state in body["hmmStates"]:
            assert state["regime"] in valid

    def test_cache_hit_returns_without_fitting(self, client: TestClient):
        cached = MagicMock()
        cached.state_date = date.today()
        cached.regime = "bull"
        cached.probabilities = [
            {"regime": "bull", "probability": 0.72},
            {"regime": "sideways", "probability": 0.21},
            {"regime": "bear", "probability": 0.05},
            {"regime": "volatile", "probability": 0.02},
        ]
        cached.metadata_ = {
            "since": "2025-11-15",
            "n_states": 4,
            "last_fitted": "2026-03-17T07:00:00+00:00",
        }

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
            patch(_FIT_HMM) as mock_fit,
        ):
            MockPortRepo.return_value.get_latest_regime.return_value = cached
            resp = client.get(f"{BASE_URL}/regime")

            mock_fit.assert_not_called()
            MockDashRepo.return_value.get_multi_ticker_prices.assert_not_called()

        assert resp.status_code == 200
        assert resp.json()["current"] == "bull"

    def test_missing_spy_data_returns_503(self, client: TestClient):
        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
        ):
            MockPortRepo.return_value.get_latest_regime.return_value = None
            MockDashRepo.return_value.get_multi_ticker_prices.return_value = (
                pd.DataFrame()
            )

            resp = client.get(f"{BASE_URL}/regime")

        assert resp.status_code == 503

    def test_fit_hmm_failure_returns_503(self, client: TestClient):
        spy_df = _make_spy_prices()

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
            patch(_FIT_HMM, side_effect=RuntimeError("convergence failed")),
        ):
            MockPortRepo.return_value.get_latest_regime.return_value = None
            MockDashRepo.return_value.get_multi_ticker_prices.return_value = spy_df

            resp = client.get(f"{BASE_URL}/regime")

        assert resp.status_code == 503
        assert "convergence failed" in resp.json()["error"]["message"]

    def test_upsert_called_on_cache_miss(self, client: TestClient):
        spy_df = _make_spy_prices()
        hmm_mock = _make_hmm_result()

        with (
            patch(_PORTFOLIO_REPO) as MockPortRepo,
            patch(_DASHBOARD_REPO) as MockDashRepo,
            patch(_FIT_HMM, return_value=hmm_mock),
        ):
            MockPortRepo.return_value.get_latest_regime.return_value = None
            MockDashRepo.return_value.get_multi_ticker_prices.return_value = spy_df

            resp = client.get(f"{BASE_URL}/regime")

        assert resp.status_code == 200
        MockPortRepo.return_value.upsert_regime_state.assert_called_once()
