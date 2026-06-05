"""Route tests for the view-generation + entropy-pooling endpoints (issue #817).

Covers ``app.api.v1.views.views``: ``POST /views/generate`` and
``POST /views/entropy-pooling``.  All BAML / solver work is mocked at the
service boundary (``fetch_factor_data`` / ``generate_views`` /
``fetch_prices_df`` / ``run_entropy_pooling``) — never a real BAML client.
Uses the conftest ``client`` fixture (DI-overridden DB), not a module-level
``TestClient``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.schemas.views.views import EntropyPoolingResponse, GenerateViewsResponse
from tests._fixtures.assertions import assert_json_safe, assert_response_matches_schema

GEN_URL = "/api/v1/views/generate"
EP_URL = "/api/v1/views/entropy-pooling"
_FETCH_FACTOR = "app.api.v1.views.views.fetch_factor_data"
_GENERATE = "app.api.v1.views.views.generate_views"
_FETCH_PRICES = "app.api.v1.views.views.fetch_prices_df"
_RUN_EP = "app.api.v1.views.views.run_entropy_pooling"


def _factor(ticker: str) -> SimpleNamespace:
    return SimpleNamespace(ticker=ticker)


def _generated_views(p: np.ndarray, q: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        view_strings=["AAPL == 0.020000"],
        P=p,
        Q=q,
        view_confidences=[0.5],
        idzorek_alphas={"AAPL": 0.5},
        asset_views=[
            SimpleNamespace(
                asset="AAPL", direction=1, magnitude_bps=200.0,
                confidence=0.5, reasoning="momentum strong",
            )
        ],
        rationale="net bullish AAPL",
    )


def _price_frame() -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=3)
    return pd.DataFrame({"AAPL": [100.0, 101.0, 102.0],
                         "MSFT": [50.0, 50.5, 51.0]}, index=idx)


# ---------------------------------------------------------------------------
# POST /views/generate
# ---------------------------------------------------------------------------


class TestGenerateViews:
    def test_when_views_generated_then_200_with_p_q_shapes(
        self, client: TestClient
    ) -> None:
        gv = _generated_views(np.array([[1.0, 0.0]]), np.array([0.02]))
        with (
            patch(_FETCH_FACTOR, return_value=[_factor("AAPL"), _factor("MSFT")]),
            patch(_GENERATE, return_value=gv),
        ):
            resp = client.post(GEN_URL, json={"tickers": ["AAPL", "MSFT"]})

        assert resp.status_code == 200
        body = resp.json()
        assert_response_matches_schema(body, GenerateViewsResponse)
        # CamelCaseModel lowercases single-token field names: P -> "p", Q -> "q".
        assert body["p"] == [[1.0, 0.0]]
        assert body["q"] == [0.02]
        assert_json_safe(body)

    def test_when_no_factor_data_then_422(self, client: TestClient) -> None:
        with patch(_FETCH_FACTOR, return_value=[]):
            resp = client.post(GEN_URL, json={"tickers": ["AAPL", "MSFT"]})

        assert resp.status_code == 422

    def test_when_llm_raises_then_502(self, client: TestClient) -> None:
        with (
            patch(_FETCH_FACTOR, return_value=[_factor("AAPL"), _factor("MSFT")]),
            patch(_GENERATE, side_effect=RuntimeError("baml down")),
        ):
            resp = client.post(GEN_URL, json={"tickers": ["AAPL", "MSFT"]})

        assert resp.status_code == 502

    def test_when_p_shape_mismatch_then_500(self, client: TestClient) -> None:
        # 1 view declared but P is (2, 2) -> shape guard fails.
        gv = _generated_views(np.zeros((2, 2)), np.array([0.02]))
        with (
            patch(_FETCH_FACTOR, return_value=[_factor("AAPL"), _factor("MSFT")]),
            patch(_GENERATE, return_value=gv),
        ):
            resp = client.post(GEN_URL, json={"tickers": ["AAPL", "MSFT"]})

        assert resp.status_code == 500

    def test_when_db_fetch_raises_then_500(self, client: TestClient) -> None:
        with patch(_FETCH_FACTOR, side_effect=RuntimeError("db boom")):
            resp = client.post(GEN_URL, json={"tickers": ["AAPL", "MSFT"]})

        assert resp.status_code == 500

    def test_when_q_shape_mismatch_then_500(self, client: TestClient) -> None:
        """Issue #827: views.py line 106 — Q shape guard → 500."""
        # P is correct (1, 2) but Q is wrong shape (2,) instead of (1,)
        gv = _generated_views(np.array([[1.0, 0.0]]), np.array([0.02, 0.03]))
        with (
            patch(_FETCH_FACTOR, return_value=[_factor("AAPL"), _factor("MSFT")]),
            patch(_GENERATE, return_value=gv),
        ):
            resp = client.post(GEN_URL, json={"tickers": ["AAPL", "MSFT"]})

        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# POST /views/entropy-pooling
# ---------------------------------------------------------------------------


class TestEntropyPooling:
    _BODY = {"tickers": ["AAPL", "MSFT"], "mean_views": ["AAPL == 0.001"]}

    def test_when_solved_then_200_with_moments(self, client: TestClient) -> None:
        result = {
            "mu": [0.001, 0.001],
            "covariance": [[0.0004, 0.0], [0.0, 0.0004]],
            "tickers": ["AAPL", "MSFT"],
        }
        with (
            patch(_FETCH_PRICES, return_value=_price_frame()),
            patch(_RUN_EP, return_value=result),
        ):
            resp = client.post(EP_URL, json=self._BODY)

        assert resp.status_code == 200
        body = resp.json()
        assert_response_matches_schema(body, EntropyPoolingResponse)
        assert_json_safe(body)

    def test_when_prices_empty_then_422(self, client: TestClient) -> None:
        with patch(_FETCH_PRICES, return_value=pd.DataFrame()):
            resp = client.post(EP_URL, json=self._BODY)

        assert resp.status_code == 422

    def test_when_solver_fails_then_500(self, client: TestClient) -> None:
        with (
            patch(_FETCH_PRICES, return_value=_price_frame()),
            patch(_RUN_EP, side_effect=RuntimeError("did not converge")),
        ):
            resp = client.post(EP_URL, json=self._BODY)

        assert resp.status_code == 500
