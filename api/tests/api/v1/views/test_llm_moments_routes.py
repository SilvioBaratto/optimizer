"""Route tests for the LLM-moments endpoints (issue #817).

Covers ``app.api.v1.views.llm_moments``: ``POST /llm-moments/calibrate-delta``,
``/adapt-factor-weights``, ``/select-cov-regime``.  Each BAML-backed service
function is mocked at the import site — never a real BAML client.  Uses the
conftest ``client`` fixture (NOT a module-level ``TestClient``).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.schemas.views.llm_moments import (
    AdaptFactorWeightsResponse,
    CalibrateDeltaResponse,
    SelectCovRegimeResponse,
)
from tests._fixtures.assertions import assert_json_safe, assert_response_matches_schema

BASE_URL = "/api/v1/llm-moments"
_CALIBRATE = "app.api.v1.views.llm_moments.calibrate_delta"
_ADAPT = "app.api.v1.views.llm_moments.adapt_factor_weights"
_SELECT = "app.api.v1.views.llm_moments.select_cov_regime"
_COV_TYPE = "app.api.v1.views.llm_moments.cov_estimator_type_str"

_MACRO_TEXT = "Fed held rates; GDP growth slowing toward trend over the quarter."


# ---------------------------------------------------------------------------
# POST /llm-moments/calibrate-delta
# ---------------------------------------------------------------------------


class TestCalibrateDelta:
    URL = f"{BASE_URL}/calibrate-delta"

    def test_when_calibrated_then_200(self, client: TestClient) -> None:
        result = SimpleNamespace(delta=3.5, rationale="moderate risk aversion")
        with patch(_CALIBRATE, return_value=result):
            resp = client.post(self.URL, json={"macro_text": _MACRO_TEXT})

        assert resp.status_code == 200
        body = resp.json()
        assert_response_matches_schema(body, CalibrateDeltaResponse)
        assert body["delta"] == 3.5
        assert_json_safe(body)

    def test_when_llm_raises_then_502(self, client: TestClient) -> None:
        with patch(_CALIBRATE, side_effect=RuntimeError("baml down")):
            resp = client.post(self.URL, json={"macro_text": _MACRO_TEXT})

        assert resp.status_code == 502


# ---------------------------------------------------------------------------
# POST /llm-moments/adapt-factor-weights
# ---------------------------------------------------------------------------


class TestAdaptFactorWeights:
    URL = f"{BASE_URL}/adapt-factor-weights"
    _BODY = {"macro_indicators": _MACRO_TEXT, "factor_groups": ["value", "momentum"]}

    def test_when_adapted_then_200(self, client: TestClient) -> None:
        result = SimpleNamespace(
            phase=SimpleNamespace(value="MID_EXPANSION"),
            weights={"value": 1.2, "momentum": 0.8},
            rationale="mid-cycle tilt",
        )
        with patch(_ADAPT, return_value=result):
            resp = client.post(self.URL, json=self._BODY)

        assert resp.status_code == 200
        body = resp.json()
        assert_response_matches_schema(body, AdaptFactorWeightsResponse)
        assert body["phase"] == "MID_EXPANSION"
        assert_json_safe(body)

    def test_when_llm_raises_then_502(self, client: TestClient) -> None:
        with patch(_ADAPT, side_effect=RuntimeError("baml down")):
            resp = client.post(self.URL, json=self._BODY)

        assert resp.status_code == 502


# ---------------------------------------------------------------------------
# POST /llm-moments/select-cov-regime
# ---------------------------------------------------------------------------


class TestSelectCovRegime:
    URL = f"{BASE_URL}/select-cov-regime"
    _BODY = {
        "news_headlines": ["Markets calm into close"],
        "avg_sentiment_score": 0.2,
        "realized_vol_30d": 0.15,
    }

    def test_when_selected_then_200_with_estimator_type(
        self, client: TestClient
    ) -> None:
        result = SimpleNamespace(
            estimator=SimpleNamespace(value="LEDOIT_WOLF"),
            confidence=0.8,
            rationale="low vol, calm sentiment",
        )
        with (
            patch(_SELECT, return_value=result),
            patch(_COV_TYPE, return_value="ledoit_wolf"),
        ):
            resp = client.post(self.URL, json=self._BODY)

        assert resp.status_code == 200
        body = resp.json()
        assert_response_matches_schema(body, SelectCovRegimeResponse)
        assert body["estimatorType"] == "ledoit_wolf"
        assert_json_safe(body)

    def test_when_llm_raises_then_502(self, client: TestClient) -> None:
        with patch(_SELECT, side_effect=RuntimeError("baml down")):
            resp = client.post(self.URL, json=self._BODY)

        assert resp.status_code == 502
