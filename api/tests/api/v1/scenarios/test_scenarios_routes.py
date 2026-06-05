"""Route tests for the scenario endpoints (issue #818).

Covers ``stress_scenarios.py`` (``POST /risk/stress-scenarios``) and
``synthetic.py`` (``POST /synthetic/scenario`` + ``GET /synthetic/scenario/
{download_id}``).  All generation work is mocked at the service boundary
(``generate_stress_scenarios`` / ``scenario_to_synthetic_data_args`` /
``generate_scenarios`` / ``store_result`` / ``load_result``).  Uses the
conftest ``client`` fixture (DI-overridden DB).
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from app.schemas.scenarios.stress_scenarios import StressScenarioResponse
from app.schemas.scenarios.synthetic import (
    ScenarioInlineResponse,
    ScenarioStoredResponse,
)
from tests._fixtures.assertions import assert_json_safe, assert_response_matches_schema

STRESS_URL = "/api/v1/risk/stress-scenarios"
SYNTH_URL = "/api/v1/synthetic/scenario"

_GEN_STRESS = "app.api.v1.scenarios.stress_scenarios.generate_stress_scenarios"
_TO_ARGS = "app.api.v1.scenarios.stress_scenarios.scenario_to_synthetic_data_args"
_GEN_SCEN = "app.api.v1.scenarios.synthetic.generate_scenarios"
_STORE = "app.api.v1.scenarios.synthetic.store_result"
_LOAD = "app.api.v1.scenarios.synthetic.load_result"


def _scenario() -> SimpleNamespace:
    return SimpleNamespace(
        name="Broad Drawdown",
        description="Equities sell off across the board.",
        shocks={"AAPL": -0.2},
        probability=0.1,
        horizon_days=21,
    )


# ---------------------------------------------------------------------------
# POST /risk/stress-scenarios
# ---------------------------------------------------------------------------


class TestStressScenarios:
    _BODY = {
        "current_portfolio": {"AAPL": 1.0},
        "macro_context": "elevated inflation, hawkish rate outlook",
        "n_scenarios": 1,
    }

    def test_when_generated_then_200_with_scenarios(
        self, client: TestClient
    ) -> None:
        with (
            patch(_GEN_STRESS, return_value=[_scenario()]),
            patch(_TO_ARGS, return_value={"conditioning": {"AAPL": -0.2}}),
        ):
            resp = client.post(STRESS_URL, json=self._BODY)

        assert resp.status_code == 200
        body = resp.json()
        assert_response_matches_schema(body, StressScenarioResponse)
        assert body["nScenarios"] == 1
        assert body["tickers"] == ["AAPL"]
        assert_json_safe(body)

    def test_when_value_error_then_422(self, client: TestClient) -> None:
        with patch(_GEN_STRESS, side_effect=ValueError("bad input")):
            resp = client.post(STRESS_URL, json=self._BODY)

        assert resp.status_code == 422

    def test_when_runtime_error_then_502(self, client: TestClient) -> None:
        with patch(_GEN_STRESS, side_effect=RuntimeError("llm down")):
            resp = client.post(STRESS_URL, json=self._BODY)

        assert resp.status_code == 502

    def test_when_n_scenarios_below_one_then_422(self, client: TestClient) -> None:
        body = {**self._BODY, "n_scenarios": 0}
        resp = client.post(STRESS_URL, json=body)

        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /synthetic/scenario
# ---------------------------------------------------------------------------


class TestSyntheticPost:
    def test_when_small_sample_then_200_inline(self, client: TestClient) -> None:
        gen = (np.array([[0.01, 0.02]]), ["AAPL", "MSFT"])
        with patch(_GEN_SCEN, return_value=gen):
            resp = client.post(
                SYNTH_URL, json={"tickers": ["AAPL", "MSFT"], "n_samples": 1000}
            )

        assert resp.status_code == 200
        body = resp.json()
        assert_response_matches_schema(body, ScenarioInlineResponse)
        assert body["scenarios"] == [[0.01, 0.02]]  # numpy -> list
        assert_json_safe(body)

    def test_when_large_sample_then_201_with_download_id(
        self, client: TestClient
    ) -> None:
        gen = (np.array([[0.01, 0.02]]), ["AAPL", "MSFT"])
        expires = datetime(2024, 1, 2, tzinfo=timezone.utc)
        with (
            patch(_GEN_SCEN, return_value=gen),
            patch(_STORE, return_value=("dl-123", expires)),
        ):
            resp = client.post(
                SYNTH_URL, json={"tickers": ["AAPL", "MSFT"], "n_samples": 20_000}
            )

        assert resp.status_code == 201
        body = resp.json()
        assert_response_matches_schema(body, ScenarioStoredResponse)
        assert body["download_id"] == "dl-123"

    def test_when_conditioning_mismatch_then_400(self, client: TestClient) -> None:
        with patch(
            _GEN_SCEN,
            side_effect=ValueError("Conditioning tickers not present in model"),
        ):
            resp = client.post(
                SYNTH_URL,
                json={"tickers": ["AAPL"], "conditioning": {"AAPL": -0.2}},
            )

        assert resp.status_code == 400

    def test_when_insufficient_data_then_422(self, client: TestClient) -> None:
        with patch(_GEN_SCEN, side_effect=ValueError("insufficient history")):
            resp = client.post(SYNTH_URL, json={"tickers": ["AAPL"]})

        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /synthetic/scenario/{download_id}
# ---------------------------------------------------------------------------


class TestSyntheticGet:
    def test_when_present_then_200(self, client: TestClient) -> None:
        stored = {"scenarios": [[0.01, 0.02]], "columns": ["AAPL", "MSFT"]}
        with patch(_LOAD, return_value=stored):
            resp = client.get(f"{SYNTH_URL}/dl-123")

        assert resp.status_code == 200
        body = resp.json()
        assert_response_matches_schema(body, ScenarioInlineResponse)
        assert_json_safe(body)

    def test_when_absent_then_404(self, client: TestClient) -> None:
        with patch(_LOAD, return_value=None):
            resp = client.get(f"{SYNTH_URL}/ghost")

        assert resp.status_code == 404
