"""Route tests for the opinion-pool endpoint (issue #817).

Covers ``app.api.v1.views.opinion_pooling``: ``POST /views/opinion-pool``.
The factor fetch and the multi-LLM pool are mocked at the service boundary
(``fetch_factor_data`` / ``build_llm_opinion_pool``) — never a real BAML
client.  Uses the conftest ``client`` fixture.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from app.schemas.views.views import OpinionPoolResponse
from tests._fixtures.assertions import assert_json_safe, assert_response_matches_schema

URL = "/api/v1/views/opinion-pool"
_FETCH_FACTOR = "app.api.v1.views.opinion_pooling.fetch_factor_data"
_BUILD_POOL = "app.api.v1.views.opinion_pooling.build_llm_opinion_pool"
_BODY = {"tickers": ["AAPL", "MSFT"]}


def _factor(ticker: str) -> SimpleNamespace:
    return SimpleNamespace(ticker=ticker)


def _pool_result() -> SimpleNamespace:
    expert = SimpleNamespace(
        persona=SimpleNamespace(value="VALUE_INVESTOR"),
        name="Value Investor",
        view_strings=["AAPL == 0.02"],
        idzorek_alphas={"AAPL": 0.5},
    )
    return SimpleNamespace(expert_results=[expert], ic_weights=np.array([1.0]))


class TestOpinionPool:
    def test_when_pooled_then_200(self, client: TestClient) -> None:
        with (
            patch(_FETCH_FACTOR, return_value=[_factor("AAPL"), _factor("MSFT")]),
            patch(_BUILD_POOL, return_value=_pool_result()),
        ):
            resp = client.post(URL, json=_BODY)

        assert resp.status_code == 200
        body = resp.json()
        assert_response_matches_schema(body, OpinionPoolResponse)
        assert body["nExperts"] == 1
        assert_json_safe(body)

    def test_when_no_factor_data_then_422(self, client: TestClient) -> None:
        with patch(_FETCH_FACTOR, return_value=[]):
            resp = client.post(URL, json=_BODY)

        assert resp.status_code == 422

    def test_when_llm_raises_then_502(self, client: TestClient) -> None:
        with (
            patch(_FETCH_FACTOR, return_value=[_factor("AAPL"), _factor("MSFT")]),
            patch(_BUILD_POOL, side_effect=RuntimeError("baml down")),
        ):
            resp = client.post(URL, json=_BODY)

        assert resp.status_code == 502

    def test_when_db_fetch_raises_then_500(self, client: TestClient) -> None:
        """Issue #827: opinion_pooling.py lines 95-97 — DB error → 500."""
        with patch(_FETCH_FACTOR, side_effect=RuntimeError("db connection lost")):
            resp = client.post(URL, json=_BODY)

        assert resp.status_code == 500
