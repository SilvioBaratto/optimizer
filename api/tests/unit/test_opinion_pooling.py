"""Unit tests for Multi-LLM Opinion Pooling service and endpoint."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from baml_client.types import AssetFactorData, AssetView, ExpertPersona, ViewOutput
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "GOOGL"]


def _make_asset_factor_data(ticker: str = "AAPL") -> AssetFactorData:
    return AssetFactorData(
        ticker=ticker,
        trailing_pe=28.0,
        price_to_book=45.0,
        momentum_12_1m=0.20,
        momentum_1m=0.02,
        rsi_14=60.0,
        return_on_equity=1.4,
        debt_to_equity=1.8,
        profit_margins=0.25,
        revenue_growth_yoy=0.06,
        earnings_growth_yoy=0.13,
        pct_from_52w_high=-0.03,
        pct_from_52w_low=0.28,
        recommendation_mean=1.9,
        target_upside=0.12,
        analyst_count=45,
    )


def _make_view_output(
    ticker: str = "AAPL",
    direction: int = 1,
    magnitude_bps: float = 200.0,
    confidence: float = 0.7,
) -> ViewOutput:
    views = [
        AssetView(
            asset=ticker,
            direction=direction,
            magnitude_bps=magnitude_bps,
            confidence=confidence,
            reasoning="Test.",
        )
    ]
    return ViewOutput(
        views=views,
        idzorek_alphas={ticker: confidence},
        rationale="Test rationale.",
    )


def _make_empty_view_output() -> ViewOutput:
    return ViewOutput(views=[], idzorek_alphas={}, rationale="No views.")


# ---------------------------------------------------------------------------
# Service layer — compute_ic_weights
# ---------------------------------------------------------------------------


class TestComputeICWeights:
    def test_weights_sum_to_one(self) -> None:
        from app.services.opinion_pooling import compute_ic_weights

        histories = [
            pd.Series([0.1, 0.2, 0.15, 0.08]),
            pd.Series([0.05, 0.0, -0.1, 0.03]),
            pd.Series([0.3, 0.25, 0.28, 0.22]),
        ]
        weights = compute_ic_weights(histories)
        assert weights.sum() == pytest.approx(1.0)

    def test_weights_non_negative(self) -> None:
        from app.services.opinion_pooling import compute_ic_weights

        histories = [pd.Series([0.1, 0.2, 0.15]), pd.Series([-0.5, -0.3, -0.4])]
        weights = compute_ic_weights(histories)
        assert np.all(weights >= 0.0)

    def test_high_icir_gets_higher_weight(self) -> None:
        from app.services.opinion_pooling import compute_ic_weights

        good = pd.Series([0.30, 0.28, 0.32, 0.29])  # high ICIR
        bad = pd.Series([0.01, -0.01, 0.02, -0.02])  # near-zero ICIR
        weights = compute_ic_weights([good, bad])
        assert weights[0] > weights[1]

    def test_zero_icir_gets_near_zero_not_hard_zero(self) -> None:
        from app.services.opinion_pooling import compute_ic_weights

        zero_ic = pd.Series([0.1, -0.1, 0.1, -0.1])  # mean IC ≈ 0
        good = pd.Series([0.3, 0.28, 0.32, 0.29])
        weights = compute_ic_weights([zero_ic, good])
        assert weights[0] > 0.0

    def test_empty_series_gets_eps_weight(self) -> None:
        from app.services.opinion_pooling import compute_ic_weights

        weights = compute_ic_weights(
            [pd.Series([], dtype=float), pd.Series([0.2, 0.3, 0.25])]
        )
        assert weights[0] > 0.0
        assert weights.sum() == pytest.approx(1.0)

    def test_equal_icir_gives_equal_weights(self) -> None:
        from app.services.opinion_pooling import compute_ic_weights

        s = pd.Series([0.2, 0.3, 0.25])
        weights = compute_ic_weights([s.copy(), s.copy()])
        assert weights[0] == pytest.approx(weights[1])

    def test_single_expert_gets_weight_one(self) -> None:
        from app.services.opinion_pooling import compute_ic_weights

        weights = compute_ic_weights([pd.Series([0.1, 0.2, 0.15])])
        assert weights[0] == pytest.approx(1.0)

    def test_all_negative_icir_still_sums_to_one(self) -> None:
        from app.services.opinion_pooling import compute_ic_weights

        histories = [pd.Series([-0.5, -0.4, -0.6]) for _ in range(3)]
        weights = compute_ic_weights(histories)
        assert weights.sum() == pytest.approx(1.0)
        assert np.all(weights > 0.0)


# ---------------------------------------------------------------------------
# Service layer — run_llm_experts
# ---------------------------------------------------------------------------


class TestRunLLMExperts:
    def _all_mock_outputs(self) -> dict[str, ViewOutput]:
        return {
            "GenerateValueView": _make_view_output("AAPL"),
            "GenerateMomentumView": _make_view_output("MSFT"),
            "GenerateMacroView": _make_view_output("GOOGL"),
        }

    def test_returns_one_result_per_persona(self) -> None:
        from app.services.opinion_pooling import ALL_PERSONAS, run_llm_experts

        assets = [_make_asset_factor_data(t) for t in TICKERS]

        with (
            patch(
                "app.services.opinion_pooling.b.GenerateValueView",
                return_value=_make_view_output("AAPL"),
            ),
            patch(
                "app.services.opinion_pooling.b.GenerateMomentumView",
                return_value=_make_view_output("MSFT"),
            ),
            patch(
                "app.services.opinion_pooling.b.GenerateMacroView",
                return_value=_make_view_output("GOOGL"),
            ),
        ):
            results = run_llm_experts(assets, TICKERS)

        assert len(results) == len(ALL_PERSONAS)

    def test_each_result_has_correct_persona(self) -> None:
        from app.services.opinion_pooling import run_llm_experts

        assets = [_make_asset_factor_data(t) for t in TICKERS]

        with (
            patch(
                "app.services.opinion_pooling.b.GenerateValueView",
                return_value=_make_view_output("AAPL"),
            ),
            patch(
                "app.services.opinion_pooling.b.GenerateMomentumView",
                return_value=_make_view_output("MSFT"),
            ),
            patch(
                "app.services.opinion_pooling.b.GenerateMacroView",
                return_value=_make_view_output("GOOGL"),
            ),
        ):
            results = run_llm_experts(assets, TICKERS)

        personas = [r.persona for r in results]
        assert ExpertPersona.VALUE_INVESTOR in personas
        assert ExpertPersona.MOMENTUM_TRADER in personas
        assert ExpertPersona.MACRO_ANALYST in personas

    def test_at_least_two_distinct_personas(self) -> None:
        from app.services.opinion_pooling import run_llm_experts

        assets = [_make_asset_factor_data(t) for t in TICKERS]
        two_personas = [ExpertPersona.VALUE_INVESTOR, ExpertPersona.MOMENTUM_TRADER]

        with (
            patch(
                "app.services.opinion_pooling.b.GenerateValueView",
                return_value=_make_view_output("AAPL"),
            ),
            patch(
                "app.services.opinion_pooling.b.GenerateMomentumView",
                return_value=_make_view_output("MSFT"),
            ),
        ):
            results = run_llm_experts(assets, TICKERS, personas=two_personas)

        assert len(results) == 2
        assert len({r.persona for r in results}) == 2

    def test_hallucinated_ticker_filtered(self) -> None:
        """LLM view on unknown ticker should be filtered out."""
        from app.services.opinion_pooling import run_llm_experts

        assets = [_make_asset_factor_data("AAPL")]
        hallucinated = ViewOutput(
            views=[
                AssetView(
                    asset="FAKE",
                    direction=1,
                    magnitude_bps=100,
                    confidence=0.5,
                    reasoning="x",
                )
            ],
            idzorek_alphas={"FAKE": 0.5},
            rationale="x",
        )

        with (
            patch(
                "app.services.opinion_pooling.b.GenerateValueView",
                return_value=hallucinated,
            ),
            patch(
                "app.services.opinion_pooling.b.GenerateMomentumView",
                return_value=_make_empty_view_output(),
            ),
            patch(
                "app.services.opinion_pooling.b.GenerateMacroView",
                return_value=_make_empty_view_output(),
            ),
        ):
            results = run_llm_experts(assets, ["AAPL"])

        assert all(
            v.asset in {"AAPL"}
            for r in results
            for v in r.view_output.views
            if r.view_strings  # only check results that kept views
        )

    def test_expert_prior_is_black_litterman(self) -> None:
        from app.services.opinion_pooling import run_llm_experts
        from skfolio.prior import BlackLitterman

        assets = [_make_asset_factor_data("AAPL")]

        with (
            patch(
                "app.services.opinion_pooling.b.GenerateValueView",
                return_value=_make_view_output("AAPL"),
            ),
            patch(
                "app.services.opinion_pooling.b.GenerateMomentumView",
                return_value=_make_empty_view_output(),
            ),
            patch(
                "app.services.opinion_pooling.b.GenerateMacroView",
                return_value=_make_empty_view_output(),
            ),
        ):
            results = run_llm_experts(assets, TICKERS)

        for r in results:
            assert isinstance(r.prior_estimator, BlackLitterman)


# ---------------------------------------------------------------------------
# Service layer — build_llm_opinion_pool
# ---------------------------------------------------------------------------


class TestBuildLLMOpinionPool:
    def _mock_all(self):
        return (
            patch(
                "app.services.opinion_pooling.b.GenerateValueView",
                return_value=_make_view_output("AAPL"),
            ),
            patch(
                "app.services.opinion_pooling.b.GenerateMomentumView",
                return_value=_make_view_output("MSFT"),
            ),
            patch(
                "app.services.opinion_pooling.b.GenerateMacroView",
                return_value=_make_view_output("GOOGL"),
            ),
        )

    def test_ic_weights_sum_to_one(self) -> None:
        from app.services.opinion_pooling import build_llm_opinion_pool

        assets = [_make_asset_factor_data(t) for t in TICKERS]

        with self._mock_all()[0], self._mock_all()[1], self._mock_all()[2]:
            result = build_llm_opinion_pool(assets, TICKERS)

        assert result.ic_weights.sum() == pytest.approx(1.0)

    def test_equal_weights_without_ic_history(self) -> None:
        from app.services.opinion_pooling import build_llm_opinion_pool

        assets = [_make_asset_factor_data(t) for t in TICKERS]

        with self._mock_all()[0], self._mock_all()[1], self._mock_all()[2]:
            result = build_llm_opinion_pool(assets, TICKERS, ic_histories=None)

        # All weights equal
        expected = 1.0 / len(result.expert_results)
        assert np.allclose(result.ic_weights, expected)

    def test_n_experts_matches_personas(self) -> None:
        from app.services.opinion_pooling import build_llm_opinion_pool

        assets = [_make_asset_factor_data(t) for t in TICKERS]

        with self._mock_all()[0], self._mock_all()[1], self._mock_all()[2]:
            result = build_llm_opinion_pool(assets, TICKERS)

        assert len(result.expert_results) == 3

    def test_ic_histories_length_mismatch_raises(self) -> None:
        from app.services.opinion_pooling import build_llm_opinion_pool

        assets = [_make_asset_factor_data(t) for t in TICKERS]
        # 3 experts but only 2 IC histories
        ic = [pd.Series([0.1, 0.2, 0.15]), pd.Series([0.05, 0.1, 0.08])]

        with self._mock_all()[0], self._mock_all()[1], self._mock_all()[2]:
            with pytest.raises(ValueError, match="ic_histories length"):
                build_llm_opinion_pool(assets, TICKERS, ic_histories=ic)

    def test_opinion_pool_is_skfolio_estimator(self) -> None:
        from app.services.opinion_pooling import build_llm_opinion_pool
        from skfolio.prior import OpinionPooling

        assets = [_make_asset_factor_data(t) for t in TICKERS]

        with self._mock_all()[0], self._mock_all()[1], self._mock_all()[2]:
            result = build_llm_opinion_pool(assets, TICKERS)

        assert isinstance(result.opinion_pool, OpinionPooling)

    def test_ic_calibrated_weights_reflect_icir(self) -> None:
        from app.services.opinion_pooling import build_llm_opinion_pool

        assets = [_make_asset_factor_data(t) for t in TICKERS]
        ic_histories = [
            pd.Series([0.30, 0.28, 0.32, 0.29]),  # high ICIR → expert 0 (value)
            pd.Series([0.01, -0.01, 0.01]),  # near-zero ICIR → expert 1 (momentum)
            pd.Series([0.15, 0.12, 0.18, 0.14]),  # medium ICIR → expert 2 (macro)
        ]

        with self._mock_all()[0], self._mock_all()[1], self._mock_all()[2]:
            result = build_llm_opinion_pool(assets, TICKERS, ic_histories=ic_histories)

        # Value expert (high ICIR) should dominate
        assert result.ic_weights[0] > result.ic_weights[1]
        assert result.ic_weights[0] > result.ic_weights[2]


# ---------------------------------------------------------------------------
# API endpoint — POST /api/v1/views/opinion-pool
# ---------------------------------------------------------------------------

URL = "/api/v1/views/opinion-pool"
_FETCH = "app.api.v1.opinion_pooling.fetch_factor_data"
_BUILD = "app.api.v1.opinion_pooling.build_llm_opinion_pool"


def _make_pool_result(n_experts: int = 3) -> OpinionPoolResult:
    from app.services.opinion_pooling import ExpertViewResult, OpinionPoolResult
    from skfolio.prior import BlackLitterman, OpinionPooling

    persona_list = [
        ExpertPersona.VALUE_INVESTOR,
        ExpertPersona.MOMENTUM_TRADER,
        ExpertPersona.MACRO_ANALYST,
    ][:n_experts]

    expert_results = [
        ExpertViewResult(
            persona=p,
            name=p.value.lower(),
            view_output=_make_view_output(TICKERS[i]),
            view_strings=[f"{TICKERS[i]} == 0.02"],
            idzorek_alphas={TICKERS[i]: 0.7},
            prior_estimator=BlackLitterman(views=[f"{TICKERS[i]} == 0.02"], tau=0.05),
        )
        for i, p in enumerate(persona_list)
    ]

    ic_weights = np.full(n_experts, 1.0 / n_experts)
    pool = OpinionPooling(
        estimators=[(er.name, er.prior_estimator) for er in expert_results],
        opinion_probabilities=ic_weights.tolist(),
    )

    return OpinionPoolResult(
        expert_results=expert_results,
        ic_weights=ic_weights,
        opinion_pool=pool,
        tickers=TICKERS,
    )


PAYLOAD = {"tickers": ["AAPL", "MSFT", "GOOGL"]}


class TestOpinionPoolEndpoint:
    def test_successful_response(self, client: TestClient) -> None:
        mock_pool = _make_pool_result()

        with (
            patch(_FETCH, return_value=[_make_asset_factor_data(t) for t in TICKERS]),
            patch(_BUILD, return_value=mock_pool),
        ):
            resp = client.post(URL, json=PAYLOAD)

        assert resp.status_code == 200
        data = resp.json()
        assert data["n_experts"] == 3
        assert len(data["experts"]) == 3
        assert len(data["ic_weights"]) == 3

    def test_ic_weights_sum_to_one(self, client: TestClient) -> None:
        mock_pool = _make_pool_result()

        with (
            patch(_FETCH, return_value=[_make_asset_factor_data(t) for t in TICKERS]),
            patch(_BUILD, return_value=mock_pool),
        ):
            resp = client.post(URL, json=PAYLOAD)

        weights = resp.json()["ic_weights"]
        assert sum(weights) == pytest.approx(1.0)

    def test_ic_weights_non_negative(self, client: TestClient) -> None:
        mock_pool = _make_pool_result()

        with (
            patch(_FETCH, return_value=[_make_asset_factor_data(t) for t in TICKERS]),
            patch(_BUILD, return_value=mock_pool),
        ):
            resp = client.post(URL, json=PAYLOAD)

        assert all(w >= 0.0 for w in resp.json()["ic_weights"])

    def test_at_least_two_distinct_personas_in_response(
        self, client: TestClient
    ) -> None:
        mock_pool = _make_pool_result(n_experts=2)

        with (
            patch(_FETCH, return_value=[_make_asset_factor_data(t) for t in TICKERS]),
            patch(_BUILD, return_value=mock_pool),
        ):
            resp = client.post(
                URL, json={**PAYLOAD, "personas": ["VALUE_INVESTOR", "MOMENTUM_TRADER"]}
            )

        assert resp.status_code == 200
        personas = {e["persona"] for e in resp.json()["experts"]}
        assert len(personas) >= 2

    def test_fewer_than_2_tickers_rejected(self, client: TestClient) -> None:
        resp = client.post(URL, json={"tickers": ["AAPL"]})
        assert resp.status_code == 422

    def test_unknown_persona_rejected(self, client: TestClient) -> None:
        payload = {**PAYLOAD, "personas": ["UNKNOWN_EXPERT"]}
        resp = client.post(URL, json=payload)
        assert resp.status_code == 422

    def test_no_db_data_returns_422(self, client: TestClient) -> None:
        with patch(_FETCH, return_value=[]):
            resp = client.post(URL, json=PAYLOAD)
        assert resp.status_code == 422

    def test_llm_error_returns_502(self, client: TestClient) -> None:
        with (
            patch(_FETCH, return_value=[_make_asset_factor_data("AAPL")]),
            patch(_BUILD, side_effect=RuntimeError("LLM timeout")),
        ):
            resp = client.post(URL, json=PAYLOAD)
        assert resp.status_code == 502

    def test_tickers_missing_reported(self, client: TestClient) -> None:
        mock_pool = _make_pool_result()

        with (
            patch(_FETCH, return_value=[_make_asset_factor_data("AAPL")]),  # only AAPL
            patch(_BUILD, return_value=mock_pool),
        ):
            resp = client.post(URL, json=PAYLOAD)

        data = resp.json()
        assert "MSFT" in data["tickers_missing_data"]
        assert "GOOGL" in data["tickers_missing_data"]
        assert "AAPL" in data["tickers_with_data"]

    def test_pooling_type_linear_default(self, client: TestClient) -> None:
        mock_pool = _make_pool_result()

        with (
            patch(_FETCH, return_value=[_make_asset_factor_data(t) for t in TICKERS]),
            patch(_BUILD, return_value=mock_pool),
        ):
            resp = client.post(URL, json=PAYLOAD)

        assert resp.json()["pooling_type"] == "linear"

    def test_pooling_type_geometric(self, client: TestClient) -> None:
        mock_pool = _make_pool_result()

        with (
            patch(_FETCH, return_value=[_make_asset_factor_data(t) for t in TICKERS]),
            patch(_BUILD, return_value=mock_pool),
        ):
            resp = client.post(URL, json={**PAYLOAD, "is_linear_pooling": False})

        assert resp.json()["pooling_type"] == "geometric"

    def test_ic_histories_passed_to_service(self, client: TestClient) -> None:
        mock_pool = _make_pool_result()
        captured: dict = {}

        def _capture(assets, tickers, ic_histories=None, **kwargs):
            captured["ic_histories"] = ic_histories
            return mock_pool

        payload = {
            **PAYLOAD,
            "ic_histories": [
                {"persona": "VALUE_INVESTOR", "ic_values": [0.1, 0.2, 0.15]},
                {"persona": "MOMENTUM_TRADER", "ic_values": [0.05, 0.1, 0.08]},
                {"persona": "MACRO_ANALYST", "ic_values": [0.3, 0.28, 0.32]},
            ],
        }

        with (
            patch(_FETCH, return_value=[_make_asset_factor_data(t) for t in TICKERS]),
            patch(_BUILD, side_effect=_capture),
        ):
            resp = client.post(URL, json=payload)

        assert resp.status_code == 200
        assert captured["ic_histories"] is not None
        assert len(captured["ic_histories"]) == 3
