"""Unit tests for LLM-augmented moment estimation service and endpoints."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from app.main import app
from baml_client.types import (
    BusinessCyclePhase,
    CovEstimatorChoice,
    CovRegimeSelection,
    DeltaCalibration,
    FactorWeightAdaptation,
)
from fastapi.testclient import TestClient

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers — mock BAML responses
# ---------------------------------------------------------------------------


def _delta_response(delta: float = 3.5) -> DeltaCalibration:
    return DeltaCalibration(delta=delta, rationale="Test rationale.")


def _factor_response(
    phase: BusinessCyclePhase = BusinessCyclePhase.MID_EXPANSION,
    weights: dict[str, float] | None = None,
) -> FactorWeightAdaptation:
    if weights is None:
        weights = {"momentum": 1.2, "value": 0.8, "quality": 1.0, "low_volatility": 1.0}
    return FactorWeightAdaptation(
        phase=phase, weights=weights, rationale="Test rationale."
    )


def _cov_response(
    estimator: CovEstimatorChoice = CovEstimatorChoice.LEDOIT_WOLF,
    confidence: float = 0.85,
) -> CovRegimeSelection:
    return CovRegimeSelection(
        estimator=estimator, confidence=confidence, rationale="Test."
    )


# ---------------------------------------------------------------------------
# Service layer — calibrate_delta
# ---------------------------------------------------------------------------


class TestCalibrateDelta:
    def test_returns_delta_calibration(self) -> None:
        from app.services.llm_moments import calibrate_delta

        with patch(
            "app.services.llm_moments.b.CalibrateDelta",
            return_value=_delta_response(4.0),
        ):
            result = calibrate_delta("Global growth slowing.")

        assert result.delta == pytest.approx(4.0)
        assert result.rationale

    def test_clamps_delta_above_max(self) -> None:
        from app.services.llm_moments import DELTA_MAX, calibrate_delta

        with patch(
            "app.services.llm_moments.b.CalibrateDelta",
            return_value=_delta_response(99.0),
        ):
            result = calibrate_delta("Extreme panic.")

        assert result.delta == pytest.approx(DELTA_MAX)

    def test_clamps_delta_below_min(self) -> None:
        from app.services.llm_moments import DELTA_MIN, calibrate_delta

        with patch(
            "app.services.llm_moments.b.CalibrateDelta",
            return_value=_delta_response(-5.0),
        ):
            result = calibrate_delta("Manic euphoria.")

        assert result.delta == pytest.approx(DELTA_MIN)

    def test_delta_boundary_values_not_clamped(self) -> None:
        from app.services.llm_moments import DELTA_MAX, DELTA_MIN, calibrate_delta

        for v in (DELTA_MIN, DELTA_MAX, 5.0):
            with patch(
                "app.services.llm_moments.b.CalibrateDelta",
                return_value=_delta_response(v),
            ):
                result = calibrate_delta("Some macro text.")
            assert result.delta == pytest.approx(v)


# ---------------------------------------------------------------------------
# Service layer — adapt_factor_weights
# ---------------------------------------------------------------------------


class TestAdaptFactorWeights:
    GROUPS = ["momentum", "value", "quality", "low_volatility"]

    def test_returns_correct_phase(self) -> None:
        from app.services.llm_moments import adapt_factor_weights

        with patch(
            "app.services.llm_moments.b.AdaptFactorWeights",
            return_value=_factor_response(BusinessCyclePhase.RECESSION),
        ):
            result = adapt_factor_weights("Recession signals.", self.GROUPS)

        assert result.phase == BusinessCyclePhase.RECESSION

    def test_weights_sum_to_n(self) -> None:
        from app.services.llm_moments import adapt_factor_weights

        with patch(
            "app.services.llm_moments.b.AdaptFactorWeights",
            return_value=_factor_response(),
        ):
            result = adapt_factor_weights("Macro text.", self.GROUPS)

        assert sum(result.weights.values()) == pytest.approx(len(self.GROUPS), rel=1e-6)

    def test_all_weights_positive(self) -> None:
        from app.services.llm_moments import adapt_factor_weights

        raw_weights = {
            "momentum": -1.0,
            "value": 0.0,
            "quality": 3.0,
            "low_volatility": 2.0,
        }
        with patch(
            "app.services.llm_moments.b.AdaptFactorWeights",
            return_value=_factor_response(weights=raw_weights),
        ):
            result = adapt_factor_weights("Macro text.", self.GROUPS)

        assert all(w > 0 for w in result.weights.values())

    def test_missing_group_defaults_to_one(self) -> None:
        from app.services.llm_moments import adapt_factor_weights

        # LLM returns only 2 of 4 groups
        partial_weights = {"momentum": 1.5, "value": 0.5}
        with patch(
            "app.services.llm_moments.b.AdaptFactorWeights",
            return_value=_factor_response(weights=partial_weights),
        ):
            result = adapt_factor_weights("Macro text.", self.GROUPS)

        assert set(result.weights.keys()) == set(self.GROUPS)
        assert sum(result.weights.values()) == pytest.approx(len(self.GROUPS), rel=1e-6)

    def test_weights_cover_all_factor_groups(self) -> None:
        from app.services.llm_moments import adapt_factor_weights

        with patch(
            "app.services.llm_moments.b.AdaptFactorWeights",
            return_value=_factor_response(),
        ):
            result = adapt_factor_weights("Macro text.", self.GROUPS)

        assert set(result.weights.keys()) == set(self.GROUPS)


# ---------------------------------------------------------------------------
# Service layer — select_cov_regime
# ---------------------------------------------------------------------------


class TestSelectCovRegime:
    HEADLINES = ["Stocks rally on Fed pivot hopes."]

    def test_returns_valid_estimator(self) -> None:
        from app.services.llm_moments import select_cov_regime

        with patch(
            "app.services.llm_moments.b.SelectCovRegime",
            return_value=_cov_response(CovEstimatorChoice.EW),
        ):
            result = select_cov_regime(self.HEADLINES, -0.5, 0.25)

        assert result.estimator == CovEstimatorChoice.EW

    def test_confidence_clamped_to_unit_interval(self) -> None:
        from app.services.llm_moments import select_cov_regime

        with patch(
            "app.services.llm_moments.b.SelectCovRegime",
            return_value=_cov_response(confidence=1.5),
        ):
            result = select_cov_regime(self.HEADLINES, 0.1, 0.12)

        assert result.confidence <= 1.0

    def test_confidence_negative_clamped(self) -> None:
        from app.services.llm_moments import select_cov_regime

        with patch(
            "app.services.llm_moments.b.SelectCovRegime",
            return_value=_cov_response(confidence=-0.2),
        ):
            result = select_cov_regime(self.HEADLINES, 0.1, 0.12)

        assert result.confidence >= 0.0

    def test_cov_estimator_type_str_mapping(self) -> None:
        from app.services.llm_moments import cov_estimator_type_str, select_cov_regime

        for choice, expected in [
            (CovEstimatorChoice.LEDOIT_WOLF, "ledoit_wolf"),
            (CovEstimatorChoice.EW, "ew"),
            (CovEstimatorChoice.GERBER, "gerber"),
            (CovEstimatorChoice.DENOISE, "denoise"),
        ]:
            with patch(
                "app.services.llm_moments.b.SelectCovRegime",
                return_value=_cov_response(choice),
            ):
                result = select_cov_regime(self.HEADLINES, 0.0, 0.15)
            assert cov_estimator_type_str(result) == expected


# ---------------------------------------------------------------------------
# API endpoint — /api/v1/llm-moments/calibrate-delta
# ---------------------------------------------------------------------------


class TestCalibrateDeltaEndpoint:
    URL = "/api/v1/llm-moments/calibrate-delta"

    def test_successful_response(self) -> None:
        with patch(
            "app.services.llm_moments.b.CalibrateDelta",
            return_value=_delta_response(5.0),
        ):
            resp = client.post(
                self.URL, json={"macro_text": "GDP fell for two consecutive quarters."}
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["delta"] == pytest.approx(5.0)
        assert "rationale" in data

    def test_rejects_short_macro_text(self) -> None:
        resp = client.post(self.URL, json={"macro_text": "short"})
        assert resp.status_code == 422

    def test_llm_error_returns_502(self) -> None:
        with patch(
            "app.services.llm_moments.b.CalibrateDelta",
            side_effect=RuntimeError("LLM timeout"),
        ):
            resp = client.post(
                self.URL, json={"macro_text": "Inflation rose sharply this quarter."}
            )

        assert resp.status_code == 502

    def test_delta_clamped_in_response(self) -> None:
        with patch(
            "app.services.llm_moments.b.CalibrateDelta",
            return_value=_delta_response(99.0),
        ):
            resp = client.post(
                self.URL, json={"macro_text": "Severe recession with credit crisis."}
            )

        assert resp.status_code == 200
        assert resp.json()["delta"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# API endpoint — /api/v1/llm-moments/adapt-factor-weights
# ---------------------------------------------------------------------------


class TestAdaptFactorWeightsEndpoint:
    URL = "/api/v1/llm-moments/adapt-factor-weights"
    PAYLOAD = {
        "macro_indicators": "PMI 55, unemployment 4.1%, yield curve +80bps.",
        "factor_groups": ["momentum", "value", "quality"],
    }

    def test_successful_response(self) -> None:
        weights = {"momentum": 1.3, "value": 0.8, "quality": 0.9}
        with patch(
            "app.services.llm_moments.b.AdaptFactorWeights",
            return_value=_factor_response(
                BusinessCyclePhase.EARLY_EXPANSION, weights=weights
            ),
        ):
            resp = client.post(self.URL, json=self.PAYLOAD)

        assert resp.status_code == 200
        data = resp.json()
        assert data["phase"] == "EARLY_EXPANSION"
        assert set(data["weights"].keys()) == {"momentum", "value", "quality"}
        assert sum(data["weights"].values()) == pytest.approx(3.0, rel=1e-4)

    def test_rejects_empty_factor_groups(self) -> None:
        payload = {**self.PAYLOAD, "factor_groups": []}
        resp = client.post(self.URL, json=payload)
        assert resp.status_code == 422

    def test_rejects_blank_factor_group_name(self) -> None:
        payload = {**self.PAYLOAD, "factor_groups": ["momentum", "  "]}
        resp = client.post(self.URL, json=payload)
        assert resp.status_code == 422

    def test_llm_error_returns_502(self) -> None:
        with patch(
            "app.services.llm_moments.b.AdaptFactorWeights",
            side_effect=RuntimeError("timeout"),
        ):
            resp = client.post(self.URL, json=self.PAYLOAD)

        assert resp.status_code == 502


# ---------------------------------------------------------------------------
# API endpoint — /api/v1/llm-moments/select-cov-regime
# ---------------------------------------------------------------------------


class TestSelectCovRegimeEndpoint:
    URL = "/api/v1/llm-moments/select-cov-regime"
    PAYLOAD = {
        "news_headlines": ["Markets tumble as inflation surges."],
        "avg_sentiment_score": -0.6,
        "realized_vol_30d": 0.22,
    }

    def test_successful_response(self) -> None:
        with patch(
            "app.services.llm_moments.b.SelectCovRegime",
            return_value=_cov_response(CovEstimatorChoice.EW, confidence=0.9),
        ):
            resp = client.post(self.URL, json=self.PAYLOAD)

        assert resp.status_code == 200
        data = resp.json()
        assert data["estimator"] == "EW"
        assert data["estimator_type"] == "ew"
        assert data["confidence"] == pytest.approx(0.9)
        assert "rationale" in data

    def test_sentiment_out_of_range_rejected(self) -> None:
        payload = {**self.PAYLOAD, "avg_sentiment_score": 1.5}
        resp = client.post(self.URL, json=payload)
        assert resp.status_code == 422

    def test_negative_vol_rejected(self) -> None:
        payload = {**self.PAYLOAD, "realized_vol_30d": -0.1}
        resp = client.post(self.URL, json=payload)
        assert resp.status_code == 422

    def test_empty_headlines_rejected(self) -> None:
        payload = {**self.PAYLOAD, "news_headlines": []}
        resp = client.post(self.URL, json=payload)
        assert resp.status_code == 422

    def test_llm_error_returns_502(self) -> None:
        with patch(
            "app.services.llm_moments.b.SelectCovRegime",
            side_effect=RuntimeError("LLM failure"),
        ):
            resp = client.post(self.URL, json=self.PAYLOAD)

        assert resp.status_code == 502

    def test_estimator_type_is_valid_optimizer_value(self) -> None:
        """estimator_type must be a valid CovEstimatorType string value."""
        from app.services.llm_moments import _COV_CHOICE_TO_ESTIMATOR_TYPE

        valid_values = set(_COV_CHOICE_TO_ESTIMATOR_TYPE.values())

        with patch(
            "app.services.llm_moments.b.SelectCovRegime",
            return_value=_cov_response(CovEstimatorChoice.GERBER),
        ):
            resp = client.post(self.URL, json=self.PAYLOAD)

        assert resp.json()["estimator_type"] in valid_values
