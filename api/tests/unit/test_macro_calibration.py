"""Unit tests for macro regime calibration service and endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from baml_client.types import BusinessCyclePhase, MacroRegimeCalibration

# ---------------------------------------------------------------------------
# Helpers — mock BAML responses
# ---------------------------------------------------------------------------


def _make_calibration(
    phase: BusinessCyclePhase = BusinessCyclePhase.MID_EXPANSION,
    delta: float = 2.75,
    tau: float = 0.025,
    confidence: float = 0.80,
    rationale: str = "Test rationale.",
) -> MacroRegimeCalibration:
    return MacroRegimeCalibration(
        phase=phase,
        delta=delta,
        tau=tau,
        confidence=confidence,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Service layer — clamping helpers
# ---------------------------------------------------------------------------


class TestClampHelpers:
    def test_clamp_delta_above_max(self) -> None:
        from app.services.macro_calibration import DELTA_MAX, _clamp_delta

        assert _clamp_delta(99.0) == pytest.approx(DELTA_MAX)

    def test_clamp_delta_below_min(self) -> None:
        from app.services.macro_calibration import DELTA_MIN, _clamp_delta

        assert _clamp_delta(-1.0) == pytest.approx(DELTA_MIN)

    def test_clamp_delta_valid_passthrough(self) -> None:
        from app.services.macro_calibration import _clamp_delta

        assert _clamp_delta(3.5) == pytest.approx(3.5)

    def test_clamp_tau_above_max(self) -> None:
        from app.services.macro_calibration import TAU_MAX, _clamp_tau

        assert _clamp_tau(1.0) == pytest.approx(TAU_MAX)

    def test_clamp_tau_below_min(self) -> None:
        from app.services.macro_calibration import TAU_MIN, _clamp_tau

        assert _clamp_tau(0.0) == pytest.approx(TAU_MIN)

    def test_clamp_tau_valid_passthrough(self) -> None:
        from app.services.macro_calibration import _clamp_tau

        assert _clamp_tau(0.05) == pytest.approx(0.05)

    def test_clamp_confidence_upper(self) -> None:
        from app.services.macro_calibration import _clamp_confidence

        assert _clamp_confidence(1.5) == pytest.approx(1.0)

    def test_clamp_confidence_lower(self) -> None:
        from app.services.macro_calibration import _clamp_confidence

        assert _clamp_confidence(-0.1) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Service layer — classify_macro_regime
# ---------------------------------------------------------------------------


class TestClassifyMacroRegime:
    SUMMARY = "GDP: +2.5%, PMI: 55, Unemployment: 4.0%, CPI: 2.3%, 10Y-2Y: +80bps"

    def test_returns_calibration_result(self) -> None:
        from app.services.macro_calibration import classify_macro_regime

        mock_session = MagicMock()
        mock_raw = _make_calibration()

        with patch("app.services.macro_calibration.b.ClassifyMacroRegime", return_value=mock_raw):
            result = classify_macro_regime(mock_session, macro_summary_override=self.SUMMARY)

        assert result.phase == BusinessCyclePhase.MID_EXPANSION
        assert result.delta == pytest.approx(2.75)
        assert result.tau == pytest.approx(0.025)

    def test_delta_clamped_to_max(self) -> None:
        from app.services.macro_calibration import DELTA_MAX, classify_macro_regime

        mock_session = MagicMock()
        mock_raw = _make_calibration(delta=999.0)

        with patch("app.services.macro_calibration.b.ClassifyMacroRegime", return_value=mock_raw):
            result = classify_macro_regime(mock_session, macro_summary_override=self.SUMMARY)

        assert result.delta == pytest.approx(DELTA_MAX)

    def test_delta_clamped_to_min(self) -> None:
        from app.services.macro_calibration import DELTA_MIN, classify_macro_regime

        mock_session = MagicMock()
        mock_raw = _make_calibration(delta=0.0)

        with patch("app.services.macro_calibration.b.ClassifyMacroRegime", return_value=mock_raw):
            result = classify_macro_regime(mock_session, macro_summary_override=self.SUMMARY)

        assert result.delta == pytest.approx(DELTA_MIN)

    def test_tau_clamped_to_max(self) -> None:
        from app.services.macro_calibration import TAU_MAX, classify_macro_regime

        mock_session = MagicMock()
        mock_raw = _make_calibration(tau=5.0)

        with patch("app.services.macro_calibration.b.ClassifyMacroRegime", return_value=mock_raw):
            result = classify_macro_regime(mock_session, macro_summary_override=self.SUMMARY)

        assert result.tau == pytest.approx(TAU_MAX)

    def test_tau_clamped_to_min(self) -> None:
        from app.services.macro_calibration import TAU_MIN, classify_macro_regime

        mock_session = MagicMock()
        mock_raw = _make_calibration(tau=0.0)

        with patch("app.services.macro_calibration.b.ClassifyMacroRegime", return_value=mock_raw):
            result = classify_macro_regime(mock_session, macro_summary_override=self.SUMMARY)

        assert result.tau == pytest.approx(TAU_MIN)

    def test_confidence_clamped(self) -> None:
        from app.services.macro_calibration import classify_macro_regime

        mock_session = MagicMock()
        mock_raw = _make_calibration(confidence=1.5)

        with patch("app.services.macro_calibration.b.ClassifyMacroRegime", return_value=mock_raw):
            result = classify_macro_regime(mock_session, macro_summary_override=self.SUMMARY)

        assert result.confidence <= 1.0

    def test_raises_on_empty_db_and_no_override(self) -> None:
        from app.services.macro_calibration import classify_macro_regime

        mock_session = MagicMock()
        mock_repo = MagicMock()
        mock_repo.get_economic_indicators.return_value = []
        mock_repo.get_te_indicators.return_value = []
        mock_repo.get_bond_yields.return_value = []

        with (
            patch(
                "app.services.macro_calibration.MacroRegimeRepository",
                return_value=mock_repo,
            ),
            pytest.raises(ValueError, match="No macro data"),
        ):
            classify_macro_regime(mock_session, country="Unknown")

    def test_macro_summary_stored_in_result(self) -> None:
        from app.services.macro_calibration import classify_macro_regime

        mock_session = MagicMock()
        mock_raw = _make_calibration()
        custom_summary = "Custom macro context."

        with patch("app.services.macro_calibration.b.ClassifyMacroRegime", return_value=mock_raw):
            result = classify_macro_regime(
                mock_session, macro_summary_override=custom_summary
            )

        assert result.macro_summary == custom_summary

    @pytest.mark.parametrize("phase,exp_delta_range,exp_tau_range", [
        (BusinessCyclePhase.EARLY_EXPANSION, (2.0, 2.5), (0.04, 0.06)),
        (BusinessCyclePhase.MID_EXPANSION, (2.5, 3.0), (0.02, 0.03)),
        (BusinessCyclePhase.LATE_EXPANSION, (3.0, 4.0), (0.005, 0.015)),
        (BusinessCyclePhase.RECESSION, (4.0, 6.0), (0.04, 0.06)),
    ])
    def test_phase_produces_expected_parameter_ranges(
        self,
        phase: BusinessCyclePhase,
        exp_delta_range: tuple[float, float],
        exp_tau_range: tuple[float, float],
    ) -> None:
        from app.services.macro_calibration import classify_macro_regime, _PHASE_DEFAULTS

        # Use phase defaults as the mock LLM output
        default_delta, default_tau = _PHASE_DEFAULTS[phase]
        mock_session = MagicMock()
        mock_raw = _make_calibration(phase=phase, delta=default_delta, tau=default_tau)

        with patch("app.services.macro_calibration.b.ClassifyMacroRegime", return_value=mock_raw):
            result = classify_macro_regime(mock_session, macro_summary_override=self.SUMMARY)

        assert exp_delta_range[0] <= result.delta <= exp_delta_range[1], (
            f"Phase {phase}: delta={result.delta} not in {exp_delta_range}"
        )
        assert exp_tau_range[0] <= result.tau <= exp_tau_range[1], (
            f"Phase {phase}: tau={result.tau} not in {exp_tau_range}"
        )


# ---------------------------------------------------------------------------
# Service layer — build_bl_config_from_calibration
# ---------------------------------------------------------------------------


class TestBuildBlConfig:
    def _make_result(self, delta: float = 3.0, tau: float = 0.025) -> "CalibrationResult":
        from app.services.macro_calibration import CalibrationResult

        return CalibrationResult(
            phase=BusinessCyclePhase.LATE_EXPANSION,
            delta=delta,
            tau=tau,
            confidence=0.75,
            rationale="Test.",
            macro_summary="Test summary.",
        )

    def test_tau_in_config(self) -> None:
        from app.services.macro_calibration import build_bl_config_from_calibration

        cfg = build_bl_config_from_calibration(self._make_result(tau=0.01))
        assert cfg["tau"] == pytest.approx(0.01)

    def test_risk_aversion_in_prior_config(self) -> None:
        from app.services.macro_calibration import build_bl_config_from_calibration

        cfg = build_bl_config_from_calibration(self._make_result(delta=3.5))
        assert cfg["prior_config"]["risk_aversion"] == pytest.approx(3.5)

    def test_mu_estimator_is_equilibrium(self) -> None:
        from app.services.macro_calibration import build_bl_config_from_calibration

        cfg = build_bl_config_from_calibration(self._make_result())
        assert cfg["prior_config"]["mu_estimator"] == "equilibrium"

    def test_compatible_with_black_litterman_config(self) -> None:
        """tau and risk_aversion wire correctly into optimizer config classes."""
        from optimizer.views._config import BlackLittermanConfig
        from optimizer.moments._config import MomentEstimationConfig, MuEstimatorType
        from app.services.macro_calibration import build_bl_config_from_calibration

        result = self._make_result(delta=3.5, tau=0.01)
        cfg = build_bl_config_from_calibration(result, views=("AAPL == 0.02",))

        prior_cfg = MomentEstimationConfig(
            mu_estimator=MuEstimatorType.EQUILIBRIUM,
            risk_aversion=cfg["prior_config"]["risk_aversion"],
        )
        bl_config = BlackLittermanConfig(
            views=tuple(cfg["views"]),
            tau=cfg["tau"],
            prior_config=prior_cfg,
        )
        assert bl_config.tau == pytest.approx(0.01)
        assert bl_config.prior_config.risk_aversion == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# Service layer — phase enum has exactly 4 values
# ---------------------------------------------------------------------------


class TestBusinessCyclePhaseEnum:
    def test_exactly_four_phases(self) -> None:
        assert len(BusinessCyclePhase) == 4

    def test_all_expected_values_present(self) -> None:
        values = {p.value for p in BusinessCyclePhase}
        assert values == {"EARLY_EXPANSION", "MID_EXPANSION", "LATE_EXPANSION", "RECESSION"}


# ---------------------------------------------------------------------------
# API endpoint — GET /api/v1/views/macro-calibration
# ---------------------------------------------------------------------------

URL = "/api/v1/views/macro-calibration"
_CLASSIFY = "app.api.v1.macro_calibration.classify_macro_regime"


class TestMacroCalibrationEndpoint:
    def _make_service_result(
        self,
        phase: BusinessCyclePhase = BusinessCyclePhase.MID_EXPANSION,
        delta: float = 2.75,
        tau: float = 0.025,
    ) -> "CalibrationResult":
        from app.services.macro_calibration import CalibrationResult

        return CalibrationResult(
            phase=phase,
            delta=delta,
            tau=tau,
            confidence=0.80,
            rationale="Test rationale.",
            macro_summary="GDP: 2.5%, PMI: 55.",
        )

    def test_successful_response(self, client: TestClient) -> None:
        mock_result = self._make_service_result()

        with patch(_CLASSIFY, return_value=mock_result):
            resp = client.get(URL, params={"macro_text": "GDP strong, PMI 55."})

        assert resp.status_code == 200
        data = resp.json()
        assert data["phase"] == "MID_EXPANSION"
        assert data["delta"] == pytest.approx(2.75)
        assert data["tau"] == pytest.approx(0.025)
        assert 0.0 <= data["confidence"] <= 1.0
        assert "rationale" in data
        assert "bl_config" in data

    def test_delta_in_valid_range(self, client: TestClient) -> None:
        mock_result = self._make_service_result(delta=5.0)

        with patch(_CLASSIFY, return_value=mock_result):
            resp = client.get(URL, params={"macro_text": "Recession indicators."})

        data = resp.json()
        assert 1.0 <= data["delta"] <= 10.0

    def test_tau_in_valid_range(self, client: TestClient) -> None:
        mock_result = self._make_service_result(tau=0.01)

        with patch(_CLASSIFY, return_value=mock_result):
            resp = client.get(URL, params={"macro_text": "Late expansion."})

        data = resp.json()
        assert 0.001 <= data["tau"] <= 0.1

    def test_bl_config_contains_tau_and_risk_aversion(self, client: TestClient) -> None:
        mock_result = self._make_service_result(delta=3.5, tau=0.01)

        with patch(_CLASSIFY, return_value=mock_result):
            resp = client.get(URL, params={"macro_text": "Late expansion."})

        bl = resp.json()["bl_config"]
        assert "tau" in bl
        assert "prior_config" in bl
        assert "risk_aversion" in bl["prior_config"]

    def test_bl_config_risk_aversion_matches_delta(self, client: TestClient) -> None:
        mock_result = self._make_service_result(delta=4.2, tau=0.05)

        with patch(_CLASSIFY, return_value=mock_result):
            resp = client.get(URL, params={"macro_text": "Recession onset."})

        data = resp.json()
        assert data["bl_config"]["prior_config"]["risk_aversion"] == pytest.approx(data["delta"])
        assert data["bl_config"]["tau"] == pytest.approx(data["tau"])

    def test_no_db_data_returns_422(self, client: TestClient) -> None:
        with patch(_CLASSIFY, side_effect=ValueError("No macro data found")):
            resp = client.get(URL)

        assert resp.status_code == 422

    def test_llm_error_returns_502(self, client: TestClient) -> None:
        with patch(_CLASSIFY, side_effect=RuntimeError("LLM timeout")):
            resp = client.get(URL, params={"macro_text": "Some macro text."})

        assert resp.status_code == 502

    def test_default_country_is_united_states(self, client: TestClient) -> None:
        mock_result = self._make_service_result()
        captured: dict = {}

        def _capture(session, country="United States", macro_summary_override=None):
            captured["country"] = country
            return mock_result

        with patch(_CLASSIFY, side_effect=_capture):
            resp = client.get(URL, params={"macro_text": "GDP: 2.5%."})

        assert resp.status_code == 200
        assert captured["country"] == "United States"

    def test_custom_country_passed_through(self, client: TestClient) -> None:
        mock_result = self._make_service_result()
        captured: dict = {}

        def _capture(session, country="United States", macro_summary_override=None):
            captured["country"] = country
            return mock_result

        with patch(_CLASSIFY, side_effect=_capture):
            resp = client.get(URL, params={"country": "Germany", "macro_text": "PMI 54."})

        assert resp.status_code == 200
        assert captured["country"] == "Germany"

    def test_all_four_phases_produce_valid_response(self, client: TestClient) -> None:
        for phase in BusinessCyclePhase:
            mock_result = self._make_service_result(phase=phase)

            with patch(_CLASSIFY, return_value=mock_result):
                resp = client.get(URL, params={"macro_text": "Test."})

            assert resp.status_code == 200
            assert resp.json()["phase"] == phase.value
