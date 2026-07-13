"""Unit tests for macro regime calibration service and endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from baml_client.types import BusinessCyclePhase, MacroRegimeCalibration

if TYPE_CHECKING:
    pass

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
        from app.services.macro.macro_calibration import DELTA_MAX, _clamp_delta

        assert _clamp_delta(99.0) == pytest.approx(DELTA_MAX)

    def test_clamp_delta_below_min(self) -> None:
        from app.services.macro.macro_calibration import DELTA_MIN, _clamp_delta

        assert _clamp_delta(-1.0) == pytest.approx(DELTA_MIN)

    def test_clamp_delta_valid_passthrough(self) -> None:
        from app.services.macro.macro_calibration import _clamp_delta

        assert _clamp_delta(3.5) == pytest.approx(3.5)

    def test_clamp_tau_above_max(self) -> None:
        from app.services.macro.macro_calibration import TAU_MAX, _clamp_tau

        assert _clamp_tau(1.0) == pytest.approx(TAU_MAX)

    def test_clamp_tau_below_min(self) -> None:
        from app.services.macro.macro_calibration import TAU_MIN, _clamp_tau

        assert _clamp_tau(0.0) == pytest.approx(TAU_MIN)

    def test_clamp_tau_valid_passthrough(self) -> None:
        from app.services.macro.macro_calibration import _clamp_tau

        assert _clamp_tau(0.05) == pytest.approx(0.05)

    def test_clamp_confidence_upper(self) -> None:
        from app.services.macro.macro_calibration import _clamp_confidence

        assert _clamp_confidence(1.5) == pytest.approx(1.0)

    def test_clamp_confidence_lower(self) -> None:
        from app.services.macro.macro_calibration import _clamp_confidence

        assert _clamp_confidence(-0.1) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Service layer — classify_macro_regime
# ---------------------------------------------------------------------------


class TestClassifyMacroRegime:
    SUMMARY = "GDP: +2.5%, PMI: 55, Unemployment: 4.0%, CPI: 2.3%, 10Y-2Y: +80bps"

    def test_returns_calibration_result(self) -> None:
        from app.services.macro.macro_calibration import classify_macro_regime

        mock_session = MagicMock()
        mock_raw = _make_calibration()

        with patch(
            "app.services.macro.macro_calibration.b.ClassifyMacroRegime",
            return_value=mock_raw,
        ):
            result = classify_macro_regime(
                mock_session, macro_summary_override=self.SUMMARY
            )

        assert result.phase == BusinessCyclePhase.MID_EXPANSION
        assert result.delta == pytest.approx(2.75)
        assert result.tau == pytest.approx(0.025)

    def test_delta_clamped_to_max(self) -> None:
        from app.services.macro.macro_calibration import (
            DELTA_MAX,
            classify_macro_regime,
        )

        mock_session = MagicMock()
        mock_raw = _make_calibration(delta=999.0)

        with patch(
            "app.services.macro.macro_calibration.b.ClassifyMacroRegime",
            return_value=mock_raw,
        ):
            result = classify_macro_regime(
                mock_session, macro_summary_override=self.SUMMARY
            )

        assert result.delta == pytest.approx(DELTA_MAX)

    def test_delta_clamped_to_min(self) -> None:
        from app.services.macro.macro_calibration import (
            DELTA_MIN,
            classify_macro_regime,
        )

        mock_session = MagicMock()
        mock_raw = _make_calibration(delta=0.0)

        with patch(
            "app.services.macro.macro_calibration.b.ClassifyMacroRegime",
            return_value=mock_raw,
        ):
            result = classify_macro_regime(
                mock_session, macro_summary_override=self.SUMMARY
            )

        assert result.delta == pytest.approx(DELTA_MIN)

    def test_tau_clamped_to_max(self) -> None:
        from app.services.macro.macro_calibration import TAU_MAX, classify_macro_regime

        mock_session = MagicMock()
        mock_raw = _make_calibration(tau=5.0)

        with patch(
            "app.services.macro.macro_calibration.b.ClassifyMacroRegime",
            return_value=mock_raw,
        ):
            result = classify_macro_regime(
                mock_session, macro_summary_override=self.SUMMARY
            )

        assert result.tau == pytest.approx(TAU_MAX)

    def test_tau_clamped_to_min(self) -> None:
        from app.services.macro.macro_calibration import TAU_MIN, classify_macro_regime

        mock_session = MagicMock()
        mock_raw = _make_calibration(tau=0.0)

        with patch(
            "app.services.macro.macro_calibration.b.ClassifyMacroRegime",
            return_value=mock_raw,
        ):
            result = classify_macro_regime(
                mock_session, macro_summary_override=self.SUMMARY
            )

        assert result.tau == pytest.approx(TAU_MIN)

    def test_confidence_clamped(self) -> None:
        from app.services.macro.macro_calibration import classify_macro_regime

        mock_session = MagicMock()
        mock_raw = _make_calibration(confidence=1.5)

        with patch(
            "app.services.macro.macro_calibration.b.ClassifyMacroRegime",
            return_value=mock_raw,
        ):
            result = classify_macro_regime(
                mock_session, macro_summary_override=self.SUMMARY
            )

        assert result.confidence <= 1.0

    def test_raises_on_empty_db_and_no_override(self) -> None:
        from app.services.macro.macro_calibration import classify_macro_regime

        mock_session = MagicMock()
        mock_repo = MagicMock()
        mock_repo.get_macro_calibration.return_value = None
        mock_repo.get_economic_indicators.return_value = []
        mock_repo.get_te_indicators.return_value = []
        mock_repo.get_bond_yields.return_value = []
        mock_repo.get_macro_news_summary.return_value = None

        with (
            patch(
                "app.services.macro.macro_calibration.MacroRegimeRepository",
                return_value=mock_repo,
            ),
            pytest.raises(ValueError, match="No macro data"),
        ):
            classify_macro_regime(mock_session, country="Unknown")

    def test_macro_summary_stored_in_result(self) -> None:
        from app.services.macro.macro_calibration import classify_macro_regime

        mock_session = MagicMock()
        mock_raw = _make_calibration()
        custom_summary = "Custom macro context."

        with patch(
            "app.services.macro.macro_calibration.b.ClassifyMacroRegime",
            return_value=mock_raw,
        ):
            result = classify_macro_regime(
                mock_session, macro_summary_override=custom_summary
            )

        assert result.macro_summary == custom_summary

    @pytest.mark.parametrize(
        "phase,exp_delta_range,exp_tau_range",
        [
            (BusinessCyclePhase.EARLY_EXPANSION, (2.0, 2.5), (0.04, 0.06)),
            (BusinessCyclePhase.MID_EXPANSION, (2.5, 3.0), (0.02, 0.03)),
            (BusinessCyclePhase.LATE_EXPANSION, (3.0, 4.0), (0.005, 0.015)),
            (BusinessCyclePhase.RECESSION, (4.0, 6.0), (0.04, 0.06)),
        ],
    )
    def test_phase_produces_expected_parameter_ranges(
        self,
        phase: BusinessCyclePhase,
        exp_delta_range: tuple[float, float],
        exp_tau_range: tuple[float, float],
    ) -> None:
        from app.services.macro.macro_calibration import (
            _PHASE_DEFAULTS,
            classify_macro_regime,
        )

        # Use phase defaults as the mock LLM output
        default_delta, default_tau = _PHASE_DEFAULTS[phase]
        mock_session = MagicMock()
        mock_raw = _make_calibration(phase=phase, delta=default_delta, tau=default_tau)

        with patch(
            "app.services.macro.macro_calibration.b.ClassifyMacroRegime",
            return_value=mock_raw,
        ):
            result = classify_macro_regime(
                mock_session, macro_summary_override=self.SUMMARY
            )

        assert exp_delta_range[0] <= result.delta <= exp_delta_range[1], (
            f"Phase {phase}: delta={result.delta} not in {exp_delta_range}"
        )
        assert exp_tau_range[0] <= result.tau <= exp_tau_range[1], (
            f"Phase {phase}: tau={result.tau} not in {exp_tau_range}"
        )


# ---------------------------------------------------------------------------
# Service layer — phase enum has exactly 4 values
# ---------------------------------------------------------------------------


class TestBusinessCyclePhaseEnum:
    def test_exactly_four_phases(self) -> None:
        assert len(BusinessCyclePhase) == 4

    def test_all_expected_values_present(self) -> None:
        values = {p.value for p in BusinessCyclePhase}
        assert values == {
            "EARLY_EXPANSION",
            "MID_EXPANSION",
            "LATE_EXPANSION",
            "RECESSION",
        }


# ---------------------------------------------------------------------------
# Service layer — _build_macro_summary news injection
# ---------------------------------------------------------------------------


class TestBuildMacroSummaryNewsInjection:
    """Tests that _build_macro_summary correctly appends the Recent News Summary section."""

    @staticmethod
    def _make_repo(
        sentiment: str | None = "BULLISH",
        sentiment_score: float | None = 0.72,
        summary: str | None = "Markets are pricing in a soft landing scenario.",
    ) -> MagicMock:
        mock_repo = MagicMock()
        mock_repo.get_economic_indicators.return_value = []
        mock_repo.get_te_indicators.return_value = []
        mock_repo.get_bond_yields.return_value = []
        news = MagicMock()
        news.sentiment = sentiment
        news.sentiment_score = sentiment_score
        news.summary = summary
        mock_repo.get_macro_news_summary.return_value = news
        return mock_repo

    @staticmethod
    def _add_te_row(mock_repo: MagicMock) -> None:
        """Add a TE indicator so sections > 1 (function returns non-empty)."""
        te_row = MagicMock()
        te_row.indicator_key = "manufacturing_pmi"
        te_row.value = 54.2
        te_row.raw_name = "Manufacturing PMI"
        te_row.unit = "index"
        mock_repo.get_te_indicators.return_value = [te_row]

    def test_news_section_appended_when_data_present(self) -> None:
        from app.services.macro.macro_calibration import _build_macro_summary

        mock_repo = self._make_repo()
        self._add_te_row(mock_repo)

        result = _build_macro_summary(mock_repo, "USA")

        assert "### Recent News Summary" in result
        assert "Sentiment: BULLISH" in result
        assert "score: 0.72" in result
        assert "Markets are pricing in a soft landing scenario." in result

    def test_news_section_omitted_when_no_summary_row(self) -> None:
        from app.services.macro.macro_calibration import _build_macro_summary

        mock_repo = self._make_repo()
        mock_repo.get_macro_news_summary.return_value = None
        self._add_te_row(mock_repo)

        result = _build_macro_summary(mock_repo, "USA")

        assert "### Recent News Summary" not in result

    def test_news_section_omitted_when_both_fields_none(self) -> None:
        from app.services.macro.macro_calibration import _build_macro_summary

        mock_repo = self._make_repo(sentiment=None, summary=None)
        self._add_te_row(mock_repo)

        result = _build_macro_summary(mock_repo, "USA")

        assert "### Recent News Summary" not in result

    def test_sentiment_score_omitted_when_none(self) -> None:
        from app.services.macro.macro_calibration import _build_macro_summary

        mock_repo = self._make_repo(sentiment_score=None)
        self._add_te_row(mock_repo)

        result = _build_macro_summary(mock_repo, "USA")

        assert "Sentiment: BULLISH" in result
        assert "score:" not in result

    def test_summary_text_omitted_when_none(self) -> None:
        from app.services.macro.macro_calibration import _build_macro_summary

        mock_repo = self._make_repo(summary=None)
        self._add_te_row(mock_repo)

        result = _build_macro_summary(mock_repo, "USA")

        assert "### Recent News Summary" in result
        assert "Sentiment: BULLISH" in result
        assert "Markets" not in result

    def test_news_section_shown_when_only_summary_text_present(self) -> None:
        """sentiment=None but summary='text' -> section still appears with summary."""
        from app.services.macro.macro_calibration import _build_macro_summary

        mock_repo = self._make_repo(sentiment=None, summary="Soft landing expected.")
        self._add_te_row(mock_repo)

        result = _build_macro_summary(mock_repo, "USA")

        assert "### Recent News Summary" in result
        assert "Soft landing expected." in result
        assert "Sentiment:" not in result

    def test_news_not_injected_when_override_active(self) -> None:
        """macro_summary_override bypasses _build_macro_summary entirely."""
        from app.services.macro.macro_calibration import classify_macro_regime

        mock_session = MagicMock()
        mock_raw = _make_calibration()

        with patch(
            "app.services.macro.macro_calibration.b.ClassifyMacroRegime",
            return_value=mock_raw,
        ) as mock_llm:
            result = classify_macro_regime(
                mock_session, macro_summary_override="Custom override text."
            )

        assert result.macro_summary == "Custom override text."
        assert "### Recent News Summary" not in result.macro_summary
        call_kwargs = mock_llm.call_args
        assert call_kwargs.kwargs["macro_summary"] == "Custom override text."
