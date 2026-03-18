"""Tests for macro regime classification and tilts."""

from __future__ import annotations

import logging

import pandas as pd

from optimizer.factors import (
    FactorGroupType,
    MacroRegime,
    RegimeThresholdConfig,
    RegimeTiltConfig,
    apply_regime_tilts,
    check_regime_disagreement,
    classify_regime,
    classify_regime_composite,
    get_regime_tilts,
)


class TestClassifyRegime:
    def test_expansion(self) -> None:
        macro = pd.DataFrame(
            {"gdp_growth": [2.0, 2.5, 3.0, 3.5]},
            index=pd.date_range("2023-01-01", periods=4, freq="QE"),
        )
        result = classify_regime(macro)
        assert result == MacroRegime.EXPANSION

    def test_recession(self) -> None:
        macro = pd.DataFrame(
            {"gdp_growth": [3.0, 2.0, 1.0, 0.5]},
            index=pd.date_range("2023-01-01", periods=4, freq="QE"),
        )
        result = classify_regime(macro)
        assert result == MacroRegime.RECESSION

    def test_recovery(self) -> None:
        macro = pd.DataFrame(
            {"gdp_growth": [3.0, 1.0, 0.5, 1.5]},
            index=pd.date_range("2023-01-01", periods=4, freq="QE"),
        )
        result = classify_regime(macro)
        assert result == MacroRegime.RECOVERY

    def test_slowdown(self) -> None:
        macro = pd.DataFrame(
            {"gdp_growth": [1.0, 2.0, 3.0, 2.5]},
            index=pd.date_range("2023-01-01", periods=4, freq="QE"),
        )
        result = classify_regime(macro)
        assert result == MacroRegime.SLOWDOWN

    def test_empty_data(self, caplog: logging.LogRecord) -> None:
        macro = pd.DataFrame()
        with caplog.at_level(logging.WARNING):
            result = classify_regime(macro)
        assert result == MacroRegime.UNKNOWN
        assert "empty" in caplog.text.lower()

    def test_yield_spread_fallback(self) -> None:
        macro = pd.DataFrame(
            {"yield_spread": [-1.0]},
            index=pd.date_range("2023-01-01", periods=1),
        )
        result = classify_regime(macro)
        assert result == MacroRegime.RECESSION


class TestGetRegimeTilts:
    def test_expansion_tilts(self) -> None:
        tilts = get_regime_tilts(MacroRegime.EXPANSION)
        assert FactorGroupType.MOMENTUM in tilts
        assert tilts[FactorGroupType.MOMENTUM] == 1.2

    def test_recession_tilts(self) -> None:
        tilts = get_regime_tilts(MacroRegime.RECESSION)
        assert FactorGroupType.LOW_RISK in tilts
        assert tilts[FactorGroupType.LOW_RISK] == 1.5

    def test_all_regimes_return_dict(self) -> None:
        for regime in MacroRegime:
            tilts = get_regime_tilts(regime)
            assert isinstance(tilts, dict)


class TestApplyRegimeTilts:
    def test_disabled_returns_copy(self) -> None:
        weights = {FactorGroupType.VALUE: 1.0, FactorGroupType.MOMENTUM: 1.0}
        config = RegimeTiltConfig.for_no_tilts()
        result = apply_regime_tilts(weights, MacroRegime.EXPANSION, config)
        assert result == weights
        assert result is not weights

    def test_enabled_applies_tilts(self) -> None:
        weights = {
            FactorGroupType.VALUE: 1.0,
            FactorGroupType.MOMENTUM: 1.0,
            FactorGroupType.LOW_RISK: 1.0,
        }
        config = RegimeTiltConfig.for_moderate_tilts()
        result = apply_regime_tilts(weights, MacroRegime.EXPANSION, config)
        # Momentum should be tilted up in expansion
        # After normalization, momentum weight > value weight
        assert result[FactorGroupType.MOMENTUM] > result[FactorGroupType.VALUE]

    def test_preserves_total_weight(self) -> None:
        weights = {
            FactorGroupType.VALUE: 2.0,
            FactorGroupType.MOMENTUM: 3.0,
            FactorGroupType.LOW_RISK: 1.0,
        }
        config = RegimeTiltConfig.for_moderate_tilts()
        result = apply_regime_tilts(weights, MacroRegime.RECESSION, config)
        assert abs(sum(result.values()) - sum(weights.values())) < 1e-10


class TestUnemploymentRegimeOverride:
    """Tests for multi-indicator regime with unemployment (issue #79)."""

    def test_slowdown_unemployment_override(self) -> None:
        """Rising unemployment + positive GDP → SLOWDOWN."""
        macro = pd.DataFrame(
            {
                "gdp_growth": [2.0, 2.5, 3.0, 3.5],
                "unemployment_rate": [3.5, 3.6, 3.8, 4.1],
            },
            index=pd.date_range("2023-01-01", periods=4, freq="QE"),
        )
        result = classify_regime(macro)
        assert result == MacroRegime.SLOWDOWN

    def test_unemployment_not_rising_no_override(self) -> None:
        """Declining unemployment → original GDP-based classification."""
        macro = pd.DataFrame(
            {
                "gdp_growth": [2.0, 2.5, 3.0, 3.5],
                "unemployment_rate": [4.1, 3.8, 3.6, 3.5],
            },
            index=pd.date_range("2023-01-01", periods=4, freq="QE"),
        )
        result = classify_regime(macro)
        # GDP is positive and rising → EXPANSION (not overridden)
        assert result == MacroRegime.EXPANSION

    def test_unemployment_missing_graceful_fallback(self) -> None:
        """No unemployment_rate column → original behavior."""
        macro = pd.DataFrame(
            {"gdp_growth": [2.0, 2.5, 3.0, 3.5]},
            index=pd.date_range("2023-01-01", periods=4, freq="QE"),
        )
        result = classify_regime(macro)
        assert result == MacroRegime.EXPANSION


class TestCompositePathReachable:
    """Verify composite path dispatch with merged indicators (#237)."""

    def test_composite_expansion(self) -> None:
        """All three composite indicators signal expansion."""
        macro = pd.DataFrame(
            {"pmi": [54.0], "spread_2s10s": [1.5], "hy_oas": [300.0]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        result = classify_regime(macro)
        assert result == MacroRegime.EXPANSION

    def test_composite_recession(self) -> None:
        """All three composite indicators signal recession."""
        macro = pd.DataFrame(
            {"pmi": [45.0], "spread_2s10s": [-0.5], "hy_oas": [600.0]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        result = classify_regime(macro)
        assert result == MacroRegime.RECESSION

    def test_composite_with_sentiment(self) -> None:
        """Composite path uses sentiment when available."""
        macro = pd.DataFrame(
            {
                "pmi": [54.0],
                "spread_2s10s": [1.5],
                "hy_oas": [300.0],
                "sentiment": [0.5],
            },
            index=pd.date_range("2024-01-01", periods=1),
        )
        result = classify_regime(macro)
        assert result == MacroRegime.EXPANSION

    def test_partial_composite_indicators(self) -> None:
        """Single composite column triggers composite dispatch."""
        macro = pd.DataFrame(
            {"spread_2s10s": [1.5]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        result = classify_regime(macro)
        # s_pmi=0, s_2s10s=1, s_hy=0 → S_t=1 → EXPANSION
        assert result == MacroRegime.EXPANSION

    def test_gdp_path_unchanged_without_composite_cols(self) -> None:
        """GDP-only data still uses the original heuristic."""
        macro = pd.DataFrame(
            {"gdp_growth": [2.0, 2.5, 3.0, 3.5]},
            index=pd.date_range("2023-01-01", periods=4, freq="QE"),
        )
        result = classify_regime(macro)
        assert result == MacroRegime.EXPANSION


class TestCheckRegimeDisagreement:
    """Tests for regime disagreement detection (issue #240)."""

    def test_agreement_returns_false(self) -> None:
        assert (
            check_regime_disagreement(MacroRegime.EXPANSION, MacroRegime.EXPANSION)
            is False
        )

    def test_disagreement_returns_true(self) -> None:
        assert (
            check_regime_disagreement(MacroRegime.EXPANSION, MacroRegime.RECESSION)
            is True
        )

    def test_logs_warning_on_disagreement(self, caplog: logging.LogRecord) -> None:
        with caplog.at_level(logging.WARNING):
            check_regime_disagreement(MacroRegime.EXPANSION, MacroRegime.RECESSION)
        assert "Regime disagreement" in caplog.text
        assert "expansion" in caplog.text
        assert "recession" in caplog.text

    def test_no_log_on_agreement(self, caplog: logging.LogRecord) -> None:
        with caplog.at_level(logging.WARNING):
            check_regime_disagreement(MacroRegime.SLOWDOWN, MacroRegime.SLOWDOWN)
        assert "Regime disagreement" not in caplog.text

    def test_custom_labels_in_log(self, caplog: logging.LogRecord) -> None:
        with caplog.at_level(logging.WARNING):
            check_regime_disagreement(
                MacroRegime.EXPANSION,
                MacroRegime.RECOVERY,
                label_a="macro_score",
                label_b="statistical_hmm",
            )
        assert "macro_score" in caplog.text
        assert "statistical_hmm" in caplog.text


class TestRegimeThresholdConfigIntegration:
    """Verify classify_regime respects injected thresholds (#241)."""

    def test_custom_pmi_threshold(self) -> None:
        """PMI=53 is expansion at default (52) but neutral at threshold 54."""
        macro = pd.DataFrame(
            {"pmi": [53.0], "spread_2s10s": [0.5], "hy_oas": [420.0]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        # Default: pmi_expansion=52 → s_pmi=+1, s_2s10s=0, s_hy=0 → S_t=1 → EXPANSION
        assert classify_regime(macro) == MacroRegime.EXPANSION
        # Raised threshold: pmi_expansion=54 → s_pmi=0, total S_t=0 → SLOWDOWN
        tight = RegimeThresholdConfig(pmi_expansion=54.0)
        assert classify_regime(macro, thresholds=tight) == MacroRegime.SLOWDOWN

    def test_custom_hy_threshold(self) -> None:
        """HY OAS=380 is neutral at default (350) but risk-on at 400."""
        macro = pd.DataFrame(
            {"hy_oas": [380.0], "pmi": [50.0], "spread_2s10s": [0.5]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        # Default: hy_oas_risk_on=350 → 380>350 not risk_on → s_hy=0, S_t=0 → SLOWDOWN
        assert classify_regime(macro) == MacroRegime.SLOWDOWN
        # Raised risk_on to 400: 380<400 → s_hy=+1 → S_t=+1 → EXPANSION
        raised = RegimeThresholdConfig(hy_oas_risk_on=400.0)
        assert classify_regime(macro, thresholds=raised) == MacroRegime.EXPANSION


class TestUnknownRegime:
    """Tests for MacroRegime.UNKNOWN introduced in issue #250."""

    def test_empty_df_returns_unknown(self, caplog: logging.LogRecord) -> None:
        with caplog.at_level(logging.WARNING):
            result = classify_regime(pd.DataFrame())
        assert result == MacroRegime.UNKNOWN
        assert "MacroRegime.UNKNOWN" in caplog.text

    def test_empty_df_composite_returns_unknown(
        self, caplog: logging.LogRecord
    ) -> None:
        with caplog.at_level(logging.WARNING):
            result = classify_regime_composite(pd.DataFrame())
        assert result == MacroRegime.UNKNOWN
        assert "MacroRegime.UNKNOWN" in caplog.text

    def test_unrecognized_columns_returns_unknown(
        self, caplog: logging.LogRecord
    ) -> None:
        macro = pd.DataFrame(
            {"inflation_rate": [2.5]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        with caplog.at_level(logging.WARNING):
            result = classify_regime(macro)
        assert result == MacroRegime.UNKNOWN
        assert "no recognized indicator columns" in caplog.text

    def test_get_regime_tilts_unknown_returns_empty_dict(self) -> None:
        tilts = get_regime_tilts(MacroRegime.UNKNOWN)
        assert tilts == {}

    def test_apply_regime_tilts_unknown_is_neutral_identity(self) -> None:
        weights = {
            FactorGroupType.VALUE: 1.0,
            FactorGroupType.MOMENTUM: 1.5,
            FactorGroupType.LOW_RISK: 0.8,
        }
        config = RegimeTiltConfig.for_moderate_tilts()
        result = apply_regime_tilts(weights, MacroRegime.UNKNOWN, config)
        for group, w in weights.items():
            assert abs(result[group] - w) < 1e-10

    def test_unknown_value_is_string(self) -> None:
        assert MacroRegime.UNKNOWN.value == "unknown"
        assert isinstance(MacroRegime.UNKNOWN, str)
