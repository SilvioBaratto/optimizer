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


class TestBoundedTilts:
    """Tests for capped tilt multipliers and weight floor (issue #279)."""

    def _weights(self) -> dict[FactorGroupType, float]:
        return {
            FactorGroupType.VALUE: 0.30,
            FactorGroupType.MOMENTUM: 0.40,
            FactorGroupType.LOW_RISK: 0.30,
        }

    # --- multiplier cap ---

    def test_extreme_tilt_is_capped(self) -> None:
        """A raw tilt of 10x is clamped to max_tilt_multiplier."""
        config = RegimeTiltConfig(
            enable=True,
            max_tilt_multiplier=2.0,
            min_post_tilt_weight=0.0,
            expansion_tilts=(("momentum", 10.0),),
        )
        result = apply_regime_tilts(self._weights(), MacroRegime.EXPANSION, config)
        # Effective momentum tilt is 2.0, not 10.0.
        # Pre-norm: value=0.30, momentum=0.40*2=0.80, low_risk=0.30
        # Total = 1.40, scale = 1.0 / 1.40
        expected_momentum = 0.80 / 1.40
        assert abs(result[FactorGroupType.MOMENTUM] - expected_momentum) < 1e-10

    def test_normal_tilt_unchanged_by_cap(self) -> None:
        """Tilts within the cap are applied without modification."""
        config = RegimeTiltConfig(
            enable=True,
            max_tilt_multiplier=2.0,
            min_post_tilt_weight=0.0,
            expansion_tilts=(
                ("momentum", 1.2),
                ("value", 0.8),
                ("low_risk", 0.8),
            ),
        )
        result = apply_regime_tilts(self._weights(), MacroRegime.EXPANSION, config)
        # Pre-norm: value=0.24, momentum=0.48, low_risk=0.24 → 0.96
        expected_momentum = 0.48 / 0.96
        assert abs(result[FactorGroupType.MOMENTUM] - expected_momentum) < 1e-10

    # --- weight floor ---

    def test_floor_prevents_near_zero_weight(self) -> None:
        """A suppressed group is raised to min_post_tilt_weight * total."""
        config = RegimeTiltConfig(
            enable=True,
            max_tilt_multiplier=2.0,
            min_post_tilt_weight=0.10,
            recession_tilts=(("momentum", 0.01),),
        )
        weights = self._weights()
        result = apply_regime_tilts(weights, MacroRegime.RECESSION, config)
        # momentum raw = 0.40 * 0.01 = 0.004, floored to 0.10 * 1.0 = 0.10
        # Pre-renorm: 0.30, 0.10, 0.30 → 0.70, scale = 1.0/0.70
        assert result[FactorGroupType.MOMENTUM] > 0.0
        assert abs(sum(result.values()) - sum(weights.values())) < 1e-10

    def test_floor_zero_allows_full_suppression(self) -> None:
        """With min_post_tilt_weight=0.0, zero tilt drives weight to 0."""
        config = RegimeTiltConfig(
            enable=True,
            max_tilt_multiplier=2.0,
            min_post_tilt_weight=0.0,
            recession_tilts=(("momentum", 0.0),),
        )
        result = apply_regime_tilts(self._weights(), MacroRegime.RECESSION, config)
        assert result[FactorGroupType.MOMENTUM] == 0.0

    # --- combined caps and floor ---

    def test_extreme_recession_tilts_bounded(self) -> None:
        """Extreme tilts: multiplier capped and floor applied."""
        config = RegimeTiltConfig(
            enable=True,
            max_tilt_multiplier=2.0,
            min_post_tilt_weight=0.05,
            recession_tilts=(
                ("low_risk", 5.0),
                ("momentum", 0.0),
            ),
        )
        weights = {
            FactorGroupType.VALUE: 1.0,
            FactorGroupType.MOMENTUM: 1.0,
            FactorGroupType.LOW_RISK: 1.0,
        }
        result = apply_regime_tilts(weights, MacroRegime.RECESSION, config)
        assert abs(sum(result.values()) - 3.0) < 1e-10
        assert result[FactorGroupType.LOW_RISK] > result[FactorGroupType.VALUE]
        assert result[FactorGroupType.MOMENTUM] > 0.0

    # --- total preservation ---

    def test_total_weight_preserved_with_bounds(self) -> None:
        """Total weight preserved even when bounds are active."""
        config = RegimeTiltConfig(
            enable=True,
            max_tilt_multiplier=1.5,
            min_post_tilt_weight=0.10,
            recession_tilts=(
                ("low_risk", 3.0),
                ("momentum", 0.1),
            ),
        )
        weights = {
            FactorGroupType.VALUE: 2.0,
            FactorGroupType.MOMENTUM: 3.0,
            FactorGroupType.LOW_RISK: 1.0,
        }
        result = apply_regime_tilts(weights, MacroRegime.RECESSION, config)
        assert abs(sum(result.values()) - 6.0) < 1e-10

    # --- validation ---

    def test_invalid_max_tilt_multiplier_raises(self) -> None:
        """max_tilt_multiplier below 1.0 is rejected."""
        import pytest

        with pytest.raises(ValueError, match="max_tilt_multiplier"):
            RegimeTiltConfig(enable=True, max_tilt_multiplier=0.9)

    def test_negative_min_post_tilt_weight_raises(self) -> None:
        """Negative min_post_tilt_weight is rejected."""
        import pytest

        with pytest.raises(ValueError, match="min_post_tilt_weight"):
            RegimeTiltConfig(enable=True, min_post_tilt_weight=-0.01)

    def test_min_post_tilt_weight_ge_one_raises(self) -> None:
        """min_post_tilt_weight >= 1.0 is rejected."""
        import pytest

        with pytest.raises(ValueError, match="min_post_tilt_weight"):
            RegimeTiltConfig(enable=True, min_post_tilt_weight=1.0)

    # --- preset ---

    def test_for_strict_bounds_preset(self) -> None:
        """for_strict_bounds() is enabled with tighter caps."""
        config = RegimeTiltConfig.for_strict_bounds()
        assert config.enable is True
        assert config.max_tilt_multiplier == 1.5
        assert config.min_post_tilt_weight == 0.10

    def test_for_strict_bounds_limits_boost(self) -> None:
        """Strict bounds cap expansion momentum at 1.5x max."""
        config = RegimeTiltConfig.for_strict_bounds()
        weights = {
            FactorGroupType.VALUE: 1.0,
            FactorGroupType.MOMENTUM: 1.0,
            FactorGroupType.LOW_RISK: 1.0,
        }
        result = apply_regime_tilts(weights, MacroRegime.EXPANSION, config)
        # Default expansion: momentum=1.2 < 1.5 cap, not capped
        # value=0.8, low_risk=0.8
        # Pre-norm: 0.8, 1.2, 0.8 → 2.8, scale=3.0/2.8
        expected = 1.2 / 2.8 * 3.0
        assert abs(result[FactorGroupType.MOMENTUM] - expected) < 1e-10
