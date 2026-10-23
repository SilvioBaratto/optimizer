"""Tests for macro regime classification and tilts."""

from __future__ import annotations

import logging
import tracemalloc
from unittest.mock import patch

import numpy as np
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
from optimizer.factors import _regime as regime_module


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
        # EXPANSION boosts cyclicals (momentum, value) and dampens defensives
        # (low_risk).  Momentum and value should outweigh low_risk.
        assert result[FactorGroupType.MOMENTUM] > result[FactorGroupType.LOW_RISK]
        assert result[FactorGroupType.VALUE] > result[FactorGroupType.LOW_RISK]

    def test_preserves_total_weight(self) -> None:
        weights = {
            FactorGroupType.VALUE: 2.0,
            FactorGroupType.MOMENTUM: 3.0,
            FactorGroupType.LOW_RISK: 1.0,
        }
        config = RegimeTiltConfig.for_moderate_tilts()
        result = apply_regime_tilts(weights, MacroRegime.RECESSION, config)
        assert abs(sum(result.values()) - sum(weights.values())) < 1e-10

    def test_when_recession_value_multiplier_is_dampened(self) -> None:
        """Cycle-2 spec: RECESSION dampens VALUE (multiplier < 1.0)."""
        config = RegimeTiltConfig()
        recession_map = dict(config.recession_tilts)
        assert recession_map["value"] < 1.0

    def test_when_expansion_value_multiplier_is_boosted(self) -> None:
        """Cycle-2 spec: EXPANSION boosts VALUE (multiplier > 1.0)."""
        config = RegimeTiltConfig()
        expansion_map = dict(config.expansion_tilts)
        assert expansion_map["value"] > 1.0

    def test_when_recession_value_weight_below_expansion_value_weight(self) -> None:
        weights = {
            FactorGroupType.VALUE: 1.0,
            FactorGroupType.MOMENTUM: 1.0,
            FactorGroupType.LOW_RISK: 1.0,
            FactorGroupType.PROFITABILITY: 1.0,
        }
        config = RegimeTiltConfig(enable=True)
        rec = apply_regime_tilts(weights, MacroRegime.RECESSION, config)
        exp = apply_regime_tilts(weights, MacroRegime.EXPANSION, config)
        assert rec[FactorGroupType.VALUE] < exp[FactorGroupType.VALUE]
        assert rec[FactorGroupType.MOMENTUM] < exp[FactorGroupType.MOMENTUM]


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
        # Default expansion (post-#529): momentum=1.2, value=1.2, low_risk=0.8
        # All multipliers < 1.5 cap, none capped
        # Pre-norm: 1.2 + 1.2 + 0.8 = 3.2; scale = 3.0 / 3.2
        expected = 1.2 / 3.2 * 3.0
        assert abs(result[FactorGroupType.MOMENTUM] - expected) < 1e-10


def _composite(**cols: float) -> pd.DataFrame:
    """Single-row macro frame for composite-score branch tests."""
    return pd.DataFrame(
        {k: [v] for k, v in cols.items()},
        index=pd.date_range("2024-01-01", periods=1),
    )


class TestCompositeScoreBranches:
    """Cover the sentiment-augmented and 3-indicator composite-score arms."""

    def test_composite_without_spread_uses_remaining_indicators(self) -> None:
        """When spread_2s10s is absent, its component score stays neutral."""
        # s_pmi=+1 (54>52), s_2s10s=0 (absent), s_hy=+1 (300<350) → S_t=2 → EXPANSION
        result = classify_regime(_composite(pmi=54.0, hy_oas=300.0))
        assert result == MacroRegime.EXPANSION

    def test_sentiment_augmented_recession(self) -> None:
        """A strongly negative augmented score classifies as RECESSION."""
        # S_t=-3, s_sent=-1 → S_aug=-4 ≤ -3 → RECESSION
        macro = _composite(pmi=45.0, spread_2s10s=-0.5, hy_oas=600.0, sentiment=-0.5)
        assert classify_regime_composite(macro) == MacroRegime.RECESSION

    def test_sentiment_augmented_expansion(self) -> None:
        """A positive (but < 3) augmented score classifies as EXPANSION."""
        # S_t=1, s_sent=+1 → S_aug=2 (>0) → EXPANSION
        macro = _composite(pmi=54.0, spread_2s10s=0.5, hy_oas=420.0, sentiment=0.5)
        assert classify_regime_composite(macro) == MacroRegime.EXPANSION

    def test_sentiment_augmented_slowdown(self) -> None:
        """An augmented score of exactly zero classifies as SLOWDOWN."""
        # S_t=1, s_sent=-1 → S_aug=0 → SLOWDOWN
        macro = _composite(pmi=54.0, spread_2s10s=0.5, hy_oas=420.0, sentiment=-0.5)
        assert classify_regime_composite(macro) == MacroRegime.SLOWDOWN

    def test_sentiment_augmented_recovery(self) -> None:
        """A mildly negative augmented score classifies as RECOVERY."""
        # S_t=0, s_sent=-1 → S_aug=-1 (>-3, <0, !=0) → RECOVERY
        macro = _composite(pmi=50.0, spread_2s10s=0.5, hy_oas=420.0, sentiment=-0.5)
        assert classify_regime_composite(macro) == MacroRegime.RECOVERY

    def test_three_indicator_recovery(self) -> None:
        """A 3-indicator score of -1 (no sentiment) classifies as RECOVERY."""
        # s_pmi=-1, s_2s10s=0, s_hy=0 → S_t=-1 → RECOVERY
        macro = _composite(pmi=45.0, spread_2s10s=0.5, hy_oas=420.0)
        assert classify_regime_composite(macro) == MacroRegime.RECOVERY


class TestClassifyRegimeYieldSpreadFallback:
    """Cover the GDP-too-short fallthrough and yield-spread fallback arms."""

    def test_gdp_single_observation_falls_through_to_yield_spread(self) -> None:
        """A single GDP observation (len < 2) defers to the yield-spread fallback."""
        macro = pd.DataFrame(
            {"gdp_growth": [2.0], "yield_spread": [1.5]},
            index=pd.date_range("2024-01-01", periods=1),
        )
        # gdp len 1 → skip GDP block; yield_spread 1.5 > 1.0 → EXPANSION
        assert classify_regime(macro) == MacroRegime.EXPANSION

    def test_yield_spread_slowdown(self) -> None:
        """A positive but flat yield spread classifies as SLOWDOWN."""
        macro = _composite(yield_spread=0.5)
        assert classify_regime(macro) == MacroRegime.SLOWDOWN

    def test_yield_spread_recovery(self) -> None:
        """A mildly inverted yield spread classifies as RECOVERY."""
        macro = _composite(yield_spread=-0.2)
        assert classify_regime(macro) == MacroRegime.RECOVERY

    def test_yield_spread_all_nan_returns_unknown(self) -> None:
        """An all-NaN yield spread leaves no usable signal → UNKNOWN."""
        macro = _composite(yield_spread=float("nan"))
        assert classify_regime(macro) == MacroRegime.UNKNOWN


class TestRegimeTiltBranchCoverage:
    """Cover invalid-group skip, default-config, and zero-total-weight arms."""

    def test_get_regime_tilts_skips_unknown_group_name(self) -> None:
        """A tilt referencing an unknown group name is silently skipped."""
        config = RegimeTiltConfig(
            enable=True, expansion_tilts=(("not_a_real_group", 1.5),)
        )
        assert get_regime_tilts(MacroRegime.EXPANSION, config) == {}

    def test_apply_regime_tilts_defaults_to_disabled_config(self) -> None:
        """With no config, the default (disabled) leaves weights unchanged."""
        weights = {FactorGroupType.VALUE: 1.0, FactorGroupType.MOMENTUM: 1.0}
        result = apply_regime_tilts(weights, MacroRegime.EXPANSION)
        assert result == weights
        assert result is not weights

    def test_apply_regime_tilts_zero_total_weights(self) -> None:
        """All-zero weights skip the floor and re-normalization steps."""
        weights = {FactorGroupType.VALUE: 0.0, FactorGroupType.MOMENTUM: 0.0}
        config = RegimeTiltConfig(enable=True)
        result = apply_regime_tilts(weights, MacroRegime.EXPANSION, config)
        # Zero total → floor and re-normalization steps are skipped; every
        # group is preserved and stays at 0.0 (no division-by-zero).
        assert set(result) == set(weights)
        assert all(value == 0.0 for value in result.values())


class TestCompositeUsesLastValidObservation:
    """A NaN in the newest row must not erase an indicator's latest print."""

    def _monthly_pmi_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "pmi": [45.0, float("nan")],
                "spread_2s10s": [1.5, 1.5],
                "hy_oas": [420.0, 420.0],
            },
            index=pd.date_range("2024-03-01", periods=2),
        )

    def test_missing_monthly_pmi_on_latest_row_carries_last_print(self) -> None:
        """PMI=45 (-1), spread=1.5 (+1), HY=420 (0) → S_t=0 → SLOWDOWN."""
        macro = self._monthly_pmi_frame()
        assert classify_regime_composite(macro.iloc[:1]) == MacroRegime.SLOWDOWN
        assert classify_regime_composite(macro) == MacroRegime.SLOWDOWN
        assert classify_regime(macro) == MacroRegime.SLOWDOWN

    def test_regime_does_not_flip_between_release_and_non_release_days(
        self,
    ) -> None:
        """Expanding windows over a daily frame with one monthly print agree."""
        macro = pd.DataFrame(
            {
                "pmi": [45.0] + [float("nan")] * 4,
                "spread_2s10s": [1.5] * 5,
                "hy_oas": [420.0] * 5,
            },
            index=pd.date_range("2024-03-01", periods=5),
        )
        regimes = {classify_regime(macro.iloc[: i + 1]) for i in range(len(macro))}
        assert regimes == {MacroRegime.SLOWDOWN}

    def test_sentiment_gap_on_latest_row_carries_last_reading(self) -> None:
        """Sentiment follows the same last-valid rule as the macro indicators."""
        macro = pd.DataFrame(
            {
                "pmi": [50.0, 50.0],
                "spread_2s10s": [0.5, 0.5],
                "hy_oas": [420.0, 420.0],
                "sentiment": [-0.5, float("nan")],
            },
            index=pd.date_range("2024-03-01", periods=2),
        )
        # S_t=0, s_sent=-1 carried → S_aug=-1 → RECOVERY (not SLOWDOWN).
        assert classify_regime_composite(macro) == MacroRegime.RECOVERY


class TestLastValidLookupCost:
    """Point-in-time validation classifies every expanding window, so finding
    the latest print must not cost a pass over the whole history."""

    @staticmethod
    def _long_macro(rows: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "pmi": [55.0] * rows,
                "spread_2s10s": [1.5] * rows,
                "hy_oas": [300.0] * rows,
                "sentiment": [0.5] * rows,
            },
            index=pd.date_range("2000-01-03", periods=rows, freq="B"),
        )

    @staticmethod
    def _peak_allocation(macro: pd.DataFrame) -> int:
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            classify_regime(macro)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    def test_classification_cost_does_not_grow_with_history_length(self) -> None:
        short = self._long_macro(1_000)
        long = self._long_macro(500_000)
        # Warm pandas' internal caches so they are not billed to either call.
        classify_regime(short)
        classify_regime(long)
        extra = self._peak_allocation(long) - self._peak_allocation(short)
        # One boolean mask over the long column alone would take ~500 kB.
        assert extra < 50_000

    def test_each_indicator_is_resolved_once_per_classification(self) -> None:
        macro = self._long_macro(50)
        with patch.object(
            regime_module, "_last_valid", wraps=regime_module._last_valid
        ) as lookup:
            assert classify_regime(macro) == MacroRegime.EXPANSION
        columns = sorted(call.args[1] for call in lookup.call_args_list)
        assert columns == ["hy_oas", "pmi", "sentiment", "spread_2s10s"]

    def test_latest_print_is_found_at_any_distance_from_the_end(self) -> None:
        rows = 300
        for gap in (0, 1, 15, 16, 17, 47, 48, 49, 150, 299):
            values = np.full(rows, np.nan)
            values[0] = 40.0
            values[rows - 1 - gap] = 51.5
            macro = pd.DataFrame(
                {"pmi": values},
                index=pd.date_range("2000-01-03", periods=rows, freq="B"),
            )
            assert regime_module._last_valid(macro, "pmi") == 51.5, gap

    def test_column_without_any_print_has_no_latest_value(self) -> None:
        macro = pd.DataFrame(
            {"pmi": [np.nan] * 300},
            index=pd.date_range("2000-01-03", periods=300, freq="B"),
        )
        assert regime_module._last_valid(macro, "pmi") is None
        assert regime_module._last_valid(macro, "not_a_column") is None

    def test_nullable_and_object_columns_are_read_as_floats(self) -> None:
        index = pd.date_range("2024-01-01", periods=3)
        nullable = pd.DataFrame(
            {"pmi": pd.array([47.0, pd.NA, pd.NA], dtype="Float64")}, index=index
        )
        mixed = pd.DataFrame({"pmi": [47.0, None, None]}, index=index, dtype=object)
        assert regime_module._last_valid(nullable, "pmi") == 47.0
        assert regime_module._last_valid(mixed, "pmi") == 47.0


class TestCompositeWithoutUsableIndicators:
    """No observed composite indicator means no information, not SLOWDOWN."""

    nan = float("nan")

    def test_all_nan_composite_row_returns_unknown(self) -> None:
        macro = _composite(pmi=self.nan, spread_2s10s=self.nan, hy_oas=self.nan)
        assert classify_regime(macro) == MacroRegime.UNKNOWN
        assert classify_regime_composite(macro) == MacroRegime.UNKNOWN

    def test_leading_all_nan_rows_are_unknown_until_first_print(self) -> None:
        macro = pd.DataFrame(
            {
                "pmi": [self.nan, self.nan, 54.0],
                "spread_2s10s": [self.nan, self.nan, 1.5],
                "hy_oas": [self.nan, self.nan, 300.0],
            },
            index=pd.date_range("2024-01-01", periods=3),
        )
        regimes = [classify_regime(macro.iloc[: i + 1]) for i in range(len(macro))]
        assert regimes == [
            MacroRegime.UNKNOWN,
            MacroRegime.UNKNOWN,
            MacroRegime.EXPANSION,
        ]

    def test_nan_composite_columns_fall_through_to_gdp_path(self) -> None:
        macro = pd.DataFrame(
            {"pmi": [self.nan] * 4, "gdp_growth": [2.0, 2.5, 3.0, 3.5]},
            index=pd.date_range("2023-01-01", periods=4, freq="QE"),
        )
        assert classify_regime(macro) == MacroRegime.EXPANSION

    def test_nan_composite_columns_fall_through_to_yield_spread(self) -> None:
        macro = _composite(pmi=self.nan, yield_spread=-1.0)
        assert classify_regime(macro) == MacroRegime.RECESSION

    def test_sentiment_alone_does_not_drive_the_regime(self) -> None:
        bullish = _composite(pmi=self.nan, sentiment=0.5)
        bearish = _composite(pmi=self.nan, sentiment=-0.5)
        assert classify_regime(bullish) == MacroRegime.UNKNOWN
        assert classify_regime(bearish) == MacroRegime.UNKNOWN
        assert classify_regime_composite(bullish) == MacroRegime.UNKNOWN
        assert classify_regime_composite(_composite(sentiment=-0.5)) == (
            MacroRegime.UNKNOWN
        )

    def test_unknown_logs_warning(self, caplog: logging.LogRecord) -> None:
        macro = _composite(pmi=self.nan)
        with caplog.at_level(logging.WARNING):
            classify_regime_composite(macro)
        assert "MacroRegime.UNKNOWN" in caplog.text


class TestTiltFloorScope:
    """The floor protects compressed groups; it must not reshape the rest."""

    def test_unknown_regime_is_identity_for_groups_below_the_floor(self) -> None:
        weights = {
            FactorGroupType.VALUE: 0.90,
            FactorGroupType.MOMENTUM: 0.08,
            FactorGroupType.SENTIMENT: 0.02,
        }
        result = apply_regime_tilts(
            weights, MacroRegime.UNKNOWN, RegimeTiltConfig(enable=True)
        )
        for group, weight in weights.items():
            assert abs(result[group] - weight) < 1e-12

    def test_zero_weight_group_stays_excluded(self) -> None:
        weights = {
            FactorGroupType.VALUE: 0.5,
            FactorGroupType.MOMENTUM: 0.5,
            FactorGroupType.LIQUIDITY: 0.0,
        }
        config = RegimeTiltConfig(enable=True)
        for regime in MacroRegime:
            result = apply_regime_tilts(weights, regime, config)
            assert result[FactorGroupType.LIQUIDITY] == 0.0, regime
            assert abs(sum(result.values()) - 1.0) < 1e-12

    def test_untilted_small_group_is_not_boosted_by_other_tilts(self) -> None:
        weights = {
            FactorGroupType.VALUE: 0.5,
            FactorGroupType.LOW_RISK: 0.48,
            FactorGroupType.DIVIDEND: 0.02,
        }
        config = RegimeTiltConfig(
            enable=True, recession_tilts=(("low_risk", 1.5), ("value", 0.8))
        )
        result = apply_regime_tilts(weights, MacroRegime.RECESSION, config)
        scale = 1.0 / (0.5 * 0.8 + 0.48 * 1.5 + 0.02)
        assert abs(result[FactorGroupType.DIVIDEND] - 0.02 * scale) < 1e-12

    def test_floor_never_lifts_a_compressed_group_above_its_base_weight(
        self,
    ) -> None:
        weights = {
            FactorGroupType.VALUE: 0.97,
            FactorGroupType.MOMENTUM: 0.03,
        }
        config = RegimeTiltConfig(
            enable=True,
            min_post_tilt_weight=0.10,
            recession_tilts=(("momentum", 0.5),),
        )
        result = apply_regime_tilts(weights, MacroRegime.RECESSION, config)
        # Floor (0.10) exceeds the base weight (0.03): a compressed group is
        # held at its base weight, never raised past it.
        assert abs(result[FactorGroupType.MOMENTUM] - 0.03) < 1e-12
        assert abs(result[FactorGroupType.VALUE] - 0.97) < 1e-12


class TestGdpHeuristicTiesAndSign:
    """The GDP fallback must not read steady growth as a recession or a
    still-contracting economy as an expansion."""

    @staticmethod
    def _gdp(values: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {"gdp_growth": values},
            index=pd.date_range("2023-01-01", periods=len(values), freq="QE"),
        )

    def test_steady_positive_growth_is_expansion(self) -> None:
        assert classify_regime(self._gdp([3.0, 3.0, 3.0, 3.0])) == MacroRegime.EXPANSION

    def test_steady_growth_is_not_split_by_float_rounding(self) -> None:
        for level in (0.1, 2.9, 3.3, 1.7):
            regime = classify_regime(self._gdp([level] * 6))
            assert regime == MacroRegime.EXPANSION, level

    def test_steady_contraction_is_recession(self) -> None:
        assert classify_regime(self._gdp([-1.0, -1.0, -1.0, -1.0])) == (
            MacroRegime.RECESSION
        )

    def test_contraction_that_is_improving_is_recovery_not_expansion(self) -> None:
        macro = self._gdp([-2.0, -1.5, -1.0, -0.5])
        assert classify_regime(macro) == MacroRegime.RECOVERY

    def test_contraction_above_trend_but_worsening_is_recession(self) -> None:
        macro = self._gdp([-3.0, -2.0, -1.0, -1.5])
        assert classify_regime(macro) == MacroRegime.RECESSION

    def test_contraction_is_not_masked_by_rising_unemployment_override(self) -> None:
        macro = self._gdp([-2.0, -1.5, -1.0, -0.5])
        macro["unemployment_rate"] = [4.0, 4.2, 4.5, 4.9]
        assert classify_regime(macro) == MacroRegime.RECOVERY

    def test_sharp_positive_deceleration_stays_recession(self) -> None:
        assert classify_regime(self._gdp([3.0, 2.0, 1.0, 0.5])) == (
            MacroRegime.RECESSION
        )

    def test_mild_deceleration_below_trend_is_a_slowdown_not_a_recession(
        self,
    ) -> None:
        for values in (
            [3.0, 3.0, 3.0, 2.99],
            [3.0, 3.0, 3.0, 2.9],
            [2.5, 3.0, 2.8, 2.4],
        ):
            regime = classify_regime(self._gdp(values))
            assert regime == MacroRegime.SLOWDOWN, values

    def test_a_vanishing_dip_does_not_jump_from_expansion_to_recession(
        self,
    ) -> None:
        steady = classify_regime(self._gdp([3.0, 3.0, 3.0, 3.0]))
        dipped = classify_regime(self._gdp([3.0, 3.0, 3.0, 3.0 - 1e-6]))
        assert steady == MacroRegime.EXPANSION
        assert dipped == MacroRegime.SLOWDOWN

    def test_deceleration_to_under_half_of_trend_is_the_recession_cut(self) -> None:
        # For [4, 4, 4, x] the four-print trend is 3 + x / 4, so growth sits
        # under half of it once x < 12 / 7 (about 1.71).
        assert classify_regime(self._gdp([4.0, 4.0, 4.0, 1.6])) == (
            MacroRegime.RECESSION
        )
        assert classify_regime(self._gdp([4.0, 4.0, 4.0, 1.8])) == (
            MacroRegime.SLOWDOWN
        )

    def test_flat_growth_below_trend_is_a_slowdown(self) -> None:
        assert classify_regime(self._gdp([4.0, 4.0, 2.5, 2.5])) == (
            MacroRegime.SLOWDOWN
        )
