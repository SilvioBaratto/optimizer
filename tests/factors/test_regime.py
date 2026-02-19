"""Tests for macro regime classification and tilts."""

from __future__ import annotations

import pandas as pd
import pytest

from optimizer.factors import (
    FactorGroupType,
    MacroRegime,
    RegimeTiltConfig,
    apply_regime_tilts,
    classify_regime,
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

    def test_empty_data(self) -> None:
        macro = pd.DataFrame()
        result = classify_regime(macro)
        assert result == MacroRegime.EXPANSION  # default

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
