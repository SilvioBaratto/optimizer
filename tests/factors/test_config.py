"""Tests for factor configuration enums and dataclasses."""

from __future__ import annotations

import pytest

from optimizer.factors import (
    FACTOR_GROUP_MAPPING,
    GROUP_WEIGHT_TIER,
    CompositeMethod,
    CompositeScoringConfig,
    FactorConstructionConfig,
    FactorGroupType,
    FactorIntegrationConfig,
    FactorType,
    FactorValidationConfig,
    GroupWeight,
    MacroRegime,
    RegimeTiltConfig,
    SelectionConfig,
    SelectionMethod,
    StandardizationConfig,
    StandardizationMethod,
)


class TestEnums:
    def test_factor_group_members(self) -> None:
        assert len(FactorGroupType) == 9
        assert set(FactorGroupType) == {
            FactorGroupType.VALUE,
            FactorGroupType.PROFITABILITY,
            FactorGroupType.INVESTMENT,
            FactorGroupType.MOMENTUM,
            FactorGroupType.LOW_RISK,
            FactorGroupType.LIQUIDITY,
            FactorGroupType.DIVIDEND,
            FactorGroupType.SENTIMENT,
            FactorGroupType.OWNERSHIP,
        }

    def test_factor_type_members(self) -> None:
        assert len(FactorType) == 17

    def test_standardization_method_members(self) -> None:
        assert set(StandardizationMethod) == {
            StandardizationMethod.Z_SCORE,
            StandardizationMethod.RANK_NORMAL,
        }

    def test_composite_method_members(self) -> None:
        assert set(CompositeMethod) == {
            CompositeMethod.EQUAL_WEIGHT,
            CompositeMethod.IC_WEIGHTED,
        }

    def test_selection_method_members(self) -> None:
        assert set(SelectionMethod) == {
            SelectionMethod.FIXED_COUNT,
            SelectionMethod.QUANTILE,
        }

    def test_macro_regime_members(self) -> None:
        assert set(MacroRegime) == {
            MacroRegime.EXPANSION,
            MacroRegime.SLOWDOWN,
            MacroRegime.RECESSION,
            MacroRegime.RECOVERY,
        }

    def test_group_weight_members(self) -> None:
        assert set(GroupWeight) == {GroupWeight.CORE, GroupWeight.SUPPLEMENTARY}

    def test_str_serialization(self) -> None:
        assert FactorGroupType.VALUE.value == "value"
        assert FactorType.MOMENTUM_12_1.value == "momentum_12_1"
        assert StandardizationMethod.Z_SCORE.value == "z_score"
        assert MacroRegime.EXPANSION.value == "expansion"


class TestMappingConstants:
    def test_every_factor_mapped(self) -> None:
        for factor in FactorType:
            assert factor in FACTOR_GROUP_MAPPING, (
                f"{factor} not in FACTOR_GROUP_MAPPING"
            )

    def test_every_group_has_weight_tier(self) -> None:
        for group in FactorGroupType:
            assert group in GROUP_WEIGHT_TIER, f"{group} not in GROUP_WEIGHT_TIER"

    def test_core_groups(self) -> None:
        core = {g for g, w in GROUP_WEIGHT_TIER.items() if w == GroupWeight.CORE}
        assert core == {
            FactorGroupType.VALUE,
            FactorGroupType.PROFITABILITY,
            FactorGroupType.MOMENTUM,
            FactorGroupType.LOW_RISK,
        }

    def test_supplementary_groups(self) -> None:
        supp = {
            g for g, w in GROUP_WEIGHT_TIER.items()
            if w == GroupWeight.SUPPLEMENTARY
        }
        assert supp == {
            FactorGroupType.INVESTMENT,
            FactorGroupType.LIQUIDITY,
            FactorGroupType.DIVIDEND,
            FactorGroupType.SENTIMENT,
            FactorGroupType.OWNERSHIP,
        }


class TestFactorConstructionConfig:
    def test_defaults(self) -> None:
        cfg = FactorConstructionConfig()
        assert len(cfg.factors) == 8
        assert cfg.momentum_lookback == 252
        assert cfg.momentum_skip == 21
        assert cfg.publication_lag == 63

    def test_frozen(self) -> None:
        cfg = FactorConstructionConfig()
        with pytest.raises(AttributeError):
            cfg.momentum_lookback = 126  # type: ignore[misc]

    def test_for_core_factors(self) -> None:
        cfg = FactorConstructionConfig.for_core_factors()
        assert len(cfg.factors) == 8

    def test_for_all_factors(self) -> None:
        cfg = FactorConstructionConfig.for_all_factors()
        assert len(cfg.factors) == 17


class TestStandardizationConfig:
    def test_defaults(self) -> None:
        cfg = StandardizationConfig()
        assert cfg.method == StandardizationMethod.Z_SCORE
        assert cfg.winsorize_lower == 0.01
        assert cfg.winsorize_upper == 0.99
        assert cfg.neutralize_sector is True

    def test_frozen(self) -> None:
        cfg = StandardizationConfig()
        with pytest.raises(AttributeError):
            cfg.method = StandardizationMethod.RANK_NORMAL  # type: ignore[misc]

    def test_for_heavy_tailed(self) -> None:
        cfg = StandardizationConfig.for_heavy_tailed()
        assert cfg.method == StandardizationMethod.RANK_NORMAL

    def test_for_normal(self) -> None:
        cfg = StandardizationConfig.for_normal()
        assert cfg.method == StandardizationMethod.Z_SCORE


class TestCompositeScoringConfig:
    def test_defaults(self) -> None:
        cfg = CompositeScoringConfig()
        assert cfg.method == CompositeMethod.EQUAL_WEIGHT
        assert cfg.ic_lookback == 36
        assert cfg.core_weight == 1.0
        assert cfg.supplementary_weight == 0.5

    def test_for_equal_weight(self) -> None:
        cfg = CompositeScoringConfig.for_equal_weight()
        assert cfg.method == CompositeMethod.EQUAL_WEIGHT

    def test_for_ic_weighted(self) -> None:
        cfg = CompositeScoringConfig.for_ic_weighted()
        assert cfg.method == CompositeMethod.IC_WEIGHTED


class TestSelectionConfig:
    def test_defaults(self) -> None:
        cfg = SelectionConfig()
        assert cfg.method == SelectionMethod.FIXED_COUNT
        assert cfg.target_count == 100
        assert cfg.buffer_fraction == 0.1
        assert cfg.sector_balance is True

    def test_for_top_100(self) -> None:
        cfg = SelectionConfig.for_top_100()
        assert cfg.target_count == 100

    def test_for_top_quintile(self) -> None:
        cfg = SelectionConfig.for_top_quintile()
        assert cfg.method == SelectionMethod.QUANTILE
        assert cfg.target_quantile == 0.8

    def test_for_concentrated(self) -> None:
        cfg = SelectionConfig.for_concentrated()
        assert cfg.target_count == 30


class TestRegimeTiltConfig:
    def test_defaults(self) -> None:
        cfg = RegimeTiltConfig()
        assert cfg.enable is False
        assert len(cfg.expansion_tilts) > 0
        assert len(cfg.recession_tilts) > 0

    def test_for_moderate_tilts(self) -> None:
        cfg = RegimeTiltConfig.for_moderate_tilts()
        assert cfg.enable is True

    def test_for_no_tilts(self) -> None:
        cfg = RegimeTiltConfig.for_no_tilts()
        assert cfg.enable is False


class TestFactorValidationConfig:
    def test_defaults(self) -> None:
        cfg = FactorValidationConfig()
        assert cfg.newey_west_lags == 6
        assert cfg.t_stat_threshold == 2.0
        assert cfg.fdr_alpha == 0.05

    def test_for_strict(self) -> None:
        cfg = FactorValidationConfig.for_strict()
        assert cfg.t_stat_threshold == 3.0
        assert cfg.fdr_alpha == 0.01


class TestFactorIntegrationConfig:
    def test_defaults(self) -> None:
        cfg = FactorIntegrationConfig()
        assert cfg.risk_free_rate == 0.04
        assert cfg.market_risk_premium == 0.05
        assert cfg.use_black_litterman is False

    def test_for_linear_mapping(self) -> None:
        cfg = FactorIntegrationConfig.for_linear_mapping()
        assert cfg.use_black_litterman is False

    def test_for_black_litterman(self) -> None:
        cfg = FactorIntegrationConfig.for_black_litterman()
        assert cfg.use_black_litterman is True
