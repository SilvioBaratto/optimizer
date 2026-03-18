"""Tests for factor configuration enums and dataclasses."""

from __future__ import annotations

import pytest

from optimizer.factors import (
    FACTOR_GROUP_MAPPING,
    GROUP_WEIGHT_TIER,
    HEAVY_TAILED_FACTORS,
    CompositeMethod,
    CompositeScoringConfig,
    FactorConstructionConfig,
    FactorGroupType,
    FactorIntegrationConfig,
    FactorType,
    FactorValidationConfig,
    GroupWeight,
    MacroRegime,
    PublicationLagConfig,
    RegimeThresholdConfig,
    RegimeTiltConfig,
    SelectionConfig,
    SelectionMethod,
    StandardizationConfig,
    StandardizationMethod,
    WinsorizeMethod,
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
            CompositeMethod.ICIR_WEIGHTED,
            CompositeMethod.RIDGE_WEIGHTED,
            CompositeMethod.GBT_WEIGHTED,
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
            MacroRegime.UNKNOWN,
        }

    def test_group_weight_members(self) -> None:
        assert set(GroupWeight) == {GroupWeight.CORE, GroupWeight.SUPPLEMENTARY}

    def test_str_serialization(self) -> None:
        assert FactorGroupType.VALUE.value == "value"
        assert FactorType.MOMENTUM_12_1.value == "momentum_12_1"
        assert StandardizationMethod.Z_SCORE.value == "z_score"
        assert MacroRegime.EXPANSION.value == "expansion"
        assert MacroRegime.UNKNOWN.value == "unknown"


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
            g for g, w in GROUP_WEIGHT_TIER.items() if w == GroupWeight.SUPPLEMENTARY
        }
        assert supp == {
            FactorGroupType.INVESTMENT,
            FactorGroupType.LIQUIDITY,
            FactorGroupType.DIVIDEND,
            FactorGroupType.SENTIMENT,
            FactorGroupType.OWNERSHIP,
        }


class TestPublicationLagConfig:
    def test_defaults(self) -> None:
        lag = PublicationLagConfig()
        assert lag.annual_days == 90
        assert lag.quarterly_days == 45
        assert lag.analyst_days == 5
        assert lag.macro_days == 63

    def test_frozen(self) -> None:
        lag = PublicationLagConfig()
        with pytest.raises(AttributeError):
            lag.annual_days = 100  # type: ignore[misc]

    def test_is_hashable(self) -> None:
        lag = PublicationLagConfig()
        assert hash(lag) is not None
        assert {lag, lag} == {lag}

    def test_uniform_applies_same_lag_to_all_sources(self) -> None:
        lag = PublicationLagConfig.uniform(30)
        assert lag.annual_days == 30
        assert lag.quarterly_days == 30
        assert lag.analyst_days == 30
        assert lag.macro_days == 30

    def test_custom_values(self) -> None:
        lag = PublicationLagConfig(annual_days=120, analyst_days=2)
        assert lag.annual_days == 120
        assert lag.analyst_days == 2
        # Other fields stay at defaults
        assert lag.quarterly_days == 45
        assert lag.macro_days == 63


class TestFactorConstructionConfig:
    def test_defaults(self) -> None:
        cfg = FactorConstructionConfig()
        assert len(cfg.factors) == 8
        assert cfg.momentum_lookback == 252
        assert cfg.momentum_skip == 21
        assert isinstance(cfg.publication_lag, PublicationLagConfig)
        assert cfg.publication_lag.annual_days == 90
        assert cfg.publication_lag.quarterly_days == 45
        assert cfg.publication_lag.analyst_days == 5
        assert cfg.publication_lag.macro_days == 63

    def test_frozen(self) -> None:
        cfg = FactorConstructionConfig()
        with pytest.raises(AttributeError):
            cfg.momentum_lookback = 126  # type: ignore[misc]

    def test_publication_lag_int_converted_to_config(self) -> None:
        """Passing a plain int converts to PublicationLagConfig.uniform(n)."""
        cfg = FactorConstructionConfig(publication_lag=100)  # type: ignore[arg-type]
        assert isinstance(cfg.publication_lag, PublicationLagConfig)
        assert cfg.publication_lag.annual_days == 100
        assert cfg.publication_lag.quarterly_days == 100
        assert cfg.publication_lag.analyst_days == 100
        assert cfg.publication_lag.macro_days == 100

    def test_publication_lag_config_accepted(self) -> None:
        lag = PublicationLagConfig(annual_days=60, analyst_days=3)
        cfg = FactorConstructionConfig(publication_lag=lag)
        assert cfg.publication_lag.annual_days == 60
        assert cfg.publication_lag.analyst_days == 3

    def test_for_core_factors(self) -> None:
        cfg = FactorConstructionConfig.for_core_factors()
        assert len(cfg.factors) == 8

    def test_for_all_factors(self) -> None:
        cfg = FactorConstructionConfig.for_all_factors()
        assert len(cfg.factors) == 17


class TestWinsorizeMethodEnum:
    def test_members(self) -> None:
        assert set(WinsorizeMethod) == {
            WinsorizeMethod.PERCENTILE,
            WinsorizeMethod.MAD,
        }

    def test_values(self) -> None:
        assert WinsorizeMethod.PERCENTILE.value == "percentile"
        assert WinsorizeMethod.MAD.value == "mad"


class TestHeavyTailedFactors:
    def test_contains_expected(self) -> None:
        expected = {
            "book_to_price",
            "earnings_yield",
            "cash_flow_yield",
            "sales_to_price",
            "ebitda_to_ev",
            "asset_growth",
            "dividend_yield",
            "amihud_illiquidity",
            "accruals",
        }
        assert expected == HEAVY_TAILED_FACTORS

    def test_is_frozenset(self) -> None:
        assert isinstance(HEAVY_TAILED_FACTORS, frozenset)

    def test_normal_factors_excluded(self) -> None:
        assert "momentum_12_1" not in HEAVY_TAILED_FACTORS
        assert "volatility" not in HEAVY_TAILED_FACTORS
        assert "beta" not in HEAVY_TAILED_FACTORS


class TestStandardizationConfig:
    def test_defaults(self) -> None:
        cfg = StandardizationConfig()
        assert cfg.method == StandardizationMethod.RANK_NORMAL
        assert cfg.winsorize_method == WinsorizeMethod.PERCENTILE
        assert cfg.winsorize_lower == 0.01
        assert cfg.winsorize_upper == 0.99
        assert cfg.neutralize_sector is True
        assert cfg.factor_method_overrides == ()

    def test_frozen(self) -> None:
        cfg = StandardizationConfig()
        with pytest.raises(AttributeError):
            cfg.method = StandardizationMethod.RANK_NORMAL  # type: ignore[misc]

    def test_is_hashable(self) -> None:
        cfg = StandardizationConfig()
        assert hash(cfg) is not None
        assert {cfg, cfg} == {cfg}

    def test_for_heavy_tailed(self) -> None:
        cfg = StandardizationConfig.for_heavy_tailed()
        assert cfg.method == StandardizationMethod.RANK_NORMAL

    def test_for_normal(self) -> None:
        cfg = StandardizationConfig.for_normal()
        assert cfg.method == StandardizationMethod.Z_SCORE

    def test_for_z_score(self) -> None:
        cfg = StandardizationConfig.for_z_score()
        assert cfg.method == StandardizationMethod.Z_SCORE

    def test_for_mad_winsorize(self) -> None:
        cfg = StandardizationConfig.for_mad_winsorize()
        assert cfg.winsorize_method == WinsorizeMethod.MAD
        assert cfg.method == StandardizationMethod.RANK_NORMAL

    def test_for_per_factor_heavy_tailed_get_rank_normal(self) -> None:
        cfg = StandardizationConfig.for_per_factor()
        overrides = dict(cfg.factor_method_overrides)
        for name in ["book_to_price", "earnings_yield", "amihud_illiquidity"]:
            assert overrides[name] == StandardizationMethod.RANK_NORMAL.value

    def test_for_per_factor_normal_get_z_score(self) -> None:
        cfg = StandardizationConfig.for_per_factor()
        overrides = dict(cfg.factor_method_overrides)
        for name in ["momentum_12_1", "volatility", "beta"]:
            assert overrides[name] == StandardizationMethod.Z_SCORE.value

    def test_for_per_factor_covers_all_factor_types(self) -> None:
        cfg = StandardizationConfig.for_per_factor()
        overrides = dict(cfg.factor_method_overrides)
        for ft in FactorType:
            assert ft.value in overrides

    def test_for_per_factor_is_hashable(self) -> None:
        cfg = StandardizationConfig.for_per_factor()
        assert hash(cfg) is not None

    def test_for_per_factor_is_frozen(self) -> None:
        cfg = StandardizationConfig.for_per_factor()
        with pytest.raises(AttributeError):
            cfg.factor_method_overrides = ()  # type: ignore[misc]


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

    def test_for_icir_weighted(self) -> None:
        cfg = CompositeScoringConfig.for_icir_weighted()
        assert cfg.method == CompositeMethod.ICIR_WEIGHTED

    def test_min_coverage_groups_default(self) -> None:
        cfg = CompositeScoringConfig()
        assert cfg.min_coverage_groups == 0

    def test_return_coverage_default(self) -> None:
        cfg = CompositeScoringConfig()
        assert cfg.return_coverage is False

    def test_for_sparse_universe_preset(self) -> None:
        cfg = CompositeScoringConfig.for_sparse_universe()
        assert cfg.min_coverage_groups == 2

    def test_for_coverage_diagnostics_preset(self) -> None:
        cfg = CompositeScoringConfig.for_coverage_diagnostics()
        assert cfg.return_coverage is True

    def test_frozen_new_fields(self) -> None:
        cfg = CompositeScoringConfig()
        with pytest.raises(AttributeError):
            cfg.min_coverage_groups = 3  # type: ignore[misc]
        with pytest.raises(AttributeError):
            cfg.return_coverage = True  # type: ignore[misc]

    def test_hashable_with_new_fields(self) -> None:
        cfg = CompositeScoringConfig(min_coverage_groups=3, return_coverage=True)
        assert hash(cfg) is not None


class TestSelectionConfig:
    def test_defaults(self) -> None:
        cfg = SelectionConfig()
        assert cfg.method == SelectionMethod.FIXED_COUNT
        assert cfg.target_count == 100
        assert cfg.buffer_fraction == 0.1
        assert cfg.sector_balance is True
        assert cfg.sector_tolerance == 0.05

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

    def test_for_low_tracking_error(self) -> None:
        cfg = SelectionConfig.for_low_tracking_error()
        assert cfg.sector_tolerance == 0.03
        assert cfg.target_count == 100


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


class TestRegimeThresholdConfig:
    def test_defaults(self) -> None:
        cfg = RegimeThresholdConfig()
        assert cfg.hy_oas_risk_on == 350.0
        assert cfg.hy_oas_risk_off == 500.0
        assert cfg.pmi_expansion == 52.0
        assert cfg.pmi_contraction == 48.0
        assert cfg.spread_2s10s_steep == 1.0
        assert cfg.spread_2s10s_inversion == 0.0
        assert cfg.sentiment_positive == 0.3
        assert cfg.sentiment_negative == -0.3

    def test_frozen(self) -> None:
        cfg = RegimeThresholdConfig()
        with pytest.raises(AttributeError):
            cfg.hy_oas_risk_on = 400.0  # type: ignore[misc]

    def test_is_hashable(self) -> None:
        cfg = RegimeThresholdConfig()
        assert hash(cfg) is not None
        assert {cfg, cfg} == {cfg}

    def test_for_empirical_returns_defaults(self) -> None:
        assert RegimeThresholdConfig.for_empirical() == RegimeThresholdConfig()

    def test_for_rolling_percentile_no_data_returns_defaults(self) -> None:
        cfg = RegimeThresholdConfig.for_rolling_percentile()
        assert cfg == RegimeThresholdConfig()

    def test_for_rolling_percentile_with_hy_series(self) -> None:
        import pandas as pd

        hy = pd.Series([200.0, 300.0, 400.0, 500.0, 600.0])
        cfg = RegimeThresholdConfig.for_rolling_percentile(
            hy_series=hy,
            hy_risk_on_pct=0.40,
            hy_risk_off_pct=0.75,
        )
        assert cfg.hy_oas_risk_on == pytest.approx(hy.quantile(0.40))
        assert cfg.hy_oas_risk_off == pytest.approx(hy.quantile(0.75))
        # Unaffected fields stay at defaults
        assert cfg.pmi_expansion == 52.0

    def test_custom_values(self) -> None:
        cfg = RegimeThresholdConfig(hy_oas_risk_on=300.0, pmi_expansion=53.0)
        assert cfg.hy_oas_risk_on == 300.0
        assert cfg.pmi_expansion == 53.0
        assert cfg.hy_oas_risk_off == 500.0  # unchanged default


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
