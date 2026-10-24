"""Tests for factor configuration enums and dataclasses."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from optimizer import factors
from optimizer.exceptions import ConfigurationError
from optimizer.factors import (
    FACTOR_DIRECTION,
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
    run_factor_validation,
)

# One alternative value per FactorValidationConfig field, chosen so that
# run_factor_validation on _validation_inputs() produces a different report.
_VALIDATION_PERTURBATIONS: dict[str, float] = {
    "newey_west_lags": 1,
    "t_stat_threshold": 1e6,
    "fdr_alpha": 1.0,
    "n_quantiles": 3,
    # 30 tickers per cross-section, so a floor of 31 drops every date.
    "composite_min_observations": 31,
    "min_ic_observations": 31,
}


def _validation_inputs() -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(3)
    n_dates, n_tickers = 48, 30
    dates = pd.date_range("2020-01-31", periods=n_dates, freq="ME")
    tickers = [f"T{i:02d}" for i in range(n_tickers)]

    def frame(values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(values, index=dates, columns=tickers)

    shape = (n_dates, n_tickers)
    signal = rng.normal(size=shape)
    factor_history = {
        "signal": frame(signal),
        "weak": frame(0.15 * signal + rng.normal(size=shape)),
        "noise": frame(rng.normal(size=shape)),
    }
    returns = frame(0.6 * signal + rng.normal(size=shape))
    composite = frame(signal + rng.normal(size=shape))
    return factor_history, returns, composite


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
        assert len(FactorType) == 18

    def test_factor_type_includes_accruals(self) -> None:
        assert FactorType.ACCRUALS.value == "accruals"

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

    def test_accruals_mapped_to_profitability_group(self) -> None:
        mapping = FACTOR_GROUP_MAPPING[FactorType.ACCRUALS]
        assert mapping == FactorGroupType.PROFITABILITY

    def test_accruals_direction_is_inverted(self) -> None:
        assert FACTOR_DIRECTION[FactorType.ACCRUALS.value] == -1

    def test_accruals_listed_heavy_tailed(self) -> None:
        assert "accruals" in HEAVY_TAILED_FACTORS


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

    def test_annual_lag_feeds_align_to_pit(self) -> None:
        import pandas as pd

        from optimizer.factors import align_to_pit

        statements = pd.DataFrame(
            {
                "ticker": ["A", "A"],
                "period_date": ["2023-12-31", "2024-12-31"],
                "net_income": [1.0, 2.0],
            }
        )
        as_of = "2025-03-31"
        default = align_to_pit(
            statements,
            "period_date",
            as_of,
            lag_days=PublicationLagConfig().annual_days,
        )
        slower = align_to_pit(
            statements,
            "period_date",
            as_of,
            lag_days=PublicationLagConfig(annual_days=120).annual_days,
        )
        assert default.loc["A", "net_income"] == 2.0
        assert slower.loc["A", "net_income"] == 1.0

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
        assert len(cfg.factors) == 11
        assert cfg.momentum_lookback == 252
        assert cfg.momentum_skip == 21

    def test_frozen(self) -> None:
        cfg = FactorConstructionConfig()
        with pytest.raises(AttributeError):
            cfg.momentum_lookback = 126  # type: ignore[misc]

    def test_min_observations_default(self) -> None:
        assert FactorConstructionConfig().min_observations == 20

    @pytest.mark.parametrize("bad", [1, 0, -5])
    def test_min_observations_below_two_rejected(self, bad: int) -> None:
        with pytest.raises(ConfigurationError, match="min_observations"):
            FactorConstructionConfig(min_observations=bad)

    def test_publication_lag_is_not_a_construction_option(self) -> None:
        import dataclasses

        # compute_* receive cross-sectional fundamentals with no period dates,
        # so a lag stored here could never be applied and would only suggest
        # point-in-time protection. Lags belong to align_to_pit(lag_days=...).
        names = {f.name for f in dataclasses.fields(FactorConstructionConfig)}
        assert "publication_lag" not in names
        with pytest.raises(TypeError):
            FactorConstructionConfig(publication_lag=120)  # type: ignore[call-arg]

    @pytest.mark.parametrize(
        "field_name",
        [
            "momentum_lookback",
            "volatility_lookback",
            "beta_lookback",
            "amihud_lookback",
        ],
    )
    @pytest.mark.parametrize("value", [0, -5])
    def test_non_positive_lookback_rejected(self, field_name: str, value: int) -> None:
        with pytest.raises(ConfigurationError, match=field_name):
            FactorConstructionConfig(**{field_name: value})  # type: ignore[arg-type]

    def test_negative_momentum_skip_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="momentum_skip"):
            FactorConstructionConfig(momentum_skip=-3)

    @pytest.mark.parametrize(
        ("lookback", "skip"),
        [(63, 126), (21, 21), (21, 20)],
    )
    def test_momentum_skip_must_leave_a_non_empty_window(
        self, lookback: int, skip: int
    ) -> None:
        # skip >= lookback - 1 makes the start and end prices coincide (zero
        # momentum) or reverse them, and a larger skip raises IndexError later.
        with pytest.raises(ConfigurationError, match="momentum_skip"):
            FactorConstructionConfig(momentum_lookback=lookback, momentum_skip=skip)

    def test_momentum_skip_zero_and_narrow_window_accepted(self) -> None:
        assert FactorConstructionConfig(momentum_skip=0).momentum_skip == 0
        cfg = FactorConstructionConfig(momentum_lookback=22, momentum_skip=20)
        assert cfg.momentum_lookback == 22

    def test_for_core_factors(self) -> None:
        cfg = FactorConstructionConfig.for_core_factors()
        assert len(cfg.factors) == 11

    def test_for_core_factors_covers_cycle2_groups(self) -> None:
        cfg = FactorConstructionConfig.for_core_factors()
        groups = {FACTOR_GROUP_MAPPING[f] for f in cfg.factors}
        assert {
            FactorGroupType.VALUE,
            FactorGroupType.PROFITABILITY,
            FactorGroupType.MOMENTUM,
            FactorGroupType.LOW_RISK,
            FactorGroupType.LIQUIDITY,
        }.issubset(groups)
        assert FactorType.ACCRUALS in cfg.factors
        assert FactorType.CASH_FLOW_YIELD in cfg.factors
        assert FactorType.BETA in cfg.factors
        assert FactorType.AMIHUD_ILLIQUIDITY in cfg.factors

    def test_for_all_factors(self) -> None:
        cfg = FactorConstructionConfig.for_all_factors()
        assert len(cfg.factors) == 18


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


class TestCompositeScoringConfigValidation:
    @pytest.mark.parametrize("ic_lookback", [0, -1, -6])
    def test_non_positive_ic_lookback_raises(self, ic_lookback: int) -> None:
        with pytest.raises(ConfigurationError, match="ic_lookback"):
            CompositeScoringConfig(
                method=CompositeMethod.IC_WEIGHTED, ic_lookback=ic_lookback
            )

    def test_ic_lookback_of_one_is_accepted(self) -> None:
        assert CompositeScoringConfig(ic_lookback=1).ic_lookback == 1

    @pytest.mark.parametrize(
        ("core_weight", "supplementary_weight", "match"),
        [
            (-1.0, 0.5, "core_weight"),
            (1.0, -0.5, "supplementary_weight"),
            (float("nan"), 0.5, "core_weight"),
            (0.0, 0.0, "tier weights"),
        ],
    )
    def test_invalid_tier_weights_raise(
        self, core_weight: float, supplementary_weight: float, match: str
    ) -> None:
        with pytest.raises(ConfigurationError, match=match):
            CompositeScoringConfig(
                core_weight=core_weight, supplementary_weight=supplementary_weight
            )

    @pytest.mark.parametrize(
        ("core_weight", "supplementary_weight"), [(1.0, 0.0), (0.0, 1.0), (2.0, 0.5)]
    )
    def test_single_tier_weight_may_be_zero(
        self, core_weight: float, supplementary_weight: float
    ) -> None:
        cfg = CompositeScoringConfig(
            core_weight=core_weight, supplementary_weight=supplementary_weight
        )
        assert cfg.core_weight == core_weight


class TestSelectionConfigMaxPerSector:
    def test_default_max_per_sector_is_zero(self) -> None:
        cfg = SelectionConfig()
        assert cfg.max_per_sector == 0

    def test_negative_max_per_sector_raises(self) -> None:
        with pytest.raises(Exception, match="max_per_sector"):
            SelectionConfig(max_per_sector=-1)


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


class TestSelectionConfigValidation:
    @pytest.mark.parametrize("target_count", [0, -1, -5])
    def test_non_positive_target_count_raises(self, target_count: int) -> None:
        with pytest.raises(ConfigurationError, match="target_count"):
            SelectionConfig(target_count=target_count)

    @pytest.mark.parametrize(
        ("field_name", "value"),
        [
            ("target_quantile", 1.5),
            ("target_quantile", -0.1),
            ("target_quantile", float("nan")),
            ("exit_quantile", -0.1),
            ("buffer_fraction", -0.5),
            ("buffer_fraction", float("nan")),
            ("sector_tolerance", -0.01),
            ("sector_tolerance", 1.5),
        ],
    )
    def test_out_of_range_fraction_raises(self, field_name: str, value: float) -> None:
        with pytest.raises(ConfigurationError, match=field_name):
            SelectionConfig(**{field_name: value})  # type: ignore[arg-type]

    def test_exit_quantile_above_target_quantile_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="exit_quantile"):
            SelectionConfig(target_quantile=0.7, exit_quantile=0.8)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"target_count": 1},
            {"target_quantile": 1.0, "exit_quantile": 1.0},
            {"target_quantile": 0.0, "exit_quantile": 0.0},
            {"buffer_fraction": 0.0, "sector_tolerance": 0.0},
            {"sector_tolerance": 1.0},
        ],
    )
    def test_boundary_values_are_accepted(self, kwargs: dict[str, float]) -> None:
        SelectionConfig(**kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "preset",
        [
            SelectionConfig.for_top_100,
            SelectionConfig.for_top_quintile,
            SelectionConfig.for_top_20,
            SelectionConfig.for_concentrated,
            SelectionConfig.for_low_tracking_error,
        ],
    )
    def test_presets_satisfy_validation(
        self, preset: Callable[[], SelectionConfig]
    ) -> None:
        assert isinstance(preset(), SelectionConfig)


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

    def test_for_rolling_percentile_keeps_conventional_inversion_threshold(
        self,
    ) -> None:
        import pandas as pd

        spread = pd.Series([-0.5, 0.2, 0.8, 1.2, 2.0])
        cfg = RegimeThresholdConfig.for_rolling_percentile(spread_series=spread)
        # Only the steepness cut-off is distribution-based; inversion is the
        # structural 10Y-2Y = 0 point, not the 35th percentile of the history.
        assert cfg.spread_2s10s_steep == pytest.approx(spread.quantile(0.65))
        assert cfg.spread_2s10s_inversion == 0.0

    def test_for_rolling_percentile_upward_sloping_curve_is_not_inverted(
        self,
    ) -> None:
        import pandas as pd

        from optimizer.factors import classify_regime_composite

        spread = pd.Series([0.1, 0.4, 0.8, 1.2, 1.6, 2.0])
        cfg = RegimeThresholdConfig.for_rolling_percentile(spread_series=spread)
        macro = pd.DataFrame({"spread_2s10s": [0.3]})
        baseline = classify_regime_composite(macro, RegimeThresholdConfig())
        assert classify_regime_composite(macro, cfg) == baseline

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

    def test_periods_per_year_defaults_to_monthly(self) -> None:
        assert FactorValidationConfig().periods_per_year == 12

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_periods_per_year_raises(self, bad: int) -> None:
        with pytest.raises(ConfigurationError, match="periods_per_year"):
            FactorValidationConfig(periods_per_year=bad)

    @pytest.mark.parametrize(
        "field", [f.name for f in dataclasses.fields(FactorValidationConfig)]
    )
    def test_every_field_changes_the_validation_report(self, field: str) -> None:
        # A knob nothing reads (e.g. mimicking-leg fractions, which the
        # mimicking builder takes as its own ``quantile`` argument) would let a
        # caller believe a stricter setting was applied. A field added without
        # an entry in _VALIDATION_PERTURBATIONS fails here on purpose.
        if field == "periods_per_year":
            pytest.skip(
                "only scales the benchmark comparison of factors that have a "
                "published spread; covered in tests/factors/test_validation.py"
            )
        assert field in _VALIDATION_PERTURBATIONS, (
            f"add a perturbation for FactorValidationConfig.{field}"
        )
        factor_history, returns, composite = _validation_inputs()
        base = FactorValidationConfig()
        perturbed = dataclasses.replace(
            base, **{field: _VALIDATION_PERTURBATIONS[field]}
        )

        def report_repr(config: FactorValidationConfig) -> str:
            # repr, not ==: the report holds NaN fields and NaN != NaN.
            return repr(
                run_factor_validation(
                    factor_history,
                    returns,
                    config=config,
                    composite_scores_history=composite,
                )
            )

        assert report_repr(perturbed) != report_repr(base)

    @pytest.mark.parametrize("removed", ["fmp_top_pct", "fmp_bottom_pct"])
    def test_removed_mimicking_leg_fractions_fail_loudly(self, removed: str) -> None:
        with pytest.raises(TypeError):
            FactorValidationConfig(**{removed: 0.1})

    def test_unconsumed_group_ic_aggregation_api_is_not_public(self) -> None:
        # No code path aggregates group ICs by t-stat weight or negative-IC
        # filter, so exporting the switches would make them silent no-ops.
        for name in (
            "GroupICAggregationConfig",
            "GroupICResult",
            "ICWeightingMethod",
            "ICNegativeFilterPolicy",
        ):
            assert name not in factors.__all__
            assert not hasattr(factors, name)


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

    def test_does_not_advertise_an_unenforced_weight_cap(self) -> None:
        """The integration returns a prior, never an optimizer, so it cannot cap.

        A silently ignored risk limit is worse than none: weight caps belong to
        ``MeanRiskConfig.max_weights``.
        """
        names = {f.name for f in dataclasses.fields(FactorIntegrationConfig)}
        assert "max_weight" not in names
        with pytest.raises(TypeError):
            FactorIntegrationConfig(max_weight=0.05)  # type: ignore[call-arg]
