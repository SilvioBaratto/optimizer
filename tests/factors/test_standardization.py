"""Tests for factor standardization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.factors import (
    StandardizationConfig,
    StandardizationMethod,
    WinsorizeMethod,
    neutralize_sector,
    orthogonalize_factors,
    rank_normal_standardize,
    standardize_all_factors,
    standardize_factor,
    winsorize_cross_section,
    winsorize_cross_section_mad,
    z_score_standardize,
)


@pytest.fixture()
def raw_scores() -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(
        rng.normal(10, 5, 100),
        index=[f"T{i:03d}" for i in range(100)],
    )


@pytest.fixture()
def sector_labels() -> pd.Series:
    sectors = ["Tech", "Finance", "Health", "Energy", "Consumer"]
    return pd.Series(
        [sectors[i % 5] for i in range(100)],
        index=[f"T{i:03d}" for i in range(100)],
    )


class TestWinsorize:
    def test_clips_extremes(self, raw_scores: pd.Series) -> None:
        result = winsorize_cross_section(raw_scores, 0.05, 0.95)
        assert result.max() <= raw_scores.quantile(0.95) + 1e-10
        assert result.min() >= raw_scores.quantile(0.05) - 1e-10

    def test_preserves_middle(self, raw_scores: pd.Series) -> None:
        result = winsorize_cross_section(raw_scores, 0.01, 0.99)
        middle = raw_scores.between(
            raw_scores.quantile(0.01),
            raw_scores.quantile(0.99),
        )
        pd.testing.assert_series_equal(result[middle], raw_scores[middle])

    def test_empty_series(self) -> None:
        result = winsorize_cross_section(pd.Series(dtype=float), 0.05, 0.95)
        assert len(result) == 0

    def test_with_nan(self) -> None:
        scores = pd.Series([1.0, np.nan, 100.0, 2.0, 3.0])
        result = winsorize_cross_section(scores, 0.1, 0.9)
        assert pd.isna(result.iloc[1])


class TestZScore:
    def test_mean_zero(self, raw_scores: pd.Series) -> None:
        result = z_score_standardize(raw_scores)
        assert abs(result.mean()) < 1e-10

    def test_std_one(self, raw_scores: pd.Series) -> None:
        result = z_score_standardize(raw_scores)
        assert abs(result.std() - 1.0) < 0.01

    def test_constant_series(self) -> None:
        scores = pd.Series([5.0] * 10)
        result = z_score_standardize(scores)
        assert (result == 0.0).all()


class TestRankNormal:
    def test_shape_preserved(self, raw_scores: pd.Series) -> None:
        result = rank_normal_standardize(raw_scores)
        assert len(result) == len(raw_scores)

    def test_approximately_normal(self, raw_scores: pd.Series) -> None:
        result = rank_normal_standardize(raw_scores)
        valid = result.dropna()
        assert abs(valid.mean()) < 0.1
        assert abs(valid.std() - 1.0) < 0.2

    def test_nan_handling(self) -> None:
        scores = pd.Series([1.0, np.nan, 3.0, 2.0])
        result = rank_normal_standardize(scores)
        assert pd.isna(result.iloc[1])
        assert result.dropna().shape[0] == 3


class TestNeutralizeSector:
    def test_sector_means_zero(
        self, raw_scores: pd.Series, sector_labels: pd.Series
    ) -> None:
        result = neutralize_sector(raw_scores, sector_labels)
        for sector in sector_labels.unique():
            mask = sector_labels == sector
            assert abs(result[mask].mean()) < 1e-10

    def test_preserves_length(
        self, raw_scores: pd.Series, sector_labels: pd.Series
    ) -> None:
        result = neutralize_sector(raw_scores, sector_labels)
        assert len(result) == len(raw_scores)


class TestStandardizeFactor:
    def test_z_score_pipeline(
        self, raw_scores: pd.Series, sector_labels: pd.Series
    ) -> None:
        config = StandardizationConfig(
            method=StandardizationMethod.Z_SCORE,
            neutralize_sector=True,
        )
        result = standardize_factor(raw_scores, config, sector_labels)
        assert len(result) == len(raw_scores)

    def test_rank_normal_pipeline(
        self, raw_scores: pd.Series, sector_labels: pd.Series
    ) -> None:
        config = StandardizationConfig.for_heavy_tailed()
        result = standardize_factor(raw_scores, config, sector_labels)
        assert len(result) == len(raw_scores)

    def test_no_neutralize_z_score(self, raw_scores: pd.Series) -> None:
        config = StandardizationConfig(
            method=StandardizationMethod.Z_SCORE, neutralize_sector=False
        )
        result = standardize_factor(raw_scores, config)
        assert abs(result.mean()) < 1e-10

    def test_no_neutralize_rank_normal(self, raw_scores: pd.Series) -> None:
        config = StandardizationConfig(neutralize_sector=False)
        result = standardize_factor(raw_scores, config)
        assert abs(result.mean()) < 0.1


class TestStandardizeAllFactors:
    def test_returns_scores_and_coverage(self) -> None:
        rng = np.random.default_rng(42)
        factors = pd.DataFrame(
            rng.normal(0, 1, (50, 3)),
            index=[f"T{i:02d}" for i in range(50)],
            columns=["factor_a", "factor_b", "factor_c"],
        )
        scores, coverage = standardize_all_factors(factors)
        assert scores.shape == factors.shape
        assert coverage.shape == factors.shape
        assert coverage.dtypes.apply(pd.api.types.is_bool_dtype).all()

    def test_nan_coverage(self) -> None:
        factors = pd.DataFrame(
            {"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, 3.0]},
            index=["X", "Y", "Z"],
        )
        _scores, coverage = standardize_all_factors(factors)
        assert not coverage.loc["Y", "a"]
        assert not coverage.loc["X", "b"]


class TestReStandardizeAfterNeutralization:
    """Tests for re_standardize_after_neutralization flag (issue #78)."""

    def test_restandardized_zero_mean_unit_std(
        self, raw_scores: pd.Series, sector_labels: pd.Series
    ) -> None:
        config = StandardizationConfig(
            neutralize_sector=True,
            re_standardize_after_neutralization=True,
        )
        result = standardize_factor(raw_scores, config, sector_labels)
        assert abs(result.mean()) < 1e-10
        assert abs(result.std() - 1.0) < 0.05

    def test_default_false_no_change(
        self, raw_scores: pd.Series, sector_labels: pd.Series
    ) -> None:
        config_without = StandardizationConfig(
            neutralize_sector=True,
            re_standardize_after_neutralization=False,
        )
        config_default = StandardizationConfig(neutralize_sector=True)
        result_without = standardize_factor(raw_scores, config_without, sector_labels)
        result_default = standardize_factor(raw_scores, config_default, sector_labels)
        pd.testing.assert_series_equal(result_without, result_default)

    def test_skipped_when_no_neutralization(self, raw_scores: pd.Series) -> None:
        config = StandardizationConfig(
            neutralize_sector=False,
            re_standardize_after_neutralization=True,
        )
        result = standardize_factor(raw_scores, config)
        # Without neutralization, re-standardize flag is ignored;
        # result should be same as plain z-score
        config_plain = StandardizationConfig(neutralize_sector=False)
        result_plain = standardize_factor(raw_scores, config_plain)
        pd.testing.assert_series_equal(result, result_plain)


# ---------------------------------------------------------------------------
# orthogonalize_factors (issue #102)
# ---------------------------------------------------------------------------


class TestOrthogonalizeFactors:
    @pytest.fixture()
    def factor_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            rng.standard_normal((100, 5)),
            index=[f"T{i:03d}" for i in range(100)],
            columns=["f1", "f2", "f3", "f4", "f5"],
        )

    def test_returns_dataframe(self, factor_df: pd.DataFrame) -> None:
        result = orthogonalize_factors(factor_df)
        assert isinstance(result, pd.DataFrame)

    def test_pc_column_names(self, factor_df: pd.DataFrame) -> None:
        result = orthogonalize_factors(factor_df)
        for col in result.columns:
            assert col.startswith("PC")

    def test_orthogonality(self, factor_df: pd.DataFrame) -> None:
        """Off-diagonal correlations should be near zero."""
        result = orthogonalize_factors(factor_df)
        clean = result.dropna()
        corr = clean.corr().to_numpy().copy()
        np.fill_diagonal(corr, 0.0)
        assert np.abs(corr).max() < 0.05

    def test_variance_filtering_reduces_dimensions(self) -> None:
        """Highly correlated factors → fewer PCs retained."""
        rng = np.random.default_rng(42)
        latent = rng.standard_normal(100)
        df = pd.DataFrame(
            {f"f{i}": latent + rng.normal(0, 0.05, 100) for i in range(5)},
            index=[f"T{i:03d}" for i in range(100)],
        )
        result = orthogonalize_factors(df, min_variance_explained=0.95)
        assert result.shape[1] < 5

    def test_preserves_index(self, factor_df: pd.DataFrame) -> None:
        result = orthogonalize_factors(factor_df)
        assert result.index.equals(factor_df.index)

    def test_nan_handling(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.standard_normal((50, 3))
        data[5, :] = np.nan
        data[10, :] = np.nan
        df = pd.DataFrame(data, columns=["a", "b", "c"])
        result = orthogonalize_factors(df)
        assert pd.isna(result.iloc[5]).all()
        assert pd.isna(result.iloc[10]).all()

    def test_single_factor_raises(self) -> None:
        df = pd.DataFrame({"f1": np.random.default_rng(0).standard_normal(50)})
        with pytest.raises(DataError, match="at least 2 factors"):
            orthogonalize_factors(df)

    def test_unsupported_method_raises(self, factor_df: pd.DataFrame) -> None:
        with pytest.raises(ConfigurationError, match="Unsupported"):
            orthogonalize_factors(factor_df, method="ica")

    def test_highly_correlated_collapses_to_one_pc(self) -> None:
        rng = np.random.default_rng(42)
        latent = rng.standard_normal(100)
        df = pd.DataFrame(
            {f"f{i}": latent + rng.normal(0, 0.01, 100) for i in range(5)},
            index=[f"T{i:03d}" for i in range(100)],
        )
        result = orthogonalize_factors(df, min_variance_explained=0.95)
        assert result.shape[1] == 1


# ---------------------------------------------------------------------------
# winsorize_cross_section_mad (issue #244)
# ---------------------------------------------------------------------------


class TestWinsorizeMAD:
    def test_clips_extreme_outlier(self) -> None:
        scores = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        result = winsorize_cross_section_mad(scores)
        assert result.iloc[-1] < 100.0

    def test_moderate_values_unchanged(self) -> None:
        scores = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = winsorize_cross_section_mad(scores)
        pd.testing.assert_series_equal(result, scores)

    def test_empty_series(self) -> None:
        result = winsorize_cross_section_mad(pd.Series(dtype=float))
        assert len(result) == 0

    def test_constant_series_unchanged(self) -> None:
        scores = pd.Series([5.0] * 10)
        result = winsorize_cross_section_mad(scores)
        pd.testing.assert_series_equal(result, scores)

    def test_nan_handling(self) -> None:
        scores = pd.Series([1.0, np.nan, 3.0, 100.0, 2.0])
        result = winsorize_cross_section_mad(scores)
        assert pd.isna(result.iloc[1])
        assert result.iloc[-1] == 2.0  # moderate value unchanged

    def test_multiplier_controls_width(self) -> None:
        scores = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 50.0])
        narrow = winsorize_cross_section_mad(scores, mad_multiplier=1.0)
        wide = winsorize_cross_section_mad(scores, mad_multiplier=5.0)
        # Narrow clips more aggressively
        assert narrow.iloc[-1] < wide.iloc[-1]

    def test_symmetric_clipping(self) -> None:
        scores = pd.Series([-100.0, 1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
        result = winsorize_cross_section_mad(scores)
        med = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]).median()
        assert result.iloc[0] > -100.0
        assert result.iloc[-1] < 100.0
        # Clips are symmetric around median
        assert abs((result.iloc[-1] - med) + (result.iloc[0] - med)) < 1e-10


# ---------------------------------------------------------------------------
# Per-factor standardization dispatch (issue #244)
# ---------------------------------------------------------------------------


class TestStandardizeFactorPerFactor:
    def test_mad_winsorize_path(self) -> None:
        rng = np.random.default_rng(42)
        scores = pd.Series(rng.normal(10, 5, 100))
        config = StandardizationConfig(
            winsorize_method=WinsorizeMethod.MAD,
            neutralize_sector=False,
        )
        result = standardize_factor(scores, config)
        assert abs(result.mean()) < 0.1

    def test_per_factor_override_applies_correct_method(self) -> None:
        rng = np.random.default_rng(42)
        scores = pd.Series(rng.normal(10, 5, 200))
        overrides = (
            ("heavy_factor", StandardizationMethod.RANK_NORMAL.value),
            ("normal_factor", StandardizationMethod.Z_SCORE.value),
        )
        config = StandardizationConfig(
            method=StandardizationMethod.Z_SCORE,
            neutralize_sector=False,
            factor_method_overrides=overrides,
        )
        # With rank_normal override
        result_heavy = standardize_factor(scores, config, factor_name="heavy_factor")
        # With z_score override
        result_normal = standardize_factor(scores, config, factor_name="normal_factor")
        # Results should differ — rank-normal vs z-score produce different values
        assert not np.allclose(
            result_heavy.to_numpy(), result_normal.to_numpy(), atol=1e-6
        )

    def test_unknown_column_falls_back_to_global(self) -> None:
        rng = np.random.default_rng(42)
        scores = pd.Series(rng.normal(10, 5, 100))
        overrides = (("known_factor", StandardizationMethod.Z_SCORE.value),)
        config = StandardizationConfig(
            method=StandardizationMethod.RANK_NORMAL,
            neutralize_sector=False,
            factor_method_overrides=overrides,
        )
        result_unknown = standardize_factor(
            scores, config, factor_name="unknown_factor"
        )
        result_global = standardize_factor(
            scores,
            StandardizationConfig(
                method=StandardizationMethod.RANK_NORMAL,
                neutralize_sector=False,
            ),
        )
        pd.testing.assert_series_equal(result_unknown, result_global)


class TestStandardizeAllFactorsPerFactor:
    def test_per_factor_preset_dispatches_correctly(self) -> None:
        rng = np.random.default_rng(42)
        factors = pd.DataFrame(
            rng.lognormal(0, 1, (200, 2)),
            index=[f"T{i:03d}" for i in range(200)],
            columns=["book_to_price", "momentum_12_1"],
        )
        config = StandardizationConfig.for_per_factor()
        scores_pf, _ = standardize_all_factors(factors, config=config)
        # Compare with uniform RANK_NORMAL
        config_uniform = StandardizationConfig(neutralize_sector=False)
        scores_uniform, _ = standardize_all_factors(factors, config=config_uniform)
        # momentum_12_1 uses Z_SCORE in per_factor → differs from RANK_NORMAL
        assert not np.allclose(
            scores_pf["momentum_12_1"].to_numpy(),
            scores_uniform["momentum_12_1"].to_numpy(),
            atol=1e-6,
        )

    def test_empty_overrides_uses_global_method(self) -> None:
        rng = np.random.default_rng(42)
        factors = pd.DataFrame(
            rng.normal(0, 1, (50, 2)),
            index=[f"T{i:02d}" for i in range(50)],
            columns=["factor_a", "factor_b"],
        )
        config_no_override = StandardizationConfig(
            method=StandardizationMethod.Z_SCORE,
            neutralize_sector=False,
            factor_method_overrides=(),
        )
        config_plain = StandardizationConfig(
            method=StandardizationMethod.Z_SCORE,
            neutralize_sector=False,
        )
        scores_no, _ = standardize_all_factors(factors, config=config_no_override)
        scores_plain, _ = standardize_all_factors(factors, config=config_plain)
        pd.testing.assert_frame_equal(scores_no, scores_plain)


# ---------------------------------------------------------------------------
# Issue #299: FACTOR_DIRECTION applied in standardize_factor()
# ---------------------------------------------------------------------------


class TestFactorDirectionInStandardization:
    """Verify FACTOR_DIRECTION flips scores for lower-is-better factors.

    After standardization, a low-volatility stock must have a *higher*
    standardized score than a high-volatility stock.  The direction flip
    is applied after winsorization and before z-scoring/rank-normal.
    """

    def test_volatility_low_vol_gets_higher_score(self) -> None:
        """After standardization, low-vol stock has higher score than high-vol."""
        rng = np.random.default_rng(0)
        n = 300
        dates = pd.bdate_range("2022-01-01", periods=n)
        low_vol_prices = pd.DataFrame(
            {"LOW": 100 * np.exp(rng.normal(0.0003, 0.005, n).cumsum())},
            index=dates,
        )
        high_vol_prices = pd.DataFrame(
            {"HIGH": 100 * np.exp(rng.normal(0.0003, 0.03, n).cumsum())},
            index=dates,
        )
        from optimizer.factors._construction import _compute_volatility

        raw = _compute_volatility(pd.concat([low_vol_prices, high_vol_prices], axis=1))
        assert raw["HIGH"] > raw["LOW"], "test setup: HIGH should have more vol"

        config = StandardizationConfig(neutralize_sector=False)
        std_scores = standardize_factor(raw, config, factor_name="volatility")
        assert std_scores["LOW"] > std_scores["HIGH"], (
            f"low-vol stock should have higher score: "
            f"LOW={std_scores['LOW']:.4f}, HIGH={std_scores['HIGH']:.4f}"
        )

    def test_beta_low_beta_gets_higher_score(self) -> None:
        """After standardization, low-beta stock has higher score than high-beta."""
        raw = pd.Series({"LOW_BETA": 0.3, "HIGH_BETA": 1.8})
        config = StandardizationConfig(neutralize_sector=False)
        std_scores = standardize_factor(raw, config, factor_name="beta")
        assert std_scores["LOW_BETA"] > std_scores["HIGH_BETA"], (
            f"low-beta stock should have higher score: "
            f"LOW={std_scores['LOW_BETA']:.4f}, HIGH={std_scores['HIGH_BETA']:.4f}"
        )

    def test_asset_growth_low_growth_gets_higher_score(self) -> None:
        """After standardization, low-growth stock has higher score than high-growth."""
        raw = pd.Series({"CONSERVATIVE": -0.05, "AGGRESSIVE": 0.30})
        config = StandardizationConfig(neutralize_sector=False)
        std_scores = standardize_factor(raw, config, factor_name="asset_growth")
        assert std_scores["CONSERVATIVE"] > std_scores["AGGRESSIVE"], (
            f"low-growth stock should have higher score: "
            f"CONSERVATIVE={std_scores['CONSERVATIVE']:.4f}, "
            f"AGGRESSIVE={std_scores['AGGRESSIVE']:.4f}"
        )

    def test_unknown_factor_name_no_direction_flip(self) -> None:
        """Unknown factor name (direction=+1) leaves ordering unchanged."""
        raw = pd.Series({"A": 1.0, "B": 3.0})
        config = StandardizationConfig(neutralize_sector=False)
        std_no_name = standardize_factor(raw, config, factor_name="")
        std_unknown = standardize_factor(raw, config, factor_name="some_new_factor")
        assert std_no_name["B"] > std_no_name["A"]
        assert std_unknown["B"] > std_unknown["A"]

    def test_momentum_no_direction_flip(self) -> None:
        """Momentum (not in FACTOR_DIRECTION) preserves natural ordering."""
        raw = pd.Series({"LOW_MOM": -0.10, "HIGH_MOM": 0.25})
        config = StandardizationConfig(neutralize_sector=False)
        std_scores = standardize_factor(raw, config, factor_name="momentum_12_1")
        assert std_scores["HIGH_MOM"] > std_scores["LOW_MOM"]

    def test_factor_direction_constant_exported(self) -> None:
        """FACTOR_DIRECTION is importable from optimizer.factors."""
        from optimizer.factors import FACTOR_DIRECTION

        assert "volatility" in FACTOR_DIRECTION
        assert "beta" in FACTOR_DIRECTION
        assert "asset_growth" in FACTOR_DIRECTION
        assert FACTOR_DIRECTION["volatility"] == -1
        assert FACTOR_DIRECTION["beta"] == -1
        assert FACTOR_DIRECTION["asset_growth"] == -1

    def test_direction_flip_applied_after_winsorize_before_zscore(self) -> None:
        """Direction flip is applied after winsorization and before z-scoring."""
        rng = np.random.default_rng(42)
        raw = pd.Series(rng.uniform(0.05, 0.80, 20))
        config = StandardizationConfig(
            method=StandardizationMethod.Z_SCORE,
            neutralize_sector=False,
            winsorize_method=WinsorizeMethod.PERCENTILE,
            winsorize_lower=0.01,
            winsorize_upper=0.99,
        )

        # Manual pipeline: winsorize -> negate -> z-score
        from optimizer.factors._standardization import (
            winsorize_cross_section,
            z_score_standardize,
        )

        winsorized = winsorize_cross_section(raw, 0.01, 0.99)
        expected = z_score_standardize(winsorized * -1)

        actual = standardize_factor(raw, config, factor_name="volatility")
        pd.testing.assert_series_equal(actual, expected)
