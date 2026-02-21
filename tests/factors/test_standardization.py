"""Tests for factor standardization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.factors import (
    StandardizationConfig,
    StandardizationMethod,
    neutralize_sector,
    orthogonalize_factors,
    rank_normal_standardize,
    standardize_all_factors,
    standardize_factor,
    winsorize_cross_section,
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

    def test_no_neutralize(self, raw_scores: pd.Series) -> None:
        config = StandardizationConfig(neutralize_sector=False)
        result = standardize_factor(raw_scores, config)
        assert abs(result.mean()) < 1e-10


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
