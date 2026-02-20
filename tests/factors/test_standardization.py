"""Tests for factor standardization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.factors import (
    StandardizationConfig,
    StandardizationMethod,
    neutralize_sector,
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
        pd.testing.assert_series_equal(
            result[middle], raw_scores[middle]
        )

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
        scores, coverage = standardize_all_factors(factors)
        assert not coverage.loc["Y", "a"]
        assert not coverage.loc["X", "b"]
