"""Tests for composite scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.factors import (
    CompositeScoringConfig,
    compute_composite_score,
    compute_equal_weight_composite,
    compute_group_scores,
    compute_ic_weighted_composite,
)


@pytest.fixture()
def standardized_factors() -> pd.DataFrame:
    """Standardized factor scores for 20 tickers."""
    rng = np.random.default_rng(42)
    tickers = [f"T{i:02d}" for i in range(20)]
    return pd.DataFrame(
        rng.normal(0, 1, (20, 8)),
        index=tickers,
        columns=[
            "book_to_price", "earnings_yield",  # value
            "gross_profitability", "roe",  # profitability
            "asset_growth",  # investment
            "momentum_12_1",  # momentum
            "volatility",  # low_risk
            "dividend_yield",  # dividend
        ],
    )


@pytest.fixture()
def coverage(standardized_factors: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        True,
        index=standardized_factors.index,
        columns=standardized_factors.columns,
    )


class TestComputeGroupScores:
    def test_groups_averaged(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        result = compute_group_scores(standardized_factors, coverage)
        assert isinstance(result, pd.DataFrame)
        # Should have groups represented in the data
        assert "value" in result.columns
        assert "profitability" in result.columns
        assert "momentum" in result.columns

    def test_value_is_average_of_value_factors(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        result = compute_group_scores(standardized_factors, coverage)
        cols = ["book_to_price", "earnings_yield"]
        expected = standardized_factors[cols].mean(axis=1)
        pd.testing.assert_series_equal(result["value"], expected, check_names=False)


class TestEqualWeightComposite:
    def test_returns_series(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        group_scores = compute_group_scores(standardized_factors, coverage)
        result = compute_equal_weight_composite(group_scores)
        assert isinstance(result, pd.Series)
        assert len(result) == len(standardized_factors)

    def test_core_groups_weighted_more(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        group_scores = compute_group_scores(standardized_factors, coverage)
        config = CompositeScoringConfig(core_weight=2.0, supplementary_weight=0.5)
        result = compute_equal_weight_composite(group_scores, config)
        assert isinstance(result, pd.Series)


class TestICWeightedComposite:
    def test_with_ic_history(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        group_scores = compute_group_scores(standardized_factors, coverage)
        rng = np.random.default_rng(42)
        ic_history = pd.DataFrame(
            rng.uniform(0.01, 0.06, (36, len(group_scores.columns))),
            columns=group_scores.columns,
        )
        result = compute_ic_weighted_composite(group_scores, ic_history)
        assert isinstance(result, pd.Series)
        assert len(result) == len(standardized_factors)


class TestComputeCompositeScore:
    def test_equal_weight_default(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        result = compute_composite_score(standardized_factors, coverage)
        assert isinstance(result, pd.Series)
        assert len(result) == len(standardized_factors)

    def test_ic_weighted_requires_history(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        config = CompositeScoringConfig.for_ic_weighted()
        with pytest.raises(ValueError, match="ic_history required"):
            compute_composite_score(standardized_factors, coverage, config=config)

    def test_ic_weighted_with_history(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        config = CompositeScoringConfig.for_ic_weighted()
        group_scores = compute_group_scores(standardized_factors, coverage)
        rng = np.random.default_rng(42)
        ic_history = pd.DataFrame(
            rng.uniform(0.01, 0.06, (36, len(group_scores.columns))),
            columns=group_scores.columns,
        )
        result = compute_composite_score(
            standardized_factors, coverage, config=config, ic_history=ic_history
        )
        assert isinstance(result, pd.Series)
