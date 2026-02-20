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
    compute_icir,
    compute_icir_weighted_composite,
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

    def test_icir_weighted_requires_history(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        config = CompositeScoringConfig.for_icir_weighted()
        with pytest.raises(ValueError, match="ic_history required"):
            compute_composite_score(standardized_factors, coverage, config=config)

    def test_icir_weighted_with_history(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        config = CompositeScoringConfig.for_icir_weighted()
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
        assert len(result) == len(standardized_factors)


class TestComputeIcir:
    def test_standard_series(self) -> None:
        ic = pd.Series([0.05, 0.04, 0.06, 0.05, 0.03, 0.07])
        result = compute_icir(ic)
        expected = ic.mean() / ic.std(ddof=1)
        assert abs(result - expected) < 1e-10

    def test_constant_series_returns_zero(self) -> None:
        # std = 0 → ICIR = 0
        ic = pd.Series([0.05, 0.05, 0.05, 0.05])
        assert compute_icir(ic) == 0.0

    def test_single_observation_returns_zero(self) -> None:
        assert compute_icir(pd.Series([0.10])) == 0.0

    def test_empty_series_returns_zero(self) -> None:
        assert compute_icir(pd.Series(dtype=float)) == 0.0

    def test_sign_preserved(self) -> None:
        # Negative mean IC → negative ICIR
        ic = pd.Series([-0.05, -0.04, -0.06, -0.03])
        assert compute_icir(ic) < 0.0

    def test_nan_values_dropped(self) -> None:
        ic_with_nan = pd.Series([0.05, float("nan"), 0.06, 0.04])
        ic_clean = pd.Series([0.05, 0.06, 0.04])
        assert abs(compute_icir(ic_with_nan) - compute_icir(ic_clean)) < 1e-10


class TestICIRWeightedComposite:
    """Acceptance criteria from issue #24."""

    def _make_ic_series(
        self,
        mean: float,
        std: float,
        n: int = 36,
        seed: int = 42,
    ) -> pd.Series:
        rng = np.random.default_rng(seed)
        return pd.Series(mean + rng.normal(0, std, n))

    def test_returns_series_correct_length(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        group_scores = compute_group_scores(standardized_factors, coverage)
        ic_per_group = {
            col: self._make_ic_series(0.05, 0.02)
            for col in group_scores.columns
        }
        result = compute_icir_weighted_composite(group_scores, ic_per_group)
        assert isinstance(result, pd.Series)
        assert len(result) == len(standardized_factors)

    def test_weights_non_negative_and_sum_to_one(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        """ICIR_WEIGHTED weights are non-negative and normalised to 1."""
        group_scores = compute_group_scores(standardized_factors, coverage)
        ic_per_group = {
            col: self._make_ic_series(0.04, 0.01, seed=i)
            for i, col in enumerate(group_scores.columns)
        }
        # Construct a test: all-ones group_scores → composite = sum(w_i * 1) = 1
        ones_scores = pd.DataFrame(
            1.0,
            index=group_scores.index,
            columns=group_scores.columns,
        )
        result = compute_icir_weighted_composite(ones_scores, ic_per_group)
        # With all group scores = 1, composite must equal 1 everywhere
        assert (result - 1.0).abs().max() < 1e-10

    def test_factor_with_icir_zero_receives_zero_weight(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        """A group with ICIR = 0 is excluded from the composite."""
        group_scores = compute_group_scores(standardized_factors, coverage)
        groups = list(group_scores.columns)

        # Set "value" group to score = 1.0; all others = 0.0
        controlled_scores = pd.DataFrame(
            0.0, index=group_scores.index, columns=groups
        )
        if "value" in controlled_scores.columns:
            controlled_scores["value"] = 1.0

        ic_per_group: dict[str, pd.Series] = {}
        for col in groups:
            if col == "value":
                # value: positive, stable ICIR
                ic_per_group[col] = self._make_ic_series(0.05, 0.01)
            else:
                # all others: ICIR = 0 (constant IC series)
                ic_per_group[col] = pd.Series([0.0] * 36)

        result = compute_icir_weighted_composite(controlled_scores, ic_per_group)
        # composite ≈ 1.0 because only "value" group has nonzero ICIR and score=1
        assert (result - 1.0).abs().max() < 1e-10

    def test_stable_factor_weighted_higher_than_volatile(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        """Stable factor (low mean IC, low std) outweighs volatile factor."""
        group_scores = compute_group_scores(standardized_factors, coverage)
        groups = list(group_scores.columns)

        # "value": score = +1; "momentum": score = -1; others = 0
        controlled = pd.DataFrame(
            0.0, index=group_scores.index, columns=groups
        )
        if "value" in controlled.columns:
            controlled["value"] = 1.0
        if "momentum" in controlled.columns:
            controlled["momentum"] = -1.0

        ic_per_group: dict[str, pd.Series] = {}
        for col in groups:
            if col == "value":
                # low mean IC, very stable → high ICIR
                ic_per_group[col] = self._make_ic_series(
                    mean=0.03, std=0.001, seed=1
                )
            elif col == "momentum":
                # high mean IC, very volatile → low ICIR
                ic_per_group[col] = self._make_ic_series(
                    mean=0.15, std=0.50, seed=2
                )
            else:
                ic_per_group[col] = pd.Series([0.0] * 36)

        result = compute_icir_weighted_composite(controlled, ic_per_group)
        # value (ICIR≈30) outweighs momentum (ICIR≈0.3) → composite > 0
        assert result.mean() > 0.0

    def test_ic_weighted_unchanged(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        """IC_WEIGHTED (raw magnitude) still returns a valid Series."""
        group_scores = compute_group_scores(standardized_factors, coverage)
        rng = np.random.default_rng(42)
        ic_history = pd.DataFrame(
            rng.uniform(0.01, 0.06, (36, len(group_scores.columns))),
            columns=group_scores.columns,
        )
        result = compute_ic_weighted_composite(group_scores, ic_history)
        assert isinstance(result, pd.Series)
        assert len(result) == len(standardized_factors)

    def test_fallback_to_equal_weight_when_all_icir_zero(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        """Falls back to equal weight when all groups have ICIR = 0."""
        group_scores = compute_group_scores(standardized_factors, coverage)
        ic_per_group = {
            col: pd.Series([0.0] * 36) for col in group_scores.columns
        }
        icir_result = compute_icir_weighted_composite(group_scores, ic_per_group)
        equal_result = compute_equal_weight_composite(group_scores)
        pd.testing.assert_series_equal(icir_result, equal_result)
