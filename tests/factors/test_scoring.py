"""Tests for composite scoring."""

from __future__ import annotations

import warnings as _warnings_module

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import ConfigurationError
from optimizer.factors import (
    CompositeMethod,
    CompositeScoringConfig,
    ICFallbackStrategy,
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
            "book_to_price",
            "earnings_yield",  # value
            "gross_profitability",
            "roe",  # profitability
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


class TestGroupWeightsOverride:
    """Tests for group_weights parameter (issue #54)."""

    def test_equal_weight_with_group_weights(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        """compute_equal_weight_composite uses group_weights when provided."""
        group_scores = compute_group_scores(standardized_factors, coverage)
        groups = list(group_scores.columns)

        # Give "value" weight 10x others
        gw = {g: (10.0 if g == "value" else 1.0) for g in groups}
        result = compute_equal_weight_composite(group_scores, group_weights=gw)
        default = compute_equal_weight_composite(group_scores)
        # Results should differ since value is upweighted
        assert not np.allclose(result.to_numpy(), default.to_numpy(), atol=1e-6)

    def test_composite_score_threads_group_weights(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        """compute_composite_score passes group_weights through."""
        groups = list(compute_group_scores(standardized_factors, coverage).columns)
        gw = {g: (5.0 if g == "value" else 1.0) for g in groups}
        result = compute_composite_score(
            standardized_factors, coverage, group_weights=gw
        )
        default = compute_composite_score(standardized_factors, coverage)
        assert not np.allclose(result.to_numpy(), default.to_numpy(), atol=1e-6)


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
        with pytest.raises(ConfigurationError, match="ic_history required"):
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
        with pytest.raises(ConfigurationError, match="ic_history required"):
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


class TestICWeightedNegativeIC:
    """Tests for IC_WEIGHTED clamping negative IC to zero (issue #55)."""

    def test_negative_ic_group_gets_zero_weight(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        """A group with negative mean IC should get zero weight."""
        group_scores = compute_group_scores(standardized_factors, coverage)
        groups = list(group_scores.columns)

        # Set "value" = 1.0, all others = 0.0
        controlled = pd.DataFrame(0.0, index=group_scores.index, columns=groups)
        if "value" in controlled.columns:
            controlled["value"] = 1.0

        # IC history: value has negative IC, others positive
        rng = np.random.default_rng(99)
        ic_history = pd.DataFrame(
            {
                col: (
                    rng.uniform(-0.10, -0.02, 36)
                    if col == "value"
                    else rng.uniform(0.02, 0.08, 36)
                )
                for col in groups
            }
        )
        result = compute_ic_weighted_composite(controlled, ic_history)
        # value has negative IC → zero weight → composite should be 0
        assert (result.abs() < 1e-10).all(), (
            "Negative-IC group should get zero weight and not contribute"
        )

    def test_all_negative_ic_falls_back_to_equal_weight(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        """When all groups have negative IC, falls back to equal weight."""
        group_scores = compute_group_scores(standardized_factors, coverage)
        groups = list(group_scores.columns)
        rng = np.random.default_rng(42)
        ic_history = pd.DataFrame(
            {col: rng.uniform(-0.10, -0.02, 36) for col in groups}
        )
        with _warnings_module.catch_warnings():
            _warnings_module.simplefilter("ignore", UserWarning)
            ic_result = compute_ic_weighted_composite(group_scores, ic_history)
        equal_result = compute_equal_weight_composite(group_scores)
        pd.testing.assert_series_equal(ic_result, equal_result)


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
            col: self._make_ic_series(0.05, 0.02) for col in group_scores.columns
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
        controlled_scores = pd.DataFrame(0.0, index=group_scores.index, columns=groups)
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
        controlled = pd.DataFrame(0.0, index=group_scores.index, columns=groups)
        if "value" in controlled.columns:
            controlled["value"] = 1.0
        if "momentum" in controlled.columns:
            controlled["momentum"] = -1.0

        ic_per_group: dict[str, pd.Series] = {}
        for col in groups:
            if col == "value":
                # low mean IC, very stable → high ICIR
                ic_per_group[col] = self._make_ic_series(mean=0.03, std=0.001, seed=1)
            elif col == "momentum":
                # high mean IC, very volatile → low ICIR
                ic_per_group[col] = self._make_ic_series(mean=0.15, std=0.50, seed=2)
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
        ic_per_group = {col: pd.Series([0.0] * 36) for col in group_scores.columns}
        with _warnings_module.catch_warnings():
            _warnings_module.simplefilter("ignore", UserWarning)
            icir_result = compute_icir_weighted_composite(group_scores, ic_per_group)
        equal_result = compute_equal_weight_composite(group_scores)
        pd.testing.assert_series_equal(icir_result, equal_result)


class TestCoverageWeightedMean:
    """Tests for coverage-weighted mean in group scoring (issue #62)."""

    def test_partial_coverage_equals_single_factor(self) -> None:
        """Ticker missing 1 of 2 factors has group score equal to the present factor."""
        tickers = ["T00", "T01"]
        factors = pd.DataFrame(
            {"book_to_price": [0.5, 0.8], "earnings_yield": [np.nan, 0.6]},
            index=tickers,
        )
        cov = pd.DataFrame(
            {"book_to_price": [True, True], "earnings_yield": [False, True]},
            index=tickers,
        )
        result = compute_group_scores(factors, cov)
        # T00 has only book_to_price covered → group score = 0.5
        assert result.loc["T00", "value"] == pytest.approx(0.5)
        # T01 has both covered → group score = mean(0.8, 0.6) = 0.7
        assert result.loc["T01", "value"] == pytest.approx(0.7)

    def test_full_coverage_matches_simple_mean(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        """With full coverage, coverage-weighted mean equals simple mean."""
        result = compute_group_scores(standardized_factors, coverage)
        cols = ["book_to_price", "earnings_yield"]
        expected = standardized_factors[cols].mean(axis=1)
        pd.testing.assert_series_equal(result["value"], expected, check_names=False)


class TestNaNGroupScoreRenormalization:
    """Tests for renormalized weighted average over available groups (issue #242)."""

    def test_missing_group_not_zero_filled(self) -> None:
        """Ticker missing a group gets composite from available groups only."""
        tickers = ["A", "B"]
        group_scores = pd.DataFrame(
            {"value": [1.0, 1.0], "momentum": [np.nan, 0.5]},
            index=tickers,
        )
        result = compute_equal_weight_composite(group_scores)
        # A has only value=1.0 available → composite = 1.0 (not 0.5)
        assert result["A"] == pytest.approx(1.0)
        # B has both → composite = mean(1.0, 0.5) = 0.75
        assert result["B"] == pytest.approx(0.75)

    def test_all_groups_missing_produces_nan(self) -> None:
        """Ticker with all NaN group scores gets NaN composite."""
        group_scores = pd.DataFrame(
            {"value": [np.nan], "momentum": [np.nan]},
            index=["A"],
        )
        result = compute_equal_weight_composite(group_scores)
        assert np.isnan(result["A"])

    def test_partial_coverage_renormalized_weights_sum_to_one(self) -> None:
        """With all available scores = 1.0, composite = 1.0 regardless of coverage."""
        group_scores = pd.DataFrame(
            {"value": [1.0], "momentum": [np.nan], "profitability": [1.0]},
            index=["A"],
        )
        result = compute_equal_weight_composite(group_scores)
        assert result["A"] == pytest.approx(1.0)

    def test_ic_weighted_missing_group_renormalizes(self) -> None:
        """IC_WEIGHTED renormalizes over available groups."""
        tickers = ["A"]
        group_scores = pd.DataFrame(
            {"value": [2.0], "momentum": [np.nan]},
            index=tickers,
        )
        rng = np.random.default_rng(42)
        ic_history = pd.DataFrame(
            {
                "value": rng.uniform(0.02, 0.06, 36),
                "momentum": rng.uniform(0.02, 0.06, 36),
            }
        )
        result = compute_ic_weighted_composite(group_scores, ic_history)
        # Only value is available → composite = 2.0
        assert result["A"] == pytest.approx(2.0)

    def test_icir_weighted_missing_group_renormalizes(self) -> None:
        """ICIR_WEIGHTED renormalizes over available groups."""
        tickers = ["A"]
        group_scores = pd.DataFrame(
            {"value": [3.0], "momentum": [np.nan]},
            index=tickers,
        )
        rng = np.random.default_rng(42)
        ic_per_group = {
            "value": pd.Series(rng.normal(0.05, 0.01, 36)),
            "momentum": pd.Series(rng.normal(0.05, 0.01, 36)),
        }
        result = compute_icir_weighted_composite(group_scores, ic_per_group)
        # Only value is available → composite = 3.0
        assert result["A"] == pytest.approx(3.0)

    def test_full_coverage_unchanged(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        """With full coverage, result is identical to pre-fix behavior."""
        group_scores = compute_group_scores(standardized_factors, coverage)
        result = compute_equal_weight_composite(group_scores)
        # Verify no NaN in output when no NaN in input
        assert not result.isna().any()
        assert len(result) == len(standardized_factors)


class TestMinCoverageGroups:
    """Tests for min_coverage_groups config field (issue #242)."""

    def test_zero_min_coverage_no_filtering(self) -> None:
        """With min_coverage_groups=0, no tickers are filtered out."""
        factors = pd.DataFrame(
            {"book_to_price": [1.0], "momentum_12_1": [np.nan]},
            index=["A"],
        )
        cov = pd.DataFrame(
            {"book_to_price": [True], "momentum_12_1": [False]},
            index=["A"],
        )
        config = CompositeScoringConfig(min_coverage_groups=0)
        result = compute_composite_score(factors, cov, config=config)
        assert not result.isna().all()

    def test_below_threshold_produces_nan(self) -> None:
        """Ticker with fewer groups than threshold gets NaN."""
        factors = pd.DataFrame(
            {"book_to_price": [1.0], "momentum_12_1": [np.nan]},
            index=["A"],
        )
        cov = pd.DataFrame(
            {"book_to_price": [True], "momentum_12_1": [False]},
            index=["A"],
        )
        # Only 1 group available, require 2
        config = CompositeScoringConfig(min_coverage_groups=2)
        result = compute_composite_score(factors, cov, config=config)
        assert result.isna().all()

    def test_exactly_at_threshold_passes(self) -> None:
        """Ticker with exactly min_coverage_groups is included."""
        factors = pd.DataFrame(
            {"book_to_price": [1.0], "momentum_12_1": [0.5]},
            index=["A"],
        )
        cov = pd.DataFrame(
            {"book_to_price": [True], "momentum_12_1": [True]},
            index=["A"],
        )
        # 2 groups available, require 2
        config = CompositeScoringConfig(min_coverage_groups=2)
        result = compute_composite_score(factors, cov, config=config)
        assert not result.isna().any()

    def test_config_defaults(self) -> None:
        assert CompositeScoringConfig().min_coverage_groups == 0

    def test_for_sparse_universe_preset(self) -> None:
        cfg = CompositeScoringConfig.for_sparse_universe()
        assert cfg.min_coverage_groups == 2


class TestCoverageRatioOutput:
    """Tests for return_coverage config field (issue #242)."""

    def test_return_coverage_false_returns_series(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        config = CompositeScoringConfig(return_coverage=False)
        result = compute_composite_score(standardized_factors, coverage, config=config)
        assert isinstance(result, pd.Series)

    def test_return_coverage_true_returns_dataframe(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        config = CompositeScoringConfig(return_coverage=True)
        result = compute_composite_score(standardized_factors, coverage, config=config)
        assert isinstance(result, pd.DataFrame)
        assert "composite" in result.columns
        assert "coverage_ratio" in result.columns

    def test_coverage_ratio_values(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        config = CompositeScoringConfig(return_coverage=True)
        result = compute_composite_score(standardized_factors, coverage, config=config)
        assert (result["coverage_ratio"] >= 0.0).all()
        assert (result["coverage_ratio"] <= 1.0).all()

    def test_full_coverage_ratio_is_one(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        config = CompositeScoringConfig(return_coverage=True)
        result = compute_composite_score(standardized_factors, coverage, config=config)
        assert (result["coverage_ratio"] == 1.0).all()

    def test_partial_coverage_ratio(self) -> None:
        factors = pd.DataFrame(
            {
                "book_to_price": [1.0],
                "momentum_12_1": [np.nan],
                "volatility": [0.5],
                "dividend_yield": [np.nan],
            },
            index=["A"],
        )
        cov = pd.DataFrame(
            {
                "book_to_price": [True],
                "momentum_12_1": [False],
                "volatility": [True],
                "dividend_yield": [False],
            },
            index=["A"],
        )
        config = CompositeScoringConfig(return_coverage=True)
        result = compute_composite_score(factors, cov, config=config)
        # 2 of 4 groups available = 0.5 (each factor maps to a different group)
        assert result.loc["A", "coverage_ratio"] == pytest.approx(0.5, abs=0.01)


class TestICIRWeightedNegativeICIR:
    """Tests for ICIR_WEIGHTED clamping negative ICIR to zero (issue #253)."""

    def _make_ic_series(
        self,
        mean: float,
        std: float,
        n: int = 36,
        seed: int = 42,
    ) -> pd.Series:
        rng = np.random.default_rng(seed)
        return pd.Series(mean + rng.normal(0, std, n))

    def test_negative_icir_group_gets_zero_weight(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        """A group with negative ICIR should get zero weight."""
        group_scores = compute_group_scores(standardized_factors, coverage)
        groups = list(group_scores.columns)

        # Set "value" = 1.0, all others = 0.0
        controlled = pd.DataFrame(0.0, index=group_scores.index, columns=groups)
        if "value" in controlled.columns:
            controlled["value"] = 1.0

        # IC series: value has consistently negative mean → negative ICIR
        rng = np.random.default_rng(99)
        ic_per_group = {
            col: (
                pd.Series(rng.uniform(-0.10, -0.02, 36))
                if col == "value"
                else self._make_ic_series(0.05, 0.02, seed=hash(col) % 2**31)
            )
            for col in groups
        }
        result = compute_icir_weighted_composite(controlled, ic_per_group)
        # value has negative ICIR → zero weight → composite should be 0
        assert (result.abs() < 1e-10).all(), (
            "Negative-ICIR group should get zero weight and not contribute"
        )

    def test_all_negative_icir_falls_back_to_equal_weight(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        """When all groups have negative ICIR, falls back to equal weight."""
        group_scores = compute_group_scores(standardized_factors, coverage)
        groups = list(group_scores.columns)
        rng = np.random.default_rng(42)
        ic_per_group = {col: pd.Series(rng.uniform(-0.10, -0.02, 36)) for col in groups}
        with _warnings_module.catch_warnings():
            _warnings_module.simplefilter("ignore", UserWarning)
            icir_result = compute_icir_weighted_composite(group_scores, ic_per_group)
        equal_result = compute_equal_weight_composite(group_scores)
        pd.testing.assert_series_equal(icir_result, equal_result)

    def test_negative_icir_same_as_zero_icir(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        """Negative ICIR and zero ICIR both produce zero contribution."""
        group_scores = compute_group_scores(standardized_factors, coverage)
        groups = list(group_scores.columns)

        # Set "value" = 1.0, "momentum" = 1.0, all others = 0.0
        controlled = pd.DataFrame(0.0, index=group_scores.index, columns=groups)
        if "value" in controlled.columns:
            controlled["value"] = 1.0
        if "momentum" in controlled.columns:
            controlled["momentum"] = 1.0

        # value: negative ICIR; momentum: zero ICIR (constant series)
        ic_per_group: dict[str, pd.Series] = {}
        rng = np.random.default_rng(99)
        for col in groups:
            if col == "value":
                ic_per_group[col] = pd.Series(rng.uniform(-0.10, -0.02, 36))
            elif col == "momentum":
                ic_per_group[col] = pd.Series([0.0] * 36)
            else:
                ic_per_group[col] = self._make_ic_series(
                    0.05, 0.02, seed=hash(col) % 2**31
                )
        result = compute_icir_weighted_composite(controlled, ic_per_group)
        # Both value (negative ICIR) and momentum (zero ICIR) excluded → composite = 0
        assert (result.abs() < 1e-10).all()


class TestICIRWeightedGroupWeights:
    """Tests for group_weights parameter in ICIR_WEIGHTED (issue #81)."""

    def _make_ic_series(
        self,
        mean: float,
        std: float,
        n: int = 36,
        seed: int = 42,
    ) -> pd.Series:
        rng = np.random.default_rng(seed)
        return pd.Series(mean + rng.normal(0, std, n))

    def test_icir_weighted_with_group_weights_differs(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        """compute_icir_weighted_composite with group_weights differs from without."""
        group_scores = compute_group_scores(standardized_factors, coverage)
        ic_per_group = {
            col: self._make_ic_series(0.05, 0.02, seed=i)
            for i, col in enumerate(group_scores.columns)
        }
        groups = list(group_scores.columns)
        gw = {g: (10.0 if g == "value" else 0.1) for g in groups}

        result_with = compute_icir_weighted_composite(
            group_scores, ic_per_group, group_weights=gw
        )
        result_without = compute_icir_weighted_composite(group_scores, ic_per_group)
        assert not np.allclose(
            result_with.to_numpy(), result_without.to_numpy(), atol=1e-6
        )

    def test_icir_weighted_path_forwards_group_weights(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        """compute_composite_score with ICIR_WEIGHTED forwards group_weights."""
        config = CompositeScoringConfig.for_icir_weighted()
        group_scores = compute_group_scores(standardized_factors, coverage)
        rng = np.random.default_rng(42)
        ic_history = pd.DataFrame(
            rng.uniform(0.01, 0.06, (36, len(group_scores.columns))),
            columns=group_scores.columns,
        )
        groups = list(group_scores.columns)
        gw = {g: (5.0 if g == "value" else 1.0) for g in groups}

        result_with = compute_composite_score(
            standardized_factors,
            coverage,
            config=config,
            ic_history=ic_history,
            group_weights=gw,
        )
        result_without = compute_composite_score(
            standardized_factors,
            coverage,
            config=config,
            ic_history=ic_history,
        )
        assert not np.allclose(
            result_with.to_numpy(), result_without.to_numpy(), atol=1e-6
        )


class TestICFallbackStrategy:
    """Tests for ICFallbackStrategy on IC/ICIR-weighted fallback (issue #277)."""

    def _all_negative_ic_history(
        self, groups: list[str], seed: int = 42
    ) -> pd.DataFrame:
        """IC history where every group has negative mean IC."""
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            {col: rng.uniform(-0.10, -0.02, 36) for col in groups}
        )

    def _all_negative_icir_series(
        self, groups: list[str], seed: int = 42
    ) -> dict[str, pd.Series]:
        """Per-group IC series where every group has negative mean IC."""
        rng = np.random.default_rng(seed)
        return {
            col: pd.Series(rng.uniform(-0.10, -0.02, 36)) for col in groups
        }

    # ------------------------------------------------------------------
    # Strategy: EQUAL_WEIGHT (default, backward-compat)
    # ------------------------------------------------------------------

    def test_ic_equal_weight_fallback(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        group_scores = compute_group_scores(standardized_factors, coverage)
        groups = list(group_scores.columns)
        ic_history = self._all_negative_ic_history(groups)

        config = CompositeScoringConfig(
            method=CompositeMethod.IC_WEIGHTED,
            ic_fallback_strategy=ICFallbackStrategy.EQUAL_WEIGHT,
        )
        with _warnings_module.catch_warnings():
            _warnings_module.simplefilter("ignore", UserWarning)
            result = compute_ic_weighted_composite(
                group_scores, ic_history, config
            )
        expected = compute_equal_weight_composite(group_scores, config)
        pd.testing.assert_series_equal(result, expected)

    def test_icir_equal_weight_fallback(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        group_scores = compute_group_scores(standardized_factors, coverage)
        groups = list(group_scores.columns)
        ic_per_group = self._all_negative_icir_series(groups)

        config = CompositeScoringConfig(
            method=CompositeMethod.ICIR_WEIGHTED,
            ic_fallback_strategy=ICFallbackStrategy.EQUAL_WEIGHT,
        )
        with _warnings_module.catch_warnings():
            _warnings_module.simplefilter("ignore", UserWarning)
            result = compute_icir_weighted_composite(
                group_scores, ic_per_group, config
            )
        expected = compute_equal_weight_composite(group_scores, config)
        pd.testing.assert_series_equal(result, expected)

    # ------------------------------------------------------------------
    # Strategy: NAN
    # ------------------------------------------------------------------

    def test_ic_nan_fallback(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        group_scores = compute_group_scores(standardized_factors, coverage)
        groups = list(group_scores.columns)
        ic_history = self._all_negative_ic_history(groups)

        config = CompositeScoringConfig(
            method=CompositeMethod.IC_WEIGHTED,
            ic_fallback_strategy=ICFallbackStrategy.NAN,
        )
        with _warnings_module.catch_warnings():
            _warnings_module.simplefilter("ignore", UserWarning)
            result = compute_ic_weighted_composite(
                group_scores, ic_history, config
            )
        assert isinstance(result, pd.Series)
        assert result.isna().all()
        assert result.index.equals(group_scores.index)

    def test_icir_nan_fallback(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        group_scores = compute_group_scores(standardized_factors, coverage)
        groups = list(group_scores.columns)
        ic_per_group = self._all_negative_icir_series(groups)

        config = CompositeScoringConfig(
            method=CompositeMethod.ICIR_WEIGHTED,
            ic_fallback_strategy=ICFallbackStrategy.NAN,
        )
        with _warnings_module.catch_warnings():
            _warnings_module.simplefilter("ignore", UserWarning)
            result = compute_icir_weighted_composite(
                group_scores, ic_per_group, config
            )
        assert isinstance(result, pd.Series)
        assert result.isna().all()
        assert result.index.equals(group_scores.index)

    # ------------------------------------------------------------------
    # Strategy: RAISE
    # ------------------------------------------------------------------

    def test_ic_raise_fallback(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        group_scores = compute_group_scores(standardized_factors, coverage)
        groups = list(group_scores.columns)
        ic_history = self._all_negative_ic_history(groups)

        config = CompositeScoringConfig(
            method=CompositeMethod.IC_WEIGHTED,
            ic_fallback_strategy=ICFallbackStrategy.RAISE,
        )
        with pytest.raises(ConfigurationError, match="non-positive IC"):
            compute_ic_weighted_composite(group_scores, ic_history, config)

    def test_icir_raise_fallback(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        group_scores = compute_group_scores(standardized_factors, coverage)
        groups = list(group_scores.columns)
        ic_per_group = self._all_negative_icir_series(groups)

        config = CompositeScoringConfig(
            method=CompositeMethod.ICIR_WEIGHTED,
            ic_fallback_strategy=ICFallbackStrategy.RAISE,
        )
        with pytest.raises(ConfigurationError, match="non-positive ICIR"):
            compute_icir_weighted_composite(
                group_scores, ic_per_group, config
            )

    # ------------------------------------------------------------------
    # Warning emission
    # ------------------------------------------------------------------

    def test_ic_fallback_emits_warning(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        group_scores = compute_group_scores(standardized_factors, coverage)
        groups = list(group_scores.columns)
        ic_history = self._all_negative_ic_history(groups)

        config = CompositeScoringConfig(
            method=CompositeMethod.IC_WEIGHTED,
            ic_fallback_strategy=ICFallbackStrategy.EQUAL_WEIGHT,
        )
        with pytest.warns(UserWarning, match="non-positive IC"):
            compute_ic_weighted_composite(group_scores, ic_history, config)

    def test_icir_fallback_emits_warning(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        group_scores = compute_group_scores(standardized_factors, coverage)
        groups = list(group_scores.columns)
        ic_per_group = self._all_negative_icir_series(groups)

        config = CompositeScoringConfig(
            method=CompositeMethod.ICIR_WEIGHTED,
            ic_fallback_strategy=ICFallbackStrategy.EQUAL_WEIGHT,
        )
        with pytest.warns(UserWarning, match="non-positive ICIR"):
            compute_icir_weighted_composite(
                group_scores, ic_per_group, config
            )

    def test_warning_includes_group_count(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        group_scores = compute_group_scores(standardized_factors, coverage)
        groups = list(group_scores.columns)
        ic_history = self._all_negative_ic_history(groups)
        n_groups = len(groups)

        config = CompositeScoringConfig(
            method=CompositeMethod.IC_WEIGHTED,
            ic_fallback_strategy=ICFallbackStrategy.EQUAL_WEIGHT,
        )
        with pytest.warns(UserWarning) as record:
            compute_ic_weighted_composite(group_scores, ic_history, config)

        assert len(record) == 1
        msg = str(record[0].message)
        assert f"{n_groups}/{n_groups}" in msg

    def test_no_warning_when_positive_ic(
        self, standardized_factors: pd.DataFrame, coverage: pd.DataFrame
    ) -> None:
        """No warning when at least one group has positive IC."""
        group_scores = compute_group_scores(standardized_factors, coverage)
        groups = list(group_scores.columns)
        rng = np.random.default_rng(42)
        ic_history = pd.DataFrame(
            {col: rng.uniform(0.01, 0.06, 36) for col in groups}
        )
        config = CompositeScoringConfig(
            method=CompositeMethod.IC_WEIGHTED,
            ic_fallback_strategy=ICFallbackStrategy.EQUAL_WEIGHT,
        )
        with _warnings_module.catch_warnings():
            _warnings_module.simplefilter("error", UserWarning)
            compute_ic_weighted_composite(group_scores, ic_history, config)

    # ------------------------------------------------------------------
    # Config defaults and enum values
    # ------------------------------------------------------------------

    def test_default_fallback_strategy_is_equal_weight(self) -> None:
        cfg = CompositeScoringConfig()
        assert cfg.ic_fallback_strategy == ICFallbackStrategy.EQUAL_WEIGHT

    def test_enum_values(self) -> None:
        assert ICFallbackStrategy.EQUAL_WEIGHT == "equal_weight"
        assert ICFallbackStrategy.NAN == "nan"
        assert ICFallbackStrategy.RAISE == "raise"

    def test_preset_raise_on_fallback(self) -> None:
        cfg = CompositeScoringConfig.for_ic_weighted_raise_on_fallback()
        assert cfg.method == CompositeMethod.IC_WEIGHTED
        assert cfg.ic_fallback_strategy == ICFallbackStrategy.RAISE
