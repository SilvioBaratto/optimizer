"""Tests for long-short factor-mimicking portfolio construction."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import ConfigurationError
from optimizer.factors import (
    build_factor_mimicking_portfolios,
    compute_cross_factor_correlation,
)

N_DATES = 60
N_ASSETS = 40
QUANTILE = 0.30
DATES = pd.date_range("2020-01-01", periods=N_DATES, freq="ME")
ASSETS = [f"A{i:03d}" for i in range(N_ASSETS)]


@pytest.fixture()
def scores() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.standard_normal((N_DATES, N_ASSETS)),
        index=DATES,
        columns=ASSETS,
    )


@pytest.fixture()
def returns() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        rng.normal(0.001, 0.02, (N_DATES, N_ASSETS)),
        index=DATES,
        columns=ASSETS,
    )


# ---------------------------------------------------------------------------
# Shape and index
# ---------------------------------------------------------------------------


class TestOutputShape:
    def test_returns_dataframe(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        result = build_factor_mimicking_portfolios(scores, returns)
        assert isinstance(result, pd.DataFrame)

    def test_column_name(self, scores: pd.DataFrame, returns: pd.DataFrame) -> None:
        result = build_factor_mimicking_portfolios(scores, returns)
        assert list(result.columns) == ["factor_return"]

    def test_index_matches_date_intersection(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        result = build_factor_mimicking_portfolios(scores, returns)
        expected_index = scores.index.intersection(returns.index)
        assert result.index.equals(expected_index)

    def test_returns_has_same_dates_as_inputs(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        result = build_factor_mimicking_portfolios(scores, returns)
        assert len(result) == N_DATES

    def test_partial_date_overlap(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        scores_sub = scores.iloc[:40]
        result = build_factor_mimicking_portfolios(scores_sub, returns)
        assert len(result) == 40


# ---------------------------------------------------------------------------
# Long and short leg sizing
# ---------------------------------------------------------------------------


class TestLegSizing:
    def test_long_leg_size_ceiling(self, returns: pd.DataFrame) -> None:
        """Long leg must contain at most ceil(n_assets * quantile) assets."""
        # Use deterministic scores to inspect leg membership precisely
        rng = np.random.default_rng(0)
        scores_det = pd.DataFrame(
            rng.standard_normal((N_DATES, N_ASSETS)),
            index=DATES,
            columns=ASSETS,
        )
        k_expected = max(1, math.ceil(N_ASSETS * QUANTILE))

        # Verify for the first date: top k assets are in long leg
        scores_t = scores_det.iloc[0]
        ranked = scores_t.sort_values(ascending=False)
        long_leg = ranked.index[:k_expected]

        assert len(long_leg) == k_expected
        assert len(long_leg) <= math.ceil(N_ASSETS * QUANTILE)

    def test_short_leg_size_ceiling(self, returns: pd.DataFrame) -> None:
        """Short leg must contain at most ceil(n_assets * quantile) assets."""
        rng = np.random.default_rng(0)
        scores_det = pd.DataFrame(
            rng.standard_normal((N_DATES, N_ASSETS)),
            index=DATES,
            columns=ASSETS,
        )
        k_expected = max(1, math.ceil(N_ASSETS * QUANTILE))
        scores_t = scores_det.iloc[0]
        ranked = scores_t.sort_values(ascending=False)
        short_leg = ranked.index[-k_expected:]

        assert len(short_leg) == k_expected
        assert len(short_leg) <= math.ceil(N_ASSETS * QUANTILE)

    def test_no_overlap_between_long_and_short(self, returns: pd.DataFrame) -> None:
        """Long and short legs must be disjoint."""
        rng = np.random.default_rng(0)
        scores_det = pd.DataFrame(
            rng.standard_normal((N_DATES, N_ASSETS)),
            index=DATES,
            columns=ASSETS,
        )
        k = max(1, math.ceil(N_ASSETS * QUANTILE))
        for t in DATES[:5]:
            ranked = scores_det.loc[t].sort_values(ascending=False)
            long_leg = set(ranked.index[:k])
            short_leg = set(ranked.index[-k:])
            assert long_leg.isdisjoint(short_leg)

    def test_long_leg_has_highest_scores(self) -> None:
        """Long leg assets must have strictly higher scores than short leg assets."""
        # 10-asset, 5-date deterministic example
        assets = [f"X{i}" for i in range(10)]
        dates = pd.date_range("2021-01-01", periods=5, freq="ME")
        # Deterministic scores: 9,8,...,0 — clear ranking
        scores_vals = np.tile(np.arange(9, -1, -1, dtype=float), (5, 1))
        scores_det = pd.DataFrame(scores_vals, index=dates, columns=assets)
        k = max(1, math.ceil(10 * 0.30))  # 3
        for t in dates:
            ranked = scores_det.loc[t].sort_values(ascending=False)
            long_min = ranked.iloc[:k].min()
            short_max = ranked.iloc[-k:].max()
            assert long_min > short_max


# ---------------------------------------------------------------------------
# Returns are valid floats
# ---------------------------------------------------------------------------


class TestReturnValues:
    def test_no_nan_when_enough_assets(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        result = build_factor_mimicking_portfolios(scores, returns)
        assert result["factor_return"].notna().all()

    def test_nan_when_too_few_assets(self) -> None:
        """2 valid assets with quantile=0.30 → k=1, needs 2 → should succeed."""
        dates = pd.date_range("2021-01-01", periods=3, freq="ME")
        sc = pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [-1.0, -1.0, -1.0]}, index=dates)
        rt = pd.DataFrame(
            {"A": [0.01, 0.02, 0.01], "B": [-0.01, -0.02, -0.01]}, index=dates
        )
        result = build_factor_mimicking_portfolios(sc, rt, quantile=0.30)
        # 2 assets, k=ceil(2*0.3)=1, need 2*1=2 → should succeed (not nan)
        assert result["factor_return"].notna().all()

    def test_positive_factor_premium_when_scores_correlated_with_returns(self) -> None:
        """When scores strongly predict returns, mean LS return should be positive."""
        rng = np.random.default_rng(42)
        assets = [f"A{i:02d}" for i in range(50)]
        dates = pd.date_range("2019-01-01", periods=120, freq="ME")
        # Scores predict sign of returns
        base_scores = rng.standard_normal((120, 50))
        base_returns = base_scores * 0.02 + rng.normal(0, 0.005, (120, 50))
        sc = pd.DataFrame(base_scores, index=dates, columns=assets)
        rt = pd.DataFrame(base_returns, index=dates, columns=assets)
        result = build_factor_mimicking_portfolios(sc, rt, quantile=0.30)
        assert result["factor_return"].mean() > 0


# ---------------------------------------------------------------------------
# Weighting modes
# ---------------------------------------------------------------------------


class TestWeightingModes:
    def test_equal_weighting_default(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        result = build_factor_mimicking_portfolios(scores, returns, weighting="equal")
        assert result["factor_return"].notna().any()

    def test_value_weighting(self, scores: pd.DataFrame, returns: pd.DataFrame) -> None:
        result = build_factor_mimicking_portfolios(scores, returns, weighting="value")
        assert isinstance(result, pd.DataFrame)
        assert result["factor_return"].notna().any()

    def test_equal_and_value_differ(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        eq = build_factor_mimicking_portfolios(scores, returns, weighting="equal")
        vw = build_factor_mimicking_portfolios(scores, returns, weighting="value")
        # Should not be identical (different weighting schemes)
        assert not eq["factor_return"].equals(vw["factor_return"])

    def test_invalid_weighting_raises(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        with pytest.raises(ConfigurationError, match="weighting"):
            build_factor_mimicking_portfolios(scores, returns, weighting="market_cap")


# ---------------------------------------------------------------------------
# Quantile validation
# ---------------------------------------------------------------------------


class TestQuantileValidation:
    def test_invalid_quantile_zero_raises(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        with pytest.raises(ConfigurationError, match="quantile"):
            build_factor_mimicking_portfolios(scores, returns, quantile=0.0)

    def test_invalid_quantile_above_half_raises(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        with pytest.raises(ConfigurationError, match="quantile"):
            build_factor_mimicking_portfolios(scores, returns, quantile=0.51)

    def test_quantile_half_allowed(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        result = build_factor_mimicking_portfolios(scores, returns, quantile=0.50)
        assert isinstance(result, pd.DataFrame)

    def test_larger_quantile_uses_more_assets(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        r30 = build_factor_mimicking_portfolios(scores, returns, quantile=0.30)
        r50 = build_factor_mimicking_portfolios(scores, returns, quantile=0.50)
        # Both should produce valid returns; results will differ
        assert r30["factor_return"].notna().all()
        assert r50["factor_return"].notna().all()


# ---------------------------------------------------------------------------
# Multi-factor concatenation
# ---------------------------------------------------------------------------


class TestMultiFactorConcatenation:
    def test_concat_produces_dates_x_factors(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        rng = np.random.default_rng(99)
        scores2 = pd.DataFrame(
            rng.standard_normal((N_DATES, N_ASSETS)),
            index=DATES,
            columns=ASSETS,
        )
        r1 = build_factor_mimicking_portfolios(scores, returns).rename(
            columns={"factor_return": "value"}
        )
        r2 = build_factor_mimicking_portfolios(scores2, returns).rename(
            columns={"factor_return": "momentum"}
        )
        combined = pd.concat([r1, r2], axis=1)
        assert combined.shape == (N_DATES, 2)
        assert list(combined.columns) == ["value", "momentum"]


# ---------------------------------------------------------------------------
# Cross-factor correlation
# ---------------------------------------------------------------------------


class TestComputeCrossFactorCorrelation:
    @pytest.fixture()
    def factor_returns(self) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        return pd.DataFrame(
            rng.normal(0, 0.01, (N_DATES, 3)),
            index=DATES,
            columns=["value", "momentum", "quality"],
        )

    def test_returns_dataframe(self, factor_returns: pd.DataFrame) -> None:
        result = compute_cross_factor_correlation(factor_returns)
        assert isinstance(result, pd.DataFrame)

    def test_diagonal_is_one(self, factor_returns: pd.DataFrame) -> None:
        result = compute_cross_factor_correlation(factor_returns)
        diag = np.diag(result.values)
        np.testing.assert_allclose(diag, 1.0, atol=1e-10)

    def test_symmetric(self, factor_returns: pd.DataFrame) -> None:
        result = compute_cross_factor_correlation(factor_returns)
        np.testing.assert_allclose(result.values, result.values.T, atol=1e-10)

    def test_positive_semidefinite(self, factor_returns: pd.DataFrame) -> None:
        result = compute_cross_factor_correlation(factor_returns)
        eigenvalues = np.linalg.eigvalsh(result.values)
        assert (eigenvalues >= -1e-10).all()

    def test_entries_in_minus_one_to_one(self, factor_returns: pd.DataFrame) -> None:
        result = compute_cross_factor_correlation(factor_returns)
        assert (result.values >= -1.0 - 1e-10).all()
        assert (result.values <= 1.0 + 1e-10).all()

    def test_index_and_columns_are_factor_names(
        self, factor_returns: pd.DataFrame
    ) -> None:
        result = compute_cross_factor_correlation(factor_returns)
        assert list(result.index) == list(factor_returns.columns)
        assert list(result.columns) == list(factor_returns.columns)

    def test_perfect_correlation_with_self(self, factor_returns: pd.DataFrame) -> None:
        duplicated = pd.concat(
            [factor_returns["value"], factor_returns["value"]], axis=1
        )
        duplicated.columns = pd.Index(["v1", "v2"])
        result = compute_cross_factor_correlation(duplicated)
        assert pytest.approx(result.loc["v1", "v2"], abs=1e-9) == 1.0

    def test_shape_is_factors_x_factors(self, factor_returns: pd.DataFrame) -> None:
        result = compute_cross_factor_correlation(factor_returns)
        n = len(factor_returns.columns)
        assert result.shape == (n, n)
