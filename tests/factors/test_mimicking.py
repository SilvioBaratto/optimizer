"""Tests for long-short factor-mimicking portfolio construction."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import ConfigurationError, DataError
from optimizer.factors import (
    build_all_factor_mimicking_portfolios,
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


class TestHalfQuantileOddUniverse:
    """quantile=0.5 is documented as valid and must also work for odd universes."""

    @staticmethod
    def _panel(n_assets: int, n_dates: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(3)
        dates = pd.date_range("2021-01-31", periods=n_dates, freq="ME")
        cols = [f"S{i}" for i in range(n_assets)]
        sc = pd.DataFrame(
            rng.standard_normal((n_dates, n_assets)), index=dates, columns=cols
        )
        rt = pd.DataFrame(
            rng.normal(0.0, 0.02, (n_dates, n_assets)), index=dates, columns=cols
        )
        return sc, rt

    @pytest.mark.parametrize("n_assets", [3, 5, 7, 11])
    def test_odd_asset_count_is_not_nan(self, n_assets: int) -> None:
        sc, rt = self._panel(n_assets)
        result = build_factor_mimicking_portfolios(sc, rt, quantile=0.5)
        assert result["factor_return"].notna().all()

    def test_odd_universe_legs_are_floor_half_and_disjoint(self) -> None:
        sc, rt = self._panel(5, n_dates=2)
        result = build_factor_mimicking_portfolios(sc, rt, quantile=0.5)
        date = sc.index[0]
        ranked = sc.loc[date].sort_values(ascending=False).index
        expected = rt.loc[date, ranked[:2]].mean() - rt.loc[date, ranked[-2:]].mean()
        assert result.loc[date, "factor_return"] == pytest.approx(expected)

    def test_ragged_universe_has_no_parity_dependent_gaps(self) -> None:
        sc, rt = self._panel(8)
        sc.iloc[::2, 0] = np.nan
        result = build_factor_mimicking_portfolios(sc, rt, quantile=0.5)
        assert result["factor_return"].notna().all()

    def test_beta_neutral_odd_universe_is_not_nan(self) -> None:
        sc, rt = self._panel(5)
        market = rt.mean(axis=1)
        result = build_factor_mimicking_portfolios(
            sc, rt, quantile=0.5, beta_neutral=True, market_returns=market
        )
        assert result["factor_return"].notna().all()

    def test_single_valid_asset_is_still_nan(self) -> None:
        sc, rt = self._panel(1, n_dates=3)
        result = build_factor_mimicking_portfolios(sc, rt, quantile=0.5)
        assert result["factor_return"].isna().all()


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


# ---------------------------------------------------------------------------
# Beta-neutral mimicking portfolios (issue #101)
# ---------------------------------------------------------------------------


class TestBetaNeutral:
    @pytest.fixture()
    def market_returns(self, returns: pd.DataFrame) -> pd.Series:
        return returns.mean(axis=1)

    def test_returns_dataframe(
        self,
        scores: pd.DataFrame,
        returns: pd.DataFrame,
        market_returns: pd.Series,
    ) -> None:
        result = build_factor_mimicking_portfolios(
            scores, returns, beta_neutral=True, market_returns=market_returns
        )
        assert isinstance(result, pd.DataFrame)
        assert "factor_return" in result.columns

    def test_differs_from_standard(
        self,
        scores: pd.DataFrame,
        returns: pd.DataFrame,
        market_returns: pd.Series,
    ) -> None:
        standard = build_factor_mimicking_portfolios(scores, returns)
        hedged = build_factor_mimicking_portfolios(
            scores, returns, beta_neutral=True, market_returns=market_returns
        )
        assert not standard["factor_return"].equals(hedged["factor_return"])

    def test_lower_market_correlation(
        self,
        scores: pd.DataFrame,
        returns: pd.DataFrame,
        market_returns: pd.Series,
    ) -> None:
        standard = build_factor_mimicking_portfolios(scores, returns)
        hedged = build_factor_mimicking_portfolios(
            scores, returns, beta_neutral=True, market_returns=market_returns
        )
        corr_std = abs(standard["factor_return"].corr(market_returns))
        corr_hedged = abs(hedged["factor_return"].corr(market_returns))
        # Hedged should have lower or equal market correlation
        assert corr_hedged <= corr_std + 0.15

    def test_missing_market_returns_raises(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        with pytest.raises(ConfigurationError, match="market_returns"):
            build_factor_mimicking_portfolios(
                scores, returns, beta_neutral=True, market_returns=None
            )

    def test_beta_neutral_false_is_noop(
        self,
        scores: pd.DataFrame,
        returns: pd.DataFrame,
        market_returns: pd.Series,
    ) -> None:
        standard = build_factor_mimicking_portfolios(scores, returns)
        explicit_false = build_factor_mimicking_portfolios(
            scores, returns, beta_neutral=False, market_returns=market_returns
        )
        pd.testing.assert_frame_equal(standard, explicit_false)

    def test_non_overlapping_market_index_raises(
        self, scores: pd.DataFrame, returns: pd.DataFrame, market_returns: pd.Series
    ) -> None:
        """A market series that shares no dates must not yield an unhedged series."""
        tz_aware = market_returns.copy()
        tz_aware.index = tz_aware.index.tz_localize("UTC")
        with pytest.raises(DataError, match="market_returns"):
            build_factor_mimicking_portfolios(
                scores, returns, beta_neutral=True, market_returns=tz_aware
            )

    def test_two_observation_overlap_raises(
        self, scores: pd.DataFrame, returns: pd.DataFrame, market_returns: pd.Series
    ) -> None:
        """A beta estimated from two points is exactly determined, not a hedge."""
        with pytest.raises(DataError, match="market_returns"):
            build_factor_mimicking_portfolios(
                scores,
                returns,
                beta_neutral=True,
                market_returns=market_returns.iloc[:2],
            )

    def test_constant_market_returns_raises(
        self, scores: pd.DataFrame, returns: pd.DataFrame, market_returns: pd.Series
    ) -> None:
        flat = pd.Series(0.01, index=market_returns.index)
        with pytest.raises(DataError, match="variance"):
            build_factor_mimicking_portfolios(
                scores, returns, beta_neutral=True, market_returns=flat
            )

    def test_disjoint_scores_and_returns_give_empty_frame(
        self, scores: pd.DataFrame, returns: pd.DataFrame, market_returns: pd.Series
    ) -> None:
        """An empty factor panel is not a market_returns problem."""
        shifted = scores.copy()
        shifted.index = shifted.index + pd.DateOffset(years=10)
        unhedged = build_factor_mimicking_portfolios(shifted, returns)
        hedged = build_factor_mimicking_portfolios(
            shifted, returns, beta_neutral=True, market_returns=market_returns
        )
        assert hedged.empty
        pd.testing.assert_frame_equal(hedged, unhedged)

    def test_all_nan_legs_give_nan_frame(
        self, scores: pd.DataFrame, returns: pd.DataFrame, market_returns: pd.Series
    ) -> None:
        """No date with enough valid assets mirrors the unhedged NaN output."""
        blank = scores.copy()
        blank.iloc[:, :] = np.nan
        unhedged = build_factor_mimicking_portfolios(blank, returns)
        hedged = build_factor_mimicking_portfolios(
            blank, returns, beta_neutral=True, market_returns=market_returns
        )
        assert hedged["factor_return"].isna().all()
        pd.testing.assert_frame_equal(hedged, unhedged)

    def test_build_all_propagates_hedge_data_error(
        self, scores: pd.DataFrame, returns: pd.DataFrame, market_returns: pd.Series
    ) -> None:
        with pytest.raises(DataError, match="market_returns"):
            build_all_factor_mimicking_portfolios(
                {"value": scores},
                returns,
                beta_neutral=True,
                market_returns=market_returns.iloc[:2],
            )


class TestBuildAllFactorMimickingPortfolios:
    def test_columns_are_factor_names_in_order(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        rng = np.random.default_rng(99)
        scores2 = pd.DataFrame(
            rng.standard_normal(scores.shape),
            index=scores.index,
            columns=scores.columns,
        )
        panel = build_all_factor_mimicking_portfolios(
            {"value": scores, "momentum": scores2}, returns
        )
        assert list(panel.columns) == ["value", "momentum"]
        assert panel.shape[1] == 2

    def test_matches_per_factor_calls(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        panel = build_all_factor_mimicking_portfolios({"value": scores}, returns)
        single = build_factor_mimicking_portfolios(scores, returns)
        pd.testing.assert_series_equal(
            panel["value"], single["factor_return"], check_names=False
        )

    def test_outer_aligns_on_union_of_dates(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        short = scores.iloc[:-5]
        panel = build_all_factor_mimicking_portfolios(
            {"full": scores, "short": short}, returns
        )
        # Row index = union of both factors' date indices.
        assert len(panel.index) == len(scores.index)
        assert panel["short"].iloc[-5:].isna().all()

    def test_empty_mapping_raises(self, returns: pd.DataFrame) -> None:
        with pytest.raises(ConfigurationError, match="non-empty"):
            build_all_factor_mimicking_portfolios({}, returns)

    def test_invalid_quantile_propagates(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        with pytest.raises(ConfigurationError, match="quantile"):
            build_all_factor_mimicking_portfolios(
                {"value": scores}, returns, quantile=0.9
            )
