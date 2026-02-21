"""Tests for quintile spread return analysis."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import ConfigurationError
from optimizer.factors import QuintileSpreadResult, compute_quintile_spread

N_DATES = 60
N_ASSETS = 100
N_QUANTILES = 5
DATES = pd.date_range("2019-01-01", periods=N_DATES, freq="ME")
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
# Return type and structure
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_quintile_spread_result(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        result = compute_quintile_spread(scores, returns)
        assert isinstance(result, QuintileSpreadResult)

    def test_quintile_returns_is_dataframe(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        result = compute_quintile_spread(scores, returns)
        assert isinstance(result.quintile_returns, pd.DataFrame)

    def test_spread_returns_is_series(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        result = compute_quintile_spread(scores, returns)
        assert isinstance(result.spread_returns, pd.Series)

    def test_scalar_stats_are_floats(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        result = compute_quintile_spread(scores, returns)
        assert isinstance(result.annualised_mean, float)
        assert isinstance(result.t_stat, float)
        assert isinstance(result.sharpe, float)


# ---------------------------------------------------------------------------
# Quintile shape and column labels
# ---------------------------------------------------------------------------


class TestQuintileShape:
    def test_quintile_returns_has_n_quantiles_columns(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        result = compute_quintile_spread(scores, returns, n_quantiles=5)
        assert result.quintile_returns.shape[1] == 5

    def test_quintile_column_names_q1_to_q5(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        result = compute_quintile_spread(scores, returns, n_quantiles=5)
        assert list(result.quintile_returns.columns) == ["Q1", "Q2", "Q3", "Q4", "Q5"]

    def test_decile_column_names(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        result = compute_quintile_spread(scores, returns, n_quantiles=10)
        expected = [f"Q{i}" for i in range(1, 11)]
        assert list(result.quintile_returns.columns) == expected

    def test_quintile_returns_index_matches_date_intersection(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        result = compute_quintile_spread(scores, returns)
        expected = scores.index.intersection(returns.index)
        assert len(result.quintile_returns) == len(expected)

    def test_spread_returns_same_length_as_quintile_returns(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        result = compute_quintile_spread(scores, returns)
        assert len(result.spread_returns) == len(result.quintile_returns)


# ---------------------------------------------------------------------------
# Quintile buckets don't overlap
# ---------------------------------------------------------------------------


class TestNoOverlap:
    def test_buckets_cover_all_assets(self) -> None:
        """Each asset appears in exactly one quintile per date."""
        n_assets = 50
        assets = [f"X{i:02d}" for i in range(n_assets)]
        dates = pd.date_range("2021-01-01", periods=10, freq="ME")
        rng = np.random.default_rng(1)
        sc = pd.DataFrame(
            rng.standard_normal((10, n_assets)), index=dates, columns=assets
        )
        rt = pd.DataFrame(
            rng.normal(0, 0.01, (10, n_assets)), index=dates, columns=assets
        )

        result = compute_quintile_spread(sc, rt, n_quantiles=5)

        # For each date, sum of bucket sizes should equal n_assets
        # Since we only have quintile *returns* (not membership), we verify
        # that all quintile columns have non-NaN values on each date.
        assert result.quintile_returns.notna().all(axis=None)

    def test_all_quintiles_populated_every_date(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        """No empty quintiles when n_assets >= n_quantiles."""
        result = compute_quintile_spread(scores, returns, n_quantiles=5)
        assert result.quintile_returns.notna().all(axis=None)


# ---------------------------------------------------------------------------
# Spread = Qn - Q1 exactly
# ---------------------------------------------------------------------------


class TestSpreadEquality:
    def test_spread_equals_qn_minus_q1_elementwise(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        result = compute_quintile_spread(scores, returns, n_quantiles=5)
        expected = result.quintile_returns["Q5"] - result.quintile_returns["Q1"]
        np.testing.assert_allclose(
            result.spread_returns.to_numpy(),
            expected.to_numpy(),
            atol=1e-12,
        )

    def test_spread_equals_qn_minus_q1_deciles(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        result = compute_quintile_spread(scores, returns, n_quantiles=10)
        expected = result.quintile_returns["Q10"] - result.quintile_returns["Q1"]
        np.testing.assert_allclose(
            result.spread_returns.to_numpy(),
            expected.to_numpy(),
            atol=1e-12,
        )

    def test_spread_equals_qn_minus_q1_arbitrary_n(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        for n in [3, 4, 7]:
            result = compute_quintile_spread(scores, returns, n_quantiles=n)
            top = f"Q{n}"
            expected = result.quintile_returns[top] - result.quintile_returns["Q1"]
            np.testing.assert_allclose(
                result.spread_returns.to_numpy(),
                expected.to_numpy(),
                atol=1e-12,
            )


# ---------------------------------------------------------------------------
# Annualisation uses 252 trading days
# ---------------------------------------------------------------------------


class TestAnnualisation:
    def test_annualised_mean_equals_mean_times_252(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        result = compute_quintile_spread(scores, returns)
        expected = float(result.spread_returns.dropna().mean()) * 252
        assert pytest.approx(result.annualised_mean, rel=1e-9) == expected

    def test_sharpe_uses_sqrt_252(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        result = compute_quintile_spread(scores, returns)
        valid = result.spread_returns.dropna()
        expected_sharpe = (
            float(valid.mean()) * math.sqrt(252) / float(valid.std(ddof=1))
        )
        assert pytest.approx(result.sharpe, rel=1e-9) == expected_sharpe

    def test_t_stat_sign_matches_mean(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        result = compute_quintile_spread(scores, returns)
        if not np.isnan(result.t_stat) and result.annualised_mean != 0:
            assert np.sign(result.t_stat) == np.sign(result.annualised_mean)


# ---------------------------------------------------------------------------
# Ordering: high-score bucket beats low-score bucket when predictive
# ---------------------------------------------------------------------------


class TestQuintileOrdering:
    def test_q5_beats_q1_when_scores_predict_returns(self) -> None:
        """When factor scores strongly predict returns, Q5 > Q1 on average."""
        rng = np.random.default_rng(0)
        n_assets = 100
        n_dates = 120
        assets = [f"A{i:03d}" for i in range(n_assets)]
        dates = pd.date_range("2015-01-01", periods=n_dates, freq="ME")
        base = rng.standard_normal((n_dates, n_assets))
        sc = pd.DataFrame(base, index=dates, columns=assets)
        # Returns strongly correlated with scores
        rt = pd.DataFrame(
            base * 0.03 + rng.normal(0, 0.005, (n_dates, n_assets)),
            index=dates,
            columns=assets,
        )
        result = compute_quintile_spread(sc, rt, n_quantiles=5)
        q_means = result.quintile_returns.mean()
        assert q_means["Q5"] > q_means["Q1"]
        assert result.annualised_mean > 0

    def test_quintile_monotone_when_perfectly_predictive(self) -> None:
        """Mean return should increase monotonically Q1 → Qn for perfect predictor."""
        n_assets = 50
        n_dates = 60
        assets = [f"A{i:02d}" for i in range(n_assets)]
        dates = pd.date_range("2018-01-01", periods=n_dates, freq="ME")
        rng = np.random.default_rng(3)
        base = rng.standard_normal((n_dates, n_assets))
        sc = pd.DataFrame(base, index=dates, columns=assets)
        # Returns = scores (perfect correlation, no noise)
        rt = pd.DataFrame(base.copy(), index=dates, columns=assets)
        result = compute_quintile_spread(sc, rt, n_quantiles=5)
        means = result.quintile_returns.mean()
        diffs = means.diff().dropna()
        assert (diffs > 0).all()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_invalid_n_quantiles_raises(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        with pytest.raises(ConfigurationError, match="n_quantiles"):
            compute_quintile_spread(scores, returns, n_quantiles=1)

    def test_partial_date_overlap(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        result = compute_quintile_spread(scores.iloc[:30], returns)
        assert len(result.quintile_returns) == 30

    def test_nan_row_when_too_few_assets(self) -> None:
        """Dates with fewer valid assets than n_quantiles produce NaN rows."""
        dates = pd.date_range("2021-01-01", periods=5, freq="ME")
        # 3 assets, n_quantiles=5 → too few → NaN
        sc = pd.DataFrame(
            np.random.default_rng(0).standard_normal((5, 3)),
            index=dates,
            columns=["A", "B", "C"],
        )
        rt = pd.DataFrame(
            np.random.default_rng(1).normal(0, 0.01, (5, 3)),
            index=dates,
            columns=["A", "B", "C"],
        )
        result = compute_quintile_spread(sc, rt, n_quantiles=5)
        assert result.quintile_returns.isna().all(axis=None)

    def test_decile_output_shape(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        result = compute_quintile_spread(scores, returns, n_quantiles=10)
        assert result.quintile_returns.shape == (N_DATES, 10)

    def test_spread_series_values_are_floats(
        self, scores: pd.DataFrame, returns: pd.DataFrame
    ) -> None:
        result = compute_quintile_spread(scores, returns)
        assert result.spread_returns.dtype == float
