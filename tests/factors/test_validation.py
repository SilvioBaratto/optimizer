"""Tests for factor validation and statistical testing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.exceptions import DataError
from optimizer.factors import (
    FACTOR_SPREAD_BENCHMARKS,
    CompositeICResult,
    CorrectedPValues,
    FactorValidationConfig,
    FactorValidationReport,
    ICStats,
    benjamini_hochberg,
    compute_composite_ic,
    compute_ic_series,
    compute_ic_stats,
    compute_icir,
    compute_monthly_ic,
    compute_newey_west_tstat,
    compute_quantile_spread,
    compute_vif,
    correct_pvalues,
    run_factor_validation,
    validate_factor_universe,
)
from optimizer.linear_model import CSLinearRegressionConfig


@pytest.fixture()
def factor_scores() -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0, 1, 100), index=[f"T{i:03d}" for i in range(100)])


@pytest.fixture()
def forward_returns() -> pd.Series:
    rng = np.random.default_rng(99)
    return pd.Series(
        rng.normal(0.001, 0.02, 100),
        index=[f"T{i:03d}" for i in range(100)],
    )


def _reference_newey_west(values: np.ndarray, n_lags: int) -> tuple[float, float]:
    """Textbook Bartlett-kernel NW t-stat: every autocovariance is sum / n."""
    from scipy import stats as sp_stats

    n = len(values)
    mean = values.mean()
    e = values - mean
    var = float(e @ e) / n
    for j in range(1, min(n_lags, n - 1) + 1):
        var += 2.0 * (1.0 - j / (n_lags + 1)) * float(e[j:] @ e[:-j]) / n
    t_stat = mean / np.sqrt(var / n)
    return float(t_stat), float(2.0 * sp_stats.t.sf(abs(t_stat), df=n - 1))


# 14 usable periods around one gap: above the 2 * lags + 1 Newey-West floor for
# both lags=5 and lags=6, so the NaN handling is what the tests exercise.
_IC_WITH_GAP: list[float] = [
    0.05, 0.02, np.nan, 0.04, 0.06, 0.03, 0.05,
    0.04, 0.02, 0.07, 0.05, 0.03, 0.06, 0.04, 0.05,
]  # fmt: skip


class TestComputeMonthlyIC:
    def test_returns_float(
        self, factor_scores: pd.Series, forward_returns: pd.Series
    ) -> None:
        ic = compute_monthly_ic(factor_scores, forward_returns)
        assert isinstance(ic, float)
        assert -1.0 <= ic <= 1.0

    def test_perfect_correlation(self) -> None:
        scores = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=list("ABCDE"))
        returns = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=list("ABCDE"))
        ic = compute_monthly_ic(scores, returns)
        assert ic > 0.99

    def test_insufficient_data(self) -> None:
        scores = pd.Series([1.0, 2.0], index=["A", "B"])
        returns = pd.Series([0.1, 0.2], index=["A", "B"])
        ic = compute_monthly_ic(scores, returns)
        assert np.isnan(ic)

    def test_min_observations_stricter(self) -> None:
        """5 observations pass default (min=3) but fail min_observations=6."""
        tickers = list("ABCDE")
        scores = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=tickers)
        returns = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=tickers)
        assert not np.isnan(compute_monthly_ic(scores, returns))
        assert np.isnan(compute_monthly_ic(scores, returns, min_observations=6))


class TestComputeICSeries:
    def test_returns_series(self) -> None:
        rng = np.random.default_rng(42)
        dates = pd.date_range("2023-01-01", periods=12, freq="ME")
        tickers = [f"T{i:02d}" for i in range(20)]

        scores_hist = pd.DataFrame(
            rng.normal(0, 1, (12, 20)), index=dates, columns=tickers
        )
        returns_hist = pd.DataFrame(
            rng.normal(0.001, 0.02, (12, 20)), index=dates, columns=tickers
        )

        result = compute_ic_series(scores_hist, returns_hist, "test_factor")
        assert isinstance(result, pd.Series)
        assert len(result) > 0

    def test_min_observations_filters_dates(self) -> None:
        """Dates with exactly 5 tickers are excluded at min_observations=6."""
        rng = np.random.default_rng(42)
        dates = pd.date_range("2023-01-01", periods=6, freq="ME")
        # 5 tickers — passes default (min=3) but fails min_observations=6
        tickers = [f"T{i}" for i in range(5)]

        scores_hist = pd.DataFrame(
            rng.normal(0, 1, (6, 5)), index=dates, columns=tickers
        )
        returns_hist = pd.DataFrame(
            rng.normal(0.001, 0.02, (6, 5)), index=dates, columns=tickers
        )

        default_result = compute_ic_series(scores_hist, returns_hist, "f")
        strict_result = compute_ic_series(
            scores_hist, returns_hist, "f", min_observations=6
        )
        assert len(default_result) > 0
        assert len(strict_result) == 0


class TestComputeICSeriesCSRegression:
    @pytest.fixture
    def panel(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(7)
        dates = pd.date_range("2023-01-01", periods=12, freq="ME")
        tickers = [f"T{i:02d}" for i in range(20)]
        scores = pd.DataFrame(rng.normal(0, 1, (12, 20)), index=dates, columns=tickers)
        returns = pd.DataFrame(
            0.05 * scores.values + rng.normal(0, 0.02, (12, 20)),
            index=dates,
            columns=tickers,
        )
        return scores, returns

    def test_when_use_cs_regression_false_then_default_unchanged(
        self,
        panel: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        scores, returns = panel
        rank = compute_ic_series(scores, returns, "f")
        explicit_rank = compute_ic_series(scores, returns, "f", use_cs_regression=False)
        pd.testing.assert_series_equal(rank, explicit_rank)

    def test_when_use_cs_regression_true_then_returns_series(
        self,
        panel: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        scores, returns = panel
        result = compute_ic_series(scores, returns, "f", use_cs_regression=True)
        assert isinstance(result, pd.Series)
        assert len(result) > 0

    def test_when_regression_then_differs_from_rank(
        self,
        panel: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        scores, returns = panel
        rank = compute_ic_series(scores, returns, "f")
        slope = compute_ic_series(scores, returns, "f", use_cs_regression=True)
        # Both indexed by date; slope IC values should differ from rank IC.
        common = rank.index.intersection(slope.index)
        assert len(common) > 0
        assert not np.allclose(rank.loc[common], slope.loc[common], atol=1e-6)

    def test_when_cs_config_none_then_uses_defaults(
        self,
        panel: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        scores, returns = panel
        explicit = compute_ic_series(
            scores, returns, "f", use_cs_regression=True, cs_config=None
        )
        assert len(explicit) > 0

    def test_when_cs_config_passed_then_used(
        self,
        panel: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        scores, returns = panel
        cfg = CSLinearRegressionConfig(fit_intercept=False)
        result = compute_ic_series(
            scores, returns, "f", use_cs_regression=True, cs_config=cfg
        )
        assert isinstance(result, pd.Series)
        assert len(result) > 0

    def test_when_regression_factor_name_preserved(
        self,
        panel: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        scores, returns = panel
        result = compute_ic_series(scores, returns, "MOMENTUM", use_cs_regression=True)
        assert result.name == "MOMENTUM"

    def test_when_regression_min_obs_filters_dates(
        self,
        panel: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Periods with fewer non-NaN pairs than min_observations are dropped."""
        scores, returns = panel
        # Wipe row 0 to <min_observations valid pairs.
        scores_sparse = scores.copy()
        scores_sparse.iloc[0, :18] = np.nan  # leave only 2 valid in row 0
        result = compute_ic_series(
            scores_sparse,
            returns,
            "f",
            min_observations=5,
            use_cs_regression=True,
        )
        assert scores_sparse.index[0] not in result.index

    def test_when_regression_all_dates_filtered_then_empty_series(self) -> None:
        rng = np.random.default_rng(0)
        dates = pd.date_range("2023-01-01", periods=4, freq="ME")
        tickers = [f"T{i}" for i in range(3)]
        scores = pd.DataFrame(rng.normal(0, 1, (4, 3)), index=dates, columns=tickers)
        returns = pd.DataFrame(
            rng.normal(0, 0.02, (4, 3)), index=dates, columns=tickers
        )
        result = compute_ic_series(
            scores,
            returns,
            "f",
            min_observations=10,
            use_cs_regression=True,
        )
        assert isinstance(result, pd.Series)
        assert len(result) == 0
        assert result.name == "f"

    def test_when_partial_coverage_kept_then_no_crash(self) -> None:
        """Kept periods with scattered NaN pairs must fit via cs_weights mask.

        Regression test: skfolio ``CSLinearRegression`` rejects positive-weight
        pairs with non-finite X/y.  Ragged per-period coverage (each kept row
        above ``min_observations`` still has some NaN pairs) previously raised
        ``ValueError``.  The 0/1 weight mask excludes invalid pairs.
        """
        rng = np.random.default_rng(11)
        dates = pd.date_range("2023-01-01", periods=12, freq="ME")
        tickers = [f"T{i:02d}" for i in range(20)]
        scores = pd.DataFrame(rng.normal(0, 1, (12, 20)), index=dates, columns=tickers)
        returns = pd.DataFrame(
            0.05 * scores.values + rng.normal(0, 0.02, (12, 20)),
            index=dates,
            columns=tickers,
        )
        # Punch 5 NaN scores into every row (different columns per row) so
        # every kept period is partial-coverage but still has 15 valid pairs.
        for r in range(12):
            drop_cols = rng.choice(20, size=5, replace=False)
            scores.iloc[r, drop_cols] = np.nan

        result = compute_ic_series(
            scores,
            returns,
            "f",
            min_observations=5,
            use_cs_regression=True,
        )
        assert isinstance(result, pd.Series)
        assert len(result) == 12  # all rows kept (15 >= 5 valid pairs)
        assert np.isfinite(result.to_numpy()).all()

    def test_when_partial_coverage_slope_matches_manual_ols(self) -> None:
        """Masked-out pairs must not affect the fitted slope."""
        dates = pd.date_range("2023-01-01", periods=1, freq="ME")
        tickers = [f"T{i}" for i in range(6)]
        # y = 2*x on the 5 valid assets; the 6th pair is NaN and must be
        # excluded (not treated as 0) so the recovered slope is exactly 2.
        x_vals = [1.0, 2.0, 3.0, 4.0, 5.0, np.nan]
        y_vals = [2.0, 4.0, 6.0, 8.0, 10.0, np.nan]
        scores = pd.DataFrame([x_vals], index=dates, columns=tickers)
        returns = pd.DataFrame([y_vals], index=dates, columns=tickers)
        result = compute_ic_series(
            scores,
            returns,
            "f",
            min_observations=3,
            use_cs_regression=True,
            cs_config=CSLinearRegressionConfig(min_observations=3),
        )
        assert len(result) == 1
        assert result.iloc[0] == pytest.approx(2.0, abs=1e-6)

    def test_when_cs_config_min_observations_exceeds_valid_names_then_periods_dropped(
        self,
        panel: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """cs_config.min_observations gates periods, not just the function arg."""
        scores, returns = panel  # 20 valid names per period
        cfg = CSLinearRegressionConfig(min_observations=30)
        result = compute_ic_series(
            scores, returns, "f", use_cs_regression=True, cs_config=cfg
        )
        assert len(result) == 0

    def test_when_cs_config_none_then_default_min_observations_applies(self) -> None:
        """None defers to CSLinearRegressionConfig defaults (10 names)."""
        rng = np.random.default_rng(8)
        dates = pd.date_range("2023-01-01", periods=6, freq="ME")
        tickers = [f"T{i}" for i in range(8)]
        scores = pd.DataFrame(rng.normal(0, 1, (6, 8)), index=dates, columns=tickers)
        returns = 0.01 * scores
        result = compute_ic_series(scores, returns, "f", use_cs_regression=True)
        assert len(result) == 0

    def test_when_function_min_observations_stricter_than_cs_config_then_it_wins(
        self,
        panel: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        scores, returns = panel  # 20 valid names per period
        cfg = CSLinearRegressionConfig(min_observations=5)
        result = compute_ic_series(
            scores,
            returns,
            "f",
            min_observations=25,
            use_cs_regression=True,
            cs_config=cfg,
        )
        assert len(result) == 0

    def test_when_return_columns_reordered_then_slope_pairs_by_ticker(self) -> None:
        """Scores and returns are paired by ticker label, not column position."""
        rng = np.random.default_rng(5)
        dates = pd.date_range("2023-01-01", periods=6, freq="ME")
        tickers = [f"T{i:02d}" for i in range(10)]
        scores = pd.DataFrame(rng.normal(0, 1, (6, 10)), index=dates, columns=tickers)
        returns = 0.01 * scores
        shuffled = returns[tickers[::-1]]

        result = compute_ic_series(scores, shuffled, "f", use_cs_regression=True)

        assert len(result) == 6
        np.testing.assert_allclose(result.to_numpy(), 0.01, atol=1e-9)

    def test_when_returns_cover_extra_tickers_then_only_common_names_used(
        self,
    ) -> None:
        """A returns panel wider than the score panel must not crash the fit."""
        rng = np.random.default_rng(6)
        dates = pd.date_range("2023-01-01", periods=6, freq="ME")
        tickers = [f"T{i:02d}" for i in range(12)]
        returns = pd.DataFrame(
            rng.normal(0, 0.02, (6, 12)), index=dates, columns=tickers
        )
        scores = returns.iloc[:, :10] / 0.01

        result = compute_ic_series(scores, returns, "f", use_cs_regression=True)

        assert len(result) == 6
        np.testing.assert_allclose(result.to_numpy(), 0.01, atol=1e-9)

    @pytest.mark.parametrize("bad", [np.inf, -np.inf])
    def test_when_score_is_infinite_then_pair_excluded_like_nan(
        self, bad: float
    ) -> None:
        """A non-finite score is masked out, not clamped to float max."""
        dates = pd.date_range("2023-01-01", periods=2, freq="ME")
        tickers = [f"T{i}" for i in range(6)]
        x_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        scores = pd.DataFrame([x_vals, x_vals], index=dates, columns=tickers)
        returns = 2.0 * scores
        scores.iloc[0, 5] = bad

        result = compute_ic_series(
            scores,
            returns,
            "f",
            min_observations=3,
            use_cs_regression=True,
            cs_config=CSLinearRegressionConfig(min_observations=3),
        )

        assert len(result) == 2
        np.testing.assert_allclose(result.to_numpy(), 2.0, atol=1e-6)

    def test_when_return_is_infinite_then_pair_excluded_like_nan(self) -> None:
        dates = pd.date_range("2023-01-01", periods=1, freq="ME")
        tickers = [f"T{i}" for i in range(6)]
        scores = pd.DataFrame(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], index=dates, columns=tickers
        )
        returns = 2.0 * scores
        returns.iloc[0, 0] = np.inf

        result = compute_ic_series(
            scores,
            returns,
            "f",
            min_observations=3,
            use_cs_regression=True,
            cs_config=CSLinearRegressionConfig(min_observations=3),
        )

        assert len(result) == 1
        assert result.iloc[0] == pytest.approx(2.0, abs=1e-6)

    def test_when_infinite_pairs_leave_too_few_names_then_period_dropped(
        self,
    ) -> None:
        dates = pd.date_range("2023-01-01", periods=1, freq="ME")
        tickers = [f"T{i}" for i in range(4)]
        scores = pd.DataFrame(
            [[1.0, 2.0, np.inf, -np.inf]], index=dates, columns=tickers
        )
        returns = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]], index=dates, columns=tickers)

        result = compute_ic_series(
            scores,
            returns,
            "f",
            min_observations=3,
            use_cs_regression=True,
            cs_config=CSLinearRegressionConfig(min_observations=3),
        )

        assert len(result) == 0

    @pytest.mark.parametrize("constant", [0.0, 0.1, 3.7])
    def test_when_period_scores_are_constant_then_period_dropped(
        self, constant: float
    ) -> None:
        """A zero-dispersion cross-section has no rank IC and no slope either."""
        rng = np.random.default_rng(21)
        dates = pd.date_range("2023-01-01", periods=6, freq="ME")
        tickers = [f"T{i:02d}" for i in range(12)]
        scores = pd.DataFrame(rng.normal(0, 1, (6, 12)), index=dates, columns=tickers)
        returns = pd.DataFrame(
            0.05 * scores.values + rng.normal(0, 0.02, (6, 12)),
            index=dates,
            columns=tickers,
        )
        scores.iloc[2] = constant

        result = compute_ic_series(scores, returns, "f", use_cs_regression=True)

        assert dates[2] not in result.index
        assert len(result) == 5

    def test_when_valid_scores_all_tied_after_masking_then_period_dropped(
        self,
    ) -> None:
        """Ties among the finite names count, not the NaN placeholders."""
        dates = pd.date_range("2023-01-01", periods=2, freq="ME")
        tickers = [f"T{i}" for i in range(6)]
        scores = pd.DataFrame(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [0.5, 0.5, 0.5, np.nan, np.nan, np.nan],
            ],
            index=dates,
            columns=tickers,
        )
        returns = pd.DataFrame(
            [[2.0, 4.0, 6.0, 8.0, 10.0, 12.0], [0.1, -0.2, 0.3, 0.0, 0.0, 0.0]],
            index=dates,
            columns=tickers,
        )

        result = compute_ic_series(
            scores,
            returns,
            "f",
            min_observations=3,
            use_cs_regression=True,
            cs_config=CSLinearRegressionConfig(min_observations=3),
        )

        assert list(result.index) == [dates[0]]
        assert result.iloc[0] == pytest.approx(2.0, abs=1e-6)

    def test_when_scores_tiny_but_dispersed_then_period_kept(self) -> None:
        """The zero-dispersion test is scale-free, not an absolute epsilon."""
        dates = pd.date_range("2023-01-01", periods=1, freq="ME")
        tickers = [f"T{i}" for i in range(6)]
        scores = pd.DataFrame(
            [[1e-9, 2e-9, 3e-9, 4e-9, 5e-9, 6e-9]], index=dates, columns=tickers
        )
        returns = 2.0e9 * scores

        result = compute_ic_series(
            scores,
            returns,
            "f",
            min_observations=3,
            use_cs_regression=True,
            cs_config=CSLinearRegressionConfig(min_observations=3),
        )

        assert len(result) == 1
        assert result.iloc[0] == pytest.approx(2.0e9, rel=1e-6)


class TestNeweyWestTStat:
    def test_significant_ic(self) -> None:
        rng = np.random.default_rng(42)
        # Strong positive IC series
        ic = pd.Series(rng.normal(0.05, 0.02, 60))
        t_stat, p_value = compute_newey_west_tstat(ic)
        assert t_stat > 2.0
        assert p_value < 0.05

    def test_insignificant_ic(self) -> None:
        rng = np.random.default_rng(42)
        # Weak, noisy IC series
        ic = pd.Series(rng.normal(0.001, 0.1, 20))
        t_stat, _p_value = compute_newey_west_tstat(ic)
        assert abs(t_stat) < 3.0

    def test_short_series(self) -> None:
        ic = pd.Series([0.05, 0.06])
        t_stat, _p_value = compute_newey_west_tstat(ic)
        assert isinstance(t_stat, float)

    @pytest.mark.parametrize("n", [4, 6, 9, 12, 30])
    def test_when_series_short_then_matches_textbook_newey_west(self, n: int) -> None:
        """Autocovariances are normalised by n so the HAC variance stays >= 0."""
        n_lags = min(6, (n - 1) // 2)
        rng = np.random.default_rng(n)
        for _ in range(25):
            values = rng.normal(0.0, 0.1, n)
            t_stat, p_value = compute_newey_west_tstat(pd.Series(values), n_lags)
            ref_t, ref_p = _reference_newey_west(values, n_lags)
            assert t_stat == pytest.approx(ref_t, rel=1e-9)
            assert p_value == pytest.approx(ref_p, rel=1e-9, abs=1e-15)

    def test_when_short_alternating_series_then_t_stat_not_inflated(self) -> None:
        t_stat, p_value = compute_newey_west_tstat(
            pd.Series([0.15, 0.05, 0.15, 0.05]), 1
        )
        assert abs(t_stat) < 100.0
        assert p_value > 1e-6

    @pytest.mark.parametrize(("n_lags", "n"), [(6, 12), (6, 5), (2, 4), (4, 8)])
    def test_when_series_shorter_than_two_lags_plus_one_then_no_evidence(
        self, n_lags: int, n: int
    ) -> None:
        """Below 2 * n_lags + 1 periods the HAC variance is not usable."""
        rng = np.random.default_rng(n)
        ic = pd.Series(0.05 + rng.normal(0.0, 1e-3, n))
        assert compute_newey_west_tstat(ic, n_lags) == (0.0, 1.0)

    @pytest.mark.parametrize("n_lags", [1, 2, 6])
    def test_when_series_has_two_lags_plus_one_periods_then_inference_runs(
        self, n_lags: int
    ) -> None:
        n = 2 * n_lags + 1
        rng = np.random.default_rng(n)
        values = 0.05 + rng.normal(0.0, 1e-2, n)

        t_stat, p_value = compute_newey_west_tstat(pd.Series(values), n_lags)
        ref_t, ref_p = _reference_newey_west(values, n_lags)

        assert t_stat == pytest.approx(ref_t, rel=1e-9)
        assert p_value == pytest.approx(ref_p, rel=1e-9, abs=1e-15)
        assert abs(t_stat) > 2.0

    def test_when_ic_history_is_pure_noise_and_short_then_never_significant(
        self,
    ) -> None:
        """The documented 2 * lags + 1 floor keeps noise out of short histories."""
        rng = np.random.default_rng(2024)
        flagged = sum(
            abs(compute_newey_west_tstat(pd.Series(rng.normal(0.0, 0.1, 12)), 6)[0])
            >= 2.0
            for _ in range(400)
        )
        assert flagged == 0

    @pytest.mark.parametrize("n", [10, 20])
    def test_when_ic_series_constant_then_no_evidence(self, n: int) -> None:
        """A constant IC series has no variance to infer from: t=0, p=1."""
        t_stat, p_value = compute_newey_west_tstat(pd.Series([0.05] * n), 6)
        assert t_stat == 0.0
        assert p_value == 1.0

    @pytest.mark.parametrize("scale", [1e-3, 1e-7, 1e-9, 1e-12, 1e3])
    def test_when_series_rescaled_then_t_stat_unchanged(self, scale: float) -> None:
        """The t-statistic is scale-free: no absolute variance floor."""
        rng = np.random.default_rng(0)
        base = pd.Series(rng.normal(2.0, 1.0, 60))
        ref_t, ref_p = compute_newey_west_tstat(base, 6)

        t_stat, p_value = compute_newey_west_tstat(base * scale, 6)

        assert t_stat == pytest.approx(ref_t, rel=1e-9)
        assert p_value == pytest.approx(ref_p, rel=1e-9, abs=1e-15)

    def test_when_cs_slopes_tiny_then_t_stat_matches_standardised_slopes(
        self,
    ) -> None:
        """Slopes on unscaled scores (raw market cap) are ~1e-12 per unit."""
        rng = np.random.default_rng(4)
        dates = pd.date_range("2020-01-31", periods=60, freq="ME")
        tickers = [f"T{i:02d}" for i in range(40)]
        caps = pd.DataFrame(
            rng.lognormal(23.0, 1.0, (60, 40)), index=dates, columns=tickers
        )
        returns = pd.DataFrame(
            1e-3 * (caps.values - caps.values.mean(axis=1, keepdims=True)) / 1e10
            + rng.normal(0.0, 1e-3, (60, 40)),
            index=dates,
            columns=tickers,
        )
        slopes = compute_ic_series(caps, returns, "cap", use_cs_regression=True)

        t_raw, _ = compute_newey_west_tstat(slopes, 6)
        t_scaled, _ = compute_newey_west_tstat(slopes * 1e10, 6)

        assert t_raw == pytest.approx(t_scaled, rel=1e-9)
        assert abs(t_raw) > 10.0

    def test_when_series_has_nan_then_matches_series_with_nan_dropped(self) -> None:
        """A gap is a missing period, not a poisoned HAC sum."""
        ic = pd.Series(_IC_WITH_GAP)

        t_stat, p_value = compute_newey_west_tstat(ic, 6)
        ref_t, ref_p = compute_newey_west_tstat(ic.dropna(), 6)

        assert np.isfinite(t_stat)
        assert t_stat > 2.0
        assert t_stat == pytest.approx(ref_t, rel=1e-12)
        assert p_value == pytest.approx(ref_p, rel=1e-12, abs=1e-15)

    def test_when_nan_leaves_fewer_than_three_values_then_no_evidence(self) -> None:
        ic = pd.Series([0.05, np.nan, 0.04, np.nan, np.nan])
        assert compute_newey_west_tstat(ic, 6) == (0.0, 1.0)


class TestQuantileSpread:
    def test_returns_float(
        self, factor_scores: pd.Series, forward_returns: pd.Series
    ) -> None:
        spread = compute_quantile_spread(factor_scores, forward_returns, n_quantiles=5)
        assert isinstance(spread, float)

    def test_insufficient_data(self) -> None:
        scores = pd.Series([1.0, 2.0], index=["A", "B"])
        returns = pd.Series([0.1, 0.2], index=["A", "B"])
        spread = compute_quantile_spread(scores, returns, n_quantiles=5)
        assert np.isnan(spread)

    def test_no_ties_top_minus_bottom(self) -> None:
        scores = pd.Series(
            [float(i) for i in range(10)],
            index=[f"S{i:02d}" for i in range(10)],
        )
        returns = pd.Series(
            [float(i) * 0.01 for i in range(10)],
            index=[f"S{i:02d}" for i in range(10)],
        )
        spread = compute_quantile_spread(scores, returns, n_quantiles=5)
        # Bottom quintile: assets 0,1 → mean return = 0.005
        # Top quintile: assets 8,9 → mean return = 0.085
        assert pytest.approx(spread, abs=1e-10) == 0.085 - 0.005

    def test_mid_range_ties_correct_top_and_bottom(self) -> None:
        n = 100
        index = [f"T{i:03d}" for i in range(n)]
        scores_vals = [0.0] * 15 + [0.5] * 70 + [1.0] * 15
        returns_vals = [-0.02] * 15 + [0.0] * 70 + [0.02] * 15
        scores = pd.Series(scores_vals, index=index)
        returns = pd.Series(returns_vals, index=index)
        spread = compute_quantile_spread(scores, returns, n_quantiles=5)
        assert not np.isnan(spread)
        assert spread > 0.0

    def test_all_identical_scores_returns_nan(self) -> None:
        index = [f"U{i:02d}" for i in range(10)]
        scores = pd.Series([1.0] * 10, index=index)
        returns = pd.Series([0.01] * 10, index=index)
        spread = compute_quantile_spread(scores, returns, n_quantiles=5)
        assert np.isnan(spread)


class TestComputeVIF:
    def test_independent_factors(self) -> None:
        rng = np.random.default_rng(42)
        factors = pd.DataFrame(
            rng.normal(0, 1, (100, 3)),
            columns=["a", "b", "c"],
        )
        vif = compute_vif(factors)
        # Independent factors should have VIF close to 1
        assert (vif < 2.0).all()

    def test_collinear_factors(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 100)
        factors = pd.DataFrame(
            {
                "a": a,
                "b": a + rng.normal(0, 0.01, 100),  # nearly identical
                "c": rng.normal(0, 1, 100),
            }
        )
        vif = compute_vif(factors)
        # Collinear factors should have high VIF
        assert vif["a"] > 5.0 or vif["b"] > 5.0

    def test_single_factor_raises(self) -> None:
        factors = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        with pytest.raises(DataError, match="at least 2 factor columns"):
            compute_vif(factors)

    def test_near_singular_r2_returns_inf(self) -> None:
        """Near-singular R² above floating-point tolerance must return inf."""
        rng = np.random.default_rng(0)
        base = rng.normal(0, 1, 200)
        # epsilon = 1e-7 makes R² ≈ 1 - (tiny noise / base variance),
        # well above the 1e-10 singularity tolerance → VIF must be inf.
        epsilon_col = base + rng.normal(0, 1e-7, 200)
        factors = pd.DataFrame(
            {"a": base, "b": epsilon_col, "c": rng.normal(0, 1, 200)}
        )
        result = compute_vif(factors)
        assert np.isinf(result["a"]) or np.isinf(result["b"]), (
            f"Expected inf for near-singular factor, "
            f"got a={result['a']}, b={result['b']}"
        )

    def test_moderate_collinearity_preserves_finite_vif(self) -> None:
        """R² ≈ 0.99 is diagnostically meaningful — VIF must remain finite."""
        rng = np.random.default_rng(1)
        base = rng.normal(0, 1, 500)
        # noise std = 0.1 → R² ≈ 0.99; 1 - R² ≈ 0.01 >> 1e-10
        noisy_col = base + rng.normal(0, 0.1, 500)
        factors = pd.DataFrame({"a": base, "b": noisy_col, "c": rng.normal(0, 1, 500)})
        result = compute_vif(factors)
        assert np.isfinite(result["a"]) and np.isfinite(result["b"]), (
            f"Expected finite VIF for R²≈0.99, got {result}"
        )
        assert result["a"] > 50 or result["b"] > 50

    def test_exact_collinearity_returns_inf_not_nan(self) -> None:
        """Exact collinearity must yield inf, never NaN."""
        rng = np.random.default_rng(2)
        base = rng.normal(0, 1, 200)
        exact = base * 2.0
        factors = pd.DataFrame({"a": base, "b": exact, "c": rng.normal(0, 1, 200)})
        result = compute_vif(factors)
        inf_vals = result[np.isinf(result)]
        assert not inf_vals.isna().any(), "inf VIF values must not be NaN"


class TestBenjaminiHochberg:
    def test_all_significant(self) -> None:
        p_values = pd.Series(
            [0.001, 0.002, 0.003],
            index=["a", "b", "c"],
        )
        result = benjamini_hochberg(p_values, alpha=0.05)
        assert result.all()

    def test_none_significant(self) -> None:
        p_values = pd.Series(
            [0.5, 0.8, 0.9],
            index=["a", "b", "c"],
        )
        result = benjamini_hochberg(p_values, alpha=0.05)
        assert not result.any()

    def test_partial_significance(self) -> None:
        p_values = pd.Series(
            [0.001, 0.5, 0.8],
            index=["a", "b", "c"],
        )
        result = benjamini_hochberg(p_values, alpha=0.05)
        assert result["a"]


class TestComputeICStats:
    def test_returns_icstats(self) -> None:
        rng = np.random.default_rng(42)
        ic = pd.Series(rng.normal(0.05, 0.02, 60))
        result = compute_ic_stats(ic)
        assert isinstance(result, ICStats)

    def test_fields_present(self) -> None:
        rng = np.random.default_rng(42)
        ic = pd.Series(rng.normal(0.05, 0.02, 60))
        result = compute_ic_stats(ic)
        assert hasattr(result, "mean")
        assert hasattr(result, "variance_nw")
        assert hasattr(result, "t_stat_nw")
        assert hasattr(result, "p_value")
        assert hasattr(result, "icir")

    def test_icir_matches_hand_calculation(self) -> None:
        rng = np.random.default_rng(7)
        ic = pd.Series(rng.normal(0.04, 0.03, 50))
        result = compute_ic_stats(ic)
        expected_icir = float(ic.mean()) / float(ic.std(ddof=1))
        assert pytest.approx(result.icir, rel=1e-6) == expected_icir

    def test_mean_matches_ic_mean(self) -> None:
        rng = np.random.default_rng(0)
        ic = pd.Series(rng.normal(0.03, 0.02, 40))
        result = compute_ic_stats(ic)
        assert pytest.approx(result.mean, rel=1e-9) == float(ic.mean())

    def test_significant_series_low_pvalue(self) -> None:
        rng = np.random.default_rng(42)
        ic = pd.Series(rng.normal(0.06, 0.015, 60))
        result = compute_ic_stats(ic)
        assert result.p_value < 0.05
        assert result.t_stat_nw > 2.0

    def test_noise_series_high_pvalue(self) -> None:
        rng = np.random.default_rng(42)
        ic = pd.Series(rng.normal(0.0, 0.1, 20))
        result = compute_ic_stats(ic)
        # noisy zero-mean series should not be significant
        assert result.p_value > 0.05

    def test_short_series_returns_nan(self) -> None:
        ic = pd.Series([0.05, 0.03])
        result = compute_ic_stats(ic)
        assert np.isnan(result.mean)
        assert result.p_value == 1.0

    def test_variance_nw_is_positive(self) -> None:
        rng = np.random.default_rng(3)
        ic = pd.Series(rng.normal(0.04, 0.02, 36))
        result = compute_ic_stats(ic)
        assert result.variance_nw > 0.0

    @pytest.mark.parametrize("n", [3, 4, 6, 9, 12, 30])
    def test_when_series_short_then_matches_textbook_newey_west(self, n: int) -> None:
        """Autocovariances are normalised by n so the HAC variance stays >= 0."""
        n_lags = min(5, (n - 1) // 2)
        rng = np.random.default_rng(100 + n)
        for _ in range(25):
            values = rng.normal(0.0, 0.1, n)
            result = compute_ic_stats(pd.Series(values), lags=n_lags)
            ref_t, ref_p = _reference_newey_west(values, n_lags)
            assert result.t_stat_nw == pytest.approx(ref_t, rel=1e-9)
            assert result.p_value == pytest.approx(ref_p, rel=1e-9, abs=1e-15)
            assert result.variance_nw > 0.0

    def test_when_short_series_then_t_stat_not_inflated(self) -> None:
        result = compute_ic_stats(pd.Series([0.1, 0.0, 0.1]), lags=1)
        assert abs(result.t_stat_nw) < 100.0
        assert result.p_value > 1e-6

    @pytest.mark.parametrize(("lags", "n"), [(5, 10), (5, 4), (2, 4), (4, 8)])
    def test_when_series_shorter_than_two_lags_plus_one_then_no_inference(
        self, lags: int, n: int
    ) -> None:
        """Below 2 * lags + 1 periods no t-test is run; descriptives still are."""
        rng = np.random.default_rng(n)
        ic = pd.Series(0.05 + rng.normal(0.0, 1e-3, n))

        result = compute_ic_stats(ic, lags=lags)

        assert result.t_stat_nw == 0.0
        assert result.p_value == 1.0
        assert result.mean == pytest.approx(float(ic.mean()), rel=1e-12)
        assert result.icir == pytest.approx(
            float(ic.mean()) / float(ic.std(ddof=1)), rel=1e-9
        )

    @pytest.mark.parametrize("lags", [1, 2, 5])
    def test_when_series_has_two_lags_plus_one_periods_then_inference_runs(
        self, lags: int
    ) -> None:
        n = 2 * lags + 1
        rng = np.random.default_rng(n)
        values = 0.05 + rng.normal(0.0, 1e-2, n)

        result = compute_ic_stats(pd.Series(values), lags=lags)
        ref_t, ref_p = _reference_newey_west(values, lags)

        assert result.t_stat_nw == pytest.approx(ref_t, rel=1e-9)
        assert result.p_value == pytest.approx(ref_p, rel=1e-9, abs=1e-15)
        assert abs(result.t_stat_nw) > 2.0

    @pytest.mark.parametrize("n", [10, 20])
    def test_when_ic_series_constant_then_no_evidence(self, n: int) -> None:
        result = compute_ic_stats(pd.Series([0.05] * n), lags=5)
        assert result.t_stat_nw == 0.0
        assert result.p_value == 1.0
        assert result.variance_nw == 0.0

    def test_when_universe_ic_history_short_then_noise_not_flagged(self) -> None:
        """Spurious p ~ 0 on short histories must not survive the corrections."""
        rng = np.random.default_rng(1)
        ic_matrix = pd.DataFrame(rng.normal(0.0, 0.1, (6, 20)))
        result = validate_factor_universe(ic_matrix, lags=5)
        assert result["p_value_raw"].min() > 1e-6
        assert result["significant_holm"].sum() == 0

    def test_when_universe_history_below_floor_then_no_factor_flagged(self) -> None:
        """A 10-period history with 5 lags is under the 2 * lags + 1 floor."""
        rng = np.random.default_rng(5)
        ic_matrix = pd.DataFrame(rng.normal(0.0, 0.1, (10, 200)))

        result = validate_factor_universe(ic_matrix, lags=5)

        assert (result["p_value_raw"] == 1.0).all()
        assert result["significant_bh"].sum() == 0

    @pytest.mark.parametrize("scale", [1e-3, 1e-9, 1e-12, 1e3])
    def test_when_series_rescaled_then_t_stat_and_icir_unchanged(
        self, scale: float
    ) -> None:
        rng = np.random.default_rng(0)
        base = pd.Series(rng.normal(2.0, 1.0, 60))
        ref = compute_ic_stats(base)

        scaled = compute_ic_stats(base * scale)

        assert scaled.t_stat_nw == pytest.approx(ref.t_stat_nw, rel=1e-9)
        assert scaled.icir == pytest.approx(ref.icir, rel=1e-9)

    def test_when_series_has_nan_then_stats_match_series_with_nan_dropped(
        self,
    ) -> None:
        """NaN is dropped before n, mean and the HAC sums, so they stay consistent."""
        ic = pd.Series(_IC_WITH_GAP)

        result = compute_ic_stats(ic, lags=5)
        ref = compute_ic_stats(ic.dropna(), lags=5)

        assert np.isfinite(result.variance_nw)
        assert result.t_stat_nw > 2.0
        assert result.p_value < 0.05
        assert result.mean == pytest.approx(ref.mean, rel=1e-12)
        assert result.variance_nw == pytest.approx(ref.variance_nw, rel=1e-12)
        assert result.t_stat_nw == pytest.approx(ref.t_stat_nw, rel=1e-12)
        assert result.p_value == pytest.approx(ref.p_value, rel=1e-12, abs=1e-15)
        assert result.icir == pytest.approx(ref.icir, rel=1e-12)

    def test_when_nan_leaves_fewer_than_three_values_then_short_series_result(
        self,
    ) -> None:
        ic = pd.Series([0.05, np.nan, 0.04, np.nan, np.nan])
        result = compute_ic_stats(ic)
        assert np.isnan(result.mean)
        assert np.isnan(result.variance_nw)
        assert result.t_stat_nw == 0.0
        assert result.p_value == 1.0


class TestICIRConstantSeries:
    """A constant IC series has no dispersion: ICIR is 0.0, not mean / noise."""

    @pytest.mark.parametrize(
        ("value", "n"), [(0.05, 20), (0.1, 13), (0.2, 7), (-0.03, 20)]
    )
    def test_when_ic_series_constant_then_compute_icir_is_zero(
        self, value: float, n: int
    ) -> None:
        assert compute_icir(pd.Series([value] * n)) == 0.0

    @pytest.mark.parametrize(
        ("value", "n"), [(0.05, 20), (0.1, 13), (0.2, 7), (-0.03, 20)]
    )
    def test_when_ic_series_constant_then_ic_stats_icir_is_zero(
        self, value: float, n: int
    ) -> None:
        assert compute_ic_stats(pd.Series([value] * n)).icir == 0.0

    def test_when_composite_ic_constant_then_icir_is_zero(self) -> None:
        rng = np.random.default_rng(0)
        n_tickers, n_dates = 8, 20
        dates = pd.date_range("2020-01-31", periods=n_dates, freq="ME")
        tickers = [f"T{i}" for i in range(n_tickers)]
        score_row = rng.normal(size=n_tickers)
        return_row = rng.normal(size=n_tickers) * 0.02
        scores = pd.DataFrame([score_row] * n_dates, index=dates, columns=tickers)
        returns = pd.DataFrame([return_row] * n_dates, index=dates, columns=tickers)

        result = compute_composite_ic(scores, returns)

        assert result.mean_ic == pytest.approx(0.238095, abs=1e-6)
        assert result.icir == 0.0

    def test_when_ic_series_small_but_dispersed_then_icir_is_finite_ratio(
        self,
    ) -> None:
        """The degenerate test is relative to the series scale."""
        rng = np.random.default_rng(3)
        ic = pd.Series(rng.normal(2.0, 1.0, 40) * 1e-9)
        expected = float(ic.mean()) / float(ic.std(ddof=1))
        assert compute_icir(ic) == pytest.approx(expected, rel=1e-12)
        assert compute_ic_stats(ic).icir == pytest.approx(expected, rel=1e-12)


class TestCorrectPValues:
    def test_returns_corrected_pvalues(self) -> None:
        raw = np.array([0.01, 0.04, 0.20, 0.50])
        result = correct_pvalues(raw)
        assert isinstance(result, CorrectedPValues)

    def test_adjusted_geq_raw_holm(self) -> None:
        rng = np.random.default_rng(0)
        raw = rng.uniform(0, 1, 20)
        result = correct_pvalues(raw)
        assert (result.holm >= raw - 1e-12).all()

    def test_adjusted_geq_raw_bh(self) -> None:
        rng = np.random.default_rng(0)
        raw = rng.uniform(0, 1, 20)
        result = correct_pvalues(raw)
        assert (result.bh >= raw - 1e-12).all()

    def test_holm_bounded_by_one(self) -> None:
        raw = np.array([0.01, 0.04, 0.20, 0.50])
        result = correct_pvalues(raw)
        assert (result.holm <= 1.0).all()

    def test_bh_bounded_by_one(self) -> None:
        raw = np.array([0.01, 0.04, 0.20, 0.50])
        result = correct_pvalues(raw)
        assert (result.bh <= 1.0).all()

    def test_holm_monotone_in_sorted_order(self) -> None:
        raw = np.sort(np.array([0.005, 0.01, 0.03, 0.10, 0.40]))
        result = correct_pvalues(raw)
        sorted_holm = result.holm[np.argsort(raw)]
        # Holm-adjusted p-values must be non-decreasing when input is sorted
        assert (np.diff(sorted_holm) >= -1e-12).all()

    def test_bh_monotone_in_sorted_order(self) -> None:
        raw = np.sort(np.array([0.005, 0.01, 0.03, 0.10, 0.40]))
        result = correct_pvalues(raw)
        sorted_bh = result.bh[np.argsort(raw)]
        assert (np.diff(sorted_bh) >= -1e-12).all()

    def test_empty_input(self) -> None:
        result = correct_pvalues(np.array([], dtype=np.float64))
        assert len(result.holm) == 0
        assert len(result.bh) == 0

    def test_single_pvalue_unchanged(self) -> None:
        raw = np.array([0.03])
        result = correct_pvalues(raw)
        # With m=1, no correction needed
        assert pytest.approx(result.holm[0], abs=1e-9) == 0.03
        assert pytest.approx(result.bh[0], abs=1e-9) == 0.03

    def test_output_length_matches_input(self) -> None:
        raw = np.array([0.01, 0.05, 0.10, 0.30, 0.80])
        result = correct_pvalues(raw)
        assert len(result.holm) == 5
        assert len(result.bh) == 5

    def test_order_preserved(self) -> None:
        # Input is unsorted; output indices must match input order
        raw = np.array([0.40, 0.01, 0.10, 0.05])
        result = correct_pvalues(raw)
        # Smallest raw (index 1) should have smallest Holm and BH
        assert result.holm[1] <= result.holm[0]
        assert result.bh[1] <= result.bh[0]


class TestValidateFactorUniverse:
    @pytest.fixture()
    def ic_matrix(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        dates = pd.date_range("2020-01-01", periods=60, freq="ME")
        # Two informative factors, one noise factor
        value_ic = pd.Series(rng.normal(0.06, 0.02, 60), index=dates)
        momentum_ic = pd.Series(rng.normal(0.05, 0.025, 60), index=dates)
        noise_ic = pd.Series(rng.normal(0.0, 0.15, 60), index=dates)
        return pd.DataFrame(
            {"value": value_ic, "momentum": momentum_ic, "noise": noise_ic}
        )

    def test_returns_dataframe(self, ic_matrix: pd.DataFrame) -> None:
        result = validate_factor_universe(ic_matrix)
        assert isinstance(result, pd.DataFrame)

    def test_index_matches_factors(self, ic_matrix: pd.DataFrame) -> None:
        result = validate_factor_universe(ic_matrix)
        assert list(result.index) == list(ic_matrix.columns)

    def test_required_columns_present(self, ic_matrix: pd.DataFrame) -> None:
        result = validate_factor_universe(ic_matrix)
        required = {
            "ic_mean",
            "icir",
            "t_stat_nw",
            "p_value_raw",
            "p_value_holm",
            "p_value_bh",
            "significant_holm",
            "significant_bh",
        }
        assert required.issubset(set(result.columns))

    def test_corrected_pvalues_geq_raw(self, ic_matrix: pd.DataFrame) -> None:
        result = validate_factor_universe(ic_matrix)
        assert (result["p_value_holm"] >= result["p_value_raw"] - 1e-12).all()
        assert (result["p_value_bh"] >= result["p_value_raw"] - 1e-12).all()

    def test_significant_flags_are_boolean_like(self, ic_matrix: pd.DataFrame) -> None:
        result = validate_factor_universe(ic_matrix)
        assert result["significant_holm"].isin([0.0, 1.0]).all()
        assert result["significant_bh"].isin([0.0, 1.0]).all()

    def test_icir_sign_matches_ic_mean(self, ic_matrix: pd.DataFrame) -> None:
        result = validate_factor_universe(ic_matrix)
        for factor in ic_matrix.columns:
            row = result.loc[factor]
            if not np.isnan(row["icir"]) and row["ic_mean"] != 0:
                assert np.sign(row["icir"]) == np.sign(row["ic_mean"])

    def test_single_factor(self) -> None:
        rng = np.random.default_rng(1)
        ic = pd.Series(rng.normal(0.05, 0.02, 36))
        ic_matrix = pd.DataFrame({"value": ic})
        result = validate_factor_universe(ic_matrix)
        assert len(result) == 1
        assert "ic_mean" in result.columns


class TestRunFactorValidation:
    def test_full_validation(self) -> None:
        rng = np.random.default_rng(42)
        dates = pd.date_range("2023-01-01", periods=36, freq="ME")
        tickers = [f"T{i:02d}" for i in range(30)]

        factor_history = {
            "value": pd.DataFrame(
                rng.normal(0, 1, (36, 30)), index=dates, columns=tickers
            ),
            "momentum": pd.DataFrame(
                rng.normal(0, 1, (36, 30)), index=dates, columns=tickers
            ),
        }
        returns_hist = pd.DataFrame(
            rng.normal(0.001, 0.02, (36, 30)), index=dates, columns=tickers
        )

        report = run_factor_validation(factor_history, returns_hist)
        assert isinstance(report, FactorValidationReport)
        assert len(report.ic_results) == 2

    def test_min_ic_observations_threads_to_run_factor_validation(self) -> None:
        """min_ic_observations in config is forwarded to compute_ic_series."""
        rng = np.random.default_rng(11)
        dates = pd.date_range("2021-01-01", periods=12, freq="ME")
        # 5 tickers: passes min_ic_observations=4 but fails min_ic_observations=6
        tickers = [f"S{i}" for i in range(5)]
        factor_history = {
            "value": pd.DataFrame(
                rng.normal(0, 1, (12, 5)), index=dates, columns=tickers
            ),
        }
        returns_hist = pd.DataFrame(
            rng.normal(0.001, 0.02, (12, 5)), index=dates, columns=tickers
        )

        # With min=4: 5 tickers >= 4 → IC series is non-empty → result has IC result
        permissive_config = FactorValidationConfig(min_ic_observations=4)
        report_permissive = run_factor_validation(
            factor_history, returns_hist, config=permissive_config
        )
        assert len(report_permissive.ic_results) == 1

        # With min=6: 5 tickers < 6 → all cross-sections dropped → factor skipped
        strict_config = FactorValidationConfig(min_ic_observations=6)
        report_strict = run_factor_validation(
            factor_history, returns_hist, config=strict_config
        )
        assert len(report_strict.ic_results) == 0


class TestFactorSpreadBenchmarks:
    """Tests for benchmark thresholds and Holm FWER (issue #80)."""

    def test_benchmark_dict_has_all_groups(self) -> None:
        expected = {
            "value",
            "profitability",
            "investment",
            "momentum",
            "low_risk",
            "liquidity",
            "dividend",
            "sentiment",
            "ownership",
        }
        assert set(FACTOR_SPREAD_BENCHMARKS.keys()) == expected

    def test_benchmark_bounds_ordered(self) -> None:
        for group, (lo, hi) in FACTOR_SPREAD_BENCHMARKS.items():
            assert lo < hi, f"{group} benchmark bounds not ordered"
            assert lo >= 0, f"{group} lower bound negative"

    def test_within_benchmark_flag(self) -> None:
        rng = np.random.default_rng(42)
        dates = pd.date_range("2023-01-01", periods=36, freq="ME")
        tickers = [f"T{i:02d}" for i in range(30)]

        factor_history = {
            "book_to_price": pd.DataFrame(
                rng.normal(0, 1, (36, 30)), index=dates, columns=tickers
            ),
        }
        returns_hist = pd.DataFrame(
            rng.normal(0.001, 0.02, (36, 30)), index=dates, columns=tickers
        )

        report = run_factor_validation(factor_history, returns_hist)
        for qs in report.quantile_spreads:
            assert isinstance(qs.within_benchmark, bool)

    @staticmethod
    def _linear_panel(
        per_period_spreads: list[float], n_tickers: int = 30
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        """Panel whose date-t quintile spread equals ``per_period_spreads[t]``."""
        n_dates = len(per_period_spreads)
        dates = pd.date_range("2021-01-31", periods=n_dates, freq="ME")
        tickers = [f"T{i:02d}" for i in range(n_tickers)]
        ranks = np.arange(n_tickers, dtype=float)
        scores = pd.DataFrame(
            np.tile(ranks, (n_dates, 1)), index=dates, columns=tickers
        )
        unit = compute_quantile_spread(scores.iloc[0], scores.iloc[0], n_quantiles=5)
        returns = pd.DataFrame(
            np.outer(np.asarray(per_period_spreads) / unit, ranks),
            index=dates,
            columns=tickers,
        )
        return {"momentum": scores}, returns

    def test_when_monthly_spread_annualises_into_band_then_within_benchmark(
        self,
    ) -> None:
        """0.5%/month is ~6% a year: inside the 4%-10% momentum band."""
        factors, returns = self._linear_panel([0.005] * 36)
        report = run_factor_validation(factors, returns)
        (qs,) = report.quantile_spreads
        assert qs.spread == pytest.approx(0.005)
        assert qs.within_benchmark is True

    def test_when_monthly_spread_matches_annual_band_numerically_then_not_within(
        self,
    ) -> None:
        """A 6% monthly spread is ~72% a year, far above the momentum band."""
        factors, returns = self._linear_panel([0.06] * 36)
        report = run_factor_validation(factors, returns)
        (qs,) = report.quantile_spreads
        assert qs.within_benchmark is False

    def test_when_spread_negative_then_not_within_benchmark(self) -> None:
        """A reversed factor (loser-minus-winner) is not inside the benchmark."""
        factors, returns = self._linear_panel([-0.06] * 36)
        report = run_factor_validation(
            factors, returns, FactorValidationConfig(periods_per_year=1)
        )
        (qs,) = report.quantile_spreads
        assert qs.spread == pytest.approx(-0.06)
        assert qs.within_benchmark is False

    @pytest.mark.filterwarnings("ignore:An input array is constant")
    def test_when_latest_date_is_noise_then_spread_averaged_over_dates(self) -> None:
        """One bad final month must not decide the benchmark flag."""
        factors, returns = self._linear_panel([0.0055] * 35 + [0.0])
        report = run_factor_validation(factors, returns)
        (qs,) = report.quantile_spreads
        assert qs.spread == pytest.approx(0.0055 * 35 / 36)
        assert qs.within_benchmark is True

    def test_when_periods_per_year_set_then_annualisation_follows_it(self) -> None:
        """Quarterly forward returns: 1.5%/quarter is 6% a year."""
        factors, returns = self._linear_panel([0.015] * 36)
        monthly = run_factor_validation(factors, returns)
        quarterly = run_factor_validation(
            factors, returns, FactorValidationConfig(periods_per_year=4)
        )
        assert monthly.quantile_spreads[0].within_benchmark is False
        assert quarterly.quantile_spreads[0].within_benchmark is True

    def test_significant_factors_holm_populated(self) -> None:
        rng = np.random.default_rng(42)
        dates = pd.date_range("2023-01-01", periods=60, freq="ME")
        tickers = [f"T{i:02d}" for i in range(50)]

        # Strong signal factors
        factor_history = {
            "value": pd.DataFrame(
                rng.normal(0, 1, (60, 50)), index=dates, columns=tickers
            ),
            "momentum": pd.DataFrame(
                rng.normal(0, 1, (60, 50)), index=dates, columns=tickers
            ),
        }
        returns_hist = pd.DataFrame(
            rng.normal(0.001, 0.02, (60, 50)), index=dates, columns=tickers
        )

        report = run_factor_validation(factor_history, returns_hist)
        assert isinstance(report.significant_factors_holm, list)

    def test_holm_subset_of_bh(self) -> None:
        rng = np.random.default_rng(42)
        dates = pd.date_range("2023-01-01", periods=60, freq="ME")
        tickers = [f"T{i:02d}" for i in range(50)]

        factor_history = {
            "value": pd.DataFrame(
                rng.normal(0, 1, (60, 50)), index=dates, columns=tickers
            ),
            "momentum": pd.DataFrame(
                rng.normal(0, 1, (60, 50)), index=dates, columns=tickers
            ),
            "quality": pd.DataFrame(
                rng.normal(0, 1, (60, 50)), index=dates, columns=tickers
            ),
        }
        returns_hist = pd.DataFrame(
            rng.normal(0.001, 0.02, (60, 50)), index=dates, columns=tickers
        )

        report = run_factor_validation(factor_history, returns_hist)
        # Holm is stricter than BH → subset
        assert set(report.significant_factors_holm).issubset(
            set(report.significant_factors)
        )

    def test_composite_ic_none_when_not_provided(self) -> None:
        rng = np.random.default_rng(42)
        dates = pd.date_range("2023-01-01", periods=36, freq="ME")
        tickers = [f"T{i:02d}" for i in range(30)]
        factor_history = {
            "value": pd.DataFrame(
                rng.normal(0, 1, (36, 30)),
                index=dates,
                columns=tickers,
            ),
        }
        returns_hist = pd.DataFrame(
            rng.normal(0.001, 0.02, (36, 30)),
            index=dates,
            columns=tickers,
        )
        report = run_factor_validation(factor_history, returns_hist)
        assert report.composite_ic_result is None

    def test_composite_ic_populated_when_provided(self) -> None:
        rng = np.random.default_rng(42)
        dates = pd.date_range("2023-01-01", periods=36, freq="ME")
        tickers = [f"T{i:02d}" for i in range(30)]
        factor_history = {
            "value": pd.DataFrame(
                rng.normal(0, 1, (36, 30)),
                index=dates,
                columns=tickers,
            ),
        }
        returns_hist = pd.DataFrame(
            rng.normal(0.001, 0.02, (36, 30)),
            index=dates,
            columns=tickers,
        )
        composite = pd.DataFrame(
            rng.normal(0, 1, (36, 30)),
            index=dates,
            columns=tickers,
        )
        report = run_factor_validation(
            factor_history,
            returns_hist,
            composite_scores_history=composite,
        )
        assert report.composite_ic_result is not None
        assert isinstance(report.composite_ic_result, CompositeICResult)
        assert not np.isnan(report.composite_ic_result.mean_ic)
        assert isinstance(
            report.composite_ic_result.outperforms_best_individual,
            bool,
        )

    def test_composite_outperforms_best_individual_flag(self) -> None:
        rng = np.random.default_rng(42)
        dates = pd.date_range("2023-01-01", periods=48, freq="ME")
        tickers = [f"T{i:02d}" for i in range(30)]
        factor_a = rng.normal(0, 1, (48, 30))
        forward_ret = factor_a + rng.normal(0, 0.1, (48, 30))
        factor_b = rng.normal(0, 1, (48, 30))

        factor_history = {
            "factor_a": pd.DataFrame(factor_a, index=dates, columns=tickers),
            "factor_b": pd.DataFrame(factor_b, index=dates, columns=tickers),
        }
        returns_hist = pd.DataFrame(forward_ret, index=dates, columns=tickers)
        composite = pd.DataFrame(
            (factor_a + factor_b) / 2,
            index=dates,
            columns=tickers,
        )
        report = run_factor_validation(
            factor_history,
            returns_hist,
            composite_scores_history=composite,
        )
        cic = report.composite_ic_result
        assert cic is not None
        # Verify best_individual_ic matches max of individual ICs
        best = max(r.mean_ic for r in report.ic_results)
        assert cic.best_individual_ic == best
        assert isinstance(cic.outperforms_best_individual, bool)


class TestCompositeICValidation:
    def test_known_strong_ic(self) -> None:
        """Strong rank correlation yields high, significant IC."""
        rng = np.random.default_rng(42)
        dates = pd.date_range("2020-01-01", periods=40, freq="ME")
        tickers = [f"T{i:02d}" for i in range(20)]
        scores = rng.normal(0, 1, (40, 20))
        returns = scores + rng.normal(0, 0.05, (40, 20))

        result = compute_composite_ic(
            pd.DataFrame(scores, index=dates, columns=tickers),
            pd.DataFrame(returns, index=dates, columns=tickers),
        )
        assert isinstance(result, CompositeICResult)
        assert result.mean_ic > 0.8
        assert result.significant is True
        assert result.p_value < 0.05
        assert result.t_stat > 2.0

    def test_known_zero_ic(self) -> None:
        """Uncorrelated scores yield near-zero, insignificant IC."""
        rng = np.random.default_rng(42)
        dates = pd.date_range("2020-01-01", periods=20, freq="ME")
        tickers = [f"T{i:02d}" for i in range(20)]
        scores = rng.normal(0, 1, (20, 20))
        rng2 = np.random.default_rng(99)
        returns = rng2.normal(0, 1, (20, 20))

        result = compute_composite_ic(
            pd.DataFrame(scores, index=dates, columns=tickers),
            pd.DataFrame(returns, index=dates, columns=tickers),
        )
        assert abs(result.mean_ic) < 0.3
        assert result.significant is False

    def test_when_ic_history_shorter_than_two_lags_plus_one_then_not_significant(
        self,
    ) -> None:
        """Noise on a 12-period history cannot be flagged with 6 HAC lags."""
        dates = pd.date_range("2020-01-01", periods=12, freq="ME")
        tickers = [f"T{i:02d}" for i in range(20)]
        flagged = 0
        for seed in range(150):
            rng = np.random.default_rng(seed)
            scores = pd.DataFrame(
                rng.normal(0, 1, (12, 20)), index=dates, columns=tickers
            )
            returns = pd.DataFrame(
                rng.normal(0, 1, (12, 20)), index=dates, columns=tickers
            )
            result = compute_composite_ic(scores, returns, newey_west_lags=6)
            flagged += int(result.significant)
            assert result.t_stat == 0.0
            assert result.p_value == 1.0
        assert flagged == 0

    def test_best_individual_ic_nan_standalone(self) -> None:
        """Direct call has NaN best_individual_ic."""
        rng = np.random.default_rng(42)
        dates = pd.date_range("2020-01-01", periods=20, freq="ME")
        tickers = [f"T{i:02d}" for i in range(10)]
        scores = pd.DataFrame(
            rng.normal(0, 1, (20, 10)),
            index=dates,
            columns=tickers,
        )
        returns = pd.DataFrame(
            rng.normal(0, 1, (20, 10)),
            index=dates,
            columns=tickers,
        )
        result = compute_composite_ic(scores, returns)
        assert np.isnan(result.best_individual_ic)
        assert result.outperforms_best_individual is False

    def test_icir_sign_matches_mean_ic(self) -> None:
        rng = np.random.default_rng(42)
        dates = pd.date_range("2020-01-01", periods=40, freq="ME")
        tickers = [f"T{i:02d}" for i in range(20)]
        scores = rng.normal(0, 1, (40, 20))
        returns = scores + rng.normal(0, 0.05, (40, 20))

        result = compute_composite_ic(
            pd.DataFrame(scores, index=dates, columns=tickers),
            pd.DataFrame(returns, index=dates, columns=tickers),
        )
        if result.mean_ic != 0:
            assert np.sign(result.icir) == np.sign(result.mean_ic)

    def test_empty_overlap_returns_degenerate(self) -> None:
        """No date overlap returns degenerate result."""
        dates_a = pd.date_range("2020-01-01", periods=10, freq="ME")
        dates_b = pd.date_range("2022-01-01", periods=10, freq="ME")
        tickers = [f"T{i:02d}" for i in range(10)]
        rng = np.random.default_rng(42)
        scores = pd.DataFrame(
            rng.normal(0, 1, (10, 10)),
            index=dates_a,
            columns=tickers,
        )
        returns = pd.DataFrame(
            rng.normal(0, 1, (10, 10)),
            index=dates_b,
            columns=tickers,
        )
        result = compute_composite_ic(scores, returns)
        assert np.isnan(result.mean_ic)
        assert result.significant is False
        assert result.p_value == 1.0

    def test_icir_finite(self) -> None:
        """ICIR is finite when IC series has positive variance."""
        rng = np.random.default_rng(42)
        dates = pd.date_range("2020-01-01", periods=36, freq="ME")
        tickers = [f"T{i:02d}" for i in range(20)]
        scores = rng.normal(0, 1, (36, 20))
        returns = scores + rng.normal(0, 0.3, (36, 20))

        result = compute_composite_ic(
            pd.DataFrame(scores, index=dates, columns=tickers),
            pd.DataFrame(returns, index=dates, columns=tickers),
        )
        assert np.isfinite(result.icir)

    def test_known_exact_ic_from_deterministic_construction(self) -> None:
        """IC is exactly 1.0 when scores perfectly rank-correlate with returns."""
        dates = pd.date_range("2020-01-01", periods=12, freq="ME")
        tickers = [f"T{i:02d}" for i in range(10)]
        scores_row = np.arange(1, 11, dtype=float)
        scores = pd.DataFrame(
            np.tile(scores_row, (12, 1)), index=dates, columns=tickers
        )
        returns_row = scores_row * 0.01
        returns = pd.DataFrame(
            np.tile(returns_row, (12, 1)), index=dates, columns=tickers
        )
        result = compute_composite_ic(scores, returns)
        assert pytest.approx(result.mean_ic, abs=1e-9) == 1.0
        # A zero-variance IC series gives the Newey-West standard error nothing
        # to estimate, so inference is undefined rather than infinitely
        # significant (a stale or copied cross-section looks exactly like this).
        assert result.significant is False
        assert result.t_stat == 0.0
        assert result.p_value == 1.0
        # Perfect IC every period → ic_std=0 → icir=0 by zero-std guard
        assert result.ic_std == 0.0
        assert result.icir == 0.0

    def test_composite_does_not_outperform_dominant_factor(self) -> None:
        """When one factor is strongly predictive and the other is noise,
        the composite average IC is lower than the dominant factor IC."""
        rng = np.random.default_rng(0)
        dates = pd.date_range("2020-01-01", periods=36, freq="ME")
        tickers = [f"T{i:02d}" for i in range(30)]
        signal = rng.normal(0, 1, (36, 30))
        noise = rng.normal(0, 1, (36, 30))
        forward_ret = signal + rng.normal(0, 0.05, (36, 30))

        factor_history = {
            "signal": pd.DataFrame(signal, index=dates, columns=tickers),
            "noise": pd.DataFrame(noise, index=dates, columns=tickers),
        }
        returns_hist = pd.DataFrame(forward_ret, index=dates, columns=tickers)
        composite = pd.DataFrame((signal + noise) / 2.0, index=dates, columns=tickers)
        report = run_factor_validation(
            factor_history,
            returns_hist,
            composite_scores_history=composite,
        )
        cic = report.composite_ic_result
        assert cic is not None
        best = max(r.mean_ic for r in report.ic_results)
        assert cic.best_individual_ic == pytest.approx(best, rel=1e-9)
        assert cic.outperforms_best_individual is False

    def test_result_field_invariants(self) -> None:
        """significant flag, ICIR sign, and bounds are internally consistent."""
        rng = np.random.default_rng(7)
        dates = pd.date_range("2020-01-01", periods=48, freq="ME")
        tickers = [f"T{i:02d}" for i in range(20)]
        scores = rng.normal(0, 1, (48, 20))
        returns = scores + rng.normal(0, 0.2, (48, 20))
        result = compute_composite_ic(
            pd.DataFrame(scores, index=dates, columns=tickers),
            pd.DataFrame(returns, index=dates, columns=tickers),
            t_stat_threshold=2.0,
        )
        assert result.significant == (abs(result.t_stat) >= 2.0)
        if result.mean_ic != 0.0:
            assert np.sign(result.icir) == np.sign(result.mean_ic)
        assert result.ic_std >= 0.0
        assert 0.0 <= result.p_value <= 1.0
        assert result.outperforms_best_individual is False
        assert np.isnan(result.best_individual_ic)

    def test_composite_min_observations_respected(self) -> None:
        """composite_min_observations config threads to compute_composite_ic."""
        rng = np.random.default_rng(5)

        # --- strict override: 4 tickers < min=5 → all cross-sections dropped ---
        dates_small = pd.date_range("2020-01-01", periods=6, freq="ME")
        tickers_small = [f"T{i}" for i in range(4)]
        factor_history_small = {
            "value": pd.DataFrame(
                rng.normal(0, 1, (6, 4)),
                index=dates_small,
                columns=tickers_small,
            ),
        }
        returns_small = pd.DataFrame(
            rng.normal(0, 1, (6, 4)), index=dates_small, columns=tickers_small
        )
        composite_small = pd.DataFrame(
            rng.normal(0, 1, (6, 4)), index=dates_small, columns=tickers_small
        )
        strict_config = FactorValidationConfig(composite_min_observations=5)
        report_strict = run_factor_validation(
            factor_history_small,
            returns_small,
            config=strict_config,
            composite_scores_history=composite_small,
        )
        # 4 tickers < min=5: all cross-sections are dropped
        assert report_strict.composite_ic_result is not None
        assert np.isnan(report_strict.composite_ic_result.mean_ic)

        # --- default config: 30 tickers ≥ 24 → IC computes normally ---
        dates_large = pd.date_range("2020-01-01", periods=36, freq="ME")
        tickers_large = [f"T{i:02d}" for i in range(30)]
        factor_history_large = {
            "value": pd.DataFrame(
                rng.normal(0, 1, (36, 30)),
                index=dates_large,
                columns=tickers_large,
            ),
        }
        returns_large = pd.DataFrame(
            rng.normal(0, 1, (36, 30)), index=dates_large, columns=tickers_large
        )
        composite_large = pd.DataFrame(
            rng.normal(0, 1, (36, 30)), index=dates_large, columns=tickers_large
        )
        report_default = run_factor_validation(
            factor_history_large,
            returns_large,
            composite_scores_history=composite_large,
        )
        # Default min=24 passes with 30 tickers
        assert report_default.composite_ic_result is not None
        assert not np.isnan(report_default.composite_ic_result.mean_ic)
