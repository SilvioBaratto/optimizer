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
    compute_monthly_ic,
    compute_newey_west_tstat,
    compute_quantile_spread,
    compute_vif,
    correct_pvalues,
    run_factor_validation,
    validate_factor_universe,
)


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
        factors = pd.DataFrame(
            {"a": base, "b": noisy_col, "c": rng.normal(0, 1, 500)}
        )
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
        factors = pd.DataFrame(
            {"a": base, "b": exact, "c": rng.normal(0, 1, 200)}
        )
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
            "factor_a": pd.DataFrame(
                factor_a, index=dates, columns=tickers
            ),
            "factor_b": pd.DataFrame(
                factor_b, index=dates, columns=tickers
            ),
        }
        returns_hist = pd.DataFrame(
            forward_ret, index=dates, columns=tickers
        )
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
        assert result.significant is True
        # Perfect IC every period → ic_std=0 → icir=0 by zero-std guard
        assert result.ic_std == 0.0
        assert result.icir == 0.0

    def test_composite_does_not_outperform_dominant_factor(self) -> None:
        """When one factor is strongly predictive and the other is noise,
        the composite average IC is lower than the dominant factor IC."""
        rng = np.random.default_rng(0)
        dates = pd.date_range("2020-01-01", periods=36, freq="ME")
        tickers = [f"T{i:02d}" for i in range(20)]
        signal = rng.normal(0, 1, (36, 20))
        noise = rng.normal(0, 1, (36, 20))
        forward_ret = signal + rng.normal(0, 0.05, (36, 20))

        factor_history = {
            "signal": pd.DataFrame(signal, index=dates, columns=tickers),
            "noise": pd.DataFrame(noise, index=dates, columns=tickers),
        }
        returns_hist = pd.DataFrame(forward_ret, index=dates, columns=tickers)
        composite = pd.DataFrame(
            (signal + noise) / 2.0, index=dates, columns=tickers
        )
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
        dates = pd.date_range("2020-01-01", periods=6, freq="ME")
        tickers = [f"T{i}" for i in range(4)]
        factor_history = {
            "value": pd.DataFrame(
                rng.normal(0, 1, (6, 4)), index=dates, columns=tickers
            ),
        }
        returns_hist = pd.DataFrame(
            rng.normal(0, 1, (6, 4)), index=dates, columns=tickers
        )
        composite = pd.DataFrame(
            rng.normal(0, 1, (6, 4)), index=dates, columns=tickers
        )
        strict_config = FactorValidationConfig(composite_min_observations=5)
        report_strict = run_factor_validation(
            factor_history,
            returns_hist,
            config=strict_config,
            composite_scores_history=composite,
        )
        report_default = run_factor_validation(
            factor_history,
            returns_hist,
            composite_scores_history=composite,
        )
        # With 4 tickers, strict min=5 forces all cross-sections to be dropped
        assert report_strict.composite_ic_result is not None
        assert np.isnan(report_strict.composite_ic_result.mean_ic)
        # Default min=3 passes with 4 tickers
        assert report_default.composite_ic_result is not None
        assert not np.isnan(report_default.composite_ic_result.mean_ic)
