"""Tests for factor validation and statistical testing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.factors import (
    FactorValidationConfig,
    FactorValidationReport,
    benjamini_hochberg,
    compute_ic_series,
    compute_monthly_ic,
    compute_newey_west_tstat,
    compute_quantile_spread,
    compute_vif,
    run_factor_validation,
)


@pytest.fixture()
def factor_scores() -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0, 1, 100), index=[f"T{i:03d}" for i in range(100)])


@pytest.fixture()
def forward_returns() -> pd.Series:
    rng = np.random.default_rng(99)
    return pd.Series(rng.normal(0.001, 0.02, 100), index=[f"T{i:03d}" for i in range(100)])


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


class TestComputeICSeries:
    def test_returns_series(self) -> None:
        rng = np.random.default_rng(42)
        dates = pd.date_range("2023-01-01", periods=12, freq="ME")
        tickers = [f"T{i:02d}" for i in range(20)]

        scores_hist = pd.DataFrame(rng.normal(0, 1, (12, 20)), index=dates, columns=tickers)
        returns_hist = pd.DataFrame(rng.normal(0.001, 0.02, (12, 20)), index=dates, columns=tickers)

        result = compute_ic_series(scores_hist, returns_hist, "test_factor")
        assert isinstance(result, pd.Series)
        assert len(result) > 0


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
        t_stat, p_value = compute_newey_west_tstat(ic)
        assert abs(t_stat) < 3.0

    def test_short_series(self) -> None:
        ic = pd.Series([0.05, 0.06])
        t_stat, p_value = compute_newey_west_tstat(ic)
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
        factors = pd.DataFrame({
            "a": a,
            "b": a + rng.normal(0, 0.01, 100),  # nearly identical
            "c": rng.normal(0, 1, 100),
        })
        vif = compute_vif(factors)
        # Collinear factors should have high VIF
        assert vif["a"] > 5.0 or vif["b"] > 5.0

    def test_single_factor(self) -> None:
        factors = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        vif = compute_vif(factors)
        assert vif["a"] == 1.0


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
