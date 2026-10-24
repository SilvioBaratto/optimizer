"""Tests for validation factory functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skfolio.model_selection import (
    CombinatorialPurgedCV,
    MultipleRandomizedCV,
    WalkForward,
)

from optimizer.validation import (
    CPCVConfig,
    CrossValFailureReport,
    MultipleRandomizedCVConfig,
    WalkForwardConfig,
    build_cpcv,
    build_multiple_randomized_cv,
    build_walk_forward,
    compute_optimal_folds,
    run_cross_val,
    summarize_cv_failures,
)


@pytest.fixture()
def returns_df() -> pd.DataFrame:
    """Synthetic return DataFrame with 20 assets and 400 observations."""
    rng = np.random.default_rng(42)
    n_obs, n_assets = 400, 20
    data = rng.normal(loc=0.001, scale=0.02, size=(n_obs, n_assets))
    tickers = [f"TICK_{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(
        data,
        columns=tickers,
        index=pd.date_range("2022-01-01", periods=n_obs, freq="B"),
    )


class TestBuildWalkForward:
    def test_default_config(self) -> None:
        cv = build_walk_forward()
        assert isinstance(cv, WalkForward)

    def test_custom_config(self) -> None:
        cfg = WalkForwardConfig(test_size=42, train_size=126, purged_size=3)
        cv = build_walk_forward(cfg)
        assert isinstance(cv, WalkForward)

    def test_expanding_window(self) -> None:
        cfg = WalkForwardConfig.for_quarterly_expanding()
        cv = build_walk_forward(cfg)
        assert isinstance(cv, WalkForward)

    def test_splits_returns(self, returns_df: pd.DataFrame) -> None:
        cfg = WalkForwardConfig(test_size=21, train_size=100)
        cv = build_walk_forward(cfg)
        splits = list(cv.split(returns_df))
        assert len(splits) > 0
        for train_idx, test_idx in splits:
            assert len(test_idx) <= 21
            assert len(train_idx) >= 100

    def test_calendar_freq_produces_splits(self, returns_df: pd.DataFrame) -> None:
        # 400 business days ≈ 19 months → 12-month train leaves several
        # monthly test folds.
        cfg = WalkForwardConfig.for_monthly_calendar()
        cv = build_walk_forward(cfg)
        splits = list(cv.split(returns_df))
        assert len(splits) > 0
        # No leakage: every train index precedes its test index.
        for train_idx, test_idx in splits:
            assert int(np.max(train_idx)) < int(np.min(test_idx))

    def test_freq_offset_string_is_converted(self, returns_df: pd.DataFrame) -> None:
        cfg = WalkForwardConfig(
            test_size=1, train_size=6, freq="MS", freq_offset="2D", previous=True
        )
        cv = build_walk_forward(cfg)
        # freq_offset carried through as a parsed pandas offset, not the raw str.
        assert not isinstance(cv.freq_offset, str)
        assert len(list(cv.split(returns_df))) > 0


class TestBuildCPCV:
    def test_default_config(self) -> None:
        cv = build_cpcv()
        assert isinstance(cv, CombinatorialPurgedCV)

    def test_custom_config(self) -> None:
        cfg = CPCVConfig(n_folds=6, n_test_folds=2, purged_size=5, embargo_size=3)
        cv = build_cpcv(cfg)
        assert isinstance(cv, CombinatorialPurgedCV)

    def test_splits_returns(self, returns_df: pd.DataFrame) -> None:
        cfg = CPCVConfig.for_small_sample()
        cv = build_cpcv(cfg)
        splits = list(cv.split(returns_df))
        assert len(splits) > 0


class TestBuildMultipleRandomizedCV:
    def test_default_config(self) -> None:
        cv = build_multiple_randomized_cv()
        assert isinstance(cv, MultipleRandomizedCV)

    def test_custom_config(self) -> None:
        cfg = MultipleRandomizedCVConfig(
            n_subsamples=5,
            asset_subset_size=8,
            random_state=123,
        )
        cv = build_multiple_randomized_cv(cfg)
        assert isinstance(cv, MultipleRandomizedCV)


class TestComputeOptimalFolds:
    def test_returns_tuple(self) -> None:
        result = compute_optimal_folds(
            n_observations=500,
            target_train_size=252,
            target_n_test_paths=10,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        n_folds, n_test_folds = result
        assert n_folds > 0
        assert n_test_folds > 0
        assert n_test_folds < n_folds


class TestRunCrossVal:
    def test_walk_forward_backtest(self, returns_df: pd.DataFrame) -> None:
        from skfolio.optimization import EqualWeighted

        cfg = WalkForwardConfig(test_size=21, train_size=100)
        cv = build_walk_forward(cfg)
        pred = run_cross_val(EqualWeighted(), returns_df, cv=cv)
        assert hasattr(pred, "sharpe_ratio")

    def test_default_cv(self, returns_df: pd.DataFrame) -> None:
        from skfolio.optimization import EqualWeighted

        pred = run_cross_val(EqualWeighted(), returns_df)
        assert hasattr(pred, "sharpe_ratio")


class TestRunCrossValCPCV:
    """CPCV integration through run_cross_val (issue #75)."""

    def test_cpcv_returns_population(self, returns_df: pd.DataFrame) -> None:
        from skfolio.optimization import EqualWeighted
        from skfolio.population import Population

        cfg = CPCVConfig.for_small_sample()
        cv = build_cpcv(cfg)
        pred = run_cross_val(EqualWeighted(), returns_df, cv=cv)
        assert isinstance(pred, Population)
        assert len(pred) > 0

    def test_cpcv_portfolios_have_composition(self, returns_df: pd.DataFrame) -> None:
        from skfolio.optimization import EqualWeighted

        cfg = CPCVConfig.for_small_sample()
        cv = build_cpcv(cfg)
        pred = run_cross_val(EqualWeighted(), returns_df, cv=cv)
        for portfolio in pred:
            assert hasattr(portfolio, "composition")

    def test_cpcv_sharpe_ratios_finite(self, returns_df: pd.DataFrame) -> None:
        from skfolio.optimization import EqualWeighted

        cfg = CPCVConfig.for_small_sample()
        cv = build_cpcv(cfg)
        pred = run_cross_val(EqualWeighted(), returns_df, cv=cv)
        for portfolio in pred:
            assert np.isfinite(portfolio.sharpe_ratio)

    def test_cpcv_with_purging(self, returns_df: pd.DataFrame) -> None:
        from skfolio.optimization import EqualWeighted

        cfg = CPCVConfig(n_folds=6, n_test_folds=2, purged_size=5)
        cv = build_cpcv(cfg)
        pred = run_cross_val(EqualWeighted(), returns_df, cv=cv)
        assert len(pred) > 0

    def test_cpcv_statistical_testing_preset(self, returns_df: pd.DataFrame) -> None:
        from skfolio.optimization import EqualWeighted

        cfg = CPCVConfig.for_statistical_testing()
        cv = build_cpcv(cfg)
        pred = run_cross_val(EqualWeighted(), returns_df, cv=cv)
        assert len(pred) >= 10


class TestRunCrossValMRCV:
    """MRCV integration through run_cross_val (issue #75)."""

    def test_mrcv_returns_population(self, returns_df: pd.DataFrame) -> None:
        from skfolio.optimization import EqualWeighted
        from skfolio.population import Population

        cfg = MultipleRandomizedCVConfig(
            walk_forward_config=WalkForwardConfig(test_size=21, train_size=100),
            n_subsamples=3,
            asset_subset_size=8,
            random_state=42,
        )
        cv = build_multiple_randomized_cv(cfg)
        pred = run_cross_val(EqualWeighted(), returns_df, cv=cv)
        assert isinstance(pred, Population)
        assert len(pred) > 0

    def test_mrcv_portfolios_have_composition(self, returns_df: pd.DataFrame) -> None:
        from skfolio.optimization import EqualWeighted

        cfg = MultipleRandomizedCVConfig(
            walk_forward_config=WalkForwardConfig(test_size=21, train_size=100),
            n_subsamples=3,
            asset_subset_size=8,
            random_state=42,
        )
        cv = build_multiple_randomized_cv(cfg)
        pred = run_cross_val(EqualWeighted(), returns_df, cv=cv)
        for portfolio in pred:
            assert hasattr(portfolio, "composition")

    def test_mrcv_robustness_check_preset(self, returns_df: pd.DataFrame) -> None:
        from skfolio.optimization import EqualWeighted

        cfg = MultipleRandomizedCVConfig.for_robustness_check(
            n_subsamples=3,
            asset_subset_size=8,
        )
        cv = build_multiple_randomized_cv(cfg)
        pred = run_cross_val(EqualWeighted(), returns_df, cv=cv)
        assert len(pred) > 0


class TestSummarizeCvFailures:
    """Resilience layer: count FailedPortfolio folds (skfolio 1.0 fallback)."""

    def test_clean_backtest_reports_no_failures(self, returns_df: pd.DataFrame) -> None:
        from skfolio.optimization import EqualWeighted

        cfg = WalkForwardConfig(test_size=21, train_size=100)
        pred = run_cross_val(EqualWeighted(), returns_df, cv=build_walk_forward(cfg))
        report = summarize_cv_failures(pred)
        assert isinstance(report, CrossValFailureReport)
        assert report.n_failed == 0
        assert report.failure_rate == 0.0
        assert report.has_failures is False
        assert report.total == len(pred.portfolios)

    def test_multi_period_portfolio_with_failed_fold(
        self, returns_df: pd.DataFrame
    ) -> None:
        from skfolio import MultiPeriodPortfolio, Portfolio
        from skfolio.portfolio import FailedPortfolio

        window = returns_df.iloc[:30]
        n = returns_df.shape[1]
        ok = Portfolio(X=window, weights=np.full(n, 1.0 / n), name="ok")
        bad = FailedPortfolio(X=window, name="fold3", optimization_error="infeasible")
        mpp = MultiPeriodPortfolio(portfolios=[ok, bad])

        report = summarize_cv_failures(mpp)
        assert report.total == 2
        assert report.n_failed == 1
        assert report.failure_rate == pytest.approx(0.5)
        assert report.has_failures is True
        assert report.failures == [("fold3", "infeasible")]

    def test_population_of_multi_period_portfolios_flattened(
        self, returns_df: pd.DataFrame
    ) -> None:
        from skfolio import MultiPeriodPortfolio, Population, Portfolio
        from skfolio.portfolio import FailedPortfolio

        window = returns_df.iloc[:30]
        n = returns_df.shape[1]
        ok = Portfolio(X=window, weights=np.full(n, 1.0 / n), name="ok")
        bad = FailedPortfolio(X=window, name="p1", optimization_error="solver failed")
        pop = Population([MultiPeriodPortfolio(portfolios=[ok, bad])])

        report = summarize_cv_failures(pop)
        assert report.total == 2
        assert report.n_failed == 1
        assert report.failures[0][0] == "p1"

    def test_flat_population_of_portfolios(self, returns_df: pd.DataFrame) -> None:
        from skfolio import Population, Portfolio
        from skfolio.portfolio import FailedPortfolio

        window = returns_df.iloc[:30]
        n = returns_df.shape[1]
        ok = Portfolio(X=window, weights=np.full(n, 1.0 / n), name="ok")
        bad = FailedPortfolio(X=window, name="p2", optimization_error="nan bounds")
        report = summarize_cv_failures(Population([ok, bad]))
        assert report.total == 2
        assert report.n_failed == 1

    def test_fallback_rescued_fold_is_reported_not_silently_clean(
        self, returns_df: pd.DataFrame
    ) -> None:
        from skfolio import MultiPeriodPortfolio, Portfolio

        window = returns_df.iloc[:30]
        n = returns_df.shape[1]
        weights = np.full(n, 1.0 / n)
        ok = Portfolio(X=window, weights=weights, name="ok")
        rescued = Portfolio(
            X=window,
            weights=weights,
            name="fold2",
            fallback_chain=[
                ("MeanRisk()", "infeasible"),
                ("previous_weights", "success"),
            ],
        )
        report = summarize_cv_failures(MultiPeriodPortfolio(portfolios=[ok, rescued]))

        assert report.total == 2
        assert report.n_failed == 0
        assert report.n_fell_back == 1
        assert report.fallback_rate == pytest.approx(0.5)
        assert report.fell_back == [("fold2", "infeasible")]
        assert report.has_fallbacks is True
        assert report.has_failures is False

    def test_fallback_rescue_in_real_walk_forward_is_counted(
        self, returns_df: pd.DataFrame
    ) -> None:
        from skfolio.optimization import EqualWeighted, MeanRisk

        # max_weights below 1/n_assets is infeasible, so every fold's primary
        # solve fails and EqualWeighted rescues it.
        opt = MeanRisk(
            max_weights=0.01,
            fallback=EqualWeighted(),
            raise_on_failure=False,
        )
        cfg = WalkForwardConfig(test_size=63, train_size=126)
        pred = run_cross_val(opt, returns_df, cv=build_walk_forward(cfg))
        report = summarize_cv_failures(pred)

        assert report.total == len(pred.portfolios)
        assert report.n_failed == 0
        assert report.n_fell_back == report.total
        assert all(error for _, error in report.fell_back)

    def test_fallback_rescue_emits_warning(
        self, returns_df: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        from skfolio import Portfolio

        window = returns_df.iloc[:30]
        n = returns_df.shape[1]
        rescued = Portfolio(
            X=window,
            weights=np.full(n, 1.0 / n),
            name="fold2",
            fallback_chain=[("MeanRisk()", "infeasible"), ("previous_weights", "ok")],
        )
        with caplog.at_level("WARNING", logger="optimizer.validation._factory"):
            summarize_cv_failures(rescued)

        assert any("fallback" in r.getMessage().lower() for r in caplog.records)

    def test_fold_positions_disambiguate_identically_named_folds(
        self, returns_df: pd.DataFrame
    ) -> None:
        from skfolio import MultiPeriodPortfolio, Portfolio
        from skfolio.portfolio import FailedPortfolio

        window = returns_df.iloc[:30]
        n = returns_df.shape[1]
        weights = np.full(n, 1.0 / n)
        chain = [("MeanRisk()", "infeasible"), ("previous_weights", "ok")]

        def ok() -> Portfolio:
            return Portfolio(X=window, weights=weights, name="MeanRisk")

        def rescued() -> Portfolio:
            return Portfolio(
                X=window, weights=weights, name="MeanRisk", fallback_chain=chain
            )

        bad = FailedPortfolio(
            X=window, name="MeanRisk", optimization_error="infeasible"
        )
        mpp = MultiPeriodPortfolio(portfolios=[ok(), rescued(), ok(), bad, rescued()])
        report = summarize_cv_failures(mpp)

        assert report.fell_back_folds == [1, 4]
        assert report.failed_folds == [3]
        assert len(report.fell_back) == len(report.fell_back_folds)
        assert len(report.failures) == len(report.failed_folds)

    def test_fallback_rescue_in_real_walk_forward_reports_fold_positions(
        self, returns_df: pd.DataFrame
    ) -> None:
        from skfolio.optimization import EqualWeighted, MeanRisk

        opt = MeanRisk(
            max_weights=0.01,
            fallback=EqualWeighted(),
            raise_on_failure=False,
        )
        cfg = WalkForwardConfig(test_size=63, train_size=126)
        pred = run_cross_val(opt, returns_df, cv=build_walk_forward(cfg))
        report = summarize_cv_failures(pred)

        assert report.fell_back_folds == list(range(len(pred.portfolios)))

    def test_fallback_warning_names_fold_positions(
        self, returns_df: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        from skfolio import MultiPeriodPortfolio, Portfolio

        window = returns_df.iloc[:30]
        n = returns_df.shape[1]
        weights = np.full(n, 1.0 / n)
        chain = [("MeanRisk()", "infeasible"), ("previous_weights", "ok")]
        ok = Portfolio(X=window, weights=weights, name="MeanRisk")
        rescued = Portfolio(
            X=window, weights=weights, name="MeanRisk", fallback_chain=chain
        )
        mpp = MultiPeriodPortfolio(portfolios=[ok, rescued, ok, rescued])
        with caplog.at_level("WARNING", logger="optimizer.validation._factory"):
            summarize_cv_failures(mpp)

        message = " ".join(r.getMessage() for r in caplog.records)
        assert "[1, 3]" in message
        assert "MeanRisk" not in message

    def test_fallback_warning_truncates_long_fold_lists(
        self, returns_df: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ) -> None:
        from skfolio import MultiPeriodPortfolio, Portfolio

        window = returns_df.iloc[:30]
        n = returns_df.shape[1]
        chain = [("MeanRisk()", "infeasible"), ("previous_weights", "ok")]
        rescued = [
            Portfolio(
                X=window,
                weights=np.full(n, 1.0 / n),
                name="MeanRisk",
                fallback_chain=chain,
            )
            for _ in range(40)
        ]
        with caplog.at_level("WARNING", logger="optimizer.validation._factory"):
            report = summarize_cv_failures(MultiPeriodPortfolio(portfolios=rescued))

        message = " ".join(r.getMessage() for r in caplog.records)
        assert report.fell_back_folds == list(range(40))
        assert "39" not in message
        assert "more" in message

    def test_clean_backtest_has_no_fallbacks(self, returns_df: pd.DataFrame) -> None:
        from skfolio.optimization import EqualWeighted

        cfg = WalkForwardConfig(test_size=21, train_size=100)
        pred = run_cross_val(EqualWeighted(), returns_df, cv=build_walk_forward(cfg))
        report = summarize_cv_failures(pred)
        assert report.n_fell_back == 0
        assert report.fallback_rate == 0.0
        assert report.has_fallbacks is False
        assert report.fell_back == []

    def test_report_is_frozen(self, returns_df: pd.DataFrame) -> None:
        report = CrossValFailureReport(total=3, n_failed=0, failure_rate=0.0)
        with pytest.raises(AttributeError):
            report.n_failed = 1  # type: ignore[misc]
