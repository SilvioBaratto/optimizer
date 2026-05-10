"""Integration tests: flat service files resolve from domain folders (issue #612).

Verifies that all ~30 flat service files are importable from their new
domain locations, and that old flat-file paths no longer resolve.
"""

from __future__ import annotations

import pytest


class TestMarketDataServices:
    def test_yfinance_data_service_importable(self) -> None:
        from app.services.market_data.yfinance_data_service import (
            YFinanceDataService,
            run_bulk_yfinance_fetch,
        )

        assert YFinanceDataService is not None
        assert run_bulk_yfinance_fetch is not None

    def test_reference_index_seeder_importable(self) -> None:
        from app.services.market_data.reference_index_seeder import (
            SeedResult,
            seed_reference_indices,
        )

        assert SeedResult is not None
        assert seed_reference_indices is not None


class TestUniverseServices:
    def test_universe_screening_service_importable(self) -> None:
        from app.services.universe.universe_screening_service import (
            run_universe_screen,
        )

        assert run_universe_screen is not None


class TestMacroServices:
    def test_macro_regime_service_importable(self) -> None:
        from app.services.macro.macro_regime_service import (
            MacroRegimeService,
            run_bulk_macro_fetch,
        )

        assert MacroRegimeService is not None
        assert run_bulk_macro_fetch is not None

    def test_macro_calibration_importable(self) -> None:
        from app.services.macro.macro_calibration import (
            CalibrationResult,
            classify_macro_regime,
            run_bulk_calibrate,
        )

        assert CalibrationResult is not None
        assert classify_macro_regime is not None
        assert run_bulk_calibrate is not None

    def test_macro_news_summary_importable(self) -> None:
        from app.services.macro.macro_news_summary import (
            run_news_summarize,
        )

        assert run_news_summarize is not None


class TestPortfolioServices:
    def test_broker_sync_service_importable(self) -> None:
        from app.services.portfolio.broker_sync_service import (
            sync_portfolio,
        )

        assert sync_portfolio is not None


class TestOptimizationServices:
    def test_optimization_service_importable(self) -> None:
        from app.services.optimization.optimization_service import (
            _build_optimizer,
            extract_results,
        )

        assert _build_optimizer is not None
        assert extract_results is not None

    def test_tuning_service_importable(self) -> None:
        from app.services.optimization.tuning_service import run_tune

        assert run_tune is not None

    def test_validation_service_importable(self) -> None:
        from app.services.optimization.validation_service import run_validation

        assert run_validation is not None


class TestBacktestServices:
    def test_backtest_service_importable(self) -> None:
        from app.services.backtest.backtest_service import (
            build_optimizer,
            run_and_persist,
        )

        assert run_and_persist is not None
        assert build_optimizer is not None


class TestFactorsServices:
    def test_factor_helpers_importable(self) -> None:
        from app.services.factors._factor_helpers import FactorDataError

        assert FactorDataError is not None

    def test_factor_compute_service_importable(self) -> None:
        from app.services.factors.factor_compute_service import run_factor_compute

        assert run_factor_compute is not None

    def test_factor_scoring_service_importable(self) -> None:
        from app.services.factors.factor_scoring_service import (
            compute_factor_scores,
            validate_factors,
        )

        assert compute_factor_scores is not None
        assert validate_factors is not None

    def test_factor_analysis_service_importable(self) -> None:
        from app.services.factors.factor_analysis_service import (
            select_stocks_for_tickers,
        )

        assert select_stocks_for_tickers is not None


class TestRiskServices:
    def test_risk_analytics_service_importable(self) -> None:
        from app.services.risk.risk_analytics_service import RiskAnalyticsService

        assert RiskAnalyticsService is not None


class TestRebalancingServices:
    def test_rebalancing_service_importable(self) -> None:
        from app.services.rebalancing.rebalancing_service import (
            build_config,
            decide,
        )

        assert build_config is not None
        assert decide is not None


class TestViewsServices:
    def test_view_generation_importable(self) -> None:
        from app.services.views.view_generation import (
            GeneratedViews,
            generate_views,
        )

        assert GeneratedViews is not None
        assert generate_views is not None

    def test_entropy_pooling_service_importable(self) -> None:
        from app.services.views.entropy_pooling_service import run_entropy_pooling

        assert run_entropy_pooling is not None

    def test_opinion_pooling_importable(self) -> None:
        from app.services.views.opinion_pooling import (
            OpinionPoolResult,
            build_llm_opinion_pool,
        )

        assert OpinionPoolResult is not None
        assert build_llm_opinion_pool is not None

    def test_llm_moments_importable(self) -> None:
        from app.services.views.llm_moments import calibrate_delta

        assert calibrate_delta is not None

    def test_sentiment_importable(self) -> None:
        from app.services.views.sentiment import compute_sentiment_signal

        assert compute_sentiment_signal is not None


class TestAttributionServices:
    def test_attribution_service_importable(self) -> None:
        from app.services.attribution.attribution_service import (
            run_brinson_attribution,
        )

        assert run_brinson_attribution is not None


class TestDashboardServices:
    def test_dashboard_service_importable(self) -> None:
        from app.services.dashboard.dashboard_service import (
            compute_performance_metrics,
        )

        assert compute_performance_metrics is not None


class TestScenariosServices:
    def test_stress_scenarios_importable(self) -> None:
        from app.services.scenarios.stress_scenarios import (
            generate_stress_scenarios,
        )

        assert generate_stress_scenarios is not None

    def test_synthetic_service_importable(self) -> None:
        from app.services.scenarios.synthetic_service import generate_scenarios

        assert generate_scenarios is not None


class TestReportsServices:
    def test_report_service_importable(self) -> None:
        from app.services.reports.report_service import ReportService

        assert ReportService is not None


class TestJobsServices:
    def test_background_job_importable(self) -> None:
        from app.services.jobs.background_job import (
            BackgroundJobService,
            JobAlreadyRunningError,
        )

        assert BackgroundJobService is not None
        assert JobAlreadyRunningError is not None

    def test_scheduler_importable(self) -> None:
        from app.services.jobs.scheduler import create_scheduler

        assert create_scheduler is not None


# ---------------------------------------------------------------------------
# Old flat-file paths must NOT resolve
# ---------------------------------------------------------------------------

_OLD_FLAT_PATHS = [
    "app.services.yfinance_data_service",
    "app.services.reference_index_seeder",
    "app.services.universe_screening_service",
    "app.services.macro_regime_service",
    "app.services.macro_calibration",
    "app.services.macro_news_summary",
    "app.services.broker_sync_service",
    "app.services.optimization_service",
    "app.services.tuning_service",
    "app.services.validation_service",
    "app.services.backtest_service",
    "app.services.factor_service",
    "app.services.factor_compute_service",
    "app.services.factor_scoring_service",
    "app.services.factor_analysis_service",
    "app.services._factor_helpers",
    "app.services.risk_analytics_service",
    "app.services.rebalancing_service",
    "app.services.view_generation",
    "app.services.entropy_pooling_service",
    "app.services.opinion_pooling",
    "app.services.llm_moments",
    "app.services.sentiment",
    "app.services.attribution_service",
    "app.services.dashboard_service",
    "app.services.stress_scenarios",
    "app.services.synthetic_service",
    "app.services.report_service",
    "app.services.background_job",
    "app.services.scheduler",
]


class TestOldFlatPathsDoNotResolve:
    @pytest.mark.parametrize("old_path", _OLD_FLAT_PATHS)
    def test_old_flat_path_raises_module_not_found(self, old_path: str) -> None:
        with pytest.raises(ModuleNotFoundError):
            import importlib

            importlib.import_module(old_path)
