"""Integration tests: route files resolve from domain folders (issue #613).

Verifies all ~26 route files are importable from their new domain locations,
and old flat-file paths no longer resolve.
"""

from __future__ import annotations

import pytest


class TestMarketDataRoutes:
    def test_yfinance_data_router_importable(self) -> None:
        from app.api.v1.market_data import yfinance_data_router

        assert yfinance_data_router is not None

    def test_reference_indices_router_importable(self) -> None:
        from app.api.v1.market_data import reference_indices_router

        assert reference_indices_router is not None


class TestUniverseRoutes:
    def test_trading212_router_importable(self) -> None:
        from app.api.v1.universe import trading212_router

        assert trading212_router is not None

    def test_universe_screen_router_importable(self) -> None:
        from app.api.v1.universe import universe_screen_router

        assert universe_screen_router is not None


class TestMacroRoutes:
    def test_macro_regime_router_importable(self) -> None:
        from app.api.v1.macro import macro_regime_router

        assert macro_regime_router is not None

    def test_macro_calibration_router_importable(self) -> None:
        from app.api.v1.macro import macro_calibration_router

        assert macro_calibration_router is not None


class TestPortfolioRoutes:
    def test_portfolio_router_importable(self) -> None:
        from app.api.v1.portfolio import portfolio_router

        assert portfolio_router is not None


class TestOptimizationRoutes:
    def test_optimize_router_importable(self) -> None:
        from app.api.v1.optimization import optimize_router

        assert optimize_router is not None

    def test_tune_router_importable(self) -> None:
        from app.api.v1.optimization import tune_router

        assert tune_router is not None

    def test_validate_router_importable(self) -> None:
        from app.api.v1.optimization import validate_router

        assert validate_router is not None


class TestBacktestRoutes:
    def test_backtest_router_importable(self) -> None:
        from app.api.v1.backtest import backtest_router

        assert backtest_router is not None


class TestFactorsRoutes:
    def test_factors_router_importable(self) -> None:
        from app.api.v1.factors import factors_router

        assert factors_router is not None


class TestRiskRoutes:
    def test_risk_router_importable(self) -> None:
        from app.api.v1.risk import risk_router

        assert risk_router is not None

    def test_risk_analytics_router_importable(self) -> None:
        from app.api.v1.risk import risk_analytics_router

        assert risk_analytics_router is not None


class TestRebalancingRoutes:
    def test_rebalance_router_importable(self) -> None:
        from app.api.v1.rebalancing import rebalance_router

        assert rebalance_router is not None

    def test_rebalance_policy_router_importable(self) -> None:
        from app.api.v1.rebalancing import rebalance_policy_router

        assert rebalance_policy_router is not None


class TestViewsRoutes:
    def test_views_router_importable(self) -> None:
        from app.api.v1.views import views_router

        assert views_router is not None

    def test_opinion_pooling_router_importable(self) -> None:
        from app.api.v1.views import opinion_pooling_router

        assert opinion_pooling_router is not None

    def test_llm_moments_router_importable(self) -> None:
        from app.api.v1.views import llm_moments_router

        assert llm_moments_router is not None


class TestAttributionRoutes:
    def test_attribution_router_importable(self) -> None:
        from app.api.v1.attribution import attribution_router

        assert attribution_router is not None


class TestDashboardRoutes:
    def test_dashboard_router_importable(self) -> None:
        from app.api.v1.dashboard import dashboard_router

        assert dashboard_router is not None

    def test_market_router_importable(self) -> None:
        from app.api.v1.dashboard import market_router

        assert market_router is not None


class TestScenariosRoutes:
    def test_stress_scenarios_router_importable(self) -> None:
        from app.api.v1.scenarios import stress_scenarios_router

        assert stress_scenarios_router is not None

    def test_synthetic_router_importable(self) -> None:
        from app.api.v1.scenarios import synthetic_router

        assert synthetic_router is not None


class TestReportsRoutes:
    def test_reports_router_importable(self) -> None:
        from app.api.v1.reports import reports_router

        assert reports_router is not None


class TestJobsRoutes:
    def test_jobs_router_importable(self) -> None:
        from app.api.v1.jobs import jobs_router

        assert jobs_router is not None

    def test_scheduler_router_importable(self) -> None:
        from app.api.v1.jobs import scheduler_router

        assert scheduler_router is not None


class TestRouterPyImportsSucceed:
    """Acceptance criterion: router.py can be imported."""

    def test_api_router_importable(self) -> None:
        from app.api.v1._shared.router import api_router

        assert api_router is not None


# ---------------------------------------------------------------------------
# Old flat-file paths must NOT resolve as direct modules.
# ---------------------------------------------------------------------------

# Flat files where the domain dir has a DIFFERENT name → ModuleNotFoundError.
_OLD_NON_MATCHING_PATHS = [
    "app.api.v1.optimize",
    "app.api.v1.rebalance",
    "app.api.v1.rebalance_policy",
    "app.api.v1.reference_indices",
    "app.api.v1.risk_analytics",
    "app.api.v1.scheduler",
    "app.api.v1.stress_scenarios",
    "app.api.v1.synthetic",
    "app.api.v1.trading212",
    "app.api.v1.tune",
    "app.api.v1.universe_screen",
    "app.api.v1.llm_moments",
    "app.api.v1.macro_calibration",
    "app.api.v1.macro_regime",
    "app.api.v1.opinion_pooling",
    "app.api.v1.validate",
    "app.api.v1.yfinance_data",
]

# Flat files whose name matches a domain dir → package import succeeds,
# but old `from X import router` must fail (__init__ exports aliased names).
_OLD_MATCHING_PATHS = [
    "app.api.v1.attribution",
    "app.api.v1.backtest",
    "app.api.v1.dashboard",
    "app.api.v1.factors",
    "app.api.v1.jobs",
    "app.api.v1.portfolio",
    "app.api.v1.reports",
    "app.api.v1.risk",
    "app.api.v1.views",
]


class TestOldFlatPathsRaiseModuleNotFound:
    """Flat files moved to differently-named domain dirs → ModuleNotFoundError."""

    @pytest.mark.parametrize("old_path", _OLD_NON_MATCHING_PATHS)
    def test_old_flat_path_raises_module_not_found(self, old_path: str) -> None:
        import importlib

        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(old_path)


class TestOldMatchingPathsDontExportRawRouter:
    """Domain dirs replaced flat files; __init__.py exports aliased names only."""

    @pytest.mark.parametrize("old_path", _OLD_MATCHING_PATHS)
    def test_direct_router_attribute_does_not_exist(self, old_path: str) -> None:
        import importlib

        mod = importlib.import_module(old_path)
        assert not hasattr(mod, "router")
