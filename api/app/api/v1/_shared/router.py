"""API v1 router - imports and includes all route modules"""

from fastapi import APIRouter

from app.api.v1._shared.database import router as database_router
from app.api.v1._shared.test import router as test_router
from app.api.v1.attribution import attribution_router
from app.api.v1.backtest import backtest_router
from app.api.v1.dashboard import dashboard_router, market_router
from app.api.v1.factors import factors_router
from app.api.v1.jobs import jobs_router, scheduler_router
from app.api.v1.macro import macro_calibration_router, macro_regime_router
from app.api.v1.market_data import reference_indices_router, yfinance_data_router
from app.api.v1.optimization import optimize_router, tune_router, validate_router
from app.api.v1.portfolio import portfolio_router
from app.api.v1.rebalancing import rebalance_policy_router, rebalance_router
from app.api.v1.reports import reports_router
from app.api.v1.risk import risk_analytics_router, risk_router
from app.api.v1.scenarios import stress_scenarios_router, synthetic_router
from app.api.v1.universe import trading212_router, universe_screen_router
from app.api.v1.views import llm_moments_router, opinion_pooling_router, views_router

# Create the main API router
api_router = APIRouter()

# Include all route modules
api_router.include_router(attribution_router)
api_router.include_router(test_router)
api_router.include_router(trading212_router)
api_router.include_router(yfinance_data_router)
api_router.include_router(macro_regime_router)
api_router.include_router(database_router)
api_router.include_router(llm_moments_router)
api_router.include_router(views_router)
api_router.include_router(macro_calibration_router)
api_router.include_router(opinion_pooling_router)
api_router.include_router(stress_scenarios_router)
api_router.include_router(synthetic_router)
api_router.include_router(risk_router)
api_router.include_router(jobs_router)
api_router.include_router(dashboard_router)
api_router.include_router(market_router)
api_router.include_router(portfolio_router)
api_router.include_router(rebalance_policy_router)
api_router.include_router(reference_indices_router)
api_router.include_router(optimize_router)
api_router.include_router(backtest_router)
api_router.include_router(rebalance_router)
api_router.include_router(tune_router)
api_router.include_router(validate_router)
api_router.include_router(universe_screen_router)
api_router.include_router(factors_router)
api_router.include_router(risk_analytics_router)
api_router.include_router(reports_router)
api_router.include_router(scheduler_router)
