"""API v1 router - imports and includes all route modules"""

from fastapi import APIRouter

from app.api.v1.attribution import router as attribution_router
from app.api.v1.reports import router as reports_router
from app.api.v1.scheduler import router as scheduler_router
from app.api.v1.backtest import router as backtest_router
from app.api.v1.factors import router as factors_router
from app.api.v1.risk_analytics import router as risk_analytics_router
from app.api.v1.rebalance import router as rebalance_router
from app.api.v1.tune import router as tune_router
from app.api.v1.validate import router as validate_router
from app.api.v1.universe_screen import router as universe_screen_router
from app.api.v1.dashboard import market_router
from app.api.v1.optimize import router as optimize_router
from app.api.v1.dashboard import router as dashboard_router
from app.api.v1.database import router as database_router
from app.api.v1.jobs import router as jobs_router
from app.api.v1.llm_moments import router as llm_moments_router
from app.api.v1.macro_calibration import router as macro_calibration_router
from app.api.v1.macro_regime import router as macro_regime_router
from app.api.v1.opinion_pooling import router as opinion_pooling_router
from app.api.v1.portfolio import router as portfolio_router
from app.api.v1.rebalance_policy import router as rebalance_policy_router
from app.api.v1.reference_indices import router as reference_indices_router
from app.api.v1.risk import router as risk_router
from app.api.v1.risk_budget import router as risk_budget_router
from app.api.v1.stress_scenarios import router as stress_scenarios_router
from app.api.v1.synthetic import router as synthetic_router
from app.api.v1.test import router as test_router
from app.api.v1.trading212 import router as trading212_router
from app.api.v1.views import router as views_router
from app.api.v1.yfinance_data import router as yfinance_data_router

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
api_router.include_router(risk_budget_router)
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
