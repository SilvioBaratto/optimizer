"""API v1 router - imports and includes all route modules"""

from fastapi import APIRouter

from app.api.v1.test import router as test_router
from app.api.v1.trading212 import router as trading212_router
from app.api.v1.yfinance_data import router as yfinance_data_router
from app.api.v1.macro_regime import router as macro_regime_router
from app.api.v1.database import router as database_router
from app.api.v1.llm_moments import router as llm_moments_router
from app.api.v1.views import router as views_router
from app.api.v1.macro_calibration import router as macro_calibration_router
from app.api.v1.opinion_pooling import router as opinion_pooling_router
from app.api.v1.stress_scenarios import router as stress_scenarios_router

# Create the main API router
api_router = APIRouter()

# Include all route modules
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
