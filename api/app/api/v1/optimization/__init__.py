"""Optimization Routes."""

from app.api.v1.optimization.optimize import router as optimize_router
from app.api.v1.optimization.tune import router as tune_router
from app.api.v1.optimization.validate import router as validate_router

__all__ = ["optimize_router", "tune_router", "validate_router"]
