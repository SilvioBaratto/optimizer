"""Scenarios Routes."""

from app.api.v1.scenarios.stress_scenarios import router as stress_scenarios_router
from app.api.v1.scenarios.synthetic import router as synthetic_router

__all__ = ["stress_scenarios_router", "synthetic_router"]
