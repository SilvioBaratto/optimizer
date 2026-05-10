"""Scenarios Services."""

from app.services.scenarios.stress_scenarios import generate_stress_scenarios
from app.services.scenarios.synthetic_service import generate_scenarios

__all__ = ["generate_scenarios", "generate_stress_scenarios"]
