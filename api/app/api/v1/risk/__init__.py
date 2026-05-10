"""Risk Routes."""

from app.api.v1.risk.risk import router as risk_router
from app.api.v1.risk.risk_analytics import router as risk_analytics_router

__all__ = ["risk_analytics_router", "risk_router"]
