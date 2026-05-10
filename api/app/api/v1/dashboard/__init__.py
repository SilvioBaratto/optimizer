"""Dashboard Routes."""

from app.api.v1.dashboard.dashboard import market_router
from app.api.v1.dashboard.dashboard import router as dashboard_router

__all__ = ["dashboard_router", "market_router"]
