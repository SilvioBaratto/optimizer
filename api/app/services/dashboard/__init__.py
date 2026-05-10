"""Dashboard Services."""

from app.services.dashboard.dashboard_service import (
    compute_equity_curve,
    compute_performance_metrics,
)

__all__ = ["compute_equity_curve", "compute_performance_metrics"]
