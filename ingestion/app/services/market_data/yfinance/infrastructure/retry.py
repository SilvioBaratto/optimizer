"""yfinance-specific retry wrappers built on shared infrastructure."""

from app.services.infrastructure.retry import (
    is_transient_network_error,
    retry_with_backoff,
)

# Keep the yfinance-specific name for backwards compat with _base.py
is_rate_limit_error = is_transient_network_error

__all__ = ["is_rate_limit_error", "retry_with_backoff"]
