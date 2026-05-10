"""Re-export from shared infrastructure — keeps yfinance import paths stable."""

from app.services.infrastructure.circuit_breaker import CircuitBreaker

__all__ = ["CircuitBreaker"]
