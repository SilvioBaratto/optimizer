"""Shim-parity tests for the yfinance infrastructure re-export layer (issue #823).

``market_data/yfinance/infrastructure/{cache,circuit_breaker,rate_limiter,retry}``
re-export the shared resilience primitives so legacy yfinance import paths stay
stable.  These tests assert the shim symbols ARE the shared symbols (identity)
and that each shim's ``__all__`` is satisfied — import + ``hasattr`` only, no
instantiation.
"""

from __future__ import annotations

import app.services.market_data.yfinance.infrastructure.cache as shim_cache
import app.services.market_data.yfinance.infrastructure.circuit_breaker as shim_cb
import app.services.market_data.yfinance.infrastructure.rate_limiter as shim_rl
import app.services.market_data.yfinance.infrastructure.retry as shim_retry
from app.services.infrastructure.cache import LRUCache
from app.services.infrastructure.circuit_breaker import CircuitBreaker
from app.services.infrastructure.rate_limiter import RateLimiter
from app.services.infrastructure.retry import (
    is_transient_network_error,
    retry_with_backoff,
)


class TestShimIdentity:
    def test_cache_shim_reexports_shared_lru_cache(self) -> None:
        assert shim_cache.LRUCache is LRUCache

    def test_circuit_breaker_shim_reexports_shared(self) -> None:
        assert shim_cb.CircuitBreaker is CircuitBreaker

    def test_rate_limiter_shim_reexports_shared(self) -> None:
        assert shim_rl.RateLimiter is RateLimiter

    def test_retry_shim_reexports_shared_retry(self) -> None:
        assert shim_retry.retry_with_backoff is retry_with_backoff

    def test_retry_shim_aliases_is_rate_limit_error_to_transient(self) -> None:
        # yfinance keeps the legacy name ``is_rate_limit_error`` for _base.py.
        assert shim_retry.is_rate_limit_error is is_transient_network_error


class TestShimAllExports:
    def test_cache_all_present(self) -> None:
        assert shim_cache.__all__ == ["LRUCache"]
        assert hasattr(shim_cache, "LRUCache")

    def test_circuit_breaker_all_present(self) -> None:
        assert shim_cb.__all__ == ["CircuitBreaker"]
        assert hasattr(shim_cb, "CircuitBreaker")

    def test_rate_limiter_all_present(self) -> None:
        assert shim_rl.__all__ == ["RateLimiter"]
        assert hasattr(shim_rl, "RateLimiter")

    def test_retry_all_present(self) -> None:
        assert set(shim_retry.__all__) == {"is_rate_limit_error", "retry_with_backoff"}
        for name in shim_retry.__all__:
            assert hasattr(shim_retry, name)
