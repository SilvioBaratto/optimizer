"""
Protocol definitions for yfinance module.

Defines abstractions for dependency injection, enabling:
- Testability through mock implementations
- Open/Closed principle compliance
- Dependency Inversion principle compliance
"""

from typing import Protocol, Any, runtime_checkable


@runtime_checkable
class CacheProtocol(Protocol):
    """
    Cache abstraction for dependency injection.

    Implementations must provide thread-safe get/put operations
    with LRU or similar eviction policy.
    """

    def get(self, key: str) -> Any | None:
        """
        Retrieve item from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        ...

    def put(self, key: str, value: Any) -> None:
        """
        Store item in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        ...

    def clear(self) -> None:
        """Clear all cached items."""
        ...

    def size(self) -> int:
        """Get current number of cached items."""
        ...


@runtime_checkable
class RateLimiterProtocol(Protocol):
    """
    Rate limiting abstraction.

    Implementations must provide thread-safe rate limiting
    with per-key tracking.
    """

    def acquire(self, key: str) -> None:
        """
        Acquire rate limit slot for a key.

        Blocks if necessary to enforce rate limit.

        Args:
            key: Rate limit key (e.g., ticker symbol)
        """
        ...


@runtime_checkable
class CircuitBreakerProtocol(Protocol):
    """
    Circuit breaker abstraction.

    Implementations must provide thread-safe circuit breaking
    with exponential backoff.
    """

    def check(self) -> None:
        """
        Check if circuit breaker is active.

        Blocks if circuit breaker is active, waiting until
        the cooldown period expires.

        Raises:
            RuntimeError: If maximum attempts exceeded
        """
        ...

    def trigger(self) -> None:
        """
        Trigger the circuit breaker.

        Activates exponential backoff for subsequent requests.
        """
        ...

    def reset(self) -> None:
        """
        Gradually reset circuit breaker after success.

        Called after successful API calls to reduce backoff.
        """
        ...

    @property
    def is_active(self) -> bool:
        """Check if circuit breaker is currently active."""
        ...
