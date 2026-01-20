"""
LRU Cache implementation for yfinance module.

Thread-safe LRU cache that complies with CacheProtocol
for dependency injection support.
"""

import threading
from collections import OrderedDict
from typing import Any


class LRUCache:
    """
    Thread-safe LRU cache for Ticker objects.
    """

    def __init__(self, capacity: int = 3000) -> None:
        """
        Initialize LRU cache.

        Args:
            capacity: Maximum number of items to cache
        """
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._capacity = capacity
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        """
        Get item from cache, moving it to end (most recent).

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        with self._lock:
            if key not in self._cache:
                return None
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: str, value: Any) -> None:
        """
        Add item to cache, evicting oldest if at capacity.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            if key in self._cache:
                # Update existing - move to end
                self._cache.move_to_end(key)
            self._cache[key] = value

            # Evict oldest if over capacity
            if len(self._cache) > self._capacity:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)

    @property
    def capacity(self) -> int:
        """Get cache capacity."""
        return self._capacity

    def contains(self, key: str) -> bool:
        """
        Check if key exists in cache without updating recency.

        Args:
            key: Cache key

        Returns:
            True if key exists in cache
        """
        with self._lock:
            return key in self._cache

    def keys(self) -> list[str]:
        """
        Get all keys in cache (most recent last).

        Returns:
            List of cache keys in LRU order
        """
        with self._lock:
            return list(self._cache.keys())
