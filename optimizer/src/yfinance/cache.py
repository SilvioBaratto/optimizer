import logging
import threading
from typing import Optional, Any
from collections import OrderedDict

logger = logging.getLogger(__name__)

class LRUCache:
    """
    Simple thread-safe LRU cache for Ticker objects.

    Uses OrderedDict to maintain insertion order and implements
    least-recently-used eviction policy.
    """

    def __init__(self, capacity: int = 3000):
        """
        Initialize LRU cache.

        Args:
            capacity: Maximum number of items to cache
        """
        self._cache = OrderedDict()
        self._capacity = capacity
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
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
                self._cache.popitem(last=False)  # Remove oldest (first item)

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
