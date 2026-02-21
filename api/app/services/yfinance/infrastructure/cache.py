import threading
import time
from collections import OrderedDict
from typing import Any


class LRUCache:
    def __init__(
        self,
        capacity: int = 3000,
        default_ttl: float | None = None,
    ) -> None:
        self._cache: OrderedDict[str, tuple[Any, float | None]] = OrderedDict()
        self._capacity = capacity
        self._default_ttl = default_ttl
        self._lock = threading.Lock()

    def _is_expired(self, expiry: float | None) -> bool:
        return expiry is not None and time.monotonic() > expiry

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key not in self._cache:
                return None

            value, expiry = self._cache[key]
            if self._is_expired(expiry):
                del self._cache[key]
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return value

    def put(self, key: str, value: Any, ttl: float | None = None) -> None:
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expiry = time.monotonic() + effective_ttl if effective_ttl is not None else None

        with self._lock:
            if key in self._cache:
                # Update existing - move to end
                self._cache.move_to_end(key)
            self._cache[key] = (value, expiry)

            # Evict oldest if over capacity
            if len(self._cache) > self._capacity:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def capacity(self) -> int:
        return self._capacity

    def contains(self, key: str) -> bool:
        with self._lock:
            if key not in self._cache:
                return False
            _, expiry = self._cache[key]
            if self._is_expired(expiry):
                del self._cache[key]
                return False
            return True

    def keys(self) -> list[str]:
        with self._lock:
            now = time.monotonic()
            expired = [
                k
                for k, (_, expiry) in self._cache.items()
                if expiry is not None and now > expiry
            ]
            for k in expired:
                del self._cache[k]
            return list(self._cache.keys())
