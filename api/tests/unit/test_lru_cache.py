"""Unit tests for the shared ``LRUCache`` resilience primitive (issue #823).

Covers the gaps left by ``test_shared_infrastructure.py``: TTL expiry, LRU
eviction at capacity, ``contains``/``keys()`` TTL respect, and thread-safe
interleaving.  The monotonic clock is mocked for deterministic TTL assertions
(no real sleeps).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from app.services.infrastructure.cache import LRUCache

_CLOCK_PATH = "app.services.infrastructure.cache.time.monotonic"


class _Clock:
    """Mutable monotonic-clock stand-in."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> Iterator[_Clock]:
    c = _Clock()
    with patch(_CLOCK_PATH, c):
        yield c


# ---------------------------------------------------------------------------
# Basic get / put
# ---------------------------------------------------------------------------


class TestGetPut:
    def test_when_key_absent_then_get_returns_none(self) -> None:
        assert LRUCache().get("missing") is None

    def test_when_put_then_get_returns_value(self) -> None:
        cache = LRUCache()
        cache.put("k", "v")
        assert cache.get("k") == "v"

    def test_when_put_twice_then_latest_value_returned(self) -> None:
        cache = LRUCache()
        cache.put("k", "v1")
        cache.put("k", "v2")
        assert cache.get("k") == "v2"

    def test_when_cleared_then_size_zero_and_miss(self) -> None:
        cache = LRUCache()
        cache.put("k", "v")
        cache.clear()
        assert cache.size() == 0
        assert cache.get("k") is None

    def test_capacity_property_reflects_constructor(self) -> None:
        assert LRUCache(capacity=42).capacity == 42


# ---------------------------------------------------------------------------
# TTL expiry
# ---------------------------------------------------------------------------


class TestTTL:
    def test_when_within_ttl_then_hit(self, clock: _Clock) -> None:
        cache = LRUCache(default_ttl=10.0)
        cache.put("k", "v")
        clock.advance(9.0)
        assert cache.get("k") == "v"

    def test_when_ttl_elapsed_then_get_returns_none(self, clock: _Clock) -> None:
        cache = LRUCache(default_ttl=10.0)
        cache.put("k", "v")
        clock.advance(11.0)
        assert cache.get("k") is None

    def test_when_per_put_ttl_overrides_default(self, clock: _Clock) -> None:
        cache = LRUCache(default_ttl=100.0)
        cache.put("k", "v", ttl=5.0)
        clock.advance(6.0)
        assert cache.get("k") is None

    def test_when_no_ttl_then_never_expires(self, clock: _Clock) -> None:
        cache = LRUCache()  # default_ttl=None
        cache.put("k", "v")
        clock.advance(1_000_000.0)
        assert cache.get("k") == "v"


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


class TestEviction:
    def test_when_over_capacity_then_oldest_evicted(self) -> None:
        cache = LRUCache(capacity=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)  # evicts "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_when_accessed_then_promoted_and_spared_from_eviction(self) -> None:
        cache = LRUCache(capacity=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.get("a")  # "a" now most-recently-used
        cache.put("c", 3)  # evicts "b", not "a"
        assert cache.get("a") == 1
        assert cache.get("b") is None

    def test_when_re_put_then_promoted(self) -> None:
        cache = LRUCache(capacity=2)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("a", 11)  # "a" promoted
        cache.put("c", 3)  # evicts "b"
        assert cache.get("a") == 11
        assert cache.get("b") is None


# ---------------------------------------------------------------------------
# contains / keys respect TTL
# ---------------------------------------------------------------------------


class TestContainsAndKeys:
    def test_contains_true_before_expiry_false_after(self, clock: _Clock) -> None:
        cache = LRUCache(default_ttl=10.0)
        cache.put("k", "v")
        assert cache.contains("k") is True
        clock.advance(11.0)
        assert cache.contains("k") is False

    def test_contains_false_for_absent_key(self) -> None:
        assert LRUCache().contains("nope") is False

    def test_keys_lists_live_keys(self) -> None:
        cache = LRUCache()
        cache.put("a", 1)
        cache.put("b", 2)
        assert set(cache.keys()) == {"a", "b"}

    def test_keys_drops_expired(self, clock: _Clock) -> None:
        cache = LRUCache(default_ttl=10.0)
        cache.put("fresh", 1)
        cache.put("stale", 2, ttl=2.0)
        clock.advance(3.0)
        assert cache.keys() == ["fresh"]
        assert cache.size() == 1  # expired entry purged


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_when_interleaved_then_no_corruption(self) -> None:
        cache = LRUCache(capacity=100)
        barrier = threading.Barrier(2)

        def worker(prefix: str) -> None:
            barrier.wait()
            for i in range(50):
                key = f"{prefix}-{i}"
                cache.put(key, i)
                cache.get(key)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, p) for p in ("A", "B")]
            for f in futures:
                f.result()  # raises if any worker threw

        assert cache.size() == 100
