"""Unit tests for TickerMappingCache.

TTL/capacity-eviction acceptance items are N/A — this cache is an ephemeral
per-build dict with no TTL or capacity bound; only the real CRUD + stats +
concurrency-safety surface is exercised.

All tests are fast (<< 1 s each).  No network is touched.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from app.services.universe.trading212.cache.ticker_cache import TickerMappingCache


class TestTickerMappingCacheMiss:
    def test_when_key_absent_then_get_mapping_returns_none(self) -> None:
        cache = TickerMappingCache()
        result = cache.get_mapping("AAPL", "NYSE")
        assert result is None

    def test_when_miss_then_misses_counter_incremented(self) -> None:
        cache = TickerMappingCache()
        cache.get_mapping("AAPL", "NYSE")
        assert cache.get_stats()["misses"] == 1

    def test_when_miss_then_hits_stays_zero(self) -> None:
        cache = TickerMappingCache()
        cache.get_mapping("AAPL", "NYSE")
        assert cache.get_stats()["hits"] == 0

    def test_when_miss_then_hit_rate_is_zero(self) -> None:
        cache = TickerMappingCache()
        cache.get_mapping("AAPL", "NYSE")
        assert cache.get_stats()["hit_rate"] == 0.0


class TestTickerMappingCachePutAndGet:
    def test_when_saved_then_get_mapping_returns_ticker(self) -> None:
        cache = TickerMappingCache()
        cache.save_mapping("AAPL", "NYSE", "AAPL")
        result = cache.get_mapping("AAPL", "NYSE")
        assert result == "AAPL"

    def test_when_hit_then_hits_counter_incremented(self) -> None:
        cache = TickerMappingCache()
        cache.save_mapping("AAPL", "NYSE", "AAPL")
        cache.get_mapping("AAPL", "NYSE")
        assert cache.get_stats()["hits"] == 1

    def test_when_one_miss_then_one_hit_then_hit_rate_is_half(self) -> None:
        cache = TickerMappingCache()
        cache.get_mapping("AAPL", "NYSE")  # miss
        cache.save_mapping("AAPL", "NYSE", "AAPL")
        cache.get_mapping("AAPL", "NYSE")  # hit
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_when_key_uses_different_exchange_then_different_slot(self) -> None:
        cache = TickerMappingCache()
        cache.save_mapping("VOD", "London Stock Exchange", "VOD.L")
        cache.save_mapping("VOD", "NYSE", "VOD")
        assert cache.get_mapping("VOD", "London Stock Exchange") == "VOD.L"
        assert cache.get_mapping("VOD", "NYSE") == "VOD"

    def test_size_reflects_number_of_distinct_entries(self) -> None:
        cache = TickerMappingCache()
        cache.save_mapping("AAPL", "NYSE", "AAPL")
        cache.save_mapping("MSFT", "NASDAQ", "MSFT")
        assert cache.get_stats()["size"] == 2

    def test_overwrite_existing_key_does_not_grow_size(self) -> None:
        cache = TickerMappingCache()
        cache.save_mapping("AAPL", "NYSE", "AAPL")
        cache.save_mapping("AAPL", "NYSE", "AAPL_UPDATED")
        assert cache.get_stats()["size"] == 1
        assert cache.get_mapping("AAPL", "NYSE") == "AAPL_UPDATED"


class TestTickerMappingCacheClear:
    def test_when_cleared_then_store_is_empty(self) -> None:
        cache = TickerMappingCache()
        cache.save_mapping("AAPL", "NYSE", "AAPL")
        cache.clear()
        assert cache.get_mapping("AAPL", "NYSE") is None

    def test_when_cleared_then_size_is_zero(self) -> None:
        cache = TickerMappingCache()
        cache.save_mapping("AAPL", "NYSE", "AAPL")
        cache.clear()
        assert cache.get_stats()["size"] == 0

    def test_when_cleared_then_hits_and_misses_reset_to_zero(self) -> None:
        cache = TickerMappingCache()
        cache.save_mapping("AAPL", "NYSE", "AAPL")
        cache.get_mapping("AAPL", "NYSE")
        cache.clear()
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_when_cleared_then_hit_rate_is_zero(self) -> None:
        cache = TickerMappingCache()
        cache.save_mapping("AAPL", "NYSE", "AAPL")
        cache.get_mapping("AAPL", "NYSE")
        cache.clear()
        assert cache.get_stats()["hit_rate"] == 0.0

    def test_fresh_cache_has_zero_hit_rate_with_no_requests(self) -> None:
        cache = TickerMappingCache()
        assert cache.get_stats()["hit_rate"] == 0.0


class TestTickerMappingCacheConcurrency:
    """Verify the cache does not raise under concurrent read/write access.

    The cache carries no internal lock; this test only asserts absence of
    exceptions and structural consistency of the final state — not strict
    linearisability.
    """

    def test_concurrent_save_and_get_raises_no_exception(self) -> None:
        cache = TickerMappingCache()
        symbols = [f"SYM{i}" for i in range(25)]

        def worker(symbol: str) -> None:
            cache.save_mapping(symbol, "NYSE", f"{symbol}_YF")
            time.sleep(0)  # yield to scheduler
            cache.get_mapping(symbol, "NYSE")

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, sym) for sym in symbols]
            for fut in futures:
                fut.result()  # re-raise any exception

        stats = cache.get_stats()
        # Every symbol was saved exactly once → size == 25
        assert stats["size"] == len(symbols)
