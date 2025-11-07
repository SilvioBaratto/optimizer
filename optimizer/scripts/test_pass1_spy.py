#!/usr/bin/env python3
"""
Test Pass 1 SPY Fetching
========================
Simulates what happens in Pass 1 when fetching SPY benchmark data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.yfinance import YFinanceClient
import pandas as pd

print("="* 100)
print("TESTING PASS 1 SPY BENCHMARK FETCHING")
print("="* 100)
print()

client = YFinanceClient.get_instance()

# Test 1: Fetch SPY directly
print("[TEST 1] Fetch SPY as standalone ticker")
print("-" * 100)
spy_data = client.fetch_history("SPY", period="2y")
if spy_data is not None and not spy_data.empty:
    print(f"✅ SUCCESS: Got {len(spy_data)} rows for SPY")
    print(f"   Date range: {spy_data.index[0]} to {spy_data.index[-1]}")
else:
    print(f"❌ FAILED: SPY returned None or empty")
print()

# Test 2: Fetch a stock + SPY benchmark (simulating Pass 1)
print("[TEST 2] Fetch AAPL + SPY benchmark (simulates Pass 1)")
print("-" * 100)
stock_hist, spy_hist, stock_info = client.fetch_price_and_benchmark(
    symbol="AAPL",
    benchmark="SPY",
    period="2y"
)

if stock_hist is not None:
    print(f"✅ AAPL: {len(stock_hist)} rows")
else:
    print(f"❌ AAPL: Failed")

if spy_hist is not None:
    print(f"✅ SPY:  {len(spy_hist)} rows")
    print(f"   Sufficient history: {len(spy_hist) >= 10}")
else:
    print(f"❌ SPY:  Failed (this is the problem!)")
    print(f"   This would trigger: 'Insufficient history data for SPY (got 0 rows, expected >=10)'")
print()

# Test 3: Check if it's a threading/bulk download issue
print("[TEST 3] Bulk download test (multiple stocks + SPY)")
print("-" * 100)
try:
    tickers = ["AAPL", "MSFT", "SPY"]
    data = client.bulk_download(tickers, period="1y")

    for ticker in tickers:
        if ticker in data and not data[ticker].empty:
            print(f"✅ {ticker}: {len(data[ticker])} rows")
        else:
            print(f"❌ {ticker}: No data or empty")
except Exception as e:
    print(f"❌ Bulk download failed: {e}")
print()

# Test 4: Check cache
print("[TEST 4] Check SPY cache status")
print("-" * 100)
stats = client.get_cache_stats()
print(f"Cache size: {stats['size']}/{stats['capacity']}")
print(f"Hit rate: {stats['hit_rate']:.1%}")

# Check if SPY ticker is cached
ticker_cached = "SPY" in str(client._ticker_cache.cache_info())
print(f"SPY in cache: {ticker_cached}")
print()

print("="*100)
print("DIAGNOSIS")
print("="*100)
print()
print("If all tests pass but Pass 1 still fails, the issue is likely:")
print("1. Race condition in parallel/async fetching")
print("2. Rate limiting hitting SPY requests specifically")
print("3. Cache corruption for SPY ticker")
print("4. Exception being swallowed somewhere in Pass 1 processing")
print()
print("RECOMMENDATION: Add explicit logging in Pass 1 to capture SPY fetch attempts")
