#!/usr/bin/env python3
"""
Test Circuit Breaker - Verify thread-safe behavior
===================================================
Simulates multiple threads detecting rate limits simultaneously.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import threading
import time
from src.yfinance import YFinanceClient

print("=" * 100)
print("CIRCUIT BREAKER TEST - Multiple Threads Triggering Simultaneously")
print("=" * 100)
print()

# Reset instance to start fresh
YFinanceClient.reset_instance()
client = YFinanceClient.get_instance()

print("[TEST] Simulating 10 parallel threads detecting rate limit...")
print()

def simulate_rate_limit_detection(thread_id: int):
    """Simulate a thread detecting rate limit"""
    print(f"  Thread {thread_id}: Detecting rate limit...")
    client._trigger_circuit_breaker()
    print(f"  Thread {thread_id}: Trigger call completed")

# Create 10 threads that all detect rate limit at same time
threads = []
for i in range(10):
    t = threading.Thread(target=simulate_rate_limit_detection, args=(i,))
    threads.append(t)

# Start all threads simultaneously
start_time = time.time()
for t in threads:
    t.start()

# Wait for all threads to complete
for t in threads:
    t.join()

elapsed = time.time() - start_time

print()
print("=" * 100)
print("RESULT")
print("=" * 100)

# Check circuit breaker state
print(f"Time elapsed: {elapsed:.2f} seconds")
print(f"Circuit breaker active: {client._circuit_breaker_active}")
print(f"Attempt counter: {client._rate_limit_attempt}")
print(f"Expected wait time: {(2 ** client._rate_limit_attempt) * 60 / 60:.1f} minutes")

print()
if client._rate_limit_attempt == 1:
    print("✅ SUCCESS: Counter only incremented once (thread-safe)")
    print("   10 parallel triggers resulted in 1 activation (correct behavior)")
else:
    print(f"❌ FAILURE: Counter incremented to {client._rate_limit_attempt}")
    print("   Expected: 1, Got: {client._rate_limit_attempt}")
    print("   Multiple threads triggered circuit breaker multiple times")

print()
print("=" * 100)
