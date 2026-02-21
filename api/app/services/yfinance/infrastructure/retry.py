"""Generic retry helper with exponential backoff.

Replaces the duplicated for-loop-with-sleep pattern used across
client.py (fetch_info, fetch_history) and news_client.py (fetch).
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

# Yahoo Finance signals throttling in two ways:
#   - HTTP 429 ("Too Many Requests") — application-layer rate limit
#   - TCP ECONNRESET (errno 104) — transport-layer throttle / hard cutoff
# Both must trigger the circuit breaker; treating only 429 caused today's
# abort when BYG.L returned a connection reset that bypassed backoff entirely.
_TRANSIENT_INDICATORS = [
    # HTTP rate limiting
    "Too Many Requests",
    "Rate limited",
    "429",
    # TCP-level throttling — Yahoo Finance RSTs the connection when overloaded
    "Connection reset by peer",
    "[Errno 104]",
    "ConnectionResetError",
    "RemoteDisconnected",
    # Other transient network failures worth treating the same way
    "ChunkedEncodingError",
    "IncompleteRead",
    "ReadTimeout",
    "ConnectTimeout",
]


def is_rate_limit_error(error: Exception) -> bool:
    """Detect transient Yahoo Finance errors: HTTP rate limits and TCP resets."""
    combined = f"{type(error).__name__}: {error}"
    return any(indicator in combined for indicator in _TRANSIENT_INDICATORS)


def _full_jitter(attempt: int, base: float, cap: float) -> float:
    """Full-jitter exponential backoff (AWS/Google SOTA recommendation).

    Returns a value uniformly sampled from [0, min(cap, base * 2**attempt)].
    Full jitter avoids thundering-herd when many tickers retry simultaneously.
    """
    return random.uniform(0, min(cap, base * (2**attempt)))


def retry_with_backoff(
    action: Callable[[], T],
    max_retries: int,
    *,
    base_delay: float = 2.0,
    max_delay: float = 120.0,
    is_valid: Callable[[T], bool] | None = None,
    is_rate_limit_error: Callable[[Exception], bool] | None = None,
    on_rate_limit: Callable[[], None] | None = None,
    on_success: Callable[[T], None] | None = None,
) -> T | None:
    """Execute *action* up to *max_retries* times with exponential-jitter backoff.

    Parameters
    ----------
    action:
        Zero-arg callable that returns the desired value or raises.
    max_retries:
        Total number of attempts (including the first).
    base_delay:
        Base delay in seconds for the exponential-jitter formula.
        Actual sleep = uniform(0, min(max_delay, base_delay * 2**attempt)).
        Defaults to 2.0 s → [0–2 s, 0–4 s, 0–8 s, …] up to max_delay.
    max_delay:
        Maximum sleep cap in seconds. Defaults to 120 s.
    is_valid:
        Optional predicate applied to the return value of *action*.
        When it returns ``False`` the attempt is treated as a failure.
    is_rate_limit_error:
        Optional predicate applied to caught exceptions. When it
        returns ``True``, *on_rate_limit* is called instead of the
        normal backoff sleep (e.g. to trigger a circuit breaker).
    on_rate_limit:
        Callback invoked when *is_rate_limit_error* fires (e.g.
        ``circuit_breaker.trigger``). The circuit breaker's own
        exponential sleep handles the long wait on the next check().
    on_success:
        Callback invoked with the valid result before returning.
    """
    for attempt in range(max_retries):
        try:
            result = action()

            if is_valid is not None and not is_valid(result):
                if attempt < max_retries - 1:
                    time.sleep(_full_jitter(attempt, base_delay, max_delay))
                    continue
                return None

            if on_success is not None:
                on_success(result)
            return result

        except Exception as exc:
            if is_rate_limit_error is not None and is_rate_limit_error(exc):
                if on_rate_limit is not None:
                    on_rate_limit()
                # Small jitter before next circuit-breaker check() to avoid
                # thread pileup; the circuit breaker handles the long sleep.
                if attempt < max_retries - 1:
                    time.sleep(_full_jitter(attempt, 1.0, 10.0))
                continue

            if attempt < max_retries - 1:
                time.sleep(_full_jitter(attempt, base_delay, max_delay))
                continue
            return None

    return None
