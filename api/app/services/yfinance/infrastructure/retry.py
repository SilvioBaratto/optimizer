"""Generic retry helper with exponential backoff.

Replaces the duplicated for-loop-with-sleep pattern used across
client.py (fetch_info, fetch_history) and news_client.py (fetch).
"""

from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")


def is_rate_limit_error(error: Exception) -> bool:
    """Single source of truth for rate-limit error detection."""
    error_msg = str(error)
    rate_limit_indicators = ["Too Many Requests", "Rate limited", "429"]
    return any(indicator in error_msg for indicator in rate_limit_indicators)


def retry_with_backoff(
    action: Callable[[], T],
    max_retries: int,
    *,
    is_valid: Callable[[T], bool] | None = None,
    is_rate_limit_error: Callable[[Exception], bool] | None = None,
    on_rate_limit: Callable[[], None] | None = None,
    on_success: Callable[[T], None] | None = None,
) -> T | None:
    """Execute *action* up to *max_retries* times with linear backoff.

    Parameters
    ----------
    action:
        Zero-arg callable that returns the desired value or raises.
    max_retries:
        Total number of attempts (including the first).
    is_valid:
        Optional predicate applied to the return value of *action*.
        When it returns ``False`` the attempt is treated as a failure.
    is_rate_limit_error:
        Optional predicate applied to caught exceptions.  When it
        returns ``True``, *on_rate_limit* is called instead of the
        normal backoff sleep.
    on_rate_limit:
        Callback invoked when *is_rate_limit_error* fires (e.g.
        ``circuit_breaker.trigger``).
    on_success:
        Callback invoked with the valid result before returning.
    """
    for attempt in range(max_retries):
        try:
            result = action()

            if is_valid is not None and not is_valid(result):
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))
                    continue
                return None

            if on_success is not None:
                on_success(result)
            return result

        except Exception as exc:
            if is_rate_limit_error is not None and is_rate_limit_error(exc):
                if on_rate_limit is not None:
                    on_rate_limit()
                continue

            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
                continue
            return None

    return None
