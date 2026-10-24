"""Resilience and caching primitives."""

from .availability import is_no_data_error, is_symbol_not_found, is_transport_error
from .cache import LRUCache
from .circuit_breaker import CircuitBreaker
from .faults import (
    FaultWatch,
    YahooTransientError,
    forget_cached_responses,
    install_fault_witness,
    note_ticker,
    watch_for_faults,
)
from .outcomes import (
    RequestOutcomes,
    record_answer,
    record_failure,
    track_request_outcomes,
)
from .rate_limiter import RateLimiter
from .retry import is_rate_limit_error, retry_with_backoff

__all__ = [
    "CircuitBreaker",
    "FaultWatch",
    "LRUCache",
    "RateLimiter",
    "RequestOutcomes",
    "YahooTransientError",
    "forget_cached_responses",
    "install_fault_witness",
    "is_no_data_error",
    "is_rate_limit_error",
    "is_symbol_not_found",
    "is_transport_error",
    "note_ticker",
    "record_answer",
    "record_failure",
    "retry_with_backoff",
    "track_request_outcomes",
    "watch_for_faults",
]
