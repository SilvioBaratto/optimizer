"""Base client providing shared ticker caching and resilience infrastructure.

All ticker-based sub-clients extend this class to share a single cache,
rate limiter, and circuit breaker instance with the main ``YFinanceClient``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import yfinance as yf

from .infrastructure import (
    FaultWatch,
    forget_cached_responses,
    install_fault_witness,
    is_no_data_error,
    is_rate_limit_error,
    is_transport_error,
    note_ticker,
    record_answer,
    record_failure,
    retry_with_backoff,
    watch_for_faults,
)
from .protocols import CacheProtocol, CircuitBreakerProtocol, RateLimiterProtocol

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RequestLostError(RuntimeError):
    """A yfinance request never got an answer: every attempt failed.

    Distinct from Yahoo answering that it holds no data. Raised only to callers
    that opt in with ``raise_on_lost``, because most fetches treat both as absent.
    """


@dataclass
class _Attempts:
    """What the attempts of one request left behind, filled in by its callbacks."""

    last_error: Exception | None = None
    answered: bool = False


def fetch_with_resilience(
    action: Callable[[], T],
    retries: int,
    *,
    circuit_breaker: CircuitBreakerProtocol,
    rate_limit_predicate: Callable[[Exception], bool] = is_rate_limit_error,
    is_valid: Callable[[T], bool] | None = None,
    raise_on_lost: bool = False,
    on_attempt_failed: Callable[[], None] | None = None,
) -> T | None:
    """Run one yfinance request, retrying transient failures only.

    Yahoo's final answers end the call at once: a no-data exception, or a
    response that *is_valid* rejects, returns ``None`` without another
    request. yfinance already turns a 404 for a data module the symbol lacks
    into an empty result, so an empty result is Yahoo's answer rather than a
    lost request. That holds only while *action* raises transient faults;
    sub-clients make yfinance's swallowed ones raise again (see
    :meth:`BaseClient._fetch_with_resilience`).

    ``None`` is also what a request returns when every attempt failed, so the
    outcome is additionally recorded in the active
    :func:`~.infrastructure.track_request_outcomes` scope: a request that got
    no answer because Yahoo was unreachable or refused service is a failure
    there, every other ending counts as answered.

    Args:
        action: Zero-argument callable performing one yfinance request.
        retries: Attempt budget, including the first attempt.
        circuit_breaker: Breaker checked before each attempt and tripped by
            rate limits. It is reset only by a result *is_valid* accepts.
        rate_limit_predicate: Recognises rate-limit errors that should trip
            the breaker.
        is_valid: Predicate a usable result must satisfy; a result it rejects
            (typically an empty frame) is returned as ``None``.
        raise_on_lost: Raise :class:`RequestLostError` when every attempt failed
            (DNS, TLS, a garbled body, throttling, an open breaker) instead of
            returning ``None``, so the caller can tell an outage from an answer.
        on_attempt_failed: Called after each failed attempt except Yahoo's
            final no-data answers. yfinance memoizes a failed request on the
            ``Ticker`` it ran on, so callers use this to discard that object
            and let the next attempt issue a real request.

    Returns:
        The result of *action*, or ``None`` when Yahoo has no usable data or
        every attempt failed (the latter only without *raise_on_lost*).

    Raises:
        RequestLostError: *raise_on_lost* is set and the last attempt failed.
    """
    attempts = _Attempts()

    def _guarded() -> T:
        try:
            circuit_breaker.check()
            result = action()
        except Exception as exc:
            attempts.last_error = exc
            if on_attempt_failed is not None and not is_no_data_error(exc):
                on_attempt_failed()
            raise
        # Only the final attempt decides: an earlier failure followed by a real
        # answer (even an empty one) is not a lost request.
        attempts.last_error = None
        return result

    def _on_success(_: T) -> None:
        # Deliberately no breaker reset: a non-exception return can still be the
        # empty frame yfinance gives for a swallowed 429.
        attempts.answered = True

    try:
        result = retry_with_backoff(
            _guarded,
            retries,
            is_rate_limit_error=rate_limit_predicate,
            on_rate_limit=circuit_breaker.trigger,
            on_success=_on_success,
            is_permanent_error=is_no_data_error,
        )
    except Exception as exc:
        logger.debug("Yahoo has no data for this request: %s", exc)
        record_answer()
        return None

    error = attempts.last_error
    if (
        not attempts.answered
        and error is not None
        and (is_transport_error(error) or rate_limit_predicate(error))
    ):
        record_failure(f"{type(error).__name__}: {error}")
    else:
        record_answer()

    if result is None and raise_on_lost and error is not None:
        msg = (
            f"yfinance request lost after {retries} attempts: "
            f"{type(error).__name__}: {error}"
        )
        raise RequestLostError(msg) from error

    if result is None or (is_valid is not None and not is_valid(result)):
        return None

    # yfinance swallows a 429 on several modules and returns an empty frame, so
    # only a result the caller accepts counts as proof Yahoo is serving again.
    circuit_breaker.reset()
    return result


class BaseClient:
    """Shared infrastructure for ticker-based sub-clients."""

    def __init__(
        self,
        cache: CacheProtocol,
        rate_limiter: RateLimiterProtocol,
        circuit_breaker: CircuitBreakerProtocol,
        default_max_retries: int = 3,
    ) -> None:
        """Initialize the base client with shared resilience infrastructure.

        Args:
            cache: Ticker object cache shared across all sub-clients.
            rate_limiter: Throttles per-symbol request frequency.
            circuit_breaker: Prevents cascading failures when Yahoo is degraded.
            default_max_retries: Retry budget used when callers omit
                ``max_retries``.
        """
        self.cache = cache
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker
        self.default_max_retries = default_max_retries
        install_fault_witness()

    def _get_ticker(self, symbol: str) -> yf.Ticker:
        """Return a cached ``yf.Ticker``, creating one on cache miss."""
        note_ticker(symbol)
        ticker = self.cache.get(symbol)
        if ticker is not None:
            logger.debug("Cache hit for ticker '%s'", symbol)
            return ticker

        logger.debug("Cache miss for ticker '%s', fetching from yfinance", symbol)
        self.rate_limiter.acquire(symbol)
        ticker = yf.Ticker(symbol)
        self.cache.put(symbol, ticker)
        return ticker

    def _fetch_with_resilience(
        self,
        action: Callable[[], T],
        max_retries: int | None = None,
        *,
        is_valid: Callable[[T], bool] | None = None,
        raise_on_lost: bool = False,
    ) -> T | None:
        """Execute *action* through :func:`fetch_with_resilience`.

        yfinance's default mode swallows rate limits and server errors into an
        empty result and caches it on the ``yf.Ticker``. When *action* ends
        with a result *is_valid* rejects after such a fault was seen, the fault
        is raised again so the attempt is retried and rate limits trip the
        circuit breaker; the tainted tickers are dropped from the cache first.
        A usable result is kept even if a side request failed on the way.

        Args:
            action: Zero-argument callable performing one yfinance request.
            max_retries: Attempt budget; ``None`` uses the client default.
            is_valid: Predicate a usable result must satisfy; a result it
                rejects (typically an empty frame) is returned as ``None``.
            raise_on_lost: Raise :class:`RequestLostError` when every attempt
                failed instead of returning ``None``.

        Returns:
            The result of *action*, or ``None`` when Yahoo has no usable data
            or every attempt failed (the latter only without *raise_on_lost*).

        Raises:
            RequestLostError: *raise_on_lost* is set and the last attempt failed.
        """

        def _watched() -> T:
            with watch_for_faults() as watch:
                try:
                    result = action()
                except Exception as exc:
                    # yfinance flags a scraper as fetched before it requests, so
                    # a ticker reused after a failed attempt would answer the
                    # retry from empty state without asking Yahoo again. A no-data
                    # answer is final and costs nothing to remember, so only
                    # failures evict.
                    if watch.fault is not None or not is_no_data_error(exc):
                        self._drop_tainted(watch)
                    raise
            usable = result is not None and (is_valid is None or is_valid(result))
            if watch.fault is not None and not usable:
                self._drop_tainted(watch)
                raise watch.fault
            return result

        return fetch_with_resilience(
            _watched,
            max_retries if max_retries is not None else self.default_max_retries,
            circuit_breaker=self.circuit_breaker,
            rate_limit_predicate=self._is_rate_limit_error,
            is_valid=is_valid,
            raise_on_lost=raise_on_lost,
        )

    def _drop_tainted(self, watch: FaultWatch) -> None:
        """Forget the tickers and cached responses a faulted call touched."""
        for symbol in watch.symbols:
            self.cache.delete(symbol)
        forget_cached_responses()

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        return is_rate_limit_error(error)
