"""Circuit breakers and request timing for the daemon's external HTTP sources.

FRED, Trading Economics, Il Sole 24 Ore and Trading 212 each have one breaker per
process, sized from ``app.config.settings`` when this module is imported, so the
clients of a source that use it share its back-off state and the metrics can
report it. :func:`timed_request` records one request on
``external_request_seconds``.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum

from app import metrics
from app.config import Settings, settings
from app.services.infrastructure.circuit_breaker import CircuitBreaker

# Trading 212 reports when a rate-limit window ends; neither its breaker window
# nor a wait taken from that report exceeds this, so a bogus header cannot stall
# a run.
TRADING_212_MAX_WAIT_SECONDS = 120
# Windows of 2, 4, 8 and 16 s at failure counts 1 to 4.
_TRADING_212_BASE_WAIT_MINUTES = 1 / 30

_OK = "ok"
_ERROR = "error"


class ExternalService(str, Enum):
    """An external HTTP source with a process-wide breaker.

    The value names the breaker in its refusal message and labels the source's
    metrics.
    """

    FRED = "FRED API"
    TRADING_ECONOMICS = "Trading Economics"
    IL_SOLE = "IlSole24Ore"
    TRADING_212 = "Trading 212"


def build_breaker(
    service: ExternalService, config: Settings | None = None
) -> CircuitBreaker:
    """Build a closed circuit breaker for ``service``.

    Args:
        service: The source the breaker guards.
        config: Settings that hold the source's ``breaker_*_max_attempts``;
            ``app.config.settings`` when omitted.

    Returns:
        A breaker named after ``service``. Trading 212's back-off windows start
        at 2 s, double with each failure counted and stop at
        ``TRADING_212_MAX_WAIT_SECONDS`` (a ``min_wait_seconds`` passed to
        ``trigger()`` can lengthen one up to that cap); the other sources keep
        the ``CircuitBreaker`` default windows.
    """
    config = config or settings
    max_attempts = {
        ExternalService.FRED: config.breaker_fred_max_attempts,
        ExternalService.TRADING_ECONOMICS: config.breaker_tradingeconomics_max_attempts,
        ExternalService.IL_SOLE: config.breaker_ilsole_max_attempts,
        ExternalService.TRADING_212: config.breaker_trading_212_max_attempts,
    }[service]
    if service is ExternalService.TRADING_212:
        return CircuitBreaker(
            service_name=service.value,
            max_attempts=max_attempts,
            base_wait_minutes=_TRADING_212_BASE_WAIT_MINUTES,
            max_wait_minutes=TRADING_212_MAX_WAIT_SECONDS / 60,
        )
    return CircuitBreaker(service_name=service.value, max_attempts=max_attempts)


_PROCESS_BREAKERS = {service: build_breaker(service) for service in ExternalService}


def process_breaker(service: ExternalService) -> CircuitBreaker:
    """Return the process-wide breaker of ``service``.

    Args:
        service: The source whose breaker to return.

    Returns:
        The same breaker on every call for the life of the process.
    """
    return _PROCESS_BREAKERS[service]


@contextmanager
def timed_request(service: ExternalService) -> Iterator[None]:
    """Record how long the enclosed request to ``service`` took.

    The duration goes to ``external_request_seconds`` with outcome ``ok`` when
    the block completes and ``error`` when it raises. The exception propagates
    unchanged.

    Args:
        service: The source the enclosed request goes to.

    Yields:
        Nothing; the block runs the request.
    """
    started = time.perf_counter()
    outcome = _ERROR
    try:
        yield
        outcome = _OK
    finally:
        metrics.external_request_seconds.labels(
            service=service.value, outcome=outcome
        ).observe(time.perf_counter() - started)
