"""Circuit breakers and request timing for the daemon's external HTTP sources.

At import this module builds one breaker per source in :class:`ExternalService`,
sized from ``app.config.settings``; the clients that use it share its state, and
the worker reports it on ``external_breaker_open`` when metrics are enabled.
:func:`timed_request` records one request on ``external_request_seconds``.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum

from app import metrics
from app.config import Settings, settings
from app.services.infrastructure.circuit_breaker import CircuitBreaker

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


def build_breaker(
    service: ExternalService, config: Settings | None = None
) -> CircuitBreaker:
    """Build a closed circuit breaker for ``service``.

    Args:
        service: The source the breaker guards.
        config: Settings that hold the source's ``breaker_*_max_attempts``;
            ``app.config.settings`` when omitted.

    Returns:
        A breaker named after ``service`` with the ``CircuitBreaker`` default
        base and cap.
    """
    config = config or settings
    max_attempts = {
        ExternalService.FRED: config.breaker_fred_max_attempts,
        ExternalService.TRADING_ECONOMICS: config.breaker_tradingeconomics_max_attempts,
        ExternalService.IL_SOLE: config.breaker_ilsole_max_attempts,
    }[service]
    return CircuitBreaker(service_name=service.value, max_attempts=max_attempts)


_PROCESS_BREAKERS = {service: build_breaker(service) for service in ExternalService}


def process_breaker(service: ExternalService) -> CircuitBreaker:
    """Return the process-wide breaker of ``service``.

    Args:
        service: The source whose breaker to return.

    Returns:
        The breaker built for ``service`` when this module was imported.
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
