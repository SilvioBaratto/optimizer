"""Prometheus metric definitions for background jobs, the database breaker and
the external HTTP sources.

All metric objects are module-level singletons. Python's import cache ensures
prometheus_client never sees duplicate registrations within the same process.

The worker exposes these over ``prometheus_client.start_http_server`` on
``settings.metrics_port``; that endpoint doubles as the container healthcheck.
"""

from __future__ import annotations

from typing import Protocol

from portopt_db.resilience import BreakerState, CircuitBreaker
from prometheus_client import Counter, Gauge, Histogram

jobs_started_total = Counter(
    "jobs_started_total",
    "Total background jobs started",
    ["domain"],
)

jobs_completed_total = Counter(
    "jobs_completed_total",
    "Total background jobs completed successfully",
    ["domain"],
)

jobs_failed_total = Counter(
    "jobs_failed_total",
    "Total background jobs that failed",
    ["domain"],
)

job_duration_seconds = Histogram(
    "job_duration_seconds",
    "Background job wall-clock duration in seconds (running to terminal)",
    ["domain"],
    buckets=[10, 30, 60, 120, 300, 600, 1800, 3600],
)

jobs_in_progress = Gauge(
    "jobs_in_progress",
    "Number of background jobs currently in the running state",
    ["domain"],
)

jobs_db_unavailable_total = Counter(
    "jobs_db_unavailable_total",
    "Background job steps stopped because the database circuit breaker was open",
    ["domain"],
)

jobs_failed_status_refused_total = Counter(
    "jobs_failed_status_refused_total",
    "Failed-status writes the open database breaker refused; once a row's lease has "
    "expired, an orphan reap the breaker lets reach the database fails it",
    ["domain"],
)

db_breaker_state = Gauge(
    "portopt_db_breaker_state",
    "Database circuit breaker state: 0 closed, 1 open, 2 half-open",
)

_BREAKER_STATE_VALUES = {
    BreakerState.CLOSED: 0,
    BreakerState.OPEN: 1,
    BreakerState.HALF_OPEN: 2,
}

_tracked_breaker: CircuitBreaker | None = None


def _tracked_breaker_state() -> float:
    breaker = _tracked_breaker
    if breaker is None:
        return _BREAKER_STATE_VALUES[BreakerState.CLOSED]
    return _BREAKER_STATE_VALUES[breaker.state]


db_breaker_state.set_function(_tracked_breaker_state)


def track_db_breaker(breaker: CircuitBreaker | None) -> CircuitBreaker | None:
    """Report ``breaker``'s state on ``portopt_db_breaker_state`` at every scrape.

    The value is read when Prometheus scrapes, so it never lags a transition and
    needs no hook in the breaker. With no breaker tracked the gauge reads 0.

    Args:
        breaker: The breaker of the manager the daemon's sessions go through, or
            ``None`` to stop tracking.

    Returns:
        The breaker tracked until now, or ``None``, so a caller can restore it.
    """
    global _tracked_breaker
    previous, _tracked_breaker = _tracked_breaker, breaker
    return previous


external_breaker_open = Gauge(
    "external_breaker_open",
    "1 while the circuit breaker of an external HTTP source refuses calls, else 0",
    ["service"],
)

external_request_seconds = Histogram(
    "external_request_seconds",
    "Duration of one HTTP request to an external source; outcome is ok or error",
    ["service", "outcome"],
    buckets=[0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60],
)


class _OpenState(Protocol):
    @property
    def is_open(self) -> bool: ...


def track_external_breaker(service: str, breaker: _OpenState) -> None:
    """Report ``breaker.is_open`` on ``external_breaker_open{service}``.

    The value is read when Prometheus scrapes, like ``portopt_db_breaker_state``.
    Tracking another breaker under the same ``service`` replaces the first.

    Args:
        service: Label value naming the source. The worker passes the values of
            ``ExternalService``, which keeps the label set fixed.
        breaker: The breaker to report.
    """
    external_breaker_open.labels(service=service).set_function(
        lambda: float(breaker.is_open)
    )


def register_domain(domain: str) -> None:
    """Create every labelled series for ``domain`` at its zero value.

    prometheus_client exposes no sample for a labelled metric until ``.labels()``
    is called, so without this the first job of a domain shows up as a series born
    at 1 and ``increase()`` / ``rate()`` miss that first step.

    Args:
        domain: Job type used as the ``domain`` label value.
    """
    jobs_started_total.labels(domain=domain)
    jobs_completed_total.labels(domain=domain)
    jobs_failed_total.labels(domain=domain)
    job_duration_seconds.labels(domain=domain)
    jobs_in_progress.labels(domain=domain)
    jobs_db_unavailable_total.labels(domain=domain)
    jobs_failed_status_refused_total.labels(domain=domain)
