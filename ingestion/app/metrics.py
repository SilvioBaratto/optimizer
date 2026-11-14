"""Prometheus metric definitions for background jobs and the database breaker.

All metric objects are module-level singletons. Python's import cache ensures
prometheus_client never sees duplicate registrations within the same process.

The worker exposes these over ``prometheus_client.start_http_server`` on
``settings.metrics_port``; that endpoint doubles as the container healthcheck.
"""

from __future__ import annotations

from portopt_db.resilience import BreakerState, DbCircuitBreaker
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

db_breaker_state = Gauge(
    "portopt_db_breaker_state",
    "Database circuit breaker state: 0 closed, 1 open, 2 half-open",
)

_BREAKER_STATE_VALUES = {
    BreakerState.CLOSED: 0,
    BreakerState.OPEN: 1,
    BreakerState.HALF_OPEN: 2,
}


def track_db_breaker(breaker: DbCircuitBreaker) -> None:
    """Report ``breaker``'s state on ``portopt_db_breaker_state`` at every scrape.

    The value is read when Prometheus scrapes, so it never lags a transition and
    needs no hook in the breaker. Binding again replaces the previous breaker.

    Args:
        breaker: The breaker of the manager the daemon's sessions go through.
    """
    db_breaker_state.set_function(lambda: _BREAKER_STATE_VALUES[breaker.state])


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
