"""Prometheus metric definitions for background-job instrumentation.

All metric objects are module-level singletons. Python's import cache ensures
prometheus_client never sees duplicate registrations within the same process.

The worker exposes these over ``prometheus_client.start_http_server`` on
``settings.metrics_port``; that endpoint doubles as the container healthcheck.
"""

from __future__ import annotations

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
