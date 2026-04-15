"""Prometheus metric definitions for background job instrumentation.

All metric objects are module-level singletons.  Import this module once;
Python's import cache ensures prometheus_client never sees duplicate
registrations within the same process.
"""

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

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request wall-clock duration in seconds",
    ["method", "path_template", "status_code"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30],
)
