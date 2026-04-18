"""Prometheus metric definitions for background job and request instrumentation.

All metric objects are module-level singletons.  Python's import cache ensures
prometheus_client never sees duplicate registrations within the same process.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Final

from fastapi import HTTPException
from prometheus_client import Counter, Gauge, Histogram

# ---------------------------------------------------------------------------
# Background-job metrics (existing — DO NOT remove or rename)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Generic HTTP-request histogram (existing)
# ---------------------------------------------------------------------------

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request wall-clock duration in seconds",
    ["method", "path_template", "status_code"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30],
)

# ---------------------------------------------------------------------------
# Per-endpoint request-handler histograms for the four execution endpoints
# (issue #441).  These measure REQUEST-HANDLER latency only, not background
# job duration — that is already covered by ``job_duration_seconds{domain}``.
#
# ``status`` semantics:
#   - ``accepted``  — handler returned 2xx (sync result OR 202 enqueue)
#   - ``rejected``  — handler raised an HTTPException with a 4xx status
#                     (e.g. 409 JobAlreadyRunningError, 422 invalid input)
#   - ``error``     — handler raised an HTTPException with a 5xx status
#                     OR an unhandled non-HTTPException
# ---------------------------------------------------------------------------

_ENDPOINT_BUCKETS: Final[tuple[float, ...]] = (
    0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30,
)


def _build_endpoint_histogram(endpoint: str) -> Histogram:
    """Construct a request-handler latency histogram for one endpoint."""
    return Histogram(
        f"{endpoint}_request_duration_seconds",
        f"Request-handler wall-clock duration in seconds for /{endpoint}",
        ["status"],
        buckets=_ENDPOINT_BUCKETS,
    )


_ENDPOINT_HISTOGRAMS: Final[dict[str, Histogram]] = {
    name: _build_endpoint_histogram(name)
    for name in ("optimize", "backtest", "validate", "tune")
}

optimize_request_duration_seconds = _ENDPOINT_HISTOGRAMS["optimize"]
backtest_request_duration_seconds = _ENDPOINT_HISTOGRAMS["backtest"]
validate_request_duration_seconds = _ENDPOINT_HISTOGRAMS["validate"]
tune_request_duration_seconds = _ENDPOINT_HISTOGRAMS["tune"]


def observe_endpoint_latency(endpoint: str, status: str, seconds: float) -> None:
    """Record a single request-handler latency observation.

    Raises ``KeyError`` for unknown endpoint names — the four execution
    endpoints are a closed set; mistyping is a programmer bug, not a
    runtime condition to swallow.
    """
    _ENDPOINT_HISTOGRAMS[endpoint].labels(status=status).observe(seconds)


def _classify_http_exception(exc: HTTPException) -> str:
    """Map an HTTPException's status code to its endpoint-histogram label."""
    return "rejected" if 400 <= exc.status_code < 500 else "error"


@contextmanager
def time_endpoint(endpoint: str) -> Iterator[None]:
    """Time a request-handler body and record the observation on exit.

    The status label is derived from how the wrapped block exits:
      - normal return        → ``accepted``
      - HTTPException 4xx    → ``rejected``
      - HTTPException 5xx    → ``error``
      - any other exception  → ``error``

    Exceptions are always re-raised; this context manager observes and
    leaves error handling to the caller.
    """
    start = time.monotonic()
    status = "accepted"
    try:
        yield
    except HTTPException as exc:
        status = _classify_http_exception(exc)
        raise
    except Exception:
        status = "error"
        raise
    finally:
        observe_endpoint_latency(endpoint, status, time.monotonic() - start)
