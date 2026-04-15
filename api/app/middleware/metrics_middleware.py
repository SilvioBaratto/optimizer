"""HTTP request latency middleware for Prometheus instrumentation.

Single Responsibility: measure wall-clock duration per request and record it
into a Histogram.  No logging, no authentication, no other concerns.

Open/Closed: the excluded-path set is injected at construction time so new
exclusions can be added without modifying dispatch logic.

Dependency Inversion: the Histogram is injected (or defaults to the
application-wide singleton from app.metrics) rather than created here.
"""

import time
from typing import Final

from prometheus_client import CollectorRegistry, Histogram
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.metrics import http_request_duration_seconds as _default_histogram

# Paths that must never be measured — they are infrastructure endpoints whose
# latency is irrelevant (or would create feedback loops with /metrics itself).
_DEFAULT_EXCLUDED_PATHS: Final[frozenset[str]] = frozenset(
    {"/metrics", "/health", "/docs", "/redoc", "/openapi.json", "/"}
)

_UNMATCHED_LABEL: Final[str] = "<unmatched>"


class MetricsMiddleware:
    """Starlette-compatible ASGI middleware that records HTTP request durations.

    Parameters
    ----------
    app:
        The downstream ASGI application.
    excluded_paths:
        Paths excluded from measurement.  Defaults to the standard set of
        infrastructure paths.  Provide a custom frozenset to extend or
        replace the defaults (OCP).
    registry:
        Prometheus ``CollectorRegistry`` to use.  When supplied, a fresh
        Histogram is registered against it — useful for test isolation.
        When omitted, the application-wide singleton is used.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        excluded_paths: frozenset[str] = _DEFAULT_EXCLUDED_PATHS,
        registry: CollectorRegistry | None = None,
    ) -> None:
        self._app = app
        self._excluded_paths = excluded_paths
        self._histogram = _build_histogram(registry)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        request = Request(scope)

        if request.url.path in self._excluded_paths:
            await self._app(scope, receive, send)
            return

        start = time.perf_counter()
        status_code = 500  # default; overwritten by _capture_status

        async def _capture_status(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        await self._app(scope, receive, _capture_status)

        duration = time.perf_counter() - start
        path_template = _resolve_path_template(scope)
        self._histogram.labels(
            method=request.method,
            path_template=path_template,
            status_code=str(status_code),
        ).observe(duration)


# ---------------------------------------------------------------------------
# Private helpers (single-purpose functions < 10 lines each)
# ---------------------------------------------------------------------------


def _resolve_path_template(scope: Scope) -> str:
    """Return the FastAPI route template or ``<unmatched>`` for 404s.

    Uses ``scope["route"].path`` to obtain the parametrised template
    (e.g. ``/api/v1/portfolio/{name}/risk``) instead of applying regex
    substitution on the raw URL, which would explode label cardinality.
    """
    route = scope.get("route")
    if route is None:
        return _UNMATCHED_LABEL
    path: str = getattr(route, "path", _UNMATCHED_LABEL)
    return path or _UNMATCHED_LABEL


def _build_histogram(registry: CollectorRegistry | None) -> Histogram:
    """Return the histogram to use, creating a registry-scoped one for tests."""
    if registry is None:
        return _default_histogram
    return Histogram(
        "http_request_duration_seconds",
        "HTTP request wall-clock duration in seconds",
        ["method", "path_template", "status_code"],
        buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30],
        registry=registry,
    )
