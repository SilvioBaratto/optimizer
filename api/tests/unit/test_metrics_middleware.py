"""Tests for MetricsMiddleware — TDD Red phase written before implementation.

Acceptance criteria verified:
- Successful requests record a histogram observation with correct labels.
- Unmatched routes (404) use the fixed path label ``<unmatched>``.
- Excluded paths produce zero histogram observations.
- Measured duration is always a positive float.
- Histogram labels are method, path_template, status_code.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

# ---------------------------------------------------------------------------
# Fixture: isolated FastAPI app with MetricsMiddleware
# ---------------------------------------------------------------------------

HISTOGRAM_NAME = "http_request_duration_seconds"


@pytest.fixture()
def metrics_app():
    """Build a minimal FastAPI app with MetricsMiddleware wired in.

    Each fixture invocation creates a fresh CollectorRegistry so that
    prometheus_client counters do not bleed between tests.
    """
    from app.middleware.metrics_middleware import MetricsMiddleware

    registry = CollectorRegistry()
    test_app = FastAPI()

    @test_app.get("/api/v1/health-check")
    def _api_health() -> dict[str, str]:
        return {"status": "ok"}

    @test_app.get("/api/v1/portfolio/{name}/risk")
    def _portfolio_risk(name: str) -> dict[str, str]:
        return {"portfolio": name}

    test_app.add_middleware(MetricsMiddleware, registry=registry)

    client = TestClient(test_app, raise_server_exceptions=False)
    return client, registry


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMetricsMiddlewareRecordsObservations:
    def test_when_successful_request_records_histogram_observation(self, metrics_app):
        """A 200 response must produce exactly one histogram observation."""
        client, registry = metrics_app

        client.get("/api/v1/health-check")

        count = self._count(
            registry,
            HISTOGRAM_NAME,
            {
                "method": "GET",
                "path_template": "/api/v1/health-check",
                "status_code": "200",
            },
        )
        assert count == 1

    def test_when_404_request_uses_unmatched_label(self, metrics_app):
        """Requests that match no route must use path_template='<unmatched>'."""
        client, registry = metrics_app

        client.get("/api/v1/does-not-exist")

        count = self._count(
            registry,
            HISTOGRAM_NAME,
            {
                "method": "GET",
                "path_template": "<unmatched>",
                "status_code": "404",
            },
        )
        assert count == 1

    def test_when_excluded_path_no_histogram_observation(self, metrics_app):
        """Excluded paths (/metrics, /health, /docs, /redoc, /openapi.json, /)
        must not produce any histogram observations."""
        client, registry = metrics_app
        excluded = ["/metrics", "/health", "/docs", "/redoc", "/openapi.json", "/"]

        before = self._total(registry, HISTOGRAM_NAME)
        for path in excluded:
            client.get(path)
        after = self._total(registry, HISTOGRAM_NAME)

        assert after == before, (
            f"Excluded paths generated {after - before} unexpected observations"
        )

    def test_histogram_duration_is_positive(self, metrics_app):
        """The recorded wall-clock duration must be strictly positive."""
        client, registry = metrics_app

        client.get("/api/v1/health-check")

        duration_sum = self._sum(
            registry,
            HISTOGRAM_NAME,
            {
                "method": "GET",
                "path_template": "/api/v1/health-check",
                "status_code": "200",
            },
        )
        assert duration_sum > 0.0, "Duration should be > 0"

    def test_histogram_labels_include_method_path_status(self, metrics_app):
        """Histogram sample labels must contain method, path_template, status_code."""
        client, registry = metrics_app

        client.get("/api/v1/health-check")

        for metric in registry.collect():
            if metric.name != HISTOGRAM_NAME:
                continue
            for sample in metric.samples:
                if sample.name.endswith("_count") and sample.value > 0:
                    assert "method" in sample.labels
                    assert "path_template" in sample.labels
                    assert "status_code" in sample.labels
                    return
        pytest.fail("No histogram sample with value > 0 found")

    def test_path_template_uses_route_pattern_not_raw_url(self, metrics_app):
        """Path template must be the FastAPI route pattern, not the concrete URL."""
        client, registry = metrics_app

        client.get("/api/v1/portfolio/my-portfolio/risk")

        count = self._count(
            registry,
            HISTOGRAM_NAME,
            {
                "method": "GET",
                "path_template": "/api/v1/portfolio/{name}/risk",
                "status_code": "200",
            },
        )
        assert count == 1

    def test_multiple_requests_accumulate_counts(self, metrics_app):
        """Multiple calls to the same endpoint must accumulate in the histogram."""
        client, registry = metrics_app

        for _ in range(3):
            client.get("/api/v1/health-check")

        count = self._count(
            registry,
            HISTOGRAM_NAME,
            {
                "method": "GET",
                "path_template": "/api/v1/health-check",
                "status_code": "200",
            },
        )
        assert count == 3

    # ------------------------------------------------------------------
    # Private helpers scoped to the isolated registry
    # ------------------------------------------------------------------

    @staticmethod
    def _count(registry: CollectorRegistry, name: str, labels: dict) -> int:
        for metric in registry.collect():
            if metric.name != name:
                continue
            for sample in metric.samples:
                if not sample.name.endswith("_count"):
                    continue
                if all(sample.labels.get(k) == v for k, v in labels.items()):
                    return int(sample.value)
        return 0

    @staticmethod
    def _sum(registry: CollectorRegistry, name: str, labels: dict) -> float:
        for metric in registry.collect():
            if metric.name != name:
                continue
            for sample in metric.samples:
                if not sample.name.endswith("_sum"):
                    continue
                if all(sample.labels.get(k) == v for k, v in labels.items()):
                    return sample.value
        return 0.0

    @staticmethod
    def _total(registry: CollectorRegistry, name: str) -> int:
        total = 0
        for metric in registry.collect():
            if metric.name != name:
                continue
            for sample in metric.samples:
                if sample.name.endswith("_count"):
                    total += int(sample.value)
        return total
