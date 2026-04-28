"""Tests for per-endpoint Prometheus latency histograms (Red phase first).

Acceptance criteria (issue #441):
- Four histograms registered in ``app.metrics`` with names
  ``optimize_request_duration_seconds``, ``backtest_request_duration_seconds``,
  ``validate_request_duration_seconds``, ``tune_request_duration_seconds``.
- Each labeled by ``status`` (accepted | rejected | error).
- Buckets reuse the same bounds as ``http_request_duration_seconds``.
- Helper ``observe_endpoint_latency(endpoint, status, seconds)`` records into
  the matching histogram.
- Context manager ``time_endpoint(endpoint)`` classifies HTTPException 4xx as
  ``rejected``, 5xx and plain Exception as ``error``, and successful exit as
  ``accepted``.
- ``/metrics`` exposes all four families with ``_count``/``_sum``/bucket lines
  after at least one observation.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest
from fastapi import HTTPException
from prometheus_client import REGISTRY

from app import metrics

ENDPOINTS = ("optimize", "backtest", "validate", "tune")
HISTOGRAM_NAMES = tuple(f"{ep}_request_duration_seconds" for ep in ENDPOINTS)
EXPECTED_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bucket_bounds(metric_name: str) -> tuple[float, ...]:
    """Return the upper-bound buckets registered for ``metric_name``."""
    bounds: list[float] = []
    for metric in REGISTRY.collect():
        if metric.name != metric_name:
            continue
        for sample in metric.samples:
            if not sample.name.endswith("_bucket"):
                continue
            le = sample.labels.get("le")
            if le == "+Inf" or le is None:
                continue
            bounds.append(float(le))
        break
    return tuple(sorted(set(bounds)))


def _count(metric_name: str, labels: dict[str, str]) -> float:
    for metric in REGISTRY.collect():
        if metric.name != metric_name:
            continue
        for sample in metric.samples:
            if not sample.name.endswith("_count"):
                continue
            if all(sample.labels.get(k) == v for k, v in labels.items()):
                return sample.value
    return 0.0


def _sum(metric_name: str, labels: dict[str, str]) -> float:
    for metric in REGISTRY.collect():
        if metric.name != metric_name:
            continue
        for sample in metric.samples:
            if not sample.name.endswith("_sum"):
                continue
            if all(sample.labels.get(k) == v for k, v in labels.items()):
                return sample.value
    return 0.0


def _label_set(metric_name: str) -> set[str]:
    for metric in REGISTRY.collect():
        if metric.name != metric_name:
            continue
        for sample in metric.samples:
            if sample.labels:
                return set(sample.labels.keys()) - {"le"}
    return set()


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


class TestHistogramRegistration:
    @pytest.mark.parametrize("name", HISTOGRAM_NAMES)
    def test_histogram_is_registered(self, name: str) -> None:
        registered = {m.name for m in REGISTRY.collect()}
        assert name in registered, f"Histogram {name!r} is not registered"

    @pytest.mark.parametrize("name", HISTOGRAM_NAMES)
    def test_histogram_uses_status_label(self, name: str) -> None:
        # Force at least one sample so labels become observable.
        endpoint = name.removesuffix("_request_duration_seconds")
        metrics.observe_endpoint_latency(endpoint, "accepted", 0.0)
        labels = _label_set(name)
        assert labels == {"status"}, (
            f"{name} should expose only the 'status' label, got {labels!r}"
        )

    @pytest.mark.parametrize("name", HISTOGRAM_NAMES)
    def test_histogram_buckets_match_http_request_duration(self, name: str) -> None:
        endpoint = name.removesuffix("_request_duration_seconds")
        metrics.observe_endpoint_latency(endpoint, "accepted", 0.0)
        assert _bucket_bounds(name) == EXPECTED_BUCKETS


# ---------------------------------------------------------------------------
# observe_endpoint_latency
# ---------------------------------------------------------------------------


class TestObserveEndpointLatency:
    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    def test_observation_increments_count(self, endpoint: str) -> None:
        name = f"{endpoint}_request_duration_seconds"
        labels = {"status": "accepted"}
        before = _count(name, labels)

        metrics.observe_endpoint_latency(endpoint, "accepted", 0.123)

        assert _count(name, labels) == before + 1

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    def test_observation_increments_sum(self, endpoint: str) -> None:
        name = f"{endpoint}_request_duration_seconds"
        labels = {"status": "accepted"}
        before = _sum(name, labels)

        metrics.observe_endpoint_latency(endpoint, "accepted", 0.5)

        assert _sum(name, labels) == pytest.approx(before + 0.5)

    def test_unknown_endpoint_raises(self) -> None:
        with pytest.raises(KeyError):
            metrics.observe_endpoint_latency("nope", "accepted", 0.1)


# ---------------------------------------------------------------------------
# time_endpoint context manager
# ---------------------------------------------------------------------------


class TestTimeEndpointContextManager:
    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    def test_successful_exit_records_accepted(self, endpoint: str) -> None:
        name = f"{endpoint}_request_duration_seconds"
        before = _count(name, {"status": "accepted"})

        with metrics.time_endpoint(endpoint):
            time.sleep(0)

        assert _count(name, {"status": "accepted"}) == before + 1

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    def test_4xx_http_exception_records_rejected(self, endpoint: str) -> None:
        name = f"{endpoint}_request_duration_seconds"
        before = _count(name, {"status": "rejected"})

        with pytest.raises(HTTPException), metrics.time_endpoint(endpoint):
            raise HTTPException(status_code=409, detail="conflict")

        assert _count(name, {"status": "rejected"}) == before + 1

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    def test_5xx_http_exception_records_error(self, endpoint: str) -> None:
        name = f"{endpoint}_request_duration_seconds"
        before = _count(name, {"status": "error"})

        with pytest.raises(HTTPException), metrics.time_endpoint(endpoint):
            raise HTTPException(status_code=500, detail="boom")

        assert _count(name, {"status": "error"}) == before + 1

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    def test_unhandled_exception_records_error(self, endpoint: str) -> None:
        name = f"{endpoint}_request_duration_seconds"
        before = _count(name, {"status": "error"})

        with pytest.raises(RuntimeError), metrics.time_endpoint(endpoint):
            raise RuntimeError("oops")

        assert _count(name, {"status": "error"}) == before + 1

    @pytest.mark.parametrize("endpoint", ENDPOINTS)
    def test_recorded_duration_is_positive(self, endpoint: str) -> None:
        name = f"{endpoint}_request_duration_seconds"
        before = _sum(name, {"status": "accepted"})

        with metrics.time_endpoint(endpoint):
            time.sleep(0.001)

        assert _sum(name, {"status": "accepted"}) > before


# ---------------------------------------------------------------------------
# /metrics scrape exposes all four families
# ---------------------------------------------------------------------------


class TestMetricsEndpointExposesHistograms:
    """A live /metrics scrape must surface all four histograms."""

    def _scrape_text(self) -> str:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from app.api.v1.metrics import router as metrics_router

        app = FastAPI()
        app.include_router(metrics_router)
        client = TestClient(app)

        # Force one observation per endpoint so the family appears.
        for endpoint in ENDPOINTS:
            metrics.observe_endpoint_latency(endpoint, "accepted", 0.05)

        response = client.get("/metrics")
        assert response.status_code == 200
        return response.text

    def test_metrics_text_contains_all_four_count_lines(self) -> None:
        body = self._scrape_text()
        for endpoint in ENDPOINTS:
            assert f"{endpoint}_request_duration_seconds_count" in body

    def test_metrics_text_contains_all_four_sum_lines(self) -> None:
        body = self._scrape_text()
        for endpoint in ENDPOINTS:
            assert f"{endpoint}_request_duration_seconds_sum" in body

    def test_metrics_text_contains_buckets(self) -> None:
        body = self._scrape_text()
        for endpoint in ENDPOINTS:
            assert f"{endpoint}_request_duration_seconds_bucket" in body

    def test_metrics_text_does_not_lose_http_request_duration_seconds(self) -> None:
        """Regression guard: the original generic histogram must still register."""
        body = self._scrape_text()
        assert "http_request_duration_seconds" in body


# ---------------------------------------------------------------------------
# Route handlers actually emit observations
# ---------------------------------------------------------------------------


class TestRouteHandlersEmitObservations:
    """Each execution route must record a sample on its dedicated histogram.

    The optimize test uses a mocked service error to trigger the ``error``
    status path inside ``time_endpoint``.  The tune test sends an invalid
    optimizer type that passes Pydantic (``TuneRequest.optimizer_type`` is
    ``str``) but is rejected in-handler, exercising the ``rejected`` path.
    """

    def test_post_optimize_records_into_optimize_histogram(self, client) -> None:
        from fastapi import HTTPException as _HTTPException

        before = _count(
            "optimize_request_duration_seconds", {"status": "error"}
        )
        with patch(
            "app.api.v1.optimize.build_pipeline",
            side_effect=_HTTPException(status_code=500, detail="forced error"),
        ):
            response = client.post(
                "/api/v1/optimize",
                json={
                    "optimizer_type": "mean_risk",
                    "tickers": ["AAPL"],
                    "start_date": "2024-01-01",
                    "end_date": "2024-02-01",
                    "config": {},
                },
            )
        assert response.status_code == 500
        after = _count(
            "optimize_request_duration_seconds", {"status": "error"}
        )
        assert after == before + 1

    def test_post_tune_records_into_tune_histogram(self, client) -> None:
        before = _count("tune_request_duration_seconds", {"status": "rejected"})
        response = client.post(
            "/api/v1/tune",
            json={
                "optimizer_type": "definitely-not-a-real-optimizer",
                "tickers": ["AAPL"],
                "start_date": "2024-01-01",
                "end_date": "2024-02-01",
                "optimizer_config": {},
                "param_grid": {"l2_coef": [0.1, 0.2]},
            },
        )
        assert response.status_code == 422
        after = _count("tune_request_duration_seconds", {"status": "rejected"})
        assert after == before + 1
