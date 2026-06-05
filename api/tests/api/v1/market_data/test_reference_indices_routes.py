"""Route tests for the reference-indices endpoints (issue #816).

Covers the 4 endpoints in ``app.api.v1.market_data.reference_indices``:
``POST /seed`` + poll ``GET /seed/{job_id}`` (job-backed) and the two read
endpoints ``GET /status`` / ``GET /configured`` (repository-backed).  The
module-level ``_job_service`` is mocked via ``mock_job_service``;
``DashboardRepository`` / ``PortfolioRepository`` are patched at their import
site; ``settings`` attributes are patched on the live singleton so the
healthy/missing/stale derivation is deterministic.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import app.api.v1.market_data.reference_indices as ri_module
from app.api.v1.market_data.reference_indices import _get_yf_client
from app.main import app
from app.schemas.market_data.reference_index import (
    ReferenceIndexConfiguredResponse,
    ReferenceIndexSeedJobResponse,
    ReferenceIndexSeedProgress,
    ReferenceIndexStatusResponse,
)
from tests._fixtures.assertions import assert_json_safe, assert_response_matches_schema
from tests._fixtures.job_service_mock import mock_job_service

BASE_URL = "/api/v1/reference-indices"
_JOB = "app.api.v1.market_data.reference_indices._job_service"
_DASH = "app.api.v1.market_data.reference_indices.DashboardRepository"
_PORT = "app.api.v1.market_data.reference_indices.PortfolioRepository"


@pytest.fixture
def override_yf_client() -> Iterator[None]:
    app.dependency_overrides[_get_yf_client] = lambda: MagicMock()
    try:
        yield
    finally:
        app.dependency_overrides.pop(_get_yf_client, None)


# ---------------------------------------------------------------------------
# POST /seed + GET /seed/{job_id}
# ---------------------------------------------------------------------------


class TestSeed:
    def test_when_started_then_202_with_job_id(
        self, client: TestClient, override_yf_client: None
    ) -> None:
        with mock_job_service(_JOB, job_id="ref-1") as job:
            resp = client.post(f"{BASE_URL}/seed", json={"tickers": ["SPY"]})

        assert resp.status_code == 202
        body = resp.json()
        assert_response_matches_schema(body, ReferenceIndexSeedJobResponse)
        assert body["job_id"] == "ref-1"
        assert body["status"] == "pending"
        job.start_background.assert_called_once()

    def test_when_already_running_then_409(
        self, client: TestClient, override_yf_client: None
    ) -> None:
        with mock_job_service(_JOB, raise_already_running=True):
            resp = client.post(f"{BASE_URL}/seed", json={"tickers": ["SPY"]})

        assert resp.status_code == 409

    def test_when_job_pending_then_200_progress(self, client: TestClient) -> None:
        with mock_job_service(_JOB, job_id="ref-2", initial_status="pending"):
            resp = client.get(f"{BASE_URL}/seed/ref-2")

        assert resp.status_code == 200
        body = resp.json()
        assert_response_matches_schema(body, ReferenceIndexSeedProgress)
        assert_json_safe(body)

    def test_when_job_unknown_then_404(self, client: TestClient) -> None:
        with patch(f"{_JOB}.get_job", return_value=None):
            resp = client.get(f"{BASE_URL}/seed/ghost")

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /status
# ---------------------------------------------------------------------------


class TestStatus:
    def test_when_coverage_mixed_then_counts_classified(
        self, client: TestClient
    ) -> None:
        today = date.today()
        coverage = {
            "AAA": (10, today),  # rows + fresh -> healthy
            "BBB": (0, None),  # no rows -> missing
            "CCC": (5, today - timedelta(days=100)),  # rows + old -> stale
        }
        with (
            patch(_PORT) as MockPort,
            patch(_DASH) as MockDash,
            patch.object(ri_module.settings, "benchmark_tickers",
                         ["AAA", "BBB", "CCC"]),
            patch.object(ri_module.settings, "scheduler_benchmark_stale_days", 7),
        ):
            MockPort.return_value.get_distinct_benchmark_tickers.return_value = []
            MockDash.return_value.get_benchmark_coverage.return_value = coverage
            resp = client.get(f"{BASE_URL}/status")

        assert resp.status_code == 200
        body = resp.json()
        assert_response_matches_schema(body, ReferenceIndexStatusResponse)
        assert body["total"] == 3
        assert body["healthyCount"] == 1
        assert body["missingCount"] == 1
        assert body["staleCount"] == 1
        assert body["staleDays"] == 7
        assert_json_safe(body)


# ---------------------------------------------------------------------------
# GET /configured
# ---------------------------------------------------------------------------


class TestConfigured:
    def test_when_called_then_union_sorted(self, client: TestClient) -> None:
        with (
            patch(_PORT) as MockPort,
            patch.object(ri_module.settings, "benchmark_tickers", ["BBB", "AAA"]),
        ):
            MockPort.return_value.get_distinct_benchmark_tickers.return_value = [
                "ZZZ",
                "AAA",
            ]
            resp = client.get(f"{BASE_URL}/configured")

        assert resp.status_code == 200
        body = resp.json()
        assert_response_matches_schema(body, ReferenceIndexConfiguredResponse)
        assert body["tickers"] == ["AAA", "BBB", "ZZZ"]
        assert body["settingsTickers"] == ["AAA", "BBB"]
        assert body["portfolioTickers"] == ["ZZZ", "AAA"]
