"""Unit tests for POST /api/v1/factors/compute and GET /api/v1/factors/compute/{job_id}.

Covers:
  - POST: valid request returns 202 + job_id
  - POST: empty tickers returns 422
  - POST: missing required fields returns 422
  - POST: JobAlreadyRunningError returns 409
  - POST: background worker is started
  - GET: existing job returns 200 with progress
  - GET: job_id returns correct data
  - GET: missing job returns 404
  - GET: running job returns status

BackgroundJobService is mocked at the module-level service instance (following
the backtest_routes and macro_regime test patterns).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

BASE_URL = "/api/v1/factors/compute"

_JOB_SVC_CREATE = "app.api.v1.factors.factors._factor_job_service.create_job"
_JOB_SVC_START = "app.api.v1.factors.factors._factor_job_service.start_background"
_JOB_SVC_GET = "app.api.v1.factors.factors._factor_job_service.get_job"

_MOCK_JOB_ID = "00000000-0000-0000-0000-000000000099"

_VALID_REQUEST: dict[str, Any] = {
    "tickers": ["AAPL", "MSFT"],
    "start_date": "2020-01-01",
    "end_date": "2024-01-01",
}


# ===========================================================================
# POST /api/v1/factors/compute
# ===========================================================================


class TestPostFactorCompute:
    """POST /api/v1/factors/compute launches a background job, returns 202."""

    def test_valid_request_returns_202(self, client: TestClient) -> None:
        with (
            patch(_JOB_SVC_CREATE, return_value=_MOCK_JOB_ID),
            patch(_JOB_SVC_START),
        ):
            resp = client.post(BASE_URL, json=_VALID_REQUEST)

        assert resp.status_code == 202

    def test_valid_request_body_contains_job_id(self, client: TestClient) -> None:
        with (
            patch(_JOB_SVC_CREATE, return_value=_MOCK_JOB_ID),
            patch(_JOB_SVC_START),
        ):
            resp = client.post(BASE_URL, json=_VALID_REQUEST)

        body = resp.json()
        assert body["job_id"] == _MOCK_JOB_ID

    def test_valid_request_body_contains_status_pending(
        self, client: TestClient
    ) -> None:
        with (
            patch(_JOB_SVC_CREATE, return_value=_MOCK_JOB_ID),
            patch(_JOB_SVC_START),
        ):
            resp = client.post(BASE_URL, json=_VALID_REQUEST)

        assert resp.json()["status"] == "pending"

    def test_valid_request_body_contains_message(self, client: TestClient) -> None:
        with (
            patch(_JOB_SVC_CREATE, return_value=_MOCK_JOB_ID),
            patch(_JOB_SVC_START),
        ):
            resp = client.post(BASE_URL, json=_VALID_REQUEST)

        assert "message" in resp.json()

    def test_empty_tickers_returns_422(self, client: TestClient) -> None:
        resp = client.post(BASE_URL, json={**_VALID_REQUEST, "tickers": []})
        assert resp.status_code == 422

    def test_missing_tickers_field_returns_422(self, client: TestClient) -> None:
        payload = {k: v for k, v in _VALID_REQUEST.items() if k != "tickers"}
        resp = client.post(BASE_URL, json=payload)
        assert resp.status_code == 422

    def test_missing_start_date_returns_422(self, client: TestClient) -> None:
        payload = {k: v for k, v in _VALID_REQUEST.items() if k != "start_date"}
        resp = client.post(BASE_URL, json=payload)
        assert resp.status_code == 422

    def test_missing_end_date_returns_422(self, client: TestClient) -> None:
        payload = {k: v for k, v in _VALID_REQUEST.items() if k != "end_date"}
        resp = client.post(BASE_URL, json=payload)
        assert resp.status_code == 422

    def test_job_already_running_returns_409(self, client: TestClient) -> None:
        from app.services.jobs.background_job import JobAlreadyRunningError

        with patch(_JOB_SVC_CREATE, side_effect=JobAlreadyRunningError(_MOCK_JOB_ID)):
            resp = client.post(BASE_URL, json=_VALID_REQUEST)

        assert resp.status_code == 409

    def test_background_worker_started(self, client: TestClient) -> None:
        with (
            patch(_JOB_SVC_CREATE, return_value=_MOCK_JOB_ID),
            patch(_JOB_SVC_START) as mock_start,
        ):
            client.post(BASE_URL, json=_VALID_REQUEST)

        mock_start.assert_called_once()

    def test_optional_factor_config_accepted(self, client: TestClient) -> None:
        payload = {**_VALID_REQUEST, "factor_config": {"method": "ic_weighted"}}
        with (
            patch(_JOB_SVC_CREATE, return_value=_MOCK_JOB_ID),
            patch(_JOB_SVC_START),
        ):
            resp = client.post(BASE_URL, json=payload)

        assert resp.status_code == 202


# ===========================================================================
# GET /api/v1/factors/compute/{job_id}
# ===========================================================================


class TestGetFactorComputeJob:
    """GET /api/v1/factors/compute/{job_id} polls progress."""

    def test_existing_pending_job_returns_200(self, client: TestClient) -> None:
        mock_job = _pending_job(_MOCK_JOB_ID)
        with patch(_JOB_SVC_GET, return_value=mock_job):
            resp = client.get(f"{BASE_URL}/{_MOCK_JOB_ID}")

        assert resp.status_code == 200

    def test_existing_job_returns_job_id(self, client: TestClient) -> None:
        mock_job = _pending_job(_MOCK_JOB_ID)
        with patch(_JOB_SVC_GET, return_value=mock_job):
            resp = client.get(f"{BASE_URL}/{_MOCK_JOB_ID}")

        assert resp.json()["job_id"] == _MOCK_JOB_ID

    def test_existing_job_returns_status(self, client: TestClient) -> None:
        mock_job = {**_pending_job(_MOCK_JOB_ID), "status": "running"}
        with patch(_JOB_SVC_GET, return_value=mock_job):
            resp = client.get(f"{BASE_URL}/{_MOCK_JOB_ID}")

        assert resp.json()["status"] == "running"

    def test_existing_job_returns_progress_fields(self, client: TestClient) -> None:
        mock_job = {**_pending_job(_MOCK_JOB_ID), "current": 3, "total": 10}
        with patch(_JOB_SVC_GET, return_value=mock_job):
            resp = client.get(f"{BASE_URL}/{_MOCK_JOB_ID}")

        body = resp.json()
        assert body["current"] == 3
        assert body["total"] == 10

    def test_missing_job_returns_404(self, client: TestClient) -> None:
        with patch(_JOB_SVC_GET, return_value=None):
            resp = client.get(f"{BASE_URL}/{_MOCK_JOB_ID}")

        assert resp.status_code == 404

    def test_completed_job_returns_result(self, client: TestClient) -> None:
        mock_job = {
            **_pending_job(_MOCK_JOB_ID),
            "status": "completed",
            "current": 17,
            "total": 17,
            "result": {"rows_written": 17},
        }
        with patch(_JOB_SVC_GET, return_value=mock_job):
            resp = client.get(f"{BASE_URL}/{_MOCK_JOB_ID}")

        body = resp.json()
        assert body["status"] == "completed"
        assert body["result"] == {"rows_written": 17}

    def test_failed_job_returns_error(self, client: TestClient) -> None:
        mock_job = {
            **_pending_job(_MOCK_JOB_ID),
            "status": "failed",
            "error": "Fundamentals data unavailable",
        }
        with patch(_JOB_SVC_GET, return_value=mock_job):
            resp = client.get(f"{BASE_URL}/{_MOCK_JOB_ID}")

        body = resp.json()
        assert body["status"] == "failed"
        assert body["error"] == "Fundamentals data unavailable"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pending_job(job_id: str) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "status": "pending",
        "current": 0,
        "total": 0,
        "errors": [],
        "result": None,
        "error": None,
    }
