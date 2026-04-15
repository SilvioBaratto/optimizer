"""Unit tests for POST /api/v1/reports/generate and related endpoints.

TDD: tests written before implementation.

Covers:
  - POST /reports/generate returns 202 + job_id
  - Request validation: empty sections → 422
  - Request validation: unknown section IDs → 422
  - Request validation: invalid template → 422
  - GET /reports/jobs/{job_id} returns job progress
  - GET /reports/jobs/{job_id} missing job → 404
  - GET /reports/{report_id}/download returns file bytes with correct headers
  - GET /reports/{report_id}/download missing report → 404

Note on background job mocking
--------------------------------
The module-level ``_report_job_service`` in ``app.api.v1.reports`` uses
``database_manager.get_session`` and bypasses FastAPI DI.  All job service
methods must be mocked at the module level (same pattern as test_backtest_routes.py).
"""

from __future__ import annotations

import os
import tempfile
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

BASE_URL = "/api/v1/reports"

_JOB_SVC_CREATE = "app.api.v1.reports._report_job_service.create_job"
_JOB_SVC_START = "app.api.v1.reports._report_job_service.start_background"
_JOB_SVC_GET = "app.api.v1.reports._report_job_service.get_job"

_MOCK_JOB_ID = "00000000-0000-0000-0000-000000000099"

_VALID_REQUEST: dict[str, Any] = {
    "template": "standard",
    "sections": ["portfolio_summary", "weights"],
    "branding": {
        "company_name": "Acme Capital",
        "primary_color": "#003366",
        "include_disclaimer": True,
    },
    "format": "pdf",
    "orientation": "portrait",
}


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


def _completed_job(job_id: str, report_id: str) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "status": "completed",
        "current": 1,
        "total": 1,
        "errors": [],
        "result": {
            "download_url": f"/api/v1/reports/{report_id}/download",
            "filename": f"report_{report_id}.pdf",
            "report_id": report_id,
        },
        "error": None,
    }


# ---------------------------------------------------------------------------
# POST /reports/generate
# ---------------------------------------------------------------------------


class TestPostGenerateReport:
    """POST /api/v1/reports/generate returns 202 and job_id."""

    def test_valid_request_returns_202(self, client: TestClient) -> None:
        with (
            patch(_JOB_SVC_CREATE, return_value=_MOCK_JOB_ID),
            patch(_JOB_SVC_START),
        ):
            resp = client.post(f"{BASE_URL}/generate", json=_VALID_REQUEST)

        assert resp.status_code == 202

    def test_valid_request_returns_job_id(self, client: TestClient) -> None:
        with (
            patch(_JOB_SVC_CREATE, return_value=_MOCK_JOB_ID),
            patch(_JOB_SVC_START),
        ):
            resp = client.post(f"{BASE_URL}/generate", json=_VALID_REQUEST)

        body = resp.json()
        assert body["job_id"] == _MOCK_JOB_ID

    def test_valid_request_returns_status_pending(self, client: TestClient) -> None:
        with (
            patch(_JOB_SVC_CREATE, return_value=_MOCK_JOB_ID),
            patch(_JOB_SVC_START),
        ):
            resp = client.post(f"{BASE_URL}/generate", json=_VALID_REQUEST)

        body = resp.json()
        assert body["status"] == "pending"

    def test_background_worker_is_started(self, client: TestClient) -> None:
        with (
            patch(_JOB_SVC_CREATE, return_value=_MOCK_JOB_ID),
            patch(_JOB_SVC_START) as mock_start,
        ):
            client.post(f"{BASE_URL}/generate", json=_VALID_REQUEST)

        mock_start.assert_called_once()

    def test_job_already_running_returns_409(self, client: TestClient) -> None:
        from app.services.background_job import JobAlreadyRunningError

        with patch(_JOB_SVC_CREATE, side_effect=JobAlreadyRunningError(_MOCK_JOB_ID)):
            resp = client.post(f"{BASE_URL}/generate", json=_VALID_REQUEST)

        assert resp.status_code == 409

    def test_all_valid_sections_accepted(self, client: TestClient) -> None:
        payload = {
            **_VALID_REQUEST,
            "sections": [
                "portfolio_summary",
                "weights",
                "performance_metrics",
                "risk_analytics",
            ],
        }
        with (
            patch(_JOB_SVC_CREATE, return_value=_MOCK_JOB_ID),
            patch(_JOB_SVC_START),
        ):
            resp = client.post(f"{BASE_URL}/generate", json=payload)

        assert resp.status_code == 202


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


class TestPostGenerateValidation:
    """POST /api/v1/reports/generate validates the request body."""

    def test_empty_sections_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            f"{BASE_URL}/generate", json={**_VALID_REQUEST, "sections": []}
        )
        assert resp.status_code == 422

    def test_unknown_section_id_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            f"{BASE_URL}/generate",
            json={**_VALID_REQUEST, "sections": ["portfolio_summary", "nonexistent_section"]},
        )
        assert resp.status_code == 422

    def test_invalid_template_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            f"{BASE_URL}/generate",
            json={**_VALID_REQUEST, "template": "fantasy_template"},
        )
        assert resp.status_code == 422

    def test_invalid_format_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            f"{BASE_URL}/generate",
            json={**_VALID_REQUEST, "format": "xlsx"},
        )
        assert resp.status_code == 422

    def test_invalid_orientation_returns_422(self, client: TestClient) -> None:
        resp = client.post(
            f"{BASE_URL}/generate",
            json={**_VALID_REQUEST, "orientation": "diagonal"},
        )
        assert resp.status_code == 422

    def test_missing_sections_field_returns_422(self, client: TestClient) -> None:
        payload = {k: v for k, v in _VALID_REQUEST.items() if k != "sections"}
        resp = client.post(f"{BASE_URL}/generate", json=payload)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /reports/jobs/{job_id}
# ---------------------------------------------------------------------------


class TestGetReportJobStatus:
    """GET /api/v1/reports/jobs/{job_id} polls job progress."""

    def test_pending_job_returns_200(self, client: TestClient) -> None:
        mock_job = _pending_job(_MOCK_JOB_ID)
        with patch(_JOB_SVC_GET, return_value=mock_job):
            resp = client.get(f"{BASE_URL}/jobs/{_MOCK_JOB_ID}")

        assert resp.status_code == 200

    def test_pending_job_returns_job_id(self, client: TestClient) -> None:
        mock_job = _pending_job(_MOCK_JOB_ID)
        with patch(_JOB_SVC_GET, return_value=mock_job):
            resp = client.get(f"{BASE_URL}/jobs/{_MOCK_JOB_ID}")

        assert resp.json()["job_id"] == _MOCK_JOB_ID

    def test_pending_job_returns_status(self, client: TestClient) -> None:
        mock_job = _pending_job(_MOCK_JOB_ID)
        with patch(_JOB_SVC_GET, return_value=mock_job):
            resp = client.get(f"{BASE_URL}/jobs/{_MOCK_JOB_ID}")

        assert resp.json()["status"] == "pending"

    def test_completed_job_includes_result(self, client: TestClient) -> None:
        report_id = "abc123"
        mock_job = _completed_job(_MOCK_JOB_ID, report_id)
        with patch(_JOB_SVC_GET, return_value=mock_job):
            resp = client.get(f"{BASE_URL}/jobs/{_MOCK_JOB_ID}")

        body = resp.json()
        assert body["result"] is not None
        assert "download_url" in body["result"]

    def test_missing_job_returns_404(self, client: TestClient) -> None:
        with patch(_JOB_SVC_GET, return_value=None):
            resp = client.get(f"{BASE_URL}/jobs/{_MOCK_JOB_ID}")

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /reports/{report_id}/download
# ---------------------------------------------------------------------------


class TestGetReportDownload:
    """GET /api/v1/reports/{report_id}/download serves the PDF file."""

    def test_existing_report_returns_200(self, client: TestClient) -> None:
        report_id = "test-report-001"
        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False, prefix=f"report_{report_id}_"
        ) as tmp:
            tmp.write(b"%PDF-1.4 fake content")
            tmp_path = tmp.name

        try:
            with patch(
                "app.api.v1.reports._find_report_path",
                return_value=tmp_path,
            ):
                resp = client.get(f"{BASE_URL}/{report_id}/download")

            assert resp.status_code == 200
        finally:
            os.unlink(tmp_path)

    def test_existing_report_has_pdf_content_type(self, client: TestClient) -> None:
        report_id = "test-report-002"
        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False, prefix=f"report_{report_id}_"
        ) as tmp:
            tmp.write(b"%PDF-1.4 fake content")
            tmp_path = tmp.name

        try:
            with patch(
                "app.api.v1.reports._find_report_path",
                return_value=tmp_path,
            ):
                resp = client.get(f"{BASE_URL}/{report_id}/download")

            assert "application/pdf" in resp.headers["content-type"]
        finally:
            os.unlink(tmp_path)

    def test_existing_report_has_content_disposition_header(
        self, client: TestClient
    ) -> None:
        report_id = "test-report-003"
        with tempfile.NamedTemporaryFile(
            suffix=".pdf", delete=False, prefix=f"report_{report_id}_"
        ) as tmp:
            tmp.write(b"%PDF-1.4 fake content")
            tmp_path = tmp.name

        try:
            with patch(
                "app.api.v1.reports._find_report_path",
                return_value=tmp_path,
            ):
                resp = client.get(f"{BASE_URL}/{report_id}/download")

            assert "content-disposition" in resp.headers
            assert "attachment" in resp.headers["content-disposition"]
        finally:
            os.unlink(tmp_path)

    def test_missing_report_returns_404(self, client: TestClient) -> None:
        with patch(
            "app.api.v1.reports._find_report_path",
            return_value=None,
        ):
            resp = client.get(f"{BASE_URL}/nonexistent-report-id/download")

        assert resp.status_code == 404
