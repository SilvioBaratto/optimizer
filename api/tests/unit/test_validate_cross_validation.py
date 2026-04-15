"""Unit tests for POST /api/v1/validate/cross-validation and GET /api/v1/validate/cross-validation/{job_id}.

Covers:
  - POST: returns 202 + job_id (happy path)
  - POST: invalid cv_type returns 422
  - POST: empty tickers returns 422
  - POST: JobAlreadyRunningError returns 409
  - GET: existing job returns progress with current_fold / total_folds
  - GET: missing job returns 404
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

BASE_URL = "/api/v1/validate/cross-validation"

_JOB_SVC_CREATE = "app.api.v1.validate._job_service.create_job"
_JOB_SVC_START = "app.api.v1.validate._job_service.start_background"
_JOB_SVC_GET = "app.api.v1.validate._job_service.get_job"

_MOCK_JOB_ID = "00000000-0000-0000-0000-000000000099"

_WALK_FORWARD_REQUEST: dict[str, Any] = {
    "tickers": ["AAPL", "MSFT"],
    "start_date": "2020-01-01",
    "end_date": "2024-01-01",
    "cv_type": "walk_forward",
    "cv_config": {"test_size": 63, "train_size": 252},
    "optimizer_type": "hrp",
    "optimizer_config": {},
}

_CPCV_REQUEST: dict[str, Any] = {
    **_WALK_FORWARD_REQUEST,
    "cv_type": "cpcv",
    "cv_config": {"n_folds": 10, "n_test_folds": 2},
}

_MULTIPLE_RANDOMIZED_REQUEST: dict[str, Any] = {
    **_WALK_FORWARD_REQUEST,
    "cv_type": "multiple_randomized",
    "cv_config": {"n_subsamples": 5, "asset_subset_size": 2},
}


# ---------------------------------------------------------------------------
# POST /validate/cross-validation
# ---------------------------------------------------------------------------


class TestPostValidateCrossValidation:
    """POST /api/v1/validate/cross-validation starts a background job."""

    def test_walk_forward_returns_202(self, client: TestClient) -> None:
        with (
            patch(_JOB_SVC_CREATE, return_value=_MOCK_JOB_ID),
            patch(_JOB_SVC_START),
        ):
            resp = client.post(BASE_URL, json=_WALK_FORWARD_REQUEST)

        assert resp.status_code == 202

    def test_walk_forward_body_contains_job_id(self, client: TestClient) -> None:
        with (
            patch(_JOB_SVC_CREATE, return_value=_MOCK_JOB_ID),
            patch(_JOB_SVC_START),
        ):
            resp = client.post(BASE_URL, json=_WALK_FORWARD_REQUEST)

        assert resp.json()["job_id"] == _MOCK_JOB_ID

    def test_cpcv_cv_type_accepted(self, client: TestClient) -> None:
        with (
            patch(_JOB_SVC_CREATE, return_value=_MOCK_JOB_ID),
            patch(_JOB_SVC_START),
        ):
            resp = client.post(BASE_URL, json=_CPCV_REQUEST)

        assert resp.status_code == 202

    def test_multiple_randomized_cv_type_accepted(self, client: TestClient) -> None:
        with (
            patch(_JOB_SVC_CREATE, return_value=_MOCK_JOB_ID),
            patch(_JOB_SVC_START),
        ):
            resp = client.post(BASE_URL, json=_MULTIPLE_RANDOMIZED_REQUEST)

        assert resp.status_code == 202

    def test_invalid_cv_type_returns_422(self, client: TestClient) -> None:
        bad_request = {**_WALK_FORWARD_REQUEST, "cv_type": "not_a_real_cv"}
        resp = client.post(BASE_URL, json=bad_request)
        assert resp.status_code == 422

    def test_empty_tickers_returns_422(self, client: TestClient) -> None:
        resp = client.post(BASE_URL, json={**_WALK_FORWARD_REQUEST, "tickers": []})
        assert resp.status_code == 422

    def test_missing_tickers_returns_422(self, client: TestClient) -> None:
        payload = {k: v for k, v in _WALK_FORWARD_REQUEST.items() if k != "tickers"}
        resp = client.post(BASE_URL, json=payload)
        assert resp.status_code == 422

    def test_job_already_running_returns_409(self, client: TestClient) -> None:
        from app.services.background_job import JobAlreadyRunningError

        with patch(_JOB_SVC_CREATE, side_effect=JobAlreadyRunningError(_MOCK_JOB_ID)):
            resp = client.post(BASE_URL, json=_WALK_FORWARD_REQUEST)

        assert resp.status_code == 409

    def test_background_worker_started(self, client: TestClient) -> None:
        with (
            patch(_JOB_SVC_CREATE, return_value=_MOCK_JOB_ID),
            patch(_JOB_SVC_START) as mock_start,
        ):
            client.post(BASE_URL, json=_WALK_FORWARD_REQUEST)

        mock_start.assert_called_once()

    def test_response_contains_status_pending(self, client: TestClient) -> None:
        with (
            patch(_JOB_SVC_CREATE, return_value=_MOCK_JOB_ID),
            patch(_JOB_SVC_START),
        ):
            resp = client.post(BASE_URL, json=_WALK_FORWARD_REQUEST)

        assert resp.json()["status"] == "pending"


# ---------------------------------------------------------------------------
# GET /validate/cross-validation/{job_id}
# ---------------------------------------------------------------------------


class TestGetValidateCrossValidation:
    """GET /api/v1/validate/cross-validation/{job_id} polls job progress."""

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

    def test_progress_includes_current_fold(self, client: TestClient) -> None:
        mock_job = {**_pending_job(_MOCK_JOB_ID), "current_fold": 3, "total_folds": 10}
        with patch(_JOB_SVC_GET, return_value=mock_job):
            resp = client.get(f"{BASE_URL}/{_MOCK_JOB_ID}")

        body = resp.json()
        assert body["current_fold"] == 3

    def test_progress_includes_total_folds(self, client: TestClient) -> None:
        mock_job = {**_pending_job(_MOCK_JOB_ID), "current_fold": 3, "total_folds": 10}
        with patch(_JOB_SVC_GET, return_value=mock_job):
            resp = client.get(f"{BASE_URL}/{_MOCK_JOB_ID}")

        body = resp.json()
        assert body["total_folds"] == 10

    def test_missing_job_returns_404(self, client: TestClient) -> None:
        with patch(_JOB_SVC_GET, return_value=None):
            resp = client.get(f"{BASE_URL}/{_MOCK_JOB_ID}")

        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pending_job(job_id: str) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "status": "pending",
        "current": 0,
        "total": 0,
        "current_fold": 0,
        "total_folds": 0,
        "errors": [],
        "result": None,
        "error": None,
    }
