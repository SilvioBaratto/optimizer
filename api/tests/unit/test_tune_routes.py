"""Unit tests for POST /api/v1/tune and GET /api/v1/tune/{job_id}.

Covers:
  - POST: returns 202 + job_id
  - POST: start_background called once
  - POST: JobAlreadyRunningError → 409
  - POST: unknown optimizer_type → 422
  - POST: empty tickers → 422
  - POST: empty param_grid → 422
  - POST: invalid top_n → 422
  - POST: invalid n_iter → 422
  - POST: search_type "grid" and "randomized" both accepted → 202
  - GET: existing job returns 200 + AsyncJobProgress
  - GET: unknown job returns 404

BackgroundJobService is patched at module level per CLAUDE.md gotcha.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

BASE_URL = "/api/v1/tune"

_JOB_SVC_CREATE = "app.api.v1.tune._tune_job_service.create_job"
_JOB_SVC_START = "app.api.v1.tune._tune_job_service.start_background"
_JOB_SVC_GET = "app.api.v1.tune._tune_job_service.get_job"

_MOCK_JOB_ID = "00000000-0000-0000-0000-000000000002"

_VALID_REQUEST: dict[str, Any] = {
    "tickers": ["AAPL", "MSFT"],
    "start_date": "2020-01-01",
    "end_date": "2024-01-01",
    "optimizer_type": "mean_risk",
    "param_grid": {"optimizer__risk_measure": ["variance", "cvar"]},
    "search_type": "grid",
    "top_n": 3,
    "n_iter": 10,
}


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


# ===========================================================================
# POST /api/v1/tune
# ===========================================================================


class TestPostTune:
    """POST /api/v1/tune always launches a background job."""

    def test_post_tune_returns_202_with_job_id(self, client: TestClient) -> None:
        with (
            patch(_JOB_SVC_CREATE, return_value=_MOCK_JOB_ID),
            patch(_JOB_SVC_START),
        ):
            resp = client.post(BASE_URL, json=_VALID_REQUEST)

        assert resp.status_code == 202
        assert resp.json()["job_id"] == _MOCK_JOB_ID

    def test_post_tune_starts_background_job(self, client: TestClient) -> None:
        with (
            patch(_JOB_SVC_CREATE, return_value=_MOCK_JOB_ID),
            patch(_JOB_SVC_START) as mock_start,
        ):
            client.post(BASE_URL, json=_VALID_REQUEST)

        mock_start.assert_called_once()

    def test_job_already_running_returns_409(self, client: TestClient) -> None:
        from app.services.background_job import JobAlreadyRunningError

        with patch(_JOB_SVC_CREATE, side_effect=JobAlreadyRunningError(_MOCK_JOB_ID)):
            resp = client.post(BASE_URL, json=_VALID_REQUEST)

        assert resp.status_code == 409

    def test_unknown_optimizer_type_returns_422(self, client: TestClient) -> None:
        bad_request = {**_VALID_REQUEST, "optimizer_type": "bad_type"}

        resp = client.post(BASE_URL, json=bad_request)

        assert resp.status_code == 422

    def test_empty_tickers_returns_422(self, client: TestClient) -> None:
        resp = client.post(BASE_URL, json={**_VALID_REQUEST, "tickers": []})

        assert resp.status_code == 422

    def test_empty_param_grid_returns_422(self, client: TestClient) -> None:
        resp = client.post(BASE_URL, json={**_VALID_REQUEST, "param_grid": {}})

        assert resp.status_code == 422

    def test_invalid_top_n_returns_422(self, client: TestClient) -> None:
        resp = client.post(BASE_URL, json={**_VALID_REQUEST, "top_n": 0})

        assert resp.status_code == 422

    def test_invalid_n_iter_returns_422(self, client: TestClient) -> None:
        resp = client.post(BASE_URL, json={**_VALID_REQUEST, "n_iter": 0})

        assert resp.status_code == 422

    def test_both_search_types_accepted_grid(self, client: TestClient) -> None:
        with (
            patch(_JOB_SVC_CREATE, return_value=_MOCK_JOB_ID),
            patch(_JOB_SVC_START),
        ):
            resp = client.post(BASE_URL, json={**_VALID_REQUEST, "search_type": "grid"})

        assert resp.status_code == 202

    def test_both_search_types_accepted_randomized(self, client: TestClient) -> None:
        with (
            patch(_JOB_SVC_CREATE, return_value=_MOCK_JOB_ID),
            patch(_JOB_SVC_START),
        ):
            resp = client.post(
                BASE_URL, json={**_VALID_REQUEST, "search_type": "randomized"}
            )

        assert resp.status_code == 202


# ===========================================================================
# GET /api/v1/tune/{job_id}
# ===========================================================================


class TestGetTuneJob:
    """GET /api/v1/tune/{job_id} polls progress for a background job."""

    def test_get_job_returns_progress(self, client: TestClient) -> None:
        mock_job = _pending_job(_MOCK_JOB_ID)
        with patch(_JOB_SVC_GET, return_value=mock_job):
            resp = client.get(f"{BASE_URL}/{_MOCK_JOB_ID}")

        assert resp.status_code == 200
        assert resp.json()["job_id"] == _MOCK_JOB_ID
        assert resp.json()["status"] == "pending"

    def test_get_unknown_job_returns_404(self, client: TestClient) -> None:
        with patch(_JOB_SVC_GET, return_value=None):
            resp = client.get(f"{BASE_URL}/{_MOCK_JOB_ID}")

        assert resp.status_code == 404
