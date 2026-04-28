"""Unit tests for POST /api/v1/optimize and GET /api/v1/optimize/{run_id}.

Covers:
  - Sync path (≤ threshold tickers): returns 200 + OptimizationRunResponse
  - Async path (> threshold tickers): returns 202 + job_id
  - Invalid optimizer_type: returns 422
  - JobAlreadyRunningError: returns 409
  - GET existing run: returns 200
  - GET missing run: returns 404

All optimizer library calls and BackgroundJobService are mocked.
BackgroundJobService is patched at module level per CLAUDE.md gotcha.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

BASE_URL = "/api/v1/optimize"

_SYNC_TICKERS = ["AAPL", "MSFT"]  # 2 tickers → sync path
_ASYNC_TICKERS = [f"TICK{i}" for i in range(60)]  # 60 → async path (> default 50)

_VALID_REQUEST: dict[str, Any] = {
    "tickers": _SYNC_TICKERS,
    "start_date": "2023-01-01",
    "end_date": "2024-01-01",
    "optimizer_type": "mean_risk",
    "config": {},
}

# Module-level patch targets per CLAUDE.md
_JOB_SVC_CREATE = "app.api.v1.optimize._job_service.create_job"
_JOB_SVC_START = "app.api.v1.optimize._job_service.start_background"
_SERVICE_BUILD = "app.api.v1.optimize.build_pipeline"
_SERVICE_EXTRACT = "app.api.v1.optimize.extract_results"


def _mock_service_calls(mock_pipeline: MagicMock | None = None):
    """Context manager patching: build_pipeline, extract_results."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        mock_pipe = mock_pipeline or MagicMock()
        returns_df = MagicMock()
        result = {
            "weights": {"AAPL": 0.6, "MSFT": 0.4},
            "metrics": {"annualized_sharpe_ratio": 1.5},
            "risk_contributions": {"AAPL": 0.4, "MSFT": 0.6},
            "efficient_frontier": None,
        }
        with (
            patch(_SERVICE_BUILD, return_value=(mock_pipe, returns_df)),
            patch(_SERVICE_EXTRACT, return_value=result),
        ):
            yield

    return _ctx()


# ===========================================================================
# POST /api/v1/optimize — sync path
# ===========================================================================


class TestPostOptimizeSync:
    """When tickers ≤ threshold, optimization runs synchronously and returns 200."""

    def test_sync_path_returns_200(self, client: TestClient) -> None:
        with _mock_service_calls():
            resp = client.post(BASE_URL, json=_VALID_REQUEST)

        assert resp.status_code == 200

    def test_sync_path_returns_weights(self, client: TestClient) -> None:
        with _mock_service_calls():
            resp = client.post(BASE_URL, json=_VALID_REQUEST)

        body = resp.json()
        assert "weights" in body
        assert body["weights"] == {"AAPL": pytest.approx(0.6), "MSFT": pytest.approx(0.4)}

    def test_sync_path_persists_run(self, client: TestClient) -> None:
        """Optimization run is written to the database."""
        with _mock_service_calls():
            resp = client.post(BASE_URL, json=_VALID_REQUEST)

        assert resp.status_code == 200
        body = resp.json()
        assert "id" in body

    def test_sync_path_status_is_completed(self, client: TestClient) -> None:
        with _mock_service_calls():
            resp = client.post(BASE_URL, json=_VALID_REQUEST)

        assert resp.json()["status"] == "completed"


# ===========================================================================
# POST /api/v1/optimize — async path
# ===========================================================================


class TestPostOptimizeAsync:
    """When tickers > threshold, job is created and 202 returned."""

    _ASYNC_REQUEST = {**_VALID_REQUEST, "tickers": _ASYNC_TICKERS}

    def test_async_path_returns_202(self, client: TestClient) -> None:
        job_id = str(uuid.uuid4())
        with (
            patch(_JOB_SVC_CREATE, return_value=job_id),
            patch(_JOB_SVC_START),
        ):
            resp = client.post(BASE_URL, json=self._ASYNC_REQUEST)

        assert resp.status_code == 202

    def test_async_path_returns_job_id(self, client: TestClient) -> None:
        job_id = str(uuid.uuid4())
        with (
            patch(_JOB_SVC_CREATE, return_value=job_id),
            patch(_JOB_SVC_START),
        ):
            resp = client.post(BASE_URL, json=self._ASYNC_REQUEST)

        assert resp.json()["job_id"] == job_id

    def test_async_path_starts_background_thread(self, client: TestClient) -> None:
        job_id = str(uuid.uuid4())
        with (
            patch(_JOB_SVC_CREATE, return_value=job_id),
            patch(_JOB_SVC_START) as mock_start,
        ):
            client.post(BASE_URL, json=self._ASYNC_REQUEST)

        mock_start.assert_called_once()


# ===========================================================================
# POST /api/v1/optimize — invalid optimizer_type
# ===========================================================================


class TestPostOptimizeInvalidType:
    """Unknown optimizer_type raises ValueError → 422."""

    def test_invalid_optimizer_type_returns_422(self, client: TestClient) -> None:
        # Route validates optimizer_type via _validate_optimizer_type before calling service
        bad_request = {**_VALID_REQUEST, "optimizer_type": "not_a_real_optimizer"}

        resp = client.post(BASE_URL, json=bad_request)

        assert resp.status_code == 422

    def test_error_message_mentions_optimizer_type(self, client: TestClient) -> None:
        bad_request = {**_VALID_REQUEST, "optimizer_type": "not_a_real_optimizer"}

        resp = client.post(BASE_URL, json=bad_request)

        assert resp.status_code == 422
        assert "optimizer_type" in str(resp.json())


# ===========================================================================
# POST /api/v1/optimize — JobAlreadyRunningError → 409
# ===========================================================================


class TestPostOptimizeJobAlreadyRunning:
    """When BackgroundJobService raises JobAlreadyRunningError, returns 409."""

    _ASYNC_REQUEST = {**_VALID_REQUEST, "tickers": _ASYNC_TICKERS}

    def test_job_already_running_returns_409(self, client: TestClient) -> None:
        from app.services.background_job import JobAlreadyRunningError

        with patch(
            _JOB_SVC_CREATE,
            side_effect=JobAlreadyRunningError("existing-job-id"),
        ):
            resp = client.post(BASE_URL, json=self._ASYNC_REQUEST)

        assert resp.status_code == 409

    def test_409_body_contains_error_message(self, client: TestClient) -> None:
        from app.services.background_job import JobAlreadyRunningError

        with patch(
            _JOB_SVC_CREATE,
            side_effect=JobAlreadyRunningError("existing-job-id"),
        ):
            resp = client.post(BASE_URL, json=self._ASYNC_REQUEST)

        # App error handler may wrap in {"error": ...} or {"detail": ...}
        body_str = str(resp.json())
        assert "existing-job-id" in body_str or "in progress" in body_str


# ===========================================================================
# GET /api/v1/optimize/{run_id}
# ===========================================================================


class TestGetOptimizeRun:
    """GET /api/v1/optimize/{run_id} retrieves a completed run or returns 404."""

    def test_existing_run_returns_200(self, client: TestClient, db_session: Session) -> None:
        """A run that exists in the DB is returned with 200."""
        from app.repositories.execution_repository import ExecutionRepository

        repo = ExecutionRepository(db_session)
        run = repo.create_optimization_run({
            "status": "completed",
            "optimizer_type": "mean_risk",
            "universe_tickers": ["AAPL", "MSFT"],
            "config": {},
            "weights": {"AAPL": 0.6, "MSFT": 0.4},
            "metrics": {"annualized_sharpe_ratio": 1.5},
            "risk_contributions": {"AAPL": 0.4, "MSFT": 0.6},
        })
        db_session.commit()

        resp = client.get(f"{BASE_URL}/{run.id}")

        assert resp.status_code == 200

    def test_existing_run_returns_correct_id(
        self, client: TestClient, db_session: Session
    ) -> None:
        from app.repositories.execution_repository import ExecutionRepository

        repo = ExecutionRepository(db_session)
        run = repo.create_optimization_run({
            "status": "completed",
            "optimizer_type": "hrp",
            "universe_tickers": ["AAPL"],
            "config": {},
            "weights": {"AAPL": 1.0},
            "metrics": {},
            "risk_contributions": {"AAPL": 1.0},
        })
        db_session.commit()

        resp = client.get(f"{BASE_URL}/{run.id}")

        assert resp.json()["id"] == str(run.id)

    def test_missing_run_returns_404(self, client: TestClient) -> None:
        """A non-existent run_id returns 404."""
        missing_id = str(uuid.uuid4())

        resp = client.get(f"{BASE_URL}/{missing_id}")

        assert resp.status_code == 404

    def test_invalid_uuid_returns_422(self, client: TestClient) -> None:
        """An invalid UUID string returns 422."""
        resp = client.get(f"{BASE_URL}/not-a-uuid")

        assert resp.status_code == 422
